from datetime import timedelta
from fractions import Fraction
import numpy as np

from ..ffmpeg.libavcodec import BSFContext
from ..ffmpeg.libavformat import FormatContext, AVMediaType, AVCodecID
from ..ffmpeg.include.libavutil  import AV_NOPTS_VALUE, AV_TIME_BASE, AVColorRange, AVColorSpace
from .compat import av2cuda, cuda2av
from ..core.decode import BaseDecoder
from ..utils.color import Converter

from ..core import cuda

import logging

log = logging.getLogger(__name__)

class Screenshot:
    def __init__(self, url, target_size = lambda h,w: (h,w), target_typestr = '|u1', device = None):
        self.fc = FormatContext(url)
        l = filter(lambda s: s.codecpar.contents.codec_type == AVMediaType.VIDEO, self.fc.streams)

        l = sorted(l, key = lambda s: s.codecpar.contents.bit_rate, reverse = True)

        assert len(l) > 0, 'file has no video stream'
        self.stream = l[0]         
        if len(l) > 1:
            log.warning(f'{url} has multiple video streams, picking the highest bitrate @ {self.stream.codecpar.contents.bit_rate}')

        self._start_time = self.stream.start_time
        if self._start_time == AV_NOPTS_VALUE:
            self._start_time = self.fc.av.start_time
            if self._start_time == AV_NOPTS_VALUE:
                start_time = self.fc.infer_start_time()
                self._start_time = int(start_time / self._time_base)
            else:
                self._start_time = int(self._start_time / AV_TIME_BASE / self._time_base)                

        self._duration = self.stream.duration
        if self._duration == AV_NOPTS_VALUE:
            # if stream duration is unknown,
            # get the whole file duration
            self._duration = self.fc.av.duration
            if self._duration == AV_NOPTS_VALUE:
                log.warning('cannot infer duration')
                self._duration = None
            else:
                # remember to convert to stream' time base
                self._duration = int(self._duration / AV_TIME_BASE / self._time_base)

        codec_id = self.stream.codecpar.contents.codec_id
        if codec_id == AVCodecID.HEVC:
            f = 'hevc_mp4toannexb'
        elif codec_id == AVCodecID.H264:
            f = 'h264_mp4toannexb'
        else:
            f = None
            # raise Exception(f'unsupported codec {codec_id}')                

        self.bsf = BSFContext(f, self.stream.codecpar.contents, self.stream.time_base)

        self.device = cuda.get_current_device(device)
        def decide(p):
            return {
                'num_pictures': p['min_num_pictures'], # to be safe
                'num_surfaces': 1, # only need one 
                # will use default surface_format
                # will use default cropping (no cropping)
                'target_size': target_size,
                # will use default target rect (no margin)
            }
        self.decoder = BaseDecoder(av2cuda(self.stream.codecpar.contents.codec_id), None, decide = decide, device = self.device)
        self.cvt = None
        self.target_typestr = target_typestr

    @property
    def _time_base(self):
        return Fraction(self.stream.time_base.num, self.stream.time_base.den)

    @property
    def width(self):
        return self.decoder.width

    @property
    def height(self):
        return self.decoder.height

    @property
    def target_width(self):
        return self.decoder.target_width

    @property
    def target_height(self):
        return self.decoder.target_height

    @property
    def duration(self):
        return timedelta(seconds = float(self._duration * self._time_base))

    def color_space(self, default = AVColorSpace.UNSPECIFIED):
        r = self.stream.codecpar.contents.color_space
        if r == AVColorSpace.UNSPECIFIED:
            log.debug(f'color space is unspecified, using {default}')
            return default
        else:
            log.debug(f'color space is {r}')
            return r
    
    def color_range(self, default = AVColorRange.UNSPECIFIED):
        r = self.stream.codecpar.contents.color_range
        if r == AVColorRange.UNSPECIFIED:
            log.debug(f'color range is unspecified, using {default}')
            return default
        else:
            log.debug(f'color range is {r}')
            return r

    '''
    convert the current surface 
    '''
    def convert(self, surface, target = 'cupy', stream : int = 0):
        if isinstance(target, str):
            shape = Converter.infer_target(surface.shape, cuda2av(surface.format))
            with cuda.Device(self.device):   
                if target == 'cupy':
                    import cupy            
                    target = cupy.empty(shape, dtype = self.target_typestr)        
                elif target == 'torch':
                    import torch
                    m = {
                        '|u1': torch.uint8,
                        '<f2': torch.float16,
                        '<f4': torch.float32,
                    }
                    target = torch.empty(shape, dtype = m[self.target_typestr], device = 'cuda') 
                else:
                    raise Exception(f'unsupported target {target}')

        if self.cvt is None:
            with cuda.Device(self.device):                
                self.cvt = Converter(
                    surface, 
                    cuda2av(surface.format),
                    self.color_space(AVColorSpace.BT470BG),
                    self.color_range(AVColorRange.MPEG),
                    target,
                    self.target_typestr
                    )
                    
        self.cvt(surface, target, stream = stream)
        return target


    def shoot(self, target : timedelta, dst = 'cupy', stream : int = 0):
        target_pts = int(target.total_seconds() / self._time_base) + self._start_time
        log.debug(f'target_pts: {target_pts}')
        self.fc.seek_file(self.stream, target_pts, max_ts = target_pts)

        found = False
        act_pts = None

        def on_recv(pic, pts):
            nonlocal dst
            nonlocal act_pts
            nonlocal found
            if pts >= target_pts:
                surface = pic.map(stream)
                pic.free()                                
                act_pts = pts
                dst = self.convert(surface, dst, stream = stream)
                surface.free()
                found = True

        self.decoder.on_recv = on_recv

        for pkt in self.bsf.filter(self.fc.read_frames(self.stream), flush = False, reuse = True):
            pts = pkt.av.pts
            log.debug(f'filtered: dts={pkt.av.dts} pts = {pkt.av.pts}')
                # log.warning(pts)
            arr = np.ctypeslib.as_array(pkt.av.data, (pkt.av.size,))
            self.decoder.send(arr, pts)
            if found:
                break
        if not found:
            raise Exception(f'{target} is too late')


        act = timedelta(seconds = float((act_pts - self._start_time) * self._time_base))
        # if abs(act - target) > timedelta(seconds = 0.1):
        #     log.warning(f'actual time {act} is not close to target time {target}')
        return act, dst

    def free(self):
        self.decoder.free()