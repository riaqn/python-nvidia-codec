from datetime import timedelta
from fractions import Fraction
import numpy as np

from ..ffmpeg.libavcodec import BitStreamFilter, BSFContext
from ..ffmpeg.libavformat import FormatContext, AVMediaType, AVCodecID
from ..ffmpeg.include.libavutil  import AV_NOPTS_VALUE, AV_TIME_BASE, AVColorRange, AVColorSpace, AVPixelFormat
from .compat import av2cuda, cuda2av
from ..core.decode import Decoder
from ..utils.color import Converter

from ..core import cuda

import logging

log = logging.getLogger(__name__)

class Screenshot:
    def __init__(self, url, target_size = lambda h,w: (h,w), target_typestr = '|u1', device = None):
        self.fc = FormatContext(url)
        l = list(filter(lambda s: s.codecpar.contents.codec_type == AVMediaType.VIDEO, self.fc.streams))

        assert len(l) > 0, 'file has no video stream'
        if len(l) > 1:
            log.warning('file has multiple video streams, picking the first one')
        self.stream = l[0] 

        self.start_time = self.stream.start_time

        if self.start_time == AV_NOPTS_VALUE:
            self.start_time = self.fc.av.start_time
            if self.start_time == AV_NOPTS_VALUE:
                start_time, time_base = self.fc.infer_start_time()
                self.start_time = int(start_time * time_base / self.time_base)
            else:
                self.start_time = int(self.start_time / AV_TIME_BASE / self.time_base)                

        self.duration = self.stream.duration
        if self.duration == AV_NOPTS_VALUE:
            # if stream duration is unknown,
            # get the whole file duration
            self.duration = self.fc.av.duration
            if self.duration == AV_NOPTS_VALUE:
                log.warning('cannot infer duration')
                self.duration = None
            else:
                # remember to convert to stream' time base
                self.duration = int(self.duration / AV_TIME_BASE / self.time_base)

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
                'num_pictures': p['min_num_pictures'],
                'num_surfaces': 1, # only need one 
                # will use default surface_format
                # will use default cropping (no cropping)
                'target_size': target_size,
                # will use default target rect (no margin)
            }
        self.decoder = Decoder(av2cuda(self.stream.codecpar.contents.codec_id), decide = decide, device = self.device)
        self.cvt = None
        self.target_typestr = target_typestr

    @property
    def time_base(self):
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
    def length(self):
        return timedelta(seconds = float(self.duration * self.time_base))

    def color_space(self, default = AVColorSpace.UNSPECIFIED):
        r = self.stream.codecpar.contents.color_space
        if r == AVColorSpace.UNSPECIFIED:
            log.warning(f'color space is unspecified, using {default}')
            return default
        else:
            log.info(f'color space is {r}')
            return r
    
    def color_range(self, default = AVColorRange.UNSPECIFIED):
        r = self.stream.codecpar.contents.color_range
        if r == AVColorRange.UNSPECIFIED:
            log.warning(f'color range is unspecified, using {default}')
            return default
        else:
            log.info(f'color range is {r}')
            return r

    def shoot(self, time : int | timedelta, target = 'cupy', stream : int = 2):

        if isinstance(time, timedelta):
            target_pts = int(time.total_seconds() / self.time_base) + self.start_time
        elif isinstance(time, int):
            target_pts = time
        else:
            raise Exception(f'unsupported target type {type(time)}')
        log.debug(f'target_pts: {target_pts}')
        self.fc.seek_file(self.stream, target_pts)

        def demux():
            for pkt in self.bsf.filter(self.fc.read_frames(self.stream)):
                pts = pkt.av.pts
                log.debug(f'filtered: dts={pkt.av.dts} pts = {pkt.av.pts}')
                # log.warning(pts)
                arr = np.ctypeslib.as_array(pkt.av.data, (pkt.av.size,))
                yield arr, pts     

        surface = None
        for pic, pts in self.decoder.decode(demux()):
            print(pic.index)
            if pts > target_pts:
                break
            del surface
            surface = pic.map(stream)
            del pic
            log.debug(f'decoded: {pts}')
            # log.warning(f'{pts}')


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