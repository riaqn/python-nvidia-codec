from datetime import timedelta
from fractions import Fraction
from queue import Queue
import numpy as np

from ..ffmpeg.libavcodec import BSFContext
from ..ffmpeg.libavformat import FormatContext, AVMediaType, AVCodecID
from ..ffmpeg.include.libavutil  import AV_NOPTS_VALUE, AV_TIME_BASE, AVColorRange, AVColorSpace
from .compat import av2cuda, cuda2av, extract_stream_ptr
from ..core.decode import BaseDecoder
from .color import Converter

import torch

import logging

log = logging.getLogger(__name__)

class Player:
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

        self.device = torch.cuda.current_device() if device is None else device
        def decide(p):
            return {
                'num_pictures': p['min_num_pictures'], # to be safe
                'num_surfaces': p['min_num_pictures'],
                # will use default surface_format
                # will use default cropping (no cropping)
                'target_size': target_size,
                # will use default target rect (no margin)
            }
        self.surface_q = Queue()          
        self.cvt = None
        self.target_typestr = target_typestr

        def on_recv(pic, pts):
            if pic is None:
                self.surface_q.put(None)
                return
            stream = extract_stream_ptr(torch.cuda.current_stream())
            surface = pic.map(stream)
            ev = torch.cuda.Event()
            ev.record()
            self.surface_q.put((pts, ev, surface))
            pic.free()            
        self.decoder = BaseDecoder(av2cuda(self.stream.codecpar.contents.codec_id), on_recv, decide = decide, device = self.device)

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

    def seek(self, target : timedelta, flush = True):
        target_pts = self.time2pts(target)
        log.debug(f'target_pts: {target_pts}')
        self.fc.seek_file(self.stream, target_pts, max_ts = target_pts)
        if flush:
            self.bsf.flush()
            self.decoder.flush()
            # wait until EOS signal feedback
            for _ in self.surfaces():
                pass

        # why no flush? because the above might not jump at all in some cases

    def time2pts(self, time: timedelta):
        return int(time.total_seconds() / self._time_base) + self._start_time

    def pts2time(self, pts : int):
        return timedelta(seconds = float((pts - self._start_time) * self._time_base))

    '''
    convert a surface 
    '''
    def convert(self, surface, target = None):
        if target is None:
            with torch.cuda.device(self.device):   
                    shape = Converter.infer_target(surface.shape, cuda2av(surface.format))                
                    m = {
                        '|u1': torch.uint8,
                        '<f2': torch.float16,
                        '<f4': torch.float32,
                    }
                    target = torch.empty(shape, dtype = m[self.target_typestr], device = 'cuda')

        if self.cvt is None:
            with torch.cuda.device(self.device):
                self.cvt = Converter(
                    surface, 
                    cuda2av(surface.format),
                    self.color_space(AVColorSpace.BT470BG),
                    self.color_range(AVColorRange.MPEG),
                    target,
                    self.target_typestr
                    )
        stream = extract_stream_ptr(torch.cuda.current_stream())
        self.cvt(surface, target, stream = stream)
        return target

    def surfaces(self):
        it = self.bsf.filter(self.fc.read_packets(self.stream), flush = False, reuse = True)
        
        while True:
            while self.surface_q.empty():           
                try:
                    pkt = next(it) # this might trigger StopIteration
                except StopIteration:
                    arr = None
                    pts = 0
                else:
                    pts = pkt.av.pts
                    arr = np.ctypeslib.as_array(pkt.av.data, (pkt.av.size,))
                self.decoder.send(arr, pts)
            p = self.surface_q.get()
            if p is None:
                break 
            pts, ev, surface = p
            yield (self.pts2time(pts), ev, surface)

    def frames(self, dst = None):
        for time, ev, surface in self.surfaces():
            ev.wait()
            frame = self.convert(surface, dst)
            surface.free()
            yield (time, frame)

    def screenshoot(self, target : timedelta, dst = None):
        self.seek(target)

        last = None
        for time, ev, surface in self.surfaces():
            if target < time:
                surface.free() # free the current surface
                time, ev, surface = last # get the last surface
                ev.wait()
                frame = self.convert(surface, dst)
                surface.free()
                return (time, frame)
            last = (time, ev, surface)

        ev.wait()
        frame = self.convert(surface, dst)
        surface.free()
        return (time, frame)

    def free(self):
        self.decoder.free()