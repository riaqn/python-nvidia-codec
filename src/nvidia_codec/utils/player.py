from datetime import timedelta
from fractions import Fraction
from queue import Queue
import numpy as np

from ..ffmpeg.libavcodec import BSFContext
from ..ffmpeg.libavformat import FormatContext, AVMediaType, AVCodecID
from ..ffmpeg.include.libavutil  import AV_NOPTS_VALUE, AV_TIME_BASE, AVColorRange, AVColorSpace
from .compat import av2cuda, cuda2av, extract_stream_ptr
from ..core.decode import BaseDecoder
from .color import convert

import torch

import logging

log = logging.getLogger(__name__)

class Player:
    def __init__(self, url, target_size = lambda h,w: (h,w), device = None, num_surfaces = 2):
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
                'num_surfaces': num_surfaces,
                # will use default surface_format
                # will use default cropping (no cropping)
                'target_size': target_size,
                # will use default target rect (no margin)
            }
        self.surface_q = Queue()          


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

    def seek(self, target : timedelta):
        target_pts = self.time2pts(target)
        log.debug(f'target_pts: {target_pts}')
        self.fc.seek_file(self.stream, target_pts, max_ts = target_pts)

        self.bsf.flush()
        self.decoder.flush()

        for surface in self._surfaces():
            surface.free()

    def time2pts(self, time: timedelta):
        return int(time.total_seconds() / self._time_base) + self._start_time

    def pts2time(self, pts : int):
        return timedelta(seconds = float((int(pts) - int(self._start_time)) * self._time_base))


    def _surfaces(self):
        it = self.bsf.filter(self.fc.read_packets(self.stream), flush = False, reuse = True)
        
        while True:
            while self.surface_q.empty():           
                try:
                    pkt = next(it) # this might trigger StopIteration
                except StopIteration:
                    # actual end of file
                    arr = None
                    pts = 0
                else:
                    pts = pkt.av.pts
                    arr = np.ctypeslib.as_array(pkt.av.data, (pkt.av.size,))
                self.decoder.send(arr, pts)
            p = self.surface_q.get()
            if p is None:
                # reset
                break 
            pts, ev, surface = p
            yield (self.pts2time(pts), ev, surface)

    def frames(self, target_dtype):
        for time, ev, surface in self._surfaces():
            ev.wait()
            frame = convert(surface, self.color_space(AVColorSpace.BT470BG), self.color_range(AVColorRange.MPEG), target_dtype)
            surface.free()
            yield (time, frame)

    def screenshoot(self, target : timedelta, dtype : torch.dtype):
        self.seek(target)

        last = None
        for time, ev, surface in self._surfaces():
            if target < time:
                surface.free() # free the current surface
                time, ev, surface = last # get the last surface
                ev.wait()
                frame = convert(surface, self.color_space(AVColorSpace.BT470BG), self.color_range(AVColorRange.MPEG), dtype)
                surface.free()
                return (time, frame)
            last = (time, ev, surface)

        ev.wait()
        frame = convert(surface, self.color_space(AVColorSpace.BT470BG), self.color_range(AVColorRange.MPEG), dtype)
        surface.free()
        return (time, frame)

    def free(self):
        self.decoder.free()