from datetime import timedelta
from fractions import Fraction
import numpy as np
from ctypes import byref

from ..ffmpeg.libavcodec import BitStreamFilter, BSFContext
from ..ffmpeg.libavformat import FormatContext, AVMediaType, AVCodecID
from ..ffmpeg.include.libavutil  import AV_NOPTS_VALUE, AV_TIME_BASE, AVColorRange, AVColorSpace
from .compat import av2cuda, cuda2av
from ..core.decode import Decoder
from ..utils.color import Converter


from ..core import cuda

import logging

log = logging.getLogger(__name__)

class Screenshot:
    def __init__(self, url, device = None):
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
            self.duration = self.fc.av.duration
            if self.duration == AV_NOPTS_VALUE:
                log.warning('cannot infer duration')
                self.duration = None
            else:
                self.duration = int(self.duration / AV_TIME_BASE / self.time_base)

        codec_id = self.stream.codecpar.contents.codec_id
#    codec_id = stream.codecpar.contents.codec_id
        if codec_id == AVCodecID.HEVC:
            f = BitStreamFilter('hevc_mp4toannexb')
        elif codec_id == AVCodecID.H264:
            f = BitStreamFilter('h264_mp4toannexb') 
        else:
            raise Exception(f'unsupported codec {codec_id}')                

        self.bsf = BSFContext(f, self.stream.codecpar.contents, self.stream.time_base)

        self.device = cuda.get_current_device(device)
        with cuda.Device(self.device):
            self.decoder = Decoder(av2cuda(self.stream.codecpar.contents.codec_id))
        self.cvt = None

    @property
    def time_base(self):
        return Fraction(self.stream.time_base.num, self.stream.time_base.den)

    @property
    def width(self):
        return self.stream.codecpar.contents.width

    @property
    def height(self):
        return self.stream.codecpar.contents.height

    def color_space(self, default = AVColorSpace.UNSPECIFIED):
        r = self.stream.codecpar.contents.color_space
        if r == AVColorSpace.UNSPECIFIED:
            return default
        else:
            return r
    
    def color_range(self, default = AVColorRange.UNSPECIFIED):
        r = self.stream.codecpar.contents.color_range
        if r == AVColorRange.UNSPECIFIED:
            return default
        else:
            return r

    def shoot(self, target : int | timedelta, array, cuda_stream : int = 2):

        if isinstance(target, timedelta):
            target_pts = int(target.total_seconds() / self.time_base) + self.start_time
        elif isinstance(target, int):
            target_pts = target
        else:
            raise Exception(f'unsupported target type {type(target)}')
        log.warning(f'target_pts: {target_pts}')
        self.fc.seek_file(self.stream, target_pts)

        last = None

        def demux():
            for pkt in self.bsf.filter(self.fc.read_frames(self.stream)):
                pts = pkt.av.pts
                log.warning(f'filtered: dts={pkt.av.dts} pts = {pkt.av.pts}')
                # log.warning(pts)
                arr = np.ctypeslib.as_array(pkt.av.data, (pkt.av.size,))
                yield arr, pts     

        for pic, pts in self.decoder.decode(demux()):
            log.warning(f'decoded: {pts}')
            # log.warning(f'{pts}')
            if pts > target_pts:
                break
            last = pic
        surface = last.map(cuda_stream)
        if self.cvt is None:
            with cuda.Device(self.device):                
                self.cvt = Converter(
                    surface, 
                    cuda2av(surface.format),
                    self.color_space(AVColorSpace.BT470BG),
                    self.color_range(AVColorRange.MPEG),
                    array
                    )
                    
        self.cvt(surface, array, stream = cuda_stream)