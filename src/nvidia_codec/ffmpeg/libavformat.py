# adapters for FFMPEG
# relies directly on the C library
# not pyav

from ctypes import *

from .include.libavutil import *
from .include.libavcodec import *
from .include.libavformat import *
from .libavcodec import Packet

from .common import call

import logging
log = logging.getLogger(__name__)

lib = cdll.LoadLibrary('libavformat.so')


class FormatContext:
    def __init__(self, url : str):
        ptr = POINTER(AVFormatContext)()
        url = c_char_p(url.encode('utf-8'))
        call(lib.avformat_open_input, byref(ptr), url, None, None)
        self.av = ptr.contents
    
    def read_frame(self, pkt : Packet):
        call(lib.av_read_frame, byref(self.av), byref(pkt.av))

    def read_frames(self, stream : AVStream):
        pkt = Packet()
        while True:
            self.read_frame(pkt)
            # log.warning(f'{pkt.av.pts}')
            if pkt.av.stream_index == stream.index:
                yield pkt

    def seek_file(self, stream : AVStream, ts : int, min_ts = -(2**63), max_ts = (2**63) - 1):
        call(lib.avformat_seek_file, byref(self.av), c_int(stream.index), c_int64(min_ts), c_int64(ts), c_int64(max_ts), c_int(0))

    @property
    def streams(self):
        for i in range(self.av.nb_streams):
            yield self.av.streams[i].contents