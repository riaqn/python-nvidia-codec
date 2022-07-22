# adapters for FFMPEG
# relies directly on the C library
# not pyav

from ctypes import *
from fractions import Fraction

from .include.libavutil import *
from .include.libavcodec import *
from .include.libavformat import *
from .libavcodec import Packet

from .common import AVException, check


import logging
log = logging.getLogger(__name__)

lib = cdll.LoadLibrary('libavformat.so')


class FormatContext:
    def __init__(self, url : str):
        ptr = POINTER(AVFormatContext)()
        url = c_char_p(url.encode('utf-8'))
        check(lib.avformat_open_input(byref(ptr), url, None, None))
        self.av = ptr.contents
    
    def read_packet(self, pkt : Packet):
        check(lib.av_read_frame(byref(self.av), byref(pkt.av)))

    def read_packets(self, stream : AVStream):
        while True:
            pkt = Packet()
            try:
                self.read_packet(pkt) # we own the packet
            except AVException as e:
                if e.errnum == AVERROR_EOF:
                    break
                raise

            if pkt.av.stream_index == stream.index:
                log.debug(f'demuxed dts={pkt.av.dts} pts={pkt.av.pts}')
                yield pkt # ownership transferred outside

    def seek_file(self, stream : AVStream, ts : int, min_ts = -(2**63), max_ts = (2**63) - 1):
        check(lib.avformat_seek_file( byref(self.av), c_int(stream.index), c_int64(min_ts), c_int64(ts), c_int64(max_ts), c_int(0)))

    def infer_start_time(self):
        # log.warning('infering start time from first packet')
                # in this case, we get the first packet
        pkt = Packet()
        self.read_packet(pkt)
        start_time = pkt.av.pts
        time_base = pkt.av.time_base
        return start_time * Fraction(time_base.num, time_base.den)

    @property
    def streams(self):
        for i in range(self.av.nb_streams):
            yield self.av.streams[i].contents