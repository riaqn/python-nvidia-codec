from ctypes import *
from .common import *
from .include.libavutil import *
from .include.libavcodec  import *

from errno import EAGAIN

import  logging
log = logging.getLogger(__name__)

lib = cdll.LoadLibrary('libavcodec.so')

class Packet:
    def __init__(self):
        av_packet_alloc = lib.av_packet_alloc
        av_packet_alloc.restype = POINTER(AVPacket)            
        self.av = av_packet_alloc().contents

class BitStreamFilter:
    def __init__(self, name):
        name = c_char_p(name.encode('utf-8'))
        func = lib.av_bsf_get_by_name
        func.restype = POINTER(AVBitStreamFilter)
        ptr = func(name)
        self.av = ptr.contents


class BSFContext:
    def __init__(self, filter : BitStreamFilter, codecpar : AVCodecParameters, time_base : AVRational):
        ptr = POINTER(AVBSFContext)()
        call(lib.av_bsf_alloc, byref(filter.av), byref(ptr))

        self.av = ptr.contents
        self.av.par_in = pointer(codecpar)
        self.av.time_base_in = time_base
        call(lib.av_bsf_init, byref(self.av))

    def filter(self, packets):
        # always flush for the first time
        call(lib.av_bsf_flush, byref(self.av))

        # packets = peekable(packets)

        pkt_out = Packet()
        while True:
            while True:
                try:
                    self.receive_packet(pkt_out)
                    yield pkt_out
                    # log.warning('bsf received')
                except AVException as e:
                    if e.errnum == AVERROR_EOF:
                        # no more to see here
                        log.warning('bsf output EOF')
                        return
                    elif e.errnum == AVERROR(EAGAIN):    
                        # need input, break
                        break
                    else:
                        raise
            # only reason we are here is because we need input
            # therefore the following cannot return eagain
            try:
                self.send_packet(next(packets))
                # log.warning('bsf sent')
            except StopIteration:               
                log.warning('bsf input EOF')
                self.send_packet(None)

    # None means EOF
    def send_packet(self, pkt : Packet = None):
        call(lib.av_bsf_send_packet, byref(self.av), byref(pkt.av) if pkt is not None else None)

    def receive_packet(self, pkt : Packet):
        call(lib.av_bsf_receive_packet, byref(self.av), byref(pkt.av))

