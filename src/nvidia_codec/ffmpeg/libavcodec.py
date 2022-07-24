from ctypes import *
from .common import *
from .include.libavutil import *
from .include.libavcodec  import *

from errno import EAGAIN

import  logging
log = logging.getLogger(__name__)

lib = cdll.LoadLibrary('libavcodec.so')


av_packet_alloc = lib.av_packet_alloc
av_packet_alloc.restype = POINTER(AVPacket)

# def packet_alloc():
#     p = av_packet_alloc()    
#     return cast(p, POINTER(AVPacket)).contents

# def packet_free(pkt):
#     p = pointer(pkt)
#     lib.av_packet_free(byref(p))

# def packet_unref(pkt):
#     lib.av_packet_unref(byref(pkt))

class Packet:
    def __init__(self):
        ptr = av_packet_alloc()
        self.av = ptr.contents

    def unref(self):
        lib.av_packet_unref(byref(self.av))

    def __del__(self):
        lib.av_packet_free(byref(pointer(self.av)))

class BSFContext:
    def __init__(self, filters : str, codecpar : AVCodecParameters, time_base : AVRational):

        ptr = POINTER(AVBSFContext)()
        if filters is None:
            check(lib.av_bsf_get_null_filter(byref(ptr)))
        else:
            check(lib.av_bsf_list_parse_str(c_char_p(filters.encode('utf-8')), byref(ptr)))

        self.av = ptr.contents
        f = lib.avcodec_parameters_alloc
        f.restype = POINTER(AVCodecParameters)
        self.av.par_in = f()
        check(lib.avcodec_parameters_copy(self.av.par_in, byref(codecpar)))
        self.av.time_base_in = time_base
        check(lib.av_bsf_init(byref(self.av)))

    def __del__(self):
        lib.av_bsf_free(byref(pointer(self.av)))

    def flush(self):
        lib.av_bsf_flush(byref(self.av))

    def filter(self, packets, flush = True, reuse = False):
        if flush:
            self.flush()

        pkt_out = Packet()
        while True:
            while True:
                try:
                    self.receive_packet(pkt_out)# we still own it
                except AVException as e:
                    if e.errnum == AVERROR_EOF:
                        # no more to see here
                        # probably as a result of a previous EOS we sent
                        log.debug('bsf signal EOF')
                        return
                    elif e.errnum == AVERROR(EAGAIN):    
                        # need input, break
                        break
                    else:
                        raise
                yield pkt_out 
                if reuse:
                    pkt_out.unref()
                else:
                    pkt_out = Packet()

            try:
                pkt = next(packets)
            except StopIteration:
                pkt = None
            # we must faithfully filter the stream
            # including End-of-stream
            self.send_packet(pkt)

    # None means EOF
    def send_packet(self, pkt : Packet = None):
        check(lib.av_bsf_send_packet(byref(self.av), byref(pkt.av) if pkt is not None else None))

    def receive_packet(self, pkt : Packet):
        check(lib.av_bsf_receive_packet(byref(self.av), byref(pkt.av)))

