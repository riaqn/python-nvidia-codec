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

class BitStreamFilter:
    def __init__(self, name):
        name = c_char_p(name.encode('utf-8'))
        func = lib.av_bsf_get_by_name
        func.restype = POINTER(AVBitStreamFilter)
        ptr = func(name)
        self.av = ptr.contents

class Packet:
    def __init__(self):
        ptr = av_packet_alloc()
        self._av = ptr.contents
        self.own = True

    @property
    def av(self):
        assert self.own, 'packet is not owned by us'
        return self._av

    def disown(self):
        assert self.own, 'packet is not owned by us'
        self.own = False

    def unref(self):
        lib.av_packet_unref(byref(self.av))

    def __del__(self):
        if self.own:
            lib.av_packet_free(byref(pointer(self._av)))

class BSFContext:
    def __init__(self, filter : BitStreamFilter, codecpar : AVCodecParameters, time_base : AVRational):
        ptr = POINTER(AVBSFContext)()
        call(lib.av_bsf_alloc, byref(filter.av), byref(ptr))

        self.av = ptr.contents
        self.av.par_in = pointer(codecpar)
        self.av.time_base_in = time_base
        call(lib.av_bsf_init, byref(self.av))

    def __del__(self):
        lib.av_bsf_free(byref(pointer(self.av)))

    def filter(self, packets, reuse = False):
        # always flush for the first time
        call(lib.av_bsf_flush, byref(self.av))

        # packets = peekable(packets)
        pkt_out = Packet()
        for pkt in packets:
            self.send_packet(pkt) 

            while True:
                try:
                    self.receive_packet(pkt_out)# we still own it
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
                yield pkt_out 
                if reuse:
                    pkt_out.unref()
                else:
                    pkt_out = Packet()

                # after the yield, the other side should be done with the packet
                
    # None means EOF
    def send_packet(self, pkt : Packet = None):
        call(lib.av_bsf_send_packet, byref(self.av), byref(pkt.av) if pkt is not None else None)
        # in case of exception, the following is not called
        pkt.disown()
            
        

    def receive_packet(self, pkt : Packet):
        call(lib.av_bsf_receive_packet, byref(self.av), byref(pkt.av))

