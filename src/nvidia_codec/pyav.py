import io
import av

from .cuviddec import cudaVideoCodec


'''
Adapts PyAV stream for CUVID decoding
'''
class PyAVStreamAdaptor:
    def __init__(self, stream):
        self.stream = stream
        self.b = io.BytesIO()            
        if stream.codec.name == 'h264':
            bsf_name = 'h264_mp4toannexb'
            self.b.name = 'muxed.h264'    
        elif stream.codec.name == 'hevc':
            bsf_name = 'hevc_mp4toannexb'
            self.b.name = 'muxed.hevc'
        else:
            raise Exception(f'unsupported codec {stream.codec.name} for bitstream')
        # the following line relies on certain patch of PyAV
        # https://github.com/PyAV-Org/PyAV/tree/bitstream
        self.bsf = av.BitStreamFilterContext(bsf_name)        

        self.out_container = av.open(self.b, 'wb')
        self.out_stream = self.out_container.add_stream(template=stream)            
        self.out_container.start_encoding()

    def translate_codec(self):
        if self.stream.codec.name == 'h264':
            return cudaVideoCodec.H264
        elif self.stream.codec.name == 'hevc':
            return cudaVideoCodec.HEVC
        else:
            raise Exception('codec not supported')

    '''
    convert a pyav packet to an iterator of (pts, bytes) where bytes is annex B packet that can be passed to decoder

    if copy is True, return a copy that the caller can use as long as it wants
    if copy is False, the underlying buffer is returned directly; 
    the caller must drop the returned reference before the next iteration
    '''
    def translate_packet(self, packet, copy = True):
        if packet.dts is None:
            # pyav generate NULL packet to flush we don't need
            return # empty iterator
        for bs in self.bsf(packet):
            #bs.stream = self.out_stream
            self.b.seek(0)
            self.b.truncate()
                                    
            self.out_container.mux_one(bs)
            self.b.flush()
            buf = self.b.getbuffer()
            yield (self.b.getvalue() if copy else self.b.getbuffer(), bs.pts)

    '''
    convert an iterator of pyav packets to an iterator of (pts, bytes) 
    where bytes is annex B packet that can be passed to decoder
    See translate_packet for copy parameter
    '''
    def translate_packets(self, packets, copy = True):
        for packet in packets:
            yield from self.translate_packet(packet, copy = copy)

def test():
    pass

if __name__ == '__main__':
    test()