import io
import av

class StreamTranslate:
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
        self.bsf = av.BitStreamFilterContext(bsf_name)        

        self.out_container = av.open(self.b, 'wb')
        self.out_stream = self.out_container.add_stream(template=stream)            
        self.out_container.start_encoding()

    '''
    convert a pyav packet to iterator of (pts, bytes) where bytes is annex B packet that can be passed to decoder

    bytes instead of memoryview, because then the packet can 
    '''
    def translate_packet(self, packet):
        if packet.dts is None:
            # pyav generate NULL packet to flush we don't need
            return # empty iterator
        for bs in self.bsf(packet):
            #bs.stream = self.out_stream
            self.b.seek(0)
            self.b.truncate()
                                    
            self.out_container.mux_one(bs)
            self.b.flush()
            yield (bs.pts, self.b.getvalue())