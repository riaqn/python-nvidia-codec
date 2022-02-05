import io
import av

from .cuviddec import cudaVideoCodec

class PyAVStreamAdaptor:
    """Adapter class for PyAV video stream to work with CUVID decoder
    """
    def __init__(self, stream):
        """
        Args:
            stream (av.VideoStream): PyAV video stream to be wrapped

        Raises:
            Exception: if codec is not supported for bitstream
        """        
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
        """return the cuvid codec enum of the wrapped video stream

        Raises:
            Exception: if codec is not supported

        Returns:
            cudaVideoCodec: the codec enum
        """        

        if self.stream.codec.name == 'h264':
            return cudaVideoCodec.H264
        elif self.stream.codec.name == 'hevc':
            return cudaVideoCodec.HEVC
        else:
            raise Exception('codec not supported')

    def translate_packet(self, packet, copy = True):
        """Convert a pyav packet to an iterator of (pts, packet) 
        where the latter is annex.B packet that can be passed to decoder

        Args:
            packet (bytes): [description]
            copy (bool, optional): If True, copy of the underlying buffer is returned; 
            the copy is owned by the user. If False, the underlying buffer is returned directly, 
            and the caller must drop the returned reference (by `del`) before the next iteration. 
            Defaults to True.


        Yields:
            (int, bytes-like object): PTS and annex.B packet
        """
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

    def translate_packets(self, packets, copy = True):
        """ Plural version of translate_packet,
            converts an iterator of pyav packets to iterator of (pts, packet)
        """        
        for packet in packets:
            yield from self.translate_packet(packet, copy = copy)

    def __del__(self):
        try:
            self.out_container.close()
        except TypeError:
            # to omit the weird error message
            pass

        self.b.close()
