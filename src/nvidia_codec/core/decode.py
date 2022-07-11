'''
User sends packets to decoder and receives Pictures from the decoder. Picture contains the data of a decoded frame. However, it is not accessible to the user. To access the data, user has to call Picture.map(). This copies and post-processes (cropping, scaling and padding) the Picture into a Surface, which can be accessed as a regular CUDA array. It is up to the user whether or not to call map() for a particular Picture. For example, when screenshoting a video, only a particular subset of frames are needed as output; the user can therefore choose to only call map() on pictures of those frames for faster processing.

User owns the Picture and Surfaces; the data they represent will be valid as long as the user holds references to them. When all references are dropped, the graphics resources will  be freed. User can also call Picture.free() and Surface.free() to free the resource early. There are limits on the number of Pictures and Surfaces a user can hold at any given time. The limits are specified by `num_pictures` and `num_surfaces` when creating the decoder, upon which that much of graphics memory is pre-allocated even before user gets any Pictures/Surfaces. All Pictures/Surfaces are backed by those pre-allocated memory. Freeing Pictures/Surfaces marks those memory reusable for future Pictures/Surfaces - still not reusable to other CUDA program. By default the bare minimum required of `num_pictures` and `num_surfaces` is specified. The user can override the limits, for following reasons:

Increase the number of pictures might speed up the decoding. User can also increase the number to establish a look-back cache. Take screenshotting as an example again. Say we want the frame at PTS=350. We start decoding the video and get Picture of PTS=0, 100, 200, 300. We have to keep going, because PTS=300 might not be the target frame since there could be a following frame of PTS=350. So we keep going and get Picture of PTS=400. Now we know that indeed PTS=300 is the target frame. Fortunately we keep a variable `old` always storing the last Picture, and we will just use `old` in this case: map it to Surface and read out its data. For this to work, we must set `num_pictures` to be `min_num_pictures + 1`, since we are using one extra picture slot for this `old` picture. This is in fact the default settings since it is a common scenario. `num_pictures` has an upper limit of `32`.

Similarly one might increase the `num_surfaces` if their applications can make use of it. `num_surfaces` does not have upper limit as long as there is sufficient graphics memory. Note that again they are allocated on creation of decoder. Freeing `Surface` and `Picture` only marked the slots as reusable by the decoder for future frames, while the corresponding memory are still owned by the decoder and not usable by other CUDA operations. 
'''
from . import cuda
from .nvcuvid import *
from .common import *
from queue import Queue
import numpy as np
from ctypes import *
from threading import Condition

import logging
log = logging.getLogger(__name__)

nvcuvid = cdll.LoadLibrary('libnvcuvid.so')

class Surface:
    '''
    A CUDA array owned by the decoder to store post-processed frames
    '''
    def __init__(self, decoder, c_devptr, c_pitch, stream):
        """
        DO NOT call this by yourself; use Picture.map() instead
        """
        self.decoder = decoder
        self.c_pitch = c_pitch
        self.c_devptr = c_devptr
        self.stream = stream

    @property
    def format(self):
        """the format of the surface

        Raises:
            ValueError: surface format not supported

        Returns:
            cudaVideoSurfaceFormat : format of the surface
        """        
        return self.decoder.surface_format

    @property
    def height(self):
        """
        Returns:
            int: height of the surface
        """        
        return self.decoder.target_height

    @property
    def width(self):
        """
        Returns:
            int: width of the surface
        """        
        return self.decoder.target_width

    @property
    def size(self):
        """
        Returns:
            dict: size of the surface, with keys 'width' and 'height'
        """        
        return {'width':self.width, 'height':self.height}

    def free(self):
        """free up the surface. 
        Note that the GPU memory involved is managed by decoder and will not be available to other CUDA operations
        free() only notifies the decoder that the surface can be reused for future frames

        free() is called automatically when the surface is garbage collected
        """        

        if self.c_devptr is None:
            return
        
        with self.decoder.surfaces_cond:
            self.decoder.surfaces_to_unmap.add(self.c_devptr.value)
            self.c_devptr = None
            self.decoder.surfaces_cond.notify_all()

    def __del__(self):
        log.debug(f'trying to unmap {self.c_devptr}')
        self.free()

    @property
    def __cuda_array_interface__(self):
        format = self.format
        if format == cudaVideoSurfaceFormat.NV12:
            assert self.height % 2 == 0
            shape = (self.height //2 * 3, self.width)
            typestr = '|u1'
            strides = (self.c_pitch.value, 1)
        elif format == cudaVideoSurfaceFormat.P016:
            assert self.height % 2 == 0
            shape = (self.height //2 * 3, self.width)
            typestr = '<u2'
            strides = (self.c_pitch.value, 2)
        elif format == cudaVideoSurfaceFormat.YUV444:
            shape = (3, self.height, self.width)
            typestr = '|u1'
            strides = (self.c_pitch.value * self.height, self.c_pitch.value, 1)
        elif format == cudaVideoSurfaceFormat.YUV444_16Bit:
            shape = (3, self.height, self.width)
            typestr = '<u2'
            strides = (self.c_pitch.value * self.height, self.c_pitch.value, 2)
        else:
            raise ValueError(f'unsupported format {format}')
        return {
            'shape': shape,
            'typestr': typestr,
            'strides': strides,
            'version': 3,
            'data': (self.c_devptr.value, False), # false = not read-only
            'stream': self.stream,
        }        
        
class Picture:
    '''
    Frame after decoding but before post-processing
    Its data is not accessible at all, but still takes up GPU memory
    User must call Picture.map() to get a Surface for the data
    '''
    def __init__(self, decoder, params):
        '''
        DO NOT call this by yourself; use Decoder.decode() instead
        '''
        self.decoder = decoder        
        self.index = params.CurrPicIdx
        with self.decoder.pictures_cond:
            # wait until picture slot is available again
            log.debug('wait_for_pictures started')
            self.decoder.pictures_cond.wait_for(lambda:self.index not in self.decoder.pictures_used)
            log.debug('wait_for_pictures finished')
            self.decoder.pictures_used.add(self.index)
            log.debug(f'picture {self.index} added to used')

        cuda.call(nvcuvid.cuvidDecodePicture, self.decoder.decoder, byref(params))

    def on_display(self, params):
        self.params = params

    def free(self):
        '''
        free up the picture.
        Note that the GPU memory involved is managed by decoder and will not be available to other CUDA operations
        free() only notifies the decoder that the picture can be reused to store future pictures

        free() is called automatically when the picture is garbage collected        
        '''
        if self.index is None:
            return
        with self.decoder.pictures_cond:
            self.decoder.pictures_used.remove(self.index)
            self.index = None # the index is released and therefore invalid
            self.decoder.pictures_cond.notify_all()    
    
    def __del__(self):
        self.free()

    def map(self, stream : int = 2):
        """post-process and output this picture to a surface which can be accessed by user as a CUDA array

        Args:
            stream (optional, int): CUDA stream to queue this map operation. 
            Defaults to 2 which means per-thread default stream

        Returns:
            Surface : Surface mapped by this picture
        """
        self.params.stream = stream
        assert self.params.stream != 0, 'must speicfy legacy default stream or per-thread default stream'

        with self.decoder.surfaces_cond:
            if self.decoder.surfaces_avail == 0:
                log.debug('wait_for surface started')
                self.decoder.surfaces_cond.wait_for(lambda: len(self.decoder.surfaces_to_unmap) > 0)
                log.debug('wait_for surface finished')
                # we now make sure we at least have one surface to unmap
                try:
                    while True:
                        devptr = self.decoder.surfaces_to_unmap.pop()
                        cuda.call(nvcuvid.cuvidUnmapVideoFrame64, self.decoder.decoder, c_ulonglong(devptr))
                        self.decoder.surfaces_avail += 1
                except KeyError:
                    pass

            c_devptr = c_ulonglong() # according to cuviddec, the argument type of cuvidmapvideoframe64
            c_pitch = c_uint()
            cuda.call(nvcuvid.cuvidMapVideoFrame64, self.decoder.decoder, self.index, byref(c_devptr), byref(c_pitch), byref(self.params))
            self.decoder.surfaces_avail -= 1

            surface = Surface(self.decoder, c_devptr, c_pitch, stream)
            return surface

def decide_surface_format(chroma_format, bit_depth, supported_surface_formats, allow_high = False):
    """decide the appropriate surface format for a given chroma format and bit depth

    Args:
        chroma_format (cudaVideoChromaFormat): chroma format of the video
        bit_depth (int): bit depth per color channel, usually 8
        supported_surface_formats (set(cudaVideoSurfaceFormat)): set of supported surface formats
        allow_high (bool, optional): If False, will use 8bit output even if the video is higher than 8bit. Defaults to False.

    Raises:
        Exception: if chroma format is not supported
        Exception: if no surface format is supported

    Returns:
        cudaVideoSurfaceFormat: the selected surface format
    """    
    if chroma_format in [cudaVideoChromaFormat.YUV420, cudaVideoChromaFormat.MONOCHROME]:
        f = cudaVideoSurfaceFormat.P016 if bit_depth > 8 and allow_high else cudaVideoSurfaceFormat.NV12
    elif chroma_format == cudaVideoChromaFormat.YUV444:
        f = cudaVideoSurfaceFormat.YUV444_16Bit if bit_depth > 8 and allow_high else cudaVideoSurfaceFormat.YUV444
    elif chroma_format == cudaVideoChromaFormat.YUV422:
        f = cudaVideoSurfaceFormat.NV12
    else:
        raise Exception(f'unexpected chroma format {chroma_format}')
    
    # check if the selected format is supported. If not, check fallback options
    if f not in supported_surface_formats:
        if cudaVideoSurfaceFormat.NV12 in supported_surface_formats:
            f = cudaVideoSurfaceFormat.NV12
        elif cudaVideoSurfaceFormat.P016 in supported_surface_formats and allow_high:
            f = cudaVideoSurfaceFormat.P016
        elif cudaVideoSurfaceFormat.YUV444 in supported_surface_formats:
            f = cudaVideoSurfaceFormat.YUV444
        elif cudaVideoSurfaceFormat.YUV444_16Bit in supported_surface_formats and allow_high:
            f = cudaVideoSurfaceFormat.YUV444_16Bit
        else:
            raise Exception("No supported surface format")

    return f

class BaseDecoder:
    """Decoder with send/on-recv paradigm. That is, the user registers a callback which will be called upon decoded frames,
    and keep sending packets to the decoder.
    """
    def catch_exception(self, func, return_on_error = 0):
        '''
        wrapper for callbacks, so they return the error code as expected by cuvid; 
        '''
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except BaseException as e: # catch keyboard interrupt as well
                log.debug(f'callback exception logged {e}')
                # we record the exception; to be checked when nvcuvid propagate the error return code
                self.exception = e
                log.debug(f'returning error code {return_on_error}')
                return return_on_error
        return wrapper

    def handleVideoSequence(self, pUserData, pVideoFormat):
        log.debug('sequence')
        vf = cast(pVideoFormat, POINTER(CUVIDEOFORMAT)).contents
        if self.decoder:
            def cmp(a,b):
                if type(a) is not type(b):
                    log.debug(f'{type(a)} {type(b)}')
                    return False
                for k,_ in a._fields_:
                    log.debug(f'checking {k}')
                    va = getattr(a, k) 
                    vb = getattr(b, k)
                    if isinstance(va, Structure) and isinstance(vb, Structure):
                        if not cmp(va, vb):
                            log.debug(f'{va} not equal to {vb}')
                            return False
                    elif va != vb:
                        log.debug(f'{va} not equal to {vb}')
                        return False
                return True

            # initlized before
            if cmp(vf, self.video_format):
                return self.decode_create_info.ulNumDecodeSurfaces
            else:
                raise Exception("decoder already initialized, please create a new decoder for new video sequence")


        caps = CUVIDDECODECAPS(
            eCodecType=vf.codec,
            eChromaFormat=vf.chroma_format,
            nBitDepthMinus8=vf.bit_depth_luma_minus8
        )

        cuda.call(nvcuvid.cuvidGetDecoderCaps, byref(caps))
        assert caps.bIsSupported == 1, "Codec not supported"
        assert vf.coded_width <= caps.nMaxWidth, "width too large"
        assert vf.coded_height <= caps.nMaxHeight, "height too large"
        assert vf.coded_width >= caps.nMinWidth, "width too small"
        assert vf.coded_height >= caps.nMinHeight, "height too small"
        assert (vf.coded_width >> 4) * (vf.coded_height >> 4) <= caps.nMaxMBCount, "too many macroblocks"
        
        supported_surface_formats = []
        for surface_format in range(4): # cudaVideoSurfaceFormat:
            if caps.nOutputFormatMask & (1 << surface_format):
                supported_surface_formats.append(cudaVideoSurfaceFormat(surface_format))

        p = {
            'chroma_format': cudaVideoChromaFormat(vf.chroma_format),
            'bit_depth' : vf.bit_depth_luma_minus8 + 8,
            'size' : {
                'width' : vf.display_area.right - vf.display_area.left,
                'height' : vf.display_area.bottom - vf.display_area.top
            },
            'supported_surface_formats' : supported_surface_formats,
            'min_num_pictures' : vf.min_num_decode_surfaces,
        }

        # the default values 
        decision = {
            'num_pictures' : p['min_num_pictures'] + 1, # for simplicity
            'num_surfaces' : 1 + 1, # for simplicity
            'surface_format' : decide_surface_format(p['chroma_format'], p['bit_depth'], p['supported_surface_formats']),
            'cropping' : {
                'left': 0,
                'right' : p['size']['width'],
                'top': 0,
                'bottom' : p['size']['height']
            },
            'target_size': {
                'width' : p['size']['width'],
                'height' : p['size']['height']
            },
            'target_rect': {
                'left': 0,
                'right' : 0,
                'top': 0,
                'bottom' : 0
            },
        }

        decision |= self.decide(p)

        assert decision['num_pictures'] <= 32, f"number of pictures {decision['num_pictures']} > 32 max"

        assert decision['surface_format'] in supported_surface_formats, f"surface format {decision['surface_format']} not supported for codec {caps.eCodecType} chroma {caps.eChromaFormat} depth {caps.nBitDepthMinus8 + 8}"
        assert decision['num_surfaces'] >= 0, "number of surfaces must be non-negative"
        da = vf.display_area
        c = decision['cropping']        
        # the provided cropping is offsetted 
        display_area = SRECT(
            left = da.left + c['left'],
            top = da.top + c['top'],
            right = da.left + c['right'],
            bottom = da.top + c['bottom']
        )
        tr = decision['target_rect']
        target_rect = SRECT(
            left = tr['left'],
            top = tr['top'],
            right = tr['right'],
            bottom = tr['bottom']
        )

        p = CUVIDDECODECREATEINFO(
            ulWidth = vf.coded_width,
            ulHeight = vf.coded_height,
            ulNumDecodeSurfaces = decision['num_pictures'],
            CodecType = vf.codec,
            ChromaFormat = vf.chroma_format,
            ulCreationFlags = cudaVideoCreateFlags.PreferCUVID,
            bitDepthMinus8 = vf.bit_depth_luma_minus8,
            ulIntraDecodeOnly = 0,
            ulMaxWidth = vf.coded_width,
            ulMaxHeight = vf.coded_height,
            display_area = display_area,
            OutputFormat = decision['surface_format'],
            DeinterlaceMode = (cudaVideoDeinterlaceMode.Weave if vf.progressive_sequence else cudaVideoDeinterlaceMode.Adaptive),
            ulTargetWidth = decision['target_size']['width'],
            ulTargetHeight = decision['target_size']['height'],
            ulNumOutputSurfaces = decision['num_surfaces'],
            vidLock = None,
            target_rect = target_rect,
            enableHistogram = 0
        )

        # for field_name, filed_type in p._fields_:
        #     print(field_name, getattr(p, field_name))

        cuda.call(nvcuvid.cuvidCreateDecoder, byref(self.decoder), byref(p))
        log.debug(f'created decoder: {p}')

        # this mirrors parser's reorder buffer
        # maps picture index to pictures
        self.reorder_buffer = {}
        # this contains all picture indices that are still being used
        # including the ones in above, and the ones passed to user via on_recv
        self.pictures_used = set() 
        self.pictures_cond = Condition()
        self.surfaces_to_unmap = set() # surfaces waiting to be unmap (their devptr)
        self.surfaces_avail = decision['num_surfaces'] # number of surfaces available
        self.surfaces_cond = Condition()
        self.decode_create_info = p # save for user use
        memmove(byref(self.video_format), byref(vf), sizeof(CUVIDEOFORMAT))
        log.debug('sequence successful')
        return decision['num_pictures']

    @property
    def codec(self):
        """
        Returns:
            cudaVideoCodec: the codec enum of the video
        """        
        return cudaVideoCodec(self.decode_create_info.CodecType)

    @property
    def coded_size(self):
        '''
        Returns:
            dict: the coded size of the video, with keys 'width' and 'height'
            note this is not the actual size of the video
        '''
        return {
            'width' : self.decode_create_info.ulWidth,
            'height' : self.decode_create_info.ulHeight
        }

    @property
    def target_width(self):
        """
        Returns:
            int: the width of the target picture
        """
        return self.decode_create_info.ulTargetWidth
    
    @property
    def target_height(self):
        """

        Returns:
            int: the height of the target picture
        """        
        return self.decode_create_info.ulTargetHeight

    @property
    def target_size(self):
        """

        Returns:
            dict: the size of the target picture, with keys 'width' and 'height'
        """        
        return {
            'width' : self.target_width,
            'height' : self.target_height
        }

    @property
    def surface_format(self):
        """

        Returns:
            cudaVideoSurfaceFormat: the output surface format of the video
        """
        return self.decode_create_info.OutputFormat

    def handlePictureDecode(self, pUserData, pPicParams):
        log.debug('decode')
        pp = cast(pPicParams, POINTER(CUVIDPICPARAMS)).contents
        assert self.decoder, "decoder not initialized"
        log.debug(f'decode picture index: {pp.CurrPicIdx}')

        p = Picture(self, pp)
        self.reorder_buffer[p.index] = p
        return 1  

    def handlePictureDisplay(self, pUserData, pDispInfo):
        log.debug('display')
        di = cast(pDispInfo, POINTER(CUVIDPARSERDISPINFO)).contents
        picture = self.reorder_buffer.pop(di.picture_index) # remove this reference        

        params = CUVIDPROCPARAMS(
            progressive_frame=di.progressive_frame,
            second_field=di.repeat_first_field + 1,
            top_field_first=di.top_field_first,
            unpaired_field=di.repeat_first_field < 0
        )
        picture.on_display(params)
        self.on_recv(picture, di.timestamp)
        return 1

    '''
    I don't understand AV1 operating point, just copying reference implementation
    '''
    def handleOperatingPoint(self, pUserData, pOPInfo):
        opi = cast(pOPInfo, POINTER(CUVIDOPERATINGPOINTINFO)).contents
        if opi.codec == cudaVideoCodec.AV1:
            if opi.av1.operating_points_cnt > 1:
                if self.operating_point >= opi.av1.operating_points_cnt:
                    self.operating_point = 0
                return self.operating_point | (1 << 10 if self.disp_all_layers else 0)
        return -1

    def __init__(self, codec: cudaVideoCodec, on_recv, decide = lambda p: {}):
        """
        Args:
            codec (cudaVideoCodec): the codec enum of the video
            on_recv (callback): a callback that will be called when a picture is ready to be displayed
            decide (callback, optional): a callback used to decide decoder parameters. Defaults to lambda p:{}.
                The callback will be called with a dict containing:
                    'chroma_format' (cudaVideoChromaFormat) : the chroma format of the video
                    'bit_depth' (int) : the bit depth of the video, usually 8
                    'size' (dict) : the size of the video, with keys 'width' and 'height'
                    'supported_surface_formats' (set) : a set of cudaVideoSurfaceFormat that are supported as output surface for this video
                    'min_num_pictures' (int) : the minimum number of pictures needed for decoding
                it should return a dict containing (all optional): 
                    'num_pictures' (int) : the number of pictures used for decoding, by default it is the same as 'min_num_pictures'. 
                    'num_surfaces' (int) : the number of surfaces used for mapping, by default 1
                    'surface_format' (cudaVideoSurfaceFormat) : the output surface format to use, by default selected by decide_surface_format
                    'cropping' (dict) : the cropping parameters, with keys 'left', 'top', 'right', 'bottom', by default no cropping
                    'target_size' (dict) : the target size of the output picture, with keys 'width' and 'height, by default the same as 'size'
                    'target_rect' (dict) : the target rectangle of the output picture, with keys 'left', 'top', 'right', 'bottom', by default no black margin
                only include keys that you want to change from the default
        """        
        self.dirty = False
        self.on_recv = on_recv
        self.decide = decide
        self.exception = None

        self.operating_point = 0
        self.disp_all_layers = False
        self.parser = CUvideoparser() # NULL, to be filled in next 
        self.decoder = CUvideodecoder() # NULL, to be filled in later        
        self.video_format = CUVIDEOFORMAT() # NULL, to be filled in later

        self.handleVideoSequenceCallback = PFNVIDSEQUENCECALLBACK(self.catch_exception(self.handleVideoSequence))
        self.handlePictureDecodeCallback = PFNVIDDECODECALLBACK(self.catch_exception(self.handlePictureDecode))
        self.handlePictureDisplayCallback = PFNVIDDISPLAYCALLBACK(self.catch_exception(self.handlePictureDisplay))
        self.handleOperatingPointCallback = PFNVIDOPPOINTCALLBACK(self.catch_exception(self.handleOperatingPoint, -1))
        p = CUVIDPARSERPARAMS(
            CodecType=codec,
            ulMaxNumDecodeSurfaces=0,
            ulErrorThreshold=0,
            ulMaxDisplayDelay=0,
            pUserData=None,
            pfnSequenceCallback=self.handleVideoSequenceCallback,
            pfnDecodePicture=self.handlePictureDecodeCallback,
            pfnDisplayPicture=self.handlePictureDisplayCallback,
            pfnGetOperatingPoint=self.handleOperatingPointCallback
            )

        cuda.call(nvcuvid.cuvidCreateVideoParser, byref(self.parser), byref(p))

    def send(self, packet, pts = 0):
        """
        Currently CUVID parser doesn't tell us in display callback whether a timestamp is attached,
        default to zero; therefore here we are always passing timestamp, default to zero

        There is no need to signal end-of-stream - 
        the parser eagerly invokes display callback on pictures as soon as their turn in reorder buffer
        so if you have used send on all packet you have and returned, then garanteed your on_recv
        is called for all packets. However, if your stream is malformed, there might be leftover packets
        in parser's reorder buffer; these have been (or will be) decoded, but their display callback 
        will never be called. In this case flush() should be called
    
        Args:
            packet (numpy.ndarray): packet is expected to be a numpy 1d-array; None means end of stream; user can reuse the packet after the call
            pts (int, optional): PTS of this packet. Defaults to 0.
        """        
        self.dirty = True
        flags = CUvideopacketflags(0)
        flags |= CUvideopacketflags.TIMESTAMP

        p = CUVIDSOURCEDATAPACKET(
            flags = flags.value,
            payload_size = packet.shape[0],
            payload = packet.ctypes.data_as(POINTER(c_uint8)),
            timestamp = pts
            )
        cuda.call(nvcuvid.cuvidParseVideoData, self.parser, byref(p))
        # catch: cuvidParseVideoData will not propagate error return code 0 of HandleVideoSequenceCallback
        # it still returns CUDA_SUCCESS and simply ignore all future incoming packets
        # therefore we must check the exception here, even if the last call succeeded
        if self.exception:
            # the exception is caused by our callback, not by cuvid
            e = self.exception
            self.exception = None
            raise e

    def flush(self):
        """
        flush the pipeline. do this before sending new packets to avoid old pictures
        """        
        p = CUVIDSOURCEDATAPACKET(
            flags = CUvideopacketflags.ENDOFSTREAM.value,
            payload_size = 0,
            payload = None,
            timestamp = 0
        )
        # this reset the parser internal state
        cuda.call(nvcuvid.cuvidParseVideoData, self.parser, byref(p))

        # frees reorder buffer
        self.reorder_buffer = {}
        # no need to set self.pictures - should be set by the previous line
        # some will remain because user still holds them.
                            

    def __del__(self):
        if self.parser:
            cuda.call(nvcuvid.cuvidDestroyVideoParser, self.parser)
        if self.decoder:
            cuda.call(nvcuvid.cuvidDestroyDecoder, self.decoder)

class Decoder(BaseDecoder):
    """A decoder with send/recv paradigm. That is, user send packets to decoder, and receive pictures from decoder. 
    A queue is used to to buffer received pictures.
    """    
    def flush(self):
        """flush the pipeline. do this before sending new packets to avoid old pictures
        """        
        super().flush()
        self.pictures = Queue()

    def recv(self):
        """get the next picture; block if empty

        Returns:
            Picture: Picture decoded
        """        
        return self.pictures.get()

    def decode(self, packets):
        """Decode packets; will flush() beforehand
    
        Args:
            packets ([(numpy.array, pts)]): iterator of (annex.B packet, timestamp). 
                Each packet will be 'del'-ed before the next packet is fetched

        Yields:
            [(Picture, int)]: iterator of (Picture, timestamp)
        """        
        self.flush()

        for packet, pts in packets:
            self.send(packet, pts)
            while not self.pictures.empty():
                yield self.pictures.get()
            del packet # to allow buffer reuse

    def warmup(self, packets):
        '''
        Decode packets until the decoder is initialized (during which user-supplied `decide` is called)
        Pictures decoded will be discarded.
        This is useful if user wants to run something right after decide is called.
        '''
        self.flush()
        for packet, pts in packets:
            self.send(packet, pts)
            while not self.pictures.empty():
                self.pictures.get()
            if self.decoder:
                break
            del packet

    def __init__(self, codec: cudaVideoCodec, decide=lambda p: {}):
        """
        Args:
            codec (cudaVideoCodec): codec enum of video
            decide (callback, optional): See `decide` in `BaseDecoder`
        """        
        self.pictures = Queue()
        on_recv = lambda pic,pts : self.pictures.put((pic, pts))
        super().__init__(codec, on_recv, decide)