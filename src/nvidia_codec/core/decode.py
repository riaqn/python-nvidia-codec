'''
User sends packets to decoder and receives Pictures from the decoder. Picture contains the data of a decoded frame. However, it is not accessible to the user. To access the data, user has to call Picture.map(). This copies and post-processes (cropping, scaling and padding) the Picture into a Surface, which can be accessed as a regular CUDA array. It is up to the user whether or not to call map() for a particular Picture. For example, when screenshoting a video, only a particular subset of frames are needed as output; the user can therefore choose to only call map() on pictures of those frames for faster processing.

User owns the Picture and Surfaces; the data they represent will be valid as long as the user holds references to them. When all references are dropped, the graphics resources will  be freed. User can also call Picture.free() and Surface.free() to free the resource early. There are limits on the number of Pictures and Surfaces a user can hold at any given time. The limits are specified by `num_pictures` and `num_surfaces` when creating the decoder, upon which that much of graphics memory is pre-allocated even before user gets any Pictures/Surfaces. All Pictures/Surfaces are backed by those pre-allocated memory. Freeing Pictures/Surfaces marks those memory reusable for future Pictures/Surfaces - still not reusable to other CUDA program. By default the bare minimum required of `num_pictures` and `num_surfaces` is specified. The user can override the limits, for following reasons:

Increase the number of pictures might speed up the decoding. User can also increase the number to establish a look-back cache. Take screenshotting as an example again. Say we want the frame at PTS=350. We start decoding the video and get Picture of PTS=0, 100, 200, 300. We have to keep going, because PTS=300 might not be the target frame since there could be a following frame of PTS=350. So we keep going and get Picture of PTS=400. Now we know that indeed PTS=300 is the target frame. Fortunately we keep a variable `old` always storing the last Picture, and we will just use `old` in this case: map it to Surface and read out its data. For this to work, we must set `num_pictures` to be `min_num_pictures + 1`, since we are using one extra picture slot for this `old` picture. This is in fact the default settings since it is a common scenario. `num_pictures` has an upper limit of `32`.

Similarly one might increase the `num_surfaces` if their applications can make use of it. `num_surfaces` does not have upper limit as long as there is sufficient graphics memory. Note that again they are allocated on creation of decoder. Freeing `Surface` and `Picture` only marked the slots as reusable by the decoder for future frames, while the corresponding memory are still owned by the decoder and not usable by other CUDA operations. 
'''
from . import cuda
from .nvcuvid import *
from .common import *
from queue import Empty, Queue
import numpy as np
from ctypes import *
from threading import Condition, Semaphore

import logging
log = logging.getLogger(__name__)

nvcuvid = cdll.LoadLibrary('libnvcuvid.so')

class Surface:
    '''
    A CUDA array owned by the decoder to store post-processed frames
    '''
    def __init__(self, decoder, index, params, stream):
        """
        DO NOT call this by yourself; use Picture.map() instead
        """

        self.decoder = decoder


        self.params = CUVIDPROCPARAMS()
        memmove(byref(self.params), byref(params), sizeof(CUVIDEOFORMAT))

        self.params.output_stream = stream        

        self.c_devptr = c_ulonglong() # according to cuviddec, the argument type of cuvidmapvideoframe64
        self.c_pitch = c_uint()

        with self.decoder.condition:
            self.decoder.condition.wait_for(lambda: self.decoder.surfaces_sem > 0)
            with cuda.Device(self.decoder.device):
                cuda.check(nvcuvid.cuvidMapVideoFrame64(self.decoder.cuvid_decoder, c_int(index), byref(self.c_devptr), byref(self.c_pitch), byref(self.params)))
                    # log.warning('mapped')
                self.decoder.surfaces_sem -= 1
            self.decoder.condition.notify_all()

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
        with self.decoder.condition:
            if self.c_devptr and self.decoder.cuvid_decoder:
                with cuda.Device(self.decoder.device):   
                    cuda.check(nvcuvid.cuvidUnmapVideoFrame64(self.decoder.cuvid_decoder, self.c_devptr))
                    self.decoder.surfaces_sem += 1
                    self.c_devptr = c_ulonglong()
                    self.decoder.condition.notify_all()

    def __del__(self):
        self.free()

    @property
    def shape(self):
        return self.__cuda_array_interface__['shape']

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
            'stream': self.params.output_stream
        }        
        
class Picture:
    '''
    Frame after decoding but before post-processing
    Its data is not accessible at all, but still takes up GPU memory
    User must call Picture.map() to get a Surface for the data
    '''
    def __init__(self, decoder, index, proc_params):
        '''
        DO NOT call this by yourself; use Decoder.decode() instead
        '''
        self.decoder = decoder
        self.index = index
        self.params = proc_params
        with self.decoder.condition:
            self.decoder.pictures_used.add(self.index)

    def free(self):
        '''
        free up the picture.
        Note that the GPU memory involved is managed by decoder and will not be available to other CUDA operations
        free() only notifies the decoder that the picture can be reused to store future pictures

        free() is called automatically when the picture is garbage collected        
        '''
        log.debug(f'freeing picture {self.index}')

        with self.decoder.condition:
            # discard instead of remove
            # so this won't trigger exception
            # which is possible
            self.decoder.pictures_used.discard(self.index)
            self.decoder.condition.notify_all()    

    def __del__(self):
        self.free()

    def map(self, stream : int = 0):
        """post-process and output this picture to a surface which can be accessed by user as a CUDA array

        Args:
            stream (optional, int): CUDA stream to queue this map operation. 
            Defaults to 2 which means per-thread default stream

        Returns:
            Surface : Surface mapped by this picture
        """
        return Surface(self.decoder, self.index, self.params, stream)

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
        vf = pVideoFormat.contents

        if self.cuvid_decoder:
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
        memmove(byref(self.video_format), byref(vf), sizeof(CUVIDEOFORMAT))
        # save for user use

        caps = CUVIDDECODECAPS(
            eCodecType=vf.codec,
            eChromaFormat=vf.chroma_format,
            nBitDepthMinus8=vf.bit_depth_luma_minus8
        )

        with cuda.Device(self.device):
            cuda.check(nvcuvid.cuvidGetDecoderCaps(byref(caps)))
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
            'chroma_format': vf.chroma_format,
            'bit_depth' : vf.bit_depth_luma_minus8 + 8,
            'supported_surface_formats' : supported_surface_formats,
            'min_num_pictures' : vf.min_num_decode_surfaces,
        }


        # the default values 
        decision = {
            'num_pictures' : p['min_num_pictures'] + 1, # for simplicity
            'num_surfaces' : 1 + 1, # for simplicity
            'surface_format' : decide_surface_format(p['chroma_format'], p['bit_depth'], p['supported_surface_formats']),
            'cropping' : lambda height,width : { # by default no cropping 
                'left': 0,
                'right' : width,
                'top': 0,
                'bottom' : height
            },
            'target_size': lambda cropped_height, cropped_width: (cropped_height,cropped_width),# by default no scaling
            'target_rect': lambda target_height, target_width : { # by default no margin
                'left': 0,
                'right' : target_width,
                'top': 0,
                'bottom' : target_height
            },
        }

        decision |= self.decide(p)
        c = decision['cropping'](self.height, self.width)
        target_height, target_width = decision['target_size'](c['bottom'] - c['top'], c['right'] - c['left'])
        tr = decision['target_rect'](target_height, target_width)

        assert decision['num_pictures'] <= 32, f"number of pictures {decision['num_pictures']} > 32 max"
        assert decision['surface_format'] in supported_surface_formats, f"surface format {decision['surface_format']} not supported for codec {caps.eCodecType} chroma {caps.eChromaFormat} depth {caps.nBitDepthMinus8 + 8}"
        assert decision['num_surfaces'] >= 0, "number of surfaces must be non-negative"
        da = vf.display_area
        # the provided cropping is offsetted 
        display_area = SRECT(
            left = da.left + c['left'],
            top = da.top + c['top'],
            right = da.left + c['right'],
            bottom = da.top + c['bottom']
        )
        target_rect = SRECT(
            left = tr['left'],
            top = tr['top'],
            right = tr['right'],
            bottom = tr['bottom']
        )

        self.decode_create_info = CUVIDDECODECREATEINFO(
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
            ulTargetWidth = target_width,
            ulTargetHeight = target_height,
            ulNumOutputSurfaces = decision['num_surfaces'],
            vidLock = None,
            target_rect = target_rect,
            enableHistogram = 0
        )

        # for field_name, filed_type in p._fields_:
        #     print(field_name, getattr(p, field_name))

        with cuda.Device(self.device):
            cuda.check(nvcuvid.cuvidCreateDecoder(byref(self.cuvid_decoder), byref(self.decode_create_info)))

        # this contains all picture indices that are still being used
        # including the ones in above, and the ones passed to user via on_recv
        self.pictures_used = set() 

        self.surfaces_sem = decision['num_surfaces'] # number of surfaces available

        log.debug('sequence successful')
        return decision['num_pictures']

    @property
    def codec(self):
        """
        Returns:
            cudaVideoCodec: the codec enum of the video
        """        
        return self.decode_create_info.CodecType

    @property
    def height(self):
        return  self.video_format.display_area.bottom - self.video_format.display_area.top

    @property
    def width(self):
        return self.video_format.display_area.right - self.video_format.display_area.left

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
    def surface_format(self):
        """

        Returns:
            cudaVideoSurfaceFormat: the output surface format of the video
        """
        return self.decode_create_info.OutputFormat

    def handlePictureDecode(self, pUserData, pPicParams):
        log.debug('decode')
        pp = pPicParams.contents
        # assert self.cuvid_decoder, "decoder not initialized"
        log.debug(f'decode picture index: {pp.CurrPicIdx}')

        if pp.CurrPicIdx in self.pictures_used:
            log.info(f'{self.pictures_used}')
        with self.condition:
            # wait until picture slot is available again
            log.debug('wait_for_pictures started')
            self.condition.wait_for(lambda:pp.CurrPicIdx not in self.pictures_used)
            log.debug('wait_for_pictures finished')

            # now we know that the user no longer need this picture
            # we can overwrite it
            with cuda.Device(self.device):
                cuda.check(nvcuvid.cuvidDecodePicture(self.cuvid_decoder, byref(pp)))

        return 1  

    def handlePictureDisplay(self, pUserData, pDispInfo):
        log.debug('display')
        if not bool(pDispInfo):
            # EOS notification
            self.on_recv(None, 0)
            return 1
        # di = cast(pDispInfo, POINTER(CUVIDPARSERDISPINFO)).contents
        di = pDispInfo.contents

        params = CUVIDPROCPARAMS(
            progressive_frame=di.progressive_frame,
            second_field=di.repeat_first_field + 1,
            top_field_first=di.top_field_first,
            unpaired_field=di.repeat_first_field < 0
        )

        picture = Picture(self, di.picture_index, params)
        self.on_recv(picture, di.timestamp)
        return 1

    '''
    I don't understand AV1 operating point, just copying reference implementation
    '''
    def handleOperatingPoint(self, pUserData, pOPInfo):
        opi = pOPInfo.contents
        if opi.codec == cudaVideoCodec.AV1:
            if opi.av1.operating_points_cnt > 1:
                if self.operating_point >= opi.av1.operating_points_cnt:
                    self.operating_point = 0
                return self.operating_point | (1 << 10 if self.disp_all_layers else 0)
        return -1

    def __init__(self, codec: cudaVideoCodec, on_recv, decide = lambda p: {}, device = None):
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
        self.condition = Condition()
      
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
        
        self.device = cuda.get_current_device(device)
        with self.condition:        
            self.cuvid_parser = CUvideoparser() # NULL, to be filled in next 
            self.cuvid_decoder = CUvideodecoder() # NULL, to be filled in later          
            cuda.check(nvcuvid.cuvidCreateVideoParser(byref(self.cuvid_parser), byref(p)))
            self.condition.notify_all()

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

        if packet is None:
            # this will reset the parser internal state
            # also triggers dummy display callback
            p = CUVIDSOURCEDATAPACKET(
                flags = (CUvideopacketflags.ENDOFSTREAM | CUvideopacketflags.NOTIFY_EOS).value,
                payload_size = 0,
                payload = None,
                timestamp = 0
            )            
        else:
            p = CUVIDSOURCEDATAPACKET(
                flags = CUvideopacketflags.TIMESTAMP.value,
                payload_size = packet.shape[0],
                payload = packet.ctypes.data_as(POINTER(c_uint8)),
                timestamp = pts
                )
        with cuda.Device(self.device):
            cuda.check(nvcuvid.cuvidParseVideoData(self.cuvid_parser, byref(p)))
        # catch: cuvidParseVideoData will not propagate error return code 0 of HandleVideoSequenceCallback
        # it still returns CUDA_SUCCESS and simply ignore all future incoming packets
        # therefore we must check the exception here, even if the last call succeeded
        if self.exception:
            # the exception is caused by our callback, not by cuvid
            e = self.exception
            self.exception = None
            raise e

    def flush(self):
        self.send(None)

    def free(self):
        with self.condition:
            if self.cuvid_parser:
                cuda.check(nvcuvid.cuvidDestroyVideoParser(self.cuvid_parser))
                self.cuvid_parser = CUvideoparser()                     
            if self.cuvid_decoder:
                with cuda.Device(self.device):
                    cuda.check(nvcuvid.cuvidDestroyDecoder(self.cuvid_decoder))
                    self.cuvid_decoder = CUvideodecoder()
            self.condition.notify_all()

    def __del__(self):
        self.free()