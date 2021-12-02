from .cuda import call as CUDAcall
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
    don't call this yourself; call Picture.map
    '''
    def __init__(self, decoder, c_devptr, c_pitch, stream):
        super().__init__()
        self.decoder = decoder
        # ctypes is pain to handle, so we convert it to python int        
        self.c_pitch = c_pitch
        self.c_devptr = c_devptr
        self.stream = stream

    @property
    def format(self):
        format = self.decoder.surface_format
        if format in [cudaVideoSurfaceFormat.NV12, cudaVideoSurfaceFormat.P016]:
            return SurfaceFormat.YUV420P
        elif format in [cudaVideoSurfaceFormat.YUV444, cudaVideoSurfaceFormat.YUV444_16Bit]:
            return SurfaceFormat.YUV444P
        else:
            raise ValueError(f'unsupported format {format}')

    @property
    def height(self):
        return self.decoder.target_height

    @property
    def width(self):
        return self.decoder.target_width

    @property
    def size(self):
        return {'width':self.width, 'height':self.height}

    def __del__(self):
        log.debug(f'trying to unmap {self.c_devptr}')
        if self.c_devptr:
            with self.decoder.surfaces_cond:
                self.decoder.surfaces_to_unmap.add(self.c_devptr.value)
                self.decoder.surfaces_cond.notify_all()

    @property
    def __cuda_array_interface__(self):
        format = self.decoder.surface_format
        if format is cudaVideoSurfaceFormat.NV12:
            shape = (self.height //2 * 3, self.width)
            typestr = '|u1'
            strides = (self.c_pitch.value, 1)
        elif format is cudaVideoSurfaceFormat.P016:
            shape = (self.height //2 * 3, self.width)
            typestr = '<u2'
            strides = (self.c_pitch.value, 2)
        elif format is cudaVideoSurfaceFormat.YUV444:
            shape = (3, self.height, self.width)
            typestr = '|u1'
            strides = (self.c_pitch.value * self.height, self.c_pitch.value, 1)
        elif format is cudaVideoSurfaceFormat.YUV444_16Bit:
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
            'data': (self.c_devptr.value, False),
            'stream': get_stream_ptr(self.stream)
        }        
        
class Picture:
    def __init__(self, decoder, params):
        self.decoder = decoder        
        self.index = params.CurrPicIdx
        with self.decoder.pictures_cond:
            # wait until picture slot is available again
            log.debug('wait_for_pictures started')
            self.decoder.pictures_cond.wait_for(lambda:self.index not in self.decoder.pictures_used)
            log.debug('wait_for_pictures finished')
            self.decoder.pictures_used.add(self.index)

        CUDAcall(nvcuvid.cuvidDecodePicture, self.decoder.decoder, byref(params))

    def on_display(self, params):
        self.params = params
    
    def __del__(self):
        with self.decoder.pictures_cond:
            self.decoder.pictures_used.remove(self.index)
            self.decoder.pictures_cond.notify_all()

    '''
    should call within approapriate cuda context
    '''
    def map(self, stream = None):
        self.params.stream = get_stream_ptr(stream)
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
                        CUDAcall(nvcuvid.cuvidUnmapVideoFrame64, self.decoder.decoder, c_ulonglong(devptr))
                        self.decoder.surfaces_avail += 1
                except KeyError:
                    pass

            c_devptr = c_ulonglong() # according to cuviddec, the argument type of cuvidmapvideoframe64
            c_pitch = c_uint()
            CUDAcall(nvcuvid.cuvidMapVideoFrame64, self.decoder.decoder, self.index, byref(c_devptr), byref(c_pitch), byref(self.params))
            self.decoder.surfaces_avail -= 1

            surface = Surface(self.decoder, c_devptr, c_pitch, stream)
            return surface

'''
allow_high: do we allow high-bit-depth? default to False
'''
def decide_surface_format(chroma_format, bit_depth, supported_surface_formats, allow_high = False):
    if chroma_format in [cudaVideoChromaFormat.YUV420, cudaVideoChromaFormat.MONOCHROME]:
        f = cudaVideoSurfaceFormat.P016 if bit_depth > 8 and allow_high else cudaVideoSurfaceFormat.NV12
    elif chroma_format is cudaVideoChromaFormat.YUV444:
        f = cudaVideoSurfaceFormat.YUV444_16Bit if bit_depth > 8 and allow_high else cudaVideoSurfaceFormat.YUV444
    elif chroma_format is cudaVideoChromaFormat.YUV422:
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
    '''
    wrapper for callbacks, so they return the error code as expected by cuvid; 
    '''
    def catch_exception(self, func, return_on_error = 0):
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

        CUDAcall(nvcuvid.cuvidGetDecoderCaps, byref(caps))
        assert caps.bIsSupported == 1, "Codec not supported"
        assert vf.coded_width <= caps.nMaxWidth, "width too large"
        assert vf.coded_height <= caps.nMaxHeight, "height too large"
        assert vf.coded_width >= caps.nMinWidth, "width too small"
        assert vf.coded_height >= caps.nMinHeight, "height too small"
        assert (vf.coded_width >> 4) * (vf.coded_height >> 4) <= caps.nMaxMBCount, "too many macroblocks"
        
        supported_surface_formats = set()
        for surface_format in cudaVideoSurfaceFormat:
            if caps.nOutputFormatMask & (1 << surface_format.value):
                supported_surface_formats.add(surface_format)

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
            'num_pictures' : p['min_num_pictures'],
            'num_surfaces' : 1,
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
            }
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

        p = CUVIDDECODECREATEINFO(
            ulWidth = vf.coded_width,
            ulHeight = vf.coded_height,
            ulNumDecodeSurfaces = decision['num_pictures'],
            CodecType = vf.codec,
            ChromaFormat = vf.chroma_format,
            ulCreationFlags = cudaVideoCreateFlags.PreferCUVID.value,
            bitDepthMinus8 = vf.bit_depth_luma_minus8,
            ulIntraDecodeOnly = 0,
            ulMaxWidth = vf.coded_width,
            ulMaxHeight = vf.coded_height,
            display_area = display_area,
            OutputFormat = decision['surface_format'].value,
            DeinterlaceMode = (cudaVideoDeinterlaceMode.Weave if vf.progressive_sequence else cudaVideoDeinterlaceMode.Adaptive).value,
            ulTargetWidth = decision['target_size']['width'],
            ulTargetHeight = decision['target_size']['height'],
            ulNumOutputSurfaces = decision['num_surfaces'],
            vidLock = None,
            enableHistogram = 0
        )

        # for field_name, filed_type in p._fields_:
        #     print(field_name, getattr(p, field_name))

        CUDAcall(nvcuvid.cuvidCreateDecoder, byref(self.decoder), byref(p))
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
        return cudaVideoCodec(self.decode_create_info.CodecType)

    @property
    def coded_size(self):
        return {
            'width' : self.decode_create_info.ulWidth,
            'height' : self.decode_create_info.ulHeight
        }

    @property
    def target_width(self):
        return self.decode_create_info.ulTargetWidth
    
    @property
    def target_height(self):
        return self.decode_create_info.ulTargetHeight

    @property
    def target_size(self):
        return {
            'width' : self.target_width,
            'height' : self.target_height
        }

    @property
    def surface_format(self):
        return cudaVideoSurfaceFormat(self.decode_create_info.OutputFormat)

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
            CodecType=codec.value,
            ulMaxNumDecodeSurfaces=0,
            ulErrorThreshold=0,
            ulMaxDisplayDelay=0,
            pUserData=None,
            pfnSequenceCallback=self.handleVideoSequenceCallback,
            pfnDecodePicture=self.handlePictureDecodeCallback,
            pfnDisplayPicture=self.handlePictureDisplayCallback,
            pfnGetOperatingPoint=self.handleOperatingPointCallback
            )

        CUDAcall(nvcuvid.cuvidCreateVideoParser, byref(self.parser), byref(p))



    '''
    packet is expected to be of buffer protocol; None means end of stream
    packet can be reused after the call

    the caller does't have to set up cuda context.

    Currently CUVID parser doesn't tell us in display callback whether a timestamp is attached,
    default to zero; therefore here we are always passing timestamp, default to zero

    There is no need to signal end-of-stream - 
    the parser eagerly invokes display callback on pictures as soon as their turn in reorder buffer
    so if you have used send on all packet you have and returned, then garanteed your on_recv
    is called for all packets. However, if your stream is malformed, there might be leftover packets
    in parser's reorder buffer; these have been (or will be) decoded, but their display callback 
    will never be called. In this case flush() should be called
    
    '''
    def send(self, packet, timestamp = 0):
        self.dirty = True
        flags = CUvideopacketflags(0)
        flags |= CUvideopacketflags.TIMESTAMP

        arr = np.frombuffer(packet, dtype=np.uint8)

        p = CUVIDSOURCEDATAPACKET(
            flags = flags.value,
            payload_size = arr.shape[0],
            payload = arr.ctypes.data_as(POINTER(c_uint8)),
            timestamp = timestamp
            )
        CUDAcall(nvcuvid.cuvidParseVideoData, self.parser, byref(p))
        # catch: cuvidParseVideoData will not propagate error return code 0 of HandleVideoSequenceCallback
        # it still returns CUDA_SUCCESS and simply ignore all future incoming packets
        # therefore we must check the exception here, even if the last call succeeded
        if self.exception:
            # the exception is caused by our callback, not by cuvid
            e = self.exception
            self.exception = None
            raise e

    '''
    flush the pipeline. do this before sending new packets to avoid old pictures
    '''
    def flush(self):
        p = CUVIDSOURCEDATAPACKET(
            flags = CUvideopacketflags.ENDOFSTREAM.value,
            payload_size = 0,
            payload = None,
            timestamp = 0
        )
        # this reset the parser internal state
        CUDAcall(nvcuvid.cuvidParseVideoData, self.parser, byref(p))

        # frees reorder buffer
        self.reorder_buffer = {}
        # no need to set self.pictures - should be set by the previous line
        # some will remain because user still holds them.
                            

    def __del__(self):
        if self.parser:
            CUDAcall(nvcuvid.cuvidDestroyVideoParser, self.parser)
        if self.decoder:
            CUDAcall(nvcuvid.cuvidDestroyDecoder, self.decoder)

class Decoder(BaseDecoder):
    def flush(self):
        super().flush()
        self.pictures = Queue()

    def recv(self):
        return self.pictures.get()
    '''
    takes an iterator of (annex.b packets buffer, timestamp)
    returns an iterator of (pictures, timestamp)
    '''
    def decode(self, packets):
        self.flush()

        for packet, pts in packets:
            self.send(packet, pts)
            while not self.pictures.empty():
                yield self.pictures.get()
            del packet # to allow buffer reuse

    def warmup(self, packets):
        self.flush()
        for packet, pts in packets:
            self.send(packet, pts)
            while not self.pictures.empty():
                self.pictures.get()
            if self.decoder:
                break
            del packet

    def __init__(self, codec: cudaVideoCodec, decide=lambda p: {}):
        self.pictures = Queue()
        on_recv = lambda pic,pts : self.pictures.put((pic, pts))
        super().__init__(codec, on_recv, decide)
