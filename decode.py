from collections import namedtuple
from .nvcuvid import * 
from .cuda import call as CUDAcall
import pycuda.driver as cuda
from queue import Queue
import numpy as np
from time import sleep
import sys

import logging
log = logging.getLogger(__name__)

nvcuvid = cdll.LoadLibrary('libnvcuvid.so')

'''
Question: how do we utilize cuda streams for parallelism?
operations that requires CUstream as parameters:
1. mapsurface: copy from decoder picture to CUDA surface
2. all following operations on the surface (color transform, filtering, etc)

solution: require the user to provide stream on which the mapsurface operation is performed. This way the user can choose where to insert mapsurface operation between their own operations, maximize parallism.
'''

'''
Question: should we take cuda context on init, and make it current on every operations? Or do we do nothing in our function, and expect the user to handle the context well?

answer: we should handle it. Because it HAS to be the same context throughout the process anyway, no need to bother user with context management everytime they call our function. The user also don't know our thread/callback structure, so it's hard to manage context. 
'''

class Surface:
    '''
    don't call this yourself; call Picture.map
    '''
    def __init__(self, decoder, index, p):
        self.decoder = decoder        
        self.devptr = c_ulonglong() # according to cuviddec, the argument type of cuvidmapvideoframe64
        self.pitch = c_uint()
        CUDAcall(nvcuvid.cuvidMapVideoFrame64, decoder.decoder, index, byref(self.devptr), byref(self.pitch), byref(p))

    def __del__(self):
        # catch: when interacting with C functions, for arguments must convert numbers to corresponding ctypes,
        # for otherwise python just pass the pointers and cause errors
        log.debug(f'trying to unmap {self.devptr}')
        if self.devptr:
            self.decoder.ctx.push()
            CUDAcall(nvcuvid.cuvidUnmapVideoFrame64, self.decoder.decoder, self.devptr)
            self.decoder.ctx.pop()

    @property
    def format(self):
        return self.decoder.decode_create_info.OutputFormat
        
    @property
    def width(self):
        return self.decoder.decode_create_info.ulTargetWidth

    @property
    def height(self):
        return self.decoder.decode_create_info.ulTargetHeight

class Picture:
    def __init__(self, decoder, params):
        self.decoder = decoder        
        self.index = params.CurrPicIdx
        assert self.index not in self.decoder.pictures, "old picture still refered, consider increasing extra_pictures"
        self.decoder.pictures.add(self.index)
        CUDAcall(nvcuvid.cuvidDecodePicture, self.decoder.decoder, byref(params))

    def on_display(self, params):
        self.params = params
    
    def __del__(self):
        self.decoder.pictures.remove(self.index)

    def map(self, stream = None):
        log.debug(f'using stream {stream.handle}')
        self.params.stream = stream.handle if stream else None
        return Surface(self.decoder, self.index, self.params)

# the default when initializing decoder
def select_surface_format(chroma_format, bit_depth, supported_surface_formats):
    if chroma_format in [cudaVideoChromaFormat.YUV420, cudaVideoChromaFormat.MONOCHROME]:
        f = cudaVideoSurfaceFormat.P016 if bit_depth > 8 else cudaVideoSurfaceFormat.NV12
    elif chroma_format == cudaVideoChromaFormat.YUV444:
        f = cudaVideoSurfaceFormat.YUV444_16Bit if bit_depth > 8 else cudaVideoSurfaceFormat.YUV444
    elif chroma_format == cudaVideoChromaFormat.YUV422:
        f = cudaVideoSurfaceFormat.NV12
    
    # check if the selected format is supported. If not, check fallback options
    if f not in supported_surface_formats:
        if cudaVideoSurfaceFormat.NV12 in supported_surface_formats:
            f = cudaVideoSurfaceFormat.NV12
        elif cudaVideoSurfaceFormat.P016 in supported_surface_formats:
            f = cudaVideoSurfaceFormat.P016
        elif cudaVideoSurfaceFormat.YUV444 in supported_surface_formats:
            f = cudaVideoSurfaceFormat.YUV444
        elif cudaVideoSurfaceFormat.YUV444_16Bit in supported_surface_formats:
            f = cudaVideoSurfaceFormat.YUV444_16Bit
        else:
            raise Exception("No supported surface format")
    return f

class Decoder:
    '''
    wrapper for callbacks, so they return the error code as expected by cuvid; 

    '''
    def catch_exception(self, func, return_on_error = 0):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except BaseException as e: # catch keyboard interrupt as well
                log.warning(f'callback exception logged {e}')
                # we record the exception; to be checked when nvcuvid propagate the error return code
                self.exception = e
                log.warning(f'returning error code {return_on_error}')
                return return_on_error
        return wrapper

    def handleVideoSequence(self, pUserData, pVideoFormat):
        log.debug('sequence')
        assert not self.decoder, "decoder already initialized, please create a new decoder for new video sequence"
        vf = cast(pVideoFormat, POINTER(CUVIDEOFORMAT)).contents
        self.ctx.push()

        caps = CUVIDDECODECAPS(
            eCodecType=vf.codec,
            eChromaFormat=vf.chroma_format,
            nBitDepthMinus8=vf.bit_depth_luma_minus8
        )

        CUDAcall(nvcuvid.cuvidGetDecoderCaps, byref(caps))
        assert caps.bIsSupported == 1, "Codec not supported"
        assert vf.coded_width <= caps.nMaxWidth, "width too large"
        assert vf.coded_height <= caps.nMaxHeight, "height too large"
        assert (vf.coded_width >> 4) * (vf.coded_height >> 4) <= caps.nMaxMBCount, "too many macroblocks"
        
        supported_output_formats = set()
        for surface_format in cudaVideoSurfaceFormat:
            if caps.nOutputFormatMask & (1 << surface_format):
                supported_output_formats.add(surface_format)
        ulNumDecodeSurfaces = vf.min_num_decode_surfaces + self.extra_pictures

        p = CUVIDDECODECREATEINFO(
            ulWidth = vf.coded_width,
            ulHeight = vf.coded_height,
            ulNumDecodeSurfaces = ulNumDecodeSurfaces,
            CodecType = vf.codec,
            ChromaFormat = vf.chroma_format,
            ulCreationFlags = cudaVideoCreateFlags.PreferCUVID,
            bitDepthMinus8 = vf.bit_depth_luma_minus8,
            ulIntraDecodeOnly = 0,
            ulMaxWidth = vf.coded_width,
            ulMaxHeight = vf.coded_height,
            display_area = SRECT(
                left=vf.display_area.left, 
                top=vf.display_area.top, 
                right=vf.display_area.right, 
                bottom=vf.display_area.bottom
                ),
            OutputFormat = self.select_surface_format(
                vf.chroma_format, 
                vf.bit_depth_luma_minus8 + 8, 
                supported_output_formats
                ),
            DeinterlaceMode = cudaVideoDeinterlaceMode.Weave if vf.progressive_sequence else cudaVideoDeinterlaceMode.Adaptive,
            ulTargetWidth = vf.coded_width,
            ulTargetHeight = vf.coded_height,
            ulNumOutputSurfaces = self.surfaces,
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
        self.pictures = set()
        self.decode_create_info = p # save for user use
        self.ctx.pop()
        log.debug('sequence successful')
        return ulNumDecodeSurfaces

    def handlePictureDecode(self, pUserData, pPicParams):
        log.debug('decode')
        pp = cast(pPicParams, POINTER(CUVIDPICPARAMS)).contents
        assert self.decoder, "decoder not initialized"
        log.info(f'decode picture index: {pp.CurrPicIdx}')

        self.ctx.push()
        p = Picture(self, pp)
        self.reorder_buffer[p.index] = p
        self.ctx.pop()
        return 1  

    def handlePictureDisplay(self, pUserData, pDispInfo):
        log.debug('display')
        di = cast(pDispInfo, POINTER(CUVIDPARSERDISPINFO)).contents
        params = CUVIDPROCPARAMS(
            progressive_frame=di.progressive_frame,
            second_field=di.repeat_first_field + 1,
            top_field_first=di.top_field_first,
            unpaired_field=di.repeat_first_field < 0
        )
        picture = self.reorder_buffer.pop(di.picture_index) # remove this reference
        picture.on_display(params)
        self.ctx.push() # context pushed for the user callback        
        self.on_recv(picture, di.timestamp)
        self.ctx.pop()
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


    '''
    select_surface_format: a function that returns the surface format to use; see the default for an example
    extra_pictures: extra pictures to allocate (in addition to the ones needed for correct decoding); can be used for lookback
    surfaces: number of surfaces to allocate
    note that both pictures and surfaces are pre-allocated when decoder is created.
    '''        

    def __init__(self, ctx: cuda.Context, select_surface_format = select_surface_format, extra_pictures = 0, surfaces = 2):
        self.dirty = False
        self.on_recv = None
        self.ctx = ctx
        assert extra_pictures >= 0, "extra pictures must be non-negative"
        self.extra_pictures = extra_pictures
        self.surfaces = surfaces
        self.exception = None

        self.operating_point = 0
        self.disp_all_layers = False
        self.parser = CUvideoparser() # NULL, to be filled in next 
        self.decoder = CUvideodecoder() # NULL, to be filled in later        

        self.handleVideoSequenceCallback = PFNVIDSEQUENCECALLBACK(self.catch_exception(self.handleVideoSequence))
        self.handlePictureDecodeCallback = PFNVIDDECODECALLBACK(self.catch_exception(self.handlePictureDecode))
        self.handlePictureDisplayCallback = PFNVIDDISPLAYCALLBACK(self.catch_exception(self.handlePictureDisplay))
        self.handleOperatingPointCallback = PFNVIDOPPOINTCALLBACK(self.catch_exception(self.handleOperatingPoint, -1))
        p = CUVIDPARSERPARAMS(
            CodecType=cudaVideoCodec.HEVC,
            ulMaxNumDecodeSurfaces=0,
            pUserData=None,
            pfnSequenceCallback=self.handleVideoSequenceCallback,
            pfnDecodePicture=self.handlePictureDecodeCallback,
            pfnDisplayPicture=self.handlePictureDisplayCallback,
            pfnGetOperatingPoint=self.handleOperatingPointCallback
            )

        self.select_surface_format = select_surface_format
        CUDAcall(nvcuvid.cuvidCreateVideoParser, byref(self.parser), byref(p))

    '''
    on_recv: will be called back with (picture, timestamp), in presentation order,
    in suitable cuda context
    '''
    def set_on_recv(self, on_recv):
        assert not self.dirty, "packets already in pipeline, unsafe change of on_recv; flush first"
        self.on_recv = on_recv

    '''
    flush the pipeline. do this before sending new packets to avoid old pictures
    '''
    def flush(self):
        p = CUVIDSOURCEDATAPACKET(
            flags = CUvideopacketflags.ENDOFSTREAM,
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
        
        self.dirty = False

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
        flags = 0
        flags |= CUvideopacketflags.TIMESTAMP

        arr = np.frombuffer(packet, dtype=np.uint8)

        p = CUVIDSOURCEDATAPACKET(
            flags = flags,
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
    takes an iterator of (annex.b packets, timestamp)
    returns an iterator of (pictures, timestamp)
    '''
    def decode(self, packets):
        self.flush()
        on_recv = self.on_recv        

        pictures = Queue()
        self.set_on_recv(lambda pts, p: pictures.put((pts, p)))        
        for packet, pts in packets:
            self.send(packet, pts)
            while not pictures.empty():
                yield pictures.get()
            del packet # to allow buffer reuse
        self.on_recv = on_recv

    def __del__(self):
        if self.parser:
            CUDAcall(nvcuvid.cuvidDestroyVideoParser, self.parser)
        if self.decoder:
            CUDAcall(nvcuvid.cuvidDestroyDecoder, self.decoder)