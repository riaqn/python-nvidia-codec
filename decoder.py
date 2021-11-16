from collections import namedtuple
from .nvcuvid import * 
from .cuda import call as CUDAcall
import pycuda.driver as cuda
from . import utils
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
    Plane = namedtuple('Plane', ['name', 'devptr', 'pitch', 'height', 'width', 'dtype'])
    def __init__(self, devptr, pitch, decoder):
        self.devptr = devptr.value
        self.pitch = pitch.value
        self.decoder = decoder

    def __del__(self):
        self.decoder.ctx.push()
        # catch: when interacting with C functions, for arguments must convert numbers to corresponding ctypes,
        # for otherwise python just pass the pointers and cause errors
        devptr = c_ulonglong(self.devptr)
        log.debug(f'trying to unmap {devptr}')
        CUDAcall(nvcuvid.cuvidUnmapVideoFrame64, self.decoder.decoder, devptr)
        self.decoder.ctx.pop()

    def planes(self):
        if self.decoder.decode_create_info.OutputFormat == cudaVideoSurfaceFormat.NV12:
            return (
                Surface.Plane('Y', self.devptr, self.pitch, self.decoder.decode_create_info.ulTargetHeight, self.decoder.decode_create_info.ulTargetWidth, np.uint8),
                Surface.Plane('UV', self.devptr + self.pitch * self.decoder.decode_create_info.ulTargetHeight, self.pitch, self.decoder.decode_create_info.ulTargetHeight // 2, self.decoder.decode_create_info.ulTargetWidth, np.uint8)
            )
        else:
            raise NotImplementedError()
        
class Picture:
    def __init__(self, index, params, timestamp, decoder):
        self.index = index
        self.params = params
        self.timestamp = timestamp
        self.decoder = decoder
    
    def __del__(self):
        self.decoder.pictures[self.index] = False

    def map(self, stream = None):
        p = self.params
        log.debug(f'using stream {stream.handle}')
        p.stream = stream.handle if stream else None
        pointer = c_ulonglong() # according to cuviddec, the argument type of cuvidmapvideoframe64
        pitch = c_uint()
        CUDAcall(nvcuvid.cuvidMapVideoFrame64, self.decoder.decoder, self.index, byref(pointer), byref(pitch), byref(p))
        return Surface(pointer, pitch, self.decoder)

class Decoder:
    cuda.init()

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
        self.pictures = [False] * ulNumDecodeSurfaces      
        self.decode_create_info = p # save for user use
        self.ctx.pop()
        log.debug('sequence successful')
        return ulNumDecodeSurfaces

    def handlePictureDecode(self, pUserData, pPicParams):
        log.debug('decode')
        pp = cast(pPicParams, POINTER(CUVIDPICPARAMS)).contents
        assert self.decoder, "decoder not initialized"
        log.info(f'decode picture index: {pp.CurrPicIdx}')
        if self.pictures[pp.CurrPicIdx]:
            # picture still refered
            raise Exception("old picture still refered, consider increasing extra_pictures")
        self.ctx.push()
        CUDAcall(nvcuvid.cuvidDecodePicture, self.decoder, byref(pp))
        self.ctx.pop()
        self.pictures[pp.CurrPicIdx] = True
        return 1  

    def handlePictureDisplay(self, pUserData, pDispInfo):
        log.debug('display')
        self.ctx.push() # context pushed for the user callback
        di = cast(pDispInfo, POINTER(CUVIDPARSERDISPINFO)).contents
        p = CUVIDPROCPARAMS(
            progressive_frame=di.progressive_frame,
            second_field=di.repeat_first_field + 1,
            top_field_first=di.top_field_first,
            unpaired_field=di.repeat_first_field < 0
        )

        self.display_callback(Picture(di.picture_index, p, di.timestamp, self))
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

    def __init__(self, ctx: cuda.Context, display_callback, select_surface_format = utils.select_surface_format, extra_pictures = 0, surfaces = 2):
        self.ctx = ctx
        assert extra_pictures >= 0, "extra pictures must be non-negative"
        self.extra_pictures = extra_pictures
        self.surfaces = surfaces
        self.exception = None

        self.display_callback = display_callback
        # self.free_picture_callback = free_picture_callback
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

    # TODO: currently packet is memoryview
    # need more efficient way to convert to pointer
    def send(self, packet, timestamp = None):
        flags = 0
        if timestamp:
            flags |= CUvideopacketflags.TIMESTAMP
        if packet is None:
            flags |= CUvideopacketflags.ENDOFSTREAM

        arr = np.frombuffer(packet, dtype=np.uint8)
        log.debug(f'sending {arr.shape}')

        p = CUVIDSOURCEDATAPACKET(
            flags = flags,
            payload_size = len(packet),
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

    def __del__(self):
        if self.parser:
            CUDAcall(nvcuvid.cuvidDestroyVideoParser, self.parser)
        if self.decoder:
            CUDAcall(nvcuvid.cuvidDestroyDecoder, self.decoder)

