"""Low-level NVDEC decoder with Picture/Surface memory management.

This module provides the core decoding interface using NVIDIA's CUVID API.
The decoding pipeline has two stages:

1. **Picture**: Raw decoded frame in GPU memory (not directly accessible)
2. **Surface**: Post-processed frame that can be accessed as a CUDA array

Workflow:
    1. Send compressed packets to BaseDecoder.send()
    2. Receive Picture objects via the on_recv callback
    3. Call Picture.map() to get a Surface (applies cropping/scaling)
    4. Access Surface data via __cuda_array_interface__ (works with PyTorch, CuPy)
    5. Call Surface.free() and Picture.free() when done (or let them be garbage collected)

Memory Management:
    - Pictures and Surfaces are backed by pre-allocated GPU memory pools
    - `num_pictures` controls the decode buffer size (max 32)
    - `num_surfaces` controls the output buffer size (no hard limit)
    - Calling free() marks the slot as reusable, but memory stays allocated
    - The decoder owns all memory until BaseDecoder.free() is called

Example:
    decoder = BaseDecoder(cudaVideoCodec.H264, decide=my_decide_callback)
    def on_recv(picture, pts, result):
        if picture is None:  # End of stream
            return result
        surface = picture.map(cuda_stream)
        tensor = torch.as_tensor(surface, device='cuda')
        picture.free()
        surface.free()
        return tensor
    result = decoder.send(packet_data, on_recv, pts=12345)
"""
from . import cuda
from .nvcuvid import *
from .common import *
from .. import CodecNotSupportedError
from queue import Empty, Queue
import numpy as np
from ctypes import *
from threading import Condition, Semaphore

import logging
log = logging.getLogger(__name__)

nvcuvid = cdll.LoadLibrary('libnvcuvid.so')

class Surface:
    """Post-processed decoded frame accessible as a CUDA array.

    A Surface contains the final decoded frame data after cropping and scaling.
    It implements __cuda_array_interface__ for zero-copy access from PyTorch,
    CuPy, and other CUDA-aware libraries.

    Do not instantiate directly; use Picture.map() instead.

    Attributes:
        format: Surface pixel format (NV12, P016, YUV444, etc.)
        width: Frame width in pixels
        height: Frame height in pixels
        shape: Array shape for __cuda_array_interface__
    """

    def __init__(self, decoder, index, params, stream):
        """Internal constructor. Use Picture.map() instead."""

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
    def format(self) -> cudaVideoSurfaceFormat:
        """Surface pixel format (NV12, P016, YUV444, or YUV444_16Bit)."""
        return self.decoder.surface_format

    @property
    def height(self):
        """Frame height in pixels."""
        return self.decoder.target_height

    @property
    def width(self):
        """Frame width in pixels."""
        return self.decoder.target_width

    @property
    def size(self):
        """Frame dimensions as {'width': int, 'height': int}."""
        return {'width':self.width, 'height':self.height}

    def free(self):
        """Release this surface back to the decoder's pool.

        The underlying GPU memory remains allocated by the decoder and will be
        reused for future frames. Called automatically on garbage collection.
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
    """Decoded frame before post-processing.

    A Picture holds a reference to a decoded frame in the decoder's internal
    buffer. The frame data is not directly accessible; call map() to create
    a Surface that can be read.

    Do not instantiate directly; Pictures are created by the decoder and
    passed to your on_recv callback.
    """

    def __init__(self, decoder, index, proc_params):
        """Internal constructor. Pictures are created by the decoder."""
        self.decoder = decoder
        self.index = index
        self.params = proc_params
        with self.decoder.condition:
            self.decoder.pictures_used.add(self.index)

    def free(self):
        """Release this picture back to the decoder's pool.

        The underlying GPU memory remains allocated by the decoder and will be
        reused for future frames. Called automatically on garbage collection.
        """
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
        """Post-process this picture and create an accessible Surface.

        Applies cropping, scaling, and color format conversion as configured
        in the decoder. The operation is queued on the specified CUDA stream.

        Args:
            stream: CUDA stream for the operation. Pass 0 or 2 for the
                per-thread default stream, or a stream pointer/handle.

        Returns:
            Surface that can be accessed via __cuda_array_interface__.
        """
        return Surface(self.decoder, self.index, self.params, stream)

def decide_surface_format(chroma_format, bit_depth, supported_surface_formats, allow_high = False):
    """Select the best surface format for the given video parameters.

    Args:
        chroma_format: Video chroma subsampling (YUV420, YUV444, etc.)
        bit_depth: Bits per color channel (usually 8 or 10)
        supported_surface_formats: Formats supported by this GPU/codec combination
        allow_high: If True, use 16-bit output for >8-bit video. If False,
            always use 8-bit output (lossy for HDR content).

    Returns:
        The selected cudaVideoSurfaceFormat.

    Raises:
        Exception: If the chroma format is unsupported or no compatible
            surface format is available.
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
    """NVDEC hardware decoder with callback-based frame delivery.

    Decodes video packets using NVIDIA's CUVID API. For each decoded frame,
    the provided on_recv callback is invoked with a Picture object. This
    is the low-level decoder; most users should use Player or Screenshoter
    from nvidia_codec.utils instead.

    The decoder manages GPU memory pools for Pictures and Surfaces. Configure
    pool sizes via the decide callback to balance memory usage and throughput.
    """

    def catch_exception(self, func, return_on_error = 0):
        """Wrap callbacks to capture exceptions and return error codes."""
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
            # Only compare fields that affect decoder configuration
            # frame_rate and other metadata can vary after seek within the same video
            essential = ['codec', 'coded_width', 'coded_height', 'chroma_format', 'bit_depth_luma_minus8']
            mismatches = [
                (f, getattr(self.video_format, f), getattr(vf, f))
                for f in essential
                if getattr(self.video_format, f) != getattr(vf, f)
            ]
            if not mismatches:
                return self.decode_create_info.ulNumDecodeSurfaces
            else:
                diff = ", ".join(f"{k}: {old} -> {new}" for k, old, new in mismatches)
                raise Exception(f"decoder already initialized with different format: {diff}")
        memmove(byref(self.video_format), byref(vf), sizeof(CUVIDEOFORMAT))
        # save for user use

        caps = CUVIDDECODECAPS(
            eCodecType=vf.codec,
            eChromaFormat=vf.chroma_format,
            nBitDepthMinus8=vf.bit_depth_luma_minus8
        )

        with cuda.Device(self.device):
            cuda.check(nvcuvid.cuvidGetDecoderCaps(byref(caps)))
        if caps.bIsSupported != 1:
            raise CodecNotSupportedError(f"Codec not supported: {cudaVideoCodec(vf.codec)}")
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
        """Video codec (cudaVideoCodec enum)."""
        return self.decode_create_info.CodecType

    @property
    def height(self):
        """Original video height in pixels (before scaling)."""
        return  self.video_format.display_area.bottom - self.video_format.display_area.top

    @property
    def width(self):
        """Original video width in pixels (before scaling)."""
        return self.video_format.display_area.right - self.video_format.display_area.left

    @property
    def target_width(self):
        """Output width in pixels (after scaling)."""
        return self.decode_create_info.ulTargetWidth

    @property
    def target_height(self):
        """Output height in pixels (after scaling)."""
        return self.decode_create_info.ulTargetHeight

    @property
    def surface_format(self):
        """Output surface format (NV12, P016, YUV444, etc.)."""
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
            self.ret = self.on_recv(None, 0, self.ret)
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
        self.ret = self.on_recv(picture, di.timestamp, self.ret)
        return 1

    def handleOperatingPoint(self, pUserData, pOPInfo):
        """Handle AV1 operating point selection (internal callback)."""
        opi = pOPInfo.contents
        if opi.codec == cudaVideoCodec.AV1:
            if opi.av1.operating_points_cnt > 1:
                if self.operating_point >= opi.av1.operating_points_cnt:
                    self.operating_point = 0
                return self.operating_point | (1 << 10 if self.disp_all_layers else 0)
        return -1

    def __init__(self, codec: cudaVideoCodec, decide = lambda p: {}, device = None, extradata = None, coded_width = 0, coded_height = 0):
        """Initialize the decoder.

        Args:
            codec: Video codec (e.g., cudaVideoCodec.H264, cudaVideoCodec.HEVC)
            decide: Callback to configure decoder parameters. Called with a dict:
                - 'chroma_format': cudaVideoChromaFormat (YUV420, YUV444, etc.)
                - 'bit_depth': int (8 or 10)
                - 'supported_surface_formats': list of cudaVideoSurfaceFormat
                - 'min_num_pictures': int (minimum decode buffer size)

                Return a dict with any of these optional overrides:
                - 'num_pictures': Decode buffer size (default: min + 1, max: 32)
                - 'num_surfaces': Output buffer size (default: 2)
                - 'surface_format': Output format (default: auto-selected)
                - 'cropping': Function (h, w) -> {'left', 'top', 'right', 'bottom'}
                - 'target_size': Function (h, w) -> (new_h, new_w) for scaling
                - 'target_rect': Function (h, w) -> {'left', 'top', 'right', 'bottom'}

            device: CUDA device ID (default: current device)
            extradata: Codec sequence header from container (required for VC1/WMV3)
            coded_width: Video width hint for codecs needing extradata
            coded_height: Video height hint for codecs needing extradata
        """
        self.dirty = False
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

        # Prepare extended video info with sequence header for codecs like VC1/WMV3
        self.ext_video_info = None
        pExtVideoInfo = None
        if extradata is not None and len(extradata) > 0:
            self.ext_video_info = CUVIDEOFORMATEX()
            # Set format info from container (needed for VC1 Simple/Main which lacks in-band headers)
            self.ext_video_info.format.codec = codec
            self.ext_video_info.format.coded_width = coded_width
            self.ext_video_info.format.coded_height = coded_height
            self.ext_video_info.format.chroma_format = 1  # YUV420 - most common
            # Copy extradata into raw_seqhdr_data (max 1024 bytes)
            copy_len = min(len(extradata), 1024)
            memmove(self.ext_video_info.raw_seqhdr_data, extradata, copy_len)
            self.ext_video_info.format.seqhdr_data_length = copy_len
            pExtVideoInfo = pointer(self.ext_video_info)
            log.debug(f"Passing {copy_len} bytes of extradata to parser, size={coded_width}x{coded_height}")

        p = CUVIDPARSERPARAMS(
            CodecType=codec,
            ulMaxNumDecodeSurfaces=0,
            ulErrorThreshold=0,
            ulMaxDisplayDelay=0,
            pUserData=None,
            pfnSequenceCallback=self.handleVideoSequenceCallback,
            pfnDecodePicture=self.handlePictureDecodeCallback,
            pfnDisplayPicture=self.handlePictureDisplayCallback,
            pfnGetOperatingPoint=self.handleOperatingPointCallback,
            pExtVideoInfo=pExtVideoInfo
            )
        
        self.device = cuda.get_current_device(device)
        with self.condition:        
            self.cuvid_parser = CUvideoparser() # NULL, to be filled in next 
            self.cuvid_decoder = CUvideodecoder() # NULL, to be filled in later          
            cuda.check(nvcuvid.cuvidCreateVideoParser(byref(self.cuvid_parser), byref(p)))
            self.condition.notify_all()

    def send(self, packet, on_recv, pts = 0):
        """Send a compressed packet to the decoder.

        The decoder will parse the packet and invoke on_recv for each decoded
        frame. Frames are delivered in display order (after B-frame reordering).

        Args:
            packet: Compressed video data as a 1D numpy array, or None to
                signal end-of-stream (triggers EOS callback with picture=None).
            on_recv: Callback function(picture, pts, accumulator) -> result.
                - picture: Picture object, or None at end of stream
                - pts: Presentation timestamp passed to send()
                - accumulator: Return value from previous on_recv call
                The final on_recv return value is returned by send().
            pts: Presentation timestamp for this packet (passed to on_recv).

        Returns:
            The final return value from on_recv, or None if no frames decoded.

        Note:
            The packet buffer can be reused immediately after send() returns.
            Call flush() to drain any remaining frames from the reorder buffer.
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
        self.ret = None
        self.exception = None
        self.on_recv = on_recv
        with cuda.Device(self.device):
            try:
                cuda.check(nvcuvid.cuvidParseVideoData(self.cuvid_parser, byref(p)))
            except cuda.CUError:
                # If a callback stored an exception, raise that instead of CUError
                if self.exception is not None:
                    raise self.exception
                raise
        # catch: cuvidParseVideoData will not propagate error return code 0 of HandleVideoSequenceCallback
        # it still returns CUDA_SUCCESS and simply ignore all future incoming packets
        # therefore we must check the exception here, even if the last call succeeded
        if self.exception is not None:
            # the exception is caused by our callback, not by cuvid
            raise self.exception
        return self.ret

    def flush(self):
        """Flush the decoder, discarding any buffered frames.

        Resets the parser state. Call this after seeking to clear any
        frames from the previous position.
        """
        self.send(None, lambda pic,pts,ret: None)

    def free(self):
        """Release all decoder resources.

        Destroys the CUVID parser and decoder, freeing all GPU memory.
        The decoder cannot be used after calling free().
        """
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