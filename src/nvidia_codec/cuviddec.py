# this file is the python ctypes version of cuviddec.h

from ctypes import *
from .cuda import *
from enum import Enum, auto

class cudaVideoSurfaceFormat(Enum):
    NV12 = 0
    P016 = auto()
    YUV444 = auto()
    YUV444_16Bit = auto()

class cudaVideoCodec(Enum):
    MPEG1 = 0
    MPEG2 = auto()
    MPEG4 = auto()
    VC1  = auto()
    H264 = auto()
    JPEG = auto()
    H264_SVC = auto()
    H264_MVC = auto()
    HEVC = auto()
    VP8 = auto()
    VP9 = auto()
    AV1 = auto()

class cudaVideoChromaFormat(Enum):
    MONOCHROME = 0
    YUV420 = auto()
    YUV422 = auto()
    YUV444 = auto()

class cudaVideoCreateFlags(Enum):
    Default = 0
    PreferCUDA = auto()
    PreferDXVA = auto()
    PreferCUVID = auto()

class cudaVideoDeinterlaceMode(Enum):
    Weave = 0
    Bob = auto()
    Adaptive = auto()


class CUVIDPROCPARAMS(Structure):
    _fields_ = [('progressive_frame', c_int),
                ('second_field', c_int),
                ('top_field_first', c_int),
                ('unpaired_field', c_int),
                ('reserved_flags', c_uint),
                ('reserved_zero', c_uint),
                ('raw_input_dptr', c_ulonglong),
                ('raw_input_pitch', c_uint),
                ('raw_input_format', c_uint),
                ('raw_output_dptr', c_ulonglong),
                ('raw_output_pitch', c_uint),
                ('Reserved1', c_uint),
                ('output_stream', CUstream),
                ('Reserved', c_uint * 46),
                ('histogram_dptr', POINTER(c_ulonglong)),
                ('Reserved2', c_void_p * 1)]

class SRECT(Structure):
    _fields_ = [('left', c_short),
                ('top', c_short),
                ('right', c_short),
                ('bottom', c_short)]
CUvideodecoder = c_void_p # opaque
CUvideoctxlock = c_void_p # opaque                

class CUVIDDECODECREATEINFO(Structure):
    _fields_ = [
        ('ulWidth', c_ulong),
        ('ulHeight', c_ulong),
        ('ulNumDecodeSurfaces', c_ulong),
        ('CodecType', c_int), # should be cudaVideoCodec
        ('ChromaFormat', c_int), # should be cudaVideoChromaFormat
        ('ulCreationFlags', c_ulong),
        ('bitDepthMinus8', c_ulong),
        ('ulIntraDecodeOnly', c_ulong),
        ('ulMaxWidth', c_ulong),
        ('ulMaxHeight', c_ulong),
        ('Reserved1', c_ulong),
        ('display_area', SRECT),
        ('OutputFormat', c_int), # should be cudaVideoSurfaceFormat        
        ('DeinterlaceMode', c_int), # should be cudaVideoDeinterlaceMode
        ('ulTargetWidth', c_ulong),
        ('ulTargetHeight', c_ulong),
        ('ulNumOutputSurfaces', c_ulong),
        ('vidLock', CUvideoctxlock),
        ('target_rect', SRECT),
        ('enableHistogram', c_ulong),
        ('Reserved2', c_ulong * 4)
    ]

class CUVIDDECODECAPS(Structure):
    _fields_ = [('eCodecType', c_int), # should be cudaVideoCodec
                ('eChromaFormat', c_int), # should be cudaVideoChromaFormat
                ('nBitDepthMinus8', c_uint),
                ('reserved1', c_uint*3),
                ('bIsSupported', c_ubyte),
                ('nNumNVDECs', c_ubyte),
                ('nOutputFormatMask', c_ushort),
                ('nMaxWidth', c_uint),
                ('nMaxHeight', c_uint),
                ('nMaxMBCount', c_uint),
                ('nMinWidth', c_ushort),
                ('nMinHeight', c_ushort),
                ('bIsHistogramSupported', c_ubyte),
                ('nCounterBitDepth', c_ubyte),
                ('nMaxHistogramBins', c_ushort),
                ('reserved3', c_uint * 10)
                ]