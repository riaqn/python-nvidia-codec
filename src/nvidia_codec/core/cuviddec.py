# this file is the python ctypes version of cuviddec.h

from ctypes import *
from .cuda import *
from ..common import TypedCEnumeration

class cudaVideoSurfaceFormat(TypedCEnumeration(c_int)):
    NV12 = 0
    P016 = 1
    YUV444 = 2
    YUV444_16Bit = 3

class cudaVideoCodec(TypedCEnumeration(c_int)):
    MPEG1 = 0
    MPEG2 = 1
    MPEG4 = 2
    VC1  = 3
    H264 = 4
    JPEG = 5
    H264_SVC = 6
    H264_MVC = 7
    HEVC = 8
    VP8 = 9
    VP9 = 10
    AV1 = 11

class cudaVideoChromaFormat(TypedCEnumeration(c_int)):
    MONOCHROME = 0
    YUV420 = 1
    YUV422 = 2
    YUV444 = 3

class cudaVideoCreateFlags(TypedCEnumeration(c_int)):
    Default = 0
    PreferCUDA = 1
    PreferDXVA = 2
    PreferCUVID = 3

class cudaVideoDeinterlaceMode(TypedCEnumeration(c_int)):
    Weave = 0
    Bob = 1
    Adaptive = 2

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
        ('CodecType', cudaVideoCodec), 
        ('ChromaFormat', cudaVideoChromaFormat), 
        ('ulCreationFlags', c_ulong),
        ('bitDepthMinus8', c_ulong),
        ('ulIntraDecodeOnly', c_ulong),
        ('ulMaxWidth', c_ulong),
        ('ulMaxHeight', c_ulong),
        ('Reserved1', c_ulong),
        ('display_area', SRECT),
        ('OutputFormat', cudaVideoSurfaceFormat), 
        ('DeinterlaceMode', cudaVideoDeinterlaceMode), 
        ('ulTargetWidth', c_ulong),
        ('ulTargetHeight', c_ulong),
        ('ulNumOutputSurfaces', c_ulong),
        ('vidLock', CUvideoctxlock),
        ('target_rect', SRECT),
        ('enableHistogram', c_ulong),
        ('Reserved2', c_ulong * 4)
    ]

class CUVIDDECODECAPS(Structure):
    _fields_ = [('eCodecType', cudaVideoCodec), 
                ('eChromaFormat', cudaVideoChromaFormat), 
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