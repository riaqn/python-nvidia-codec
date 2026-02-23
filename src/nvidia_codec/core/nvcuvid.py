# This file is the python ctypes version of nvcuvid.h

from .cuviddec import *
from ctypes import *

from enum import Flag, auto


# class CUVIDAV1SEQHDR(Structure):
#     _fields = [('max_width', c_uint),
#                ('max_height', c_uint),
#                ('reserved', c_ubyte * 1016)]

# class _U(Union):
#     _fields_ = [('av1', CUVIDAV1SEQHDR),
#                 ('raw_seqhdr_data', c_ubyte * 1024)]
class FRAMERATE(Structure):
    _fields_ = [('numerator', c_uint),
                ('denominator', c_uint)]

class IRECT(Structure):
    _fields_ = [('left', c_int),
                ('top', c_int),
                ('right', c_int),
                ('bottom', c_int)]
                
class DISPLAYASPECTRATIO(Structure):
    _fields_ = [('x', c_int), 
                ('y', c_int)]

class VIDEO_SIGNAL_DESCRIPTION(Structure):
    _fields_ = [('video_format', c_ubyte, 3),
                ('video_full_range_flag', c_ubyte, 1),
                ('reserved_zero_bits', c_ubyte, 4),
                ('color_primaries', c_ubyte),
                ('transfer_characteristics', c_ubyte),
                ('matrix_coefficients', c_ubyte)]

class CUVIDEOFORMAT(Structure):
    _fields_ = [('codec', c_int), # should be cudaVideoCodec
                ('frame_rate', FRAMERATE),
                ('progressive_sequence', c_ubyte),
                ('bit_depth_luma_minus8', c_ubyte),
                ('bit_depth_chroma_minus8',c_ubyte),
                ('min_num_decode_surfaces', c_ubyte),
                ('coded_width', c_uint),
                ('coded_height', c_uint),
                ('display_area', IRECT),
                ('chroma_format', c_int), # should be cudaVideoChromaFormat
                # ('bitrate', c_uint),
                # ('display_aspect_ratio', DISPLAYASPECTRATIO),
                # ('video_signal_description', VIDEO_SIGNAL_DESCRIPTION),
                # ('seqhdr_data_length', c_uint)
                ]


class CUVIDPICPARAMS(Structure):
    # note: fields incomplete
    # because we only want to access CurrPicIdx
    # and we never need to allocate this structtime
    _fields_ = [
        ('PicWidthInMbs', c_int),
        ('FrameHeightInMbs', c_int),
        ('CurrPicIdx', c_int)
    ]

class AV1(Structure):
    _fields_ = [('operating_points_cnt', c_ubyte),
                ('reserved24_bits', c_ubyte * 3),
                ('operating_points_idc', c_ushort * 32)
    ]

class ANON(Union):
    _fields_ = [('av1', AV1),
                ('CodecReserved', c_ubyte * 1024)]

class CUVIDOPERATINGPOINTINFO(Structure):
    _anonymous_ = ('u',)
    _fields_ = [('codec', c_int), # should be cudaVideoCodec
                ('u', ANON)]

CUvideotimestamp = c_longlong
class CUVIDPARSERDISPINFO(Structure):
    _fields_ = [('picture_index', c_int),
                ('progressive_frame', c_int),
                ('top_field_first', c_int),
                ('repeat_first_field', c_int),
                ('timestamp', CUvideotimestamp)
                ]

PFNVIDSEQUENCECALLBACK = PYFUNCTYPE(CUresult, c_void_p, POINTER(CUVIDEOFORMAT))
PFNVIDDECODECALLBACK = PYFUNCTYPE(CUresult, c_void_p, POINTER(CUVIDPICPARAMS))
PFNVIDDISPLAYCALLBACK = PYFUNCTYPE(CUresult, c_void_p, POINTER(CUVIDPARSERDISPINFO))
PFNVIDOPPOINTCALLBACK = PYFUNCTYPE(CUresult, c_void_p, POINTER(CUVIDOPERATINGPOINTINFO))

class CUVIDPARSERPARAMS(Structure):
    _fields_ = [('CodecType', c_int), #should be cudaVideoCodec
                ('ulMaxNumDecodeSurfaces', c_uint),
                ('ulClockRate', c_uint),
                ('ulErrorThreshold', c_uint),
                ('ulMaxDisplayDelay', c_uint),
                ('bAnnexb', c_uint, 1),
                ('uReserved', c_uint, 31),
                ('uReserved', c_uint * 4),
                ('pUserData', c_void_p),
                #the following four will be overwritten to the C callbacks
                # which invokes the actual python callbacks
                # included in pUserData
                ('pfnSequenceCallback', PFNVIDSEQUENCECALLBACK),
                ('pfnDecodePicture', PFNVIDDECODECALLBACK),
                ('pfnDisplayPicture', PFNVIDDISPLAYCALLBACK),
                ('pfnGetOperatingPoint', PFNVIDOPPOINTCALLBACK),
                ('pvReserved2', c_void_p*6),
                ('pExtVideoInfo', c_void_p) # not used in examples, just leave empty
    ]
CUstream = c_void_p # according to /opt/cuda/include/cuda.h
CUvideoparser = c_void_p # according to nvcuvid.h
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
                ('Reserved2', c_void_p*1) ]
class CUvideopacketflags(Flag):
    ENDOFSTREAM = 1
    TIMESTAMP = auto()
    DISCOUNTINUITY = auto()
    ENDOFPICTURE = auto()
    NOTIFY_EOS = auto()

class CUVIDSOURCEDATAPACKET(Structure):
    _fields_ = [('flags', c_ulong),
                ('payload_size', c_ulong),
                ('payload', POINTER(c_ubyte)),
                ('timestamp', CUvideotimestamp)
                ]