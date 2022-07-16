from ...common import TypedCEnumeration
from ctypes import *
from .libavutil import *

class AVBitStreamFilter(Structure):
    pass

class AVMediaType(TypedCEnumeration(c_int)):
    UNKNOWN = -1
    VIDEO = 0
    AUDIO = 1
    DATA = 2
    SUBTITLE = 3
    ATTACHMENT = 4
    NB = 5

class AVCodecID(TypedCEnumeration(c_int)):
    MPEG1 = 1
    MPEG2 = 2
    MPEG4 = 12
    VC1 = 70
    H264 = 27
    JPEG = 88    
    HEVC = 173
    VP8 = 139
    VP9 = 167
    AV1 = 226

class AVCodecParameters(Structure):
    _fields_ = [
        ('codec_type', AVMediaType),
        ('codec_id', AVCodecID),
        ('codec_tag', c_uint32),
        ('extradata', POINTER(c_uint8)),
        ('extradata_size', c_int),
        ('format', c_int),
        ('bitrate', c_int64),
        ('bits_per_coded_sample', c_int),
        ('bits_per_raw_sample', c_int),
        ('profile', c_int),
        ('level', c_int),
        ('width', c_int),
        ('height', c_int),
        ('sample_aspect_ratio', AVRational),
        ('field_order', c_int),
        ('color_range', AVColorRange),
        ('color_primaries', c_int),
        ('color_trc', c_int),
        ('color_space', AVColorSpace)
        # following are skipped
    ]    


class AVBSFContext(Structure):
    _fields_ = [
        ('av_class', c_void_p),
        ('filter', POINTER(AVBitStreamFilter)),
        ('priv_data', c_void_p),
        ('par_in', POINTER(AVCodecParameters)),
        ('par_out', POINTER(AVCodecParameters)),
        ('time_base_in', AVRational),
        ('time_base_out', AVRational)
    ]


class AVPacket(Structure):
    _fields_ = [
        ('buf', c_void_p),
        ('pts', c_int64),
        ('dts', c_int64),
        ('data', POINTER(c_uint8)),
        ('size', c_int),
        ('stream_index', c_int),
        ('flags', c_int),
        ('side_data', c_void_p),
        ('side_data_elems', c_int),
        ('duration', c_int64),
        ('pos', c_int64),
        ('opaque', c_void_p),
        ('opaque_ref', c_void_p),
        ('time_base', AVRational)
    ]
