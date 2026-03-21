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
    WMV2 = 18
    H264 = 27
    RV40 = 69
    VC1 = 70
    WMV3 = 71
    MJPEG = 7
    DVVIDEO = 24
    PNG = 61
    JPEG2000 = 88
    VP6F = 92
    GIF = 97
    PRORES = 147
    WEBP = 171
    VP8 = 139
    VP9 = 167
    HEVC = 173
    AV1 = 225
    # Audio codecs — values must match ffmpeg's avcodec.h
    PCM_S16LE = 0x10000
    PCM_S16BE = 0x10001
    MP2 = 0x15000
    COOK = 0x15014
    MP3 = 0x15001
    AAC = 0x15002
    AC3 = 0x15003
    DTS = 0x15004
    VORBIS = 0x15005
    WMAV2 = 0x15008
    FLAC = 0x1500C
    ALAC = 0x15010
    WAVPACK = 0x15019
    MLP = 0x1501D
    SPEEX = 0x15023
    WMAPRO = 0x15025
    WMALOSSLESS = 0x15026
    EAC3 = 0x15028
    TRUEHD = 0x1502C
    AAC_LATM = 0x15031
    OPUS = 0x1503C

class AVCodecParameters(Structure):
    _fields_ = [
        ('codec_type', AVMediaType),
        ('codec_id', AVCodecID),
        ('codec_tag', c_uint32),
        ('extradata', POINTER(c_uint8)),
        ('extradata_size', c_int),
        ('coded_side_data', c_void_p),
        ('nb_coded_side_data', c_int),
        ('format', c_int),
        ('bit_rate', c_int64),
        ('bits_per_coded_sample', c_int),
        ('bits_per_raw_sample', c_int),
        ('profile', c_int),
        ('level', c_int),
        ('width', c_int),
        ('height', c_int),
        ('sample_aspect_ratio', AVRational),
        ('framerate', AVRational),
        ('field_order', c_int),
        ('color_range', AVColorRange),
        ('color_primaries', c_int),
        ('color_trc', c_int),
        ('color_space', AVColorSpace),
        ('chroma_location', c_int),
        ('video_delay', c_int),
        # AVChannelLayout: order(4) + nb_channels(4) + union(8) + opaque(8) = 24
        ('_ch_order', c_int),
        ('nb_channels', c_int),
        ('_ch_union', c_uint8 * 8),
        ('_ch_opaque', c_void_p),
        ('sample_rate', c_int),
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


AV_PKT_FLAG_KEY     = 0x0001
AV_PKT_FLAG_CORRUPT = 0x0002
AV_PKT_FLAG_DISCARD = 0x0004
AV_PKT_FLAG_TRUSTED = 0x0008
AV_PKT_FLAG_DISPOSABLE = 0x0010

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
