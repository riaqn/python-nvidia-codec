from ctypes import *
from .libavutil import *
from .libavcodec import *

class AVStream(Structure):
    _fields_ = [
        ('av_class', c_void_p),
        ('index', c_int),
        ('id', c_int),
        ('codecpar', POINTER(AVCodecParameters)),
        ('priv_data', c_void_p),
        ('time_base', AVRational),
        ('start_time', c_int64),
        ('duration', c_int64),
        ('nb_frames', c_int64),
        ('disposition', c_int),
        ('discard', c_int),
        ('sample_aspect_ratio', AVRational),
        ('metadata', c_void_p),
        ('avg_frame_rate', AVRational),
        ('attached_pic', AVPacket),
    ]

class AVFormatContext(Structure):
    _fields_ = [
        ('av_class', c_void_p),
        ('iformat', c_void_p),
        ('oformat', c_void_p),
        ('priv_data', c_void_p),
        ('pb', c_void_p),
        ('ctx_flags', c_int),
        ('nb_streams', c_uint),
        ('streams', POINTER(POINTER(AVStream))),
        ('url', c_char_p),
        ('start_time', c_int64),
        ('duration', c_int64),
        ('bit_rate', c_int64),
    ]
