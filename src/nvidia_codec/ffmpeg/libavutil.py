from ctypes import *
from .include.libavutil import AVDictionaryEntry

lib = cdll.LoadLibrary('libavutil.so')

lib.av_dict_get.restype = POINTER(AVDictionaryEntry)
lib.av_dict_get.argtypes = [c_void_p, c_char_p, c_void_p, c_int]

# Suppress FFmpeg stderr output globally (e.g. H.264 parser "non-existing PPS").
# av_log_set_level alone doesn't work — some messages bypass the level check.
# We set a no-op callback instead. We check errors via return codes.
_AV_LOG_CALLBACK = CFUNCTYPE(None, c_void_p, c_int, c_char_p, c_void_p)

@_AV_LOG_CALLBACK
def _silent_log(ptr, level, fmt, vl):
    pass

lib.av_log_set_callback(_silent_log)

def dict_get(metadata, key):
    """Get a metadata tag value. Returns string or None."""
    entry = lib.av_dict_get(metadata, key.encode('utf-8'), None, 0)
    if entry:
        return entry.contents.value.decode('utf-8')
    return None

def strerror(errnum):
    buf = (c_char * 256)()
    lib.av_strerror(c_int(errnum), buf, 256)
    return buf.value.decode('utf-8')


# def typestr(self):
#     if self in [AVPixelFormat.RGB24, AVPixelFormat.YUV420P, AVPixelFormat.YUV444P]:
#         return '|u1'
#     elif self in [AVPixelFormat.YUV420P16LE, AVPixelFormat.YUV444P16LE]:
#         return '<u2'
#     else:
#         raise Exception(f'unsupported pixel format {self}')

def atom_size(self):
    return int(self.typestr[2:])