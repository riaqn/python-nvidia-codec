from ctypes import *

from .include.libavutil import *

lib = cdll.LoadLibrary('libavutil.so')

def strerror(errnum):
    buf = (c_char * 256)()
    lib.av_strerror(errnum, buf, 256)
    return buf.value.decode('utf-8')

def shape2size(self, shape):
    if self == AVPixelFormat.RGB24:
        c, h, w = shape
        assert c == 3
    elif self in [AVPixelFormat.YUV420P, AVPixelFormat.YUV420P16LE]:
        h, w = shape
        assert h % 3 == 0 
        h = h // 3 * 2
    elif self in [AVPixelFormat.YUV444P, AVPixelFormat.YUV444P16LE]:
        c, h, w = shape
        assert c == 3
    else:
        raise Exception(f'unsupported pixel format {self}')
    return (h, w)

def size2shape(self, size):
    h,w = size
    if self == AVPixelFormat.RGB24:
        return (3, h, w)
    elif self in [AVPixelFormat.YUV420P, AVPixelFormat.YUV420P16LE]:
        assert h % 2 == 0
        return (h//2 * 3, w)
    elif self in [AVPixelFormat.YUV444P, AVPixelFormat.YUV444P16LE]:
        return (3, h, w)
    else:
        raise Exception(f'unsupported pixel format {self}')

def typestr(self):
    if self in [AVPixelFormat.RGB24, AVPixelFormat.YUV420P, AVPixelFormat.YUV444P]:
        return '|u1'
    elif self in [AVPixelFormat.YUV420P16LE, AVPixelFormat.YUV444P16LE]:
        return '<u2'
    else:
        raise Exception(f'unsupported pixel format {self}')

def atom_size(self):
    return int(self.typestr[2:])