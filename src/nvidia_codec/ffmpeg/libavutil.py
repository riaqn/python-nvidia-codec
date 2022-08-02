from ctypes import *

lib = cdll.LoadLibrary('libavutil.so')

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