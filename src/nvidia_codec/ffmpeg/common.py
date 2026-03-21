from ctypes import *
from errno import ENOENT
import struct
from .libavutil import *

AVERROR_ENOENT = -ENOENT
AVERROR_INVALIDDATA = -(ord('I') | (ord('N') << 8) | (ord('D') << 16) | (ord('A') << 24))

class AVException(Exception):
    def __init__(self, errnum):
        self.errnum = errnum

    def __str__(self):
        return strerror(self.errnum)

def check(result):
    if result < 0:
        raise AVException(result)