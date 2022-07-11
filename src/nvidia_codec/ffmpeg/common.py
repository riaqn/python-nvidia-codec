from ctypes import *
from .libavutil import *

class AVException(BaseException):
    def __init__(self, errnum):
        self.errnum = errnum
        super().__init__(strerror(errnum))

def call(func, *args):
    val = func(*args)
    if val < 0:
        raise AVException(val)