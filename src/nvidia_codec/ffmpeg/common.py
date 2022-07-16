from ctypes import *
from .libavutil import *

class AVException(BaseException):
    def __init__(self, errnum):
        self.errnum = errnum

    def __str__(self):
        return strerror(self.errnum)

def check(result):
    if result < 0:
        raise AVException(result)