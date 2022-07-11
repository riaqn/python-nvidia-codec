from ctypes import * 
CUstream = c_void_p
CUresult = c_int

lib = cdll.LoadLibrary('libcuda.so')

class Error(Exception):
    def __init__(self, curesult):
        self.curesult = curesult

    def __str__(self):
        p = c_char_p()
        lib.cuGetErrorString(self.curesult, byref(p))
        return p.value.decode('utf-8')

def call(cuda_func, *args):
    curesult = cuda_func(*args)
    if curesult > 0:
        raise Error(curesult)