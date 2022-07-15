from ctypes import * 
import logging

log = logging.getLogger(__name__)

CUstream = c_void_p
CUcontext = c_void_p
CUresult = c_int
cudaError = c_int

lib = cdll.LoadLibrary('libcuda.so')
librt = cdll.LoadLibrary('libcudart.so')

class CUError(Exception):
    def __init__(self, cuResult):
        self.cuResult = cuResult

    def __str__(self):
        p = c_char_p()
        lib.cuGetErrorString(self.cuResult, byref(p))
        return p.value.decode('utf-8')

def get_current_device(device = None):
    '''
    get the current device, unless specified by device
    '''
    if device is None:
        device = c_int()
        check_rt(librt.cudaGetDevice(byref(device)))
        return device.value
    else:
        return device

cudaGetErrorString = librt.cudaGetErrorString
cudaGetErrorString.restype = c_char_p

class CUDAError(Exception):
    def __init__(self, cudaError):
        self.cudaError = cudaError

    def __str__(self):
        p = cudaGetErrorString(c_int(self.cudaError))
        return p.decode('utf-8')


def check(cuResult):
    if cuResult > 0:
        raise CUError(cuResult)

def check_rt(cudaError):
    if cudaError > 0:
        raise CUDAError(cudaError)

class Device:
    def __init__(self, idx : int):
        self.idx = idx
        self.prev = c_int(-1)

    def __int__(self):
        return self.idx

    def __enter__(self):
        check_rt(librt.cudaGetDevice(byref(self.prev)))
        log.debug(f'entering {self.prev.value} -> {self.idx} ')
        check_rt(librt.cudaSetDevice(self.idx))
        check_rt(librt.cudaFree(0)) # ensure init
    
    def __exit__(self,  type, value, traceback):
        log.debug(f'exiting {self.idx} -> {self.prev.value} ')
        check_rt(librt.cudaSetDevice(self.prev))

# def get_current_context():
#     k = CUcontext()
#     check(lib.cuCtxGetCurrent(byref(k)))
#     return k
    
# class Context:
#     def __init__(self, ctx):
#         self.ctx = ctx

#     def __int__(self):
#         return self.ctx.value()

#     def __enter__(self):
#         check(lib.cuCtxPushCurrent(self.ctx))
    
#     def __exit__(self,  type, value, traceback):
#         check(lib.cuCtxPopCurrent(byref(self.prev)))

