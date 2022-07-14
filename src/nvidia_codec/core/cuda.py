from ctypes import * 
CUstream = c_void_p
CUresult = c_int

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

class CUDAError(Exception):
    def __init__(self, cudaError):
        self.cudaError = cudaError

    def __str__(self):
        p = librt.cudaGetErrorString(self.cudaError)
        return p.value.decode('utf-8')

def check(cuResult):
    if cuResult > 0:
        raise CUError(cuResult)

def check_rt(cudaError):
    if cudaError > 0:
        raise CUDAError(cudaError)

class Device:
    def __init__(self, idx):
        self.idx = idx
        self.prev = c_int(-1)

    def __int__(self):
        return self.idx

    def __enter__(self):
        check_rt(librt.cudaGetDevice(byref(self.prev)))
        if self.idx != self.prev:
            check_rt(librt.cudaSetDevice(self.idx))
    
    def __exit__(self,  type, value, traceback):
        if self.idx != self.prev:
            check_rt(librt.cudaSetDevice(self.prev))