import pycuda.driver  as cuda
from .cuda import call as CUDAcall
import numpy as np
import logging

log = logging.getLogger(__name__)

# extra surface formats
class Surface:
    def calculate_pitch(self):
        device = cuda.Context.get_device()
        texture_alignment = device.get_attribute(cuda.device_attribute.TEXTURE_ALIGNMENT)
        return (self.width_in_bytes + texture_alignment - 1) // texture_alignment * texture_alignment    
        
        # elif self.format == SurfaceFormat.P016:
        #     assert self.height % 2 == 0
        #     return cuda.mem_alloc_pitch(self.width * 2, self.height * 1.5, 2)
        # elif self.format == SurfaceFormat.YUV444P:
        #     return cuda.mem_alloc_pitch(self.width, self.height * 3, 1)
        # elif self.format == SurfaceFormat.YUV444P16:
        #     return cuda.mem_alloc_pitch(self.width * 2, self.height * 3, 2)
    '''
    width and height are in pixels
    dry_run: if True, don't actually allocate memory, just calculate the pitch
    '''
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def download(self, stream = None):
        arr = cuda.pagelocked_empty((self.height_in_rows, self.width_in_bytes), dtype=np.uint8)
        m = cuda.Memcpy2D()
        m.set_src_device(self.alloc)
        m.src_pitch = self.pitch
        m.set_dst_host(arr)
        m.dst_pitch = arr.strides[0]
        m.width_in_bytes = self.width_in_bytes
        m.height = self.height_in_rows
        m(stream)
        return arr        

    def __del__(self):
        ## no need to free self.dev_alloc; ref-count anyway 
        pass

class SurfaceRGB24(Surface):
    @property
    def width_in_bytes(self):
        return self.width * 3

    @property
    def height_in_rows(self):
        return self.height

class SurfaceRGB444P(Surface):
    @property
    def width_in_bytes(self):
        return self.width

    @property
    def height_in_rows(self):
        return self.height * 3

class SurfaceNV12(Surface):
    @property
    def width_in_bytes(self):
        return self.width 

    @property
    def height_in_rows(self):
        # assert self.height % 2 == 0
        return self.height // 2 * 3

class SurfaceP016(Surface):
    @property
    def width_in_bytes(self):
        return self.width * 2

    @property
    def height_in_rows(self):
        return self.height // 2 * 3

class SurfaceYUV444(Surface):
    @property
    def width_in_bytes(self):
        return self.width

    @property
    def height_in_rows(self):
        return self.height * 3

class SurfaceYUV444_16Bit(Surface):
    @property
    def width_in_bytes(self):
        return self.width * 2
    
    @property
    def height_in_rows(self):
        return self.height * 3