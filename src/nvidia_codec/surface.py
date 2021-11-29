from enum import Enum, auto
import pycuda.driver as cuda
import numpy as np
import logging

log = logging.getLogger(__name__)

class SurfaceFormat(Enum):
    RGB24 = auto()
    RGB48 = auto()
    RGB444P = auto()
    RGB444P16 = auto()
    RGB444P16F = auto()
    YUV420P = auto()
    YUV420P16 = auto()
    YUV444P = auto()
    YUV444P16 = auto()

# extra surface formats
class Surface:
    format2wib = {
        SurfaceFormat.RGB24 : 3,
        SurfaceFormat.RGB48 : 6,
        SurfaceFormat.RGB444P : 1,
        SurfaceFormat.RGB444P16 : 2,
        SurfaceFormat.RGB444P16F : 2,
        SurfaceFormat.YUV420P : 1,
        SurfaceFormat.YUV420P16 : 2,
        SurfaceFormat.YUV444P : 1,
        SurfaceFormat.YUV444P16 : 1,
    }

    format2hir = {
        SurfaceFormat.RGB24 : 1,
        SurfaceFormat.RGB48 : 1,
        SurfaceFormat.RGB444P : 3,
        SurfaceFormat.RGB444P16 : 3,
        SurfaceFormat.RGB444P16F : 3,
        SurfaceFormat.YUV420P : 1.5,
        SurfaceFormat.YUV420P16 : 1.5,
        SurfaceFormat.YUV444P : 3,
        SurfaceFormat.YUV444P16 : 3,
    }

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
    def __init__(self, width, height, format):
        self.width = width
        self.height = height
        self.format = format

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

    @property
    def width_in_bytes(self):
        return round(self.format2wib[self.format] * self.width)

    @property
    def height_in_rows(self):
        return round(self.format2hir[self.format] * self.height)