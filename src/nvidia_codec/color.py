from pycuda.compiler import SourceModule
import pycuda.driver as cuda
from . import decode
import numpy as np
from .cuda import call as CUDAcall
from .surface import *
from .decode import *

import cppyy

cppyy.c_include('libavutil/avutil.h')

c = cppyy.gbl


'''
We are doing some metaprogramming; according to pycuda, 
constant is much faster than variable in e.g. multipllication
it's therefore worthwhile to compile the specialized code
this should be faster than the official color cvt in NPP

source_template should be an object of a subclass of extras.Surface or decode.Surface

color space and color ranges are defined as in ffmpeg pixfmt.h

It doesn't handle cuda context switching - 
ensure it operates under the same context
'''
class Converter:
    def __init__(self, source_template, source_space, source_range, 
    target_format = SurfaceRGB24, target_space = c.AVCOL_SPC_RGB, target_range = c.AVCOL_RANGE_JPEG):
        # template is typically a Surface
        st = source_template
        w = self.width = st.width
        h = self.height = st.height
        sf = self.source_format = type(source_template)
        sp = self.source_pitch = st.pitch
        log.debug(f'source pitch {sp}')

        assert sf is SurfaceNV12,  "Only support NV12 as input"
        assert target_format is SurfaceRGB24, "Only support RGB24 as output"
        assert target_space == c.AVCOL_SPC_RGB, "Only support RGB as output"
        assert target_range == c.AVCOL_RANGE_JPEG, "Only support JPEG (full range) as output"

        if source_space in [c.AVCOL_SPC_BT470BG, c.AVCOL_SPC_SMPTE170M]:
            m = [[1, 0, 1.402], [1, -0.34414, -0.71414], [1, 1.772, 0]]
        elif source_space == c.AVCOL_SPC_BT709:
            m = [[1, 0, 1.5748], [1, -0.1873, -0.4681], [1, 1.8556, 0]]
        else:
            raise Exception(f"Unsupported color space {source_space}")

        if source_range == c.AVCOL_RANGE_MPEG:
            # limited range: Y is [16, 235], Cb and Cr are [16, 240]            
            f = False 
        else:
            f = True # source is full range

        os = target_format(w, h)
        tf = self.target_format = target_format
        tp = self.target_pitch = os.calculate_pitch()
        log.debug(f'target pitch = {tp}')

        self.mod =  SourceModule(f"""
        #include <stdint.h>
    __global__ void convert(uint8_t *src, uint8_t *dst) {{
        int x = blockIdx.x * blockDim.x + threadIdx.x; // row
        if (x >= {h}) return;
        int y = blockIdx.y * blockDim.y + threadIdx.y; // column
        if (y >= {w}) return;
        float Y = src[x * {sp} + y];
        float Y_ = {'Y/256' if f else '(Y - 16)/220'};
        uint8_t *src_uv = src + {sp} * {h};
        float U = src_uv[x/2 * {sp} + y/2*2];
        float U_ = {'(U-128)/256' if f else '(U - 128)/225'};
        float V = src_uv[x/2 * {sp} + y/2*2 + 1];
        float V_ = {'(V-128)/256' if f else '(V - 128)/225'};
        // Y_, U_, V_ are in range [0, 1)

        float R_ = {m[0][0]} * Y_ + {m[0][1]} * U_ + {m[0][2]} * V_;
        float G_ = {m[1][0]} * Y_ + {m[1][1]} * U_ + {m[1][2]} * V_;
        float B_ = {m[2][0]} * Y_ + {m[2][1]} * U_ + {m[2][2]} * V_;
        // R_, G_, B_ should be in range [0, 1)

        dst[x * {tp} + y * 3] = min(max(R_ * 256, .5), 255.5);
        dst[x * {tp} + y * 3 + 1] = min(max(G_ * 256, .5), 255.5);
        dst[x * {tp} + y * 3 + 2] = min(max(B_ * 256, .5), 255.5);
    }}
    """)
        self.convert = self.mod.get_function("convert")

    def __call__(self, surface, block = (32, 32, 1), check = True, **kwargs):
        if check:
            assert surface.height == self.height, "Surface height mismatch"
            assert surface.width == self.width, "Surface width mismatch"
            assert surface.pitch == self.source_pitch, f"Surface pitch mismatch, {surface.pitch} {self.source_pitch}"
            assert type(surface) is self.source_format, "Surface format mismatch"

        grid = ((surface.height - 1) // block[0] + 1, (surface.width - 1) // block[1] + 1)
        os = self.target_format(self.width, self.height)
        alloc = cuda.mem_alloc(self.target_pitch * os.height_in_rows)
        os.alloc = alloc
        os.pitch = self.target_pitch
        log.debug(f'{type(surface.alloc)}, {type(os.alloc)}')
        self.convert(np.ulonglong(surface.alloc), np.ulonglong(os.alloc), block=block, grid = grid, **kwargs)
        return os