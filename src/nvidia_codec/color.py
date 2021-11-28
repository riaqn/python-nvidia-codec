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

        os = target_format(w, h)
        tf = self.target_format = target_format
        tp = self.target_pitch = os.calculate_pitch()        
        log.debug(f'source pitch {sp}')

        assert target_space == c.AVCOL_SPC_RGB, "Only support RGB as output"

        if sf in [SurfaceP016, SurfaceYUV444_16Bit]:
            stype = 'uint16_t'
        elif sf in [SurfaceNV12, SurfaceYUV444]:
            stype = 'uint8_t'
        else:
            raise Exception(f"Unsupported source format {sf}")

        if tf in [SurfaceRGB48, SurfaceRGB444P16]:
            ttype = 'uint16_t'
            normalize_rgb = f'''
            {ttype} R = min(max(R_ * 65536, .5), 65536 - .5);
            {ttype} G = min(max(G_ * 65536, .5), 65536 - .5);
            {ttype} B = min(max(B_ * 65536, .5), 65536 - .5);
            '''
        elif tf in [SurfaceRGB24, SurfaceRGB444P]:
            ttype = 'uint8_t'
            normalize_rgb = f'''
            {ttype} R = min(max(R_ * 256, .5), 256 - .5);
            {ttype} G = min(max(G_ * 256, .5), 256 - .5);
            {ttype} B = min(max(B_ * 256, .5), 256 - .5);
            '''
        else:
            raise Exception(f"Unsupported target format {tf}")

        if stype == 'uint16_t' and ttype == 'uint16_t':
            mtype = 'double'
        else:
            mtype = 'float'

        if sf in [SurfaceNV12, SurfaceP016]:
            load_yuv = f'''
            {stype} Y = (({stype}*)(src + x * {sp}))[y];
            uint8_t *src_uv = src + {sp} * {h};
            {stype} U = (({stype}*)(src_uv + x/2 * {sp}))[y/2*2];
            {stype} V = (({stype}*)(src_uv + x/2 * {sp}))[y/2*2 + 1];
            '''
        else:
            raise Exception(f"Unsupported source format {sf}")

        if source_range == c.AVCOL_RANGE_MPEG:
            # partial range
            normalize_yuv = f'''
                {mtype} Y_ = ({mtype})(Y - 16) / 220;
                {mtype} U_ = ({mtype})(U - 128) / 225;
                {mtype} V_ = ({mtype})(V - 128) / 225;
            '''
        else:
            # full range
            normalize_yuv = f'''
                {mtype} Y_ = ({mtype})(Y) / 256;
                {mtype} U_ = ({mtype})(U - 128) / 256;
                {mtype} V_ = ({mtype})(V - 128) / 256;
            '''

        if source_space in [c.AVCOL_SPC_BT470BG, c.AVCOL_SPC_SMPTE170M]:
            m = [[1, 0, 1.402], [1, -0.34414, -0.71414], [1, 1.772, 0]]
        elif source_space == c.AVCOL_SPC_BT709:
            m = [[1, 0, 1.5748], [1, -0.1873, -0.4681], [1, 1.8556, 0]]
        elif source_space == c.AVCOL_SPC_BT2020_NCL:
            # https://gist.github.com/yohhoy/dafa5a47dade85d8b40625261af3776a
            m = [[1, 0, 1.4746], [1, -0.1646, -0.5714], [1, 1.8814, 0]]
        else:
            raise Exception(f"Unsupported color space {source_space}")

        yuv_to_rgb = f'''
        {mtype} R_ = {m[0][0]} * Y_ + {m[0][1]} * U_ + {m[0][2]} * V_;
        {mtype} G_ = {m[1][0]} * Y_ + {m[1][1]} * U_ + {m[1][2]} * V_;
        {mtype} B_ = {m[2][0]} * Y_ + {m[2][1]} * U_ + {m[2][2]} * V_;
        '''

        if tf in [SurfaceRGB24, SurfaceRGB48]:
            store_rgb = f'''
            (({ttype}*)(dst + x * {tp}))[y * 3] = R;
            (({ttype}*)(dst + x * {tp}))[y * 3 + 1] = G;
            (({ttype}*)(dst + x * {tp}))[y * 3 + 2] = B;
            '''
        elif tf in [SurfaceRGB444P, SurfaceRGB444P16]:
            store_rgb = f'''
            (({ttype}*)(dst + x * {tp}))[y] = R;
            (({ttype}*)(dst + h * {tp} + x * {tp}))[y] = G;
            (({ttype}*)(dst + h * {tp} * 2 + x * {tp}))[y] = B;
            '''
        else:
            raise Exception(f"Unsupported target format {tf}")

        self.mod =  SourceModule(f"""
        #include <stdint.h>
    __global__ void convert(uint8_t *src, uint8_t *dst) {{
        int x = blockIdx.x * blockDim.x + threadIdx.x; // row
        if (x >= {h}) return;
        int y = blockIdx.y * blockDim.y + threadIdx.y; // column
        if (y >= {w}) return;
        {load_yuv}
        {normalize_yuv}
        {yuv_to_rgb}
        {normalize_rgb}
        {store_rgb}
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