from pycuda.compiler import SourceModule
import pycuda.driver as cuda
import numpy as np
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
    target_format = SurfaceFormat.RGB24, target_space = c.AVCOL_SPC_RGB, target_range = c.AVCOL_RANGE_JPEG):
        # template is typically a Surface
        st = source_template
        w = self.width = st.width
        h = self.height = st.height
        sf = self.source_format = st.format
        sp = self.source_pitch = st.pitch

        ts = Surface(w, h, target_format)
        tf = self.target_format = target_format
        tp = self.target_pitch = ts.calculate_pitch()        
        log.debug(f'source pitch {sp}')

        assert target_space == c.AVCOL_SPC_RGB, "Only support RGB as output"
        assert target_range == c.AVCOL_RANGE_JPEG, "Only support JPEG (full range) as output"

        if sf in [SurfaceFormat.YUV444P16, SurfaceFormat.YUV420P16]:
            stype = 'uint16_t'
        elif sf in [SurfaceFormat.YUV444P, SurfaceFormat.YUV420P]:
            stype = 'uint8_t'
        else:
            raise Exception(f"Unsupported source format {sf}")

        if tf in [SurfaceFormat.RGB48, SurfaceFormat.RGB444P16]:
            ttype = 'uint16_t'
            normalize_rgb = f'''
            {ttype} R = min(max(R_ * 65536, .5), 65536 - .5);
            {ttype} G = min(max(G_ * 65536, .5), 65536 - .5);
            {ttype} B = min(max(B_ * 65536, .5), 65536 - .5);
            '''
        elif tf in [SurfaceFormat.RGB24, SurfaceFormat.RGB444P]:
            ttype = 'uint8_t'
            normalize_rgb = f'''
            {ttype} R = min(max(R_ * 256, .5), 256 - .5);
            {ttype} G = min(max(G_ * 256, .5), 256 - .5);
            {ttype} B = min(max(B_ * 256, .5), 256 - .5);
            '''
        elif tf in [SurfaceFormat.RGB444P16F]:
            ttype = '__half'
            normalize_rgb = f'''
            {ttype} R = __float2half(min(max(R_, 0.0), 1.0));
            {ttype} G = __float2half(min(max(G_, 0.0), 1.0));
            {ttype} B = __float2half(min(max(B_, 0.0), 1.0));
            '''
        else:
            raise Exception(f"Unsupported target format {tf}")

        mtype = 'float' # maybe use __half when either sf or tf is low res

        if sf in [SurfaceFormat.YUV420P, SurfaceFormat.YUV420P16]:
            load_yuv = f'''
            {stype} Y = (({stype}*)(src + x * {sp}))[y];
            uint8_t *src_uv = src + {sp} * {h};
            {stype} U = (({stype}*)(src_uv + x/2 * {sp}))[y/2*2];
            {stype} V = (({stype}*)(src_uv + x/2 * {sp}))[y/2*2 + 1];
            '''
        elif sf in [SurfaceFormat.YUV444P, SurfaceFormat.YUV444P16]:
            load_yuv = f'''
            {stype} Y = (({ttype}*)(src + x * {sp}))[y];
            {stype} U = (({ttype}*)(src + h * {sp} + x * {sp}))[y];
            {stype} V = (({ttype}*)(src + h * {sp} * 2 + x * {sp}))[y];
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

        if tf in [SurfaceFormat.RGB24, SurfaceFormat.RGB48]:
            store_rgb = f'''
            (({ttype}*)(dst + x * {tp}))[y * 3] = R;
            (({ttype}*)(dst + x * {tp}))[y * 3 + 1] = G;
            (({ttype}*)(dst + x * {tp}))[y * 3 + 2] = B;
            '''
        elif tf in [SurfaceFormat.RGB444P, SurfaceFormat.RGB444P16, SurfaceFormat.RGB444P16F]:
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

    def __call__(self, surface, target = 'pycuda', check = True, block = (32, 32, 1), **kwargs):
        if check:
            assert surface.height == self.height, "Surface height mismatch"
            assert surface.width == self.width, "Surface width mismatch"
            assert surface.pitch == self.source_pitch, f"Surface pitch mismatch, {surface.pitch} {self.source_pitch}"
            assert surface.format is self.source_format, "Surface format mismatch"

        grid = ((surface.height - 1) // block[0] + 1, (surface.width - 1) // block[1] + 1)
        if target == 'pycuda':
            ts = Surface(self.width, self.height, self.target_format)
            alloc = cuda.mem_alloc(self.target_pitch * ts.height_in_rows)
            ts.alloc = alloc
            ts.pitch = self.target_pitch
            log.debug(f'{type(surface.alloc)}, {type(ts.alloc)}')
            self.convert(np.ulonglong(surface.alloc), np.ulonglong(ts.alloc), block=block, grid = grid, **kwargs)
        else:
            raise Exception(f"Unsupported target {target}")
        return ts