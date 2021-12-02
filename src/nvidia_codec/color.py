from ctypes import c_ulonglong, c_void_p
from pycuda.compiler import SourceModule
import pycuda.driver as cuda
import numpy as np
from .common import *

import cppyy


cppyy.c_include('libavutil/avutil.h')

c = cppyy.gbl


'''
We are doing some metaprogramming; according to pycuda, 
constant is much faster than variable in e.g. multipllication
it's therefore worthwhile to compile the specialized code
this should be faster than the official color cvt in NPP
'''
class Converter:

    @staticmethod
    def idx(base, dtype, stride, *indices):
        code = f'((uint8_t *)({base}))'
        # index is string 
        # stide is integer
        assert len(indices) <= len(stride)
        for index, strid in zip(indices, stride):
            code += f'+ ({index}) * {strid}'
        return f'(*({dtype} *)({code}))'
        
    def __init__(self, source_template, source_format, source_space, source_range, 
    target_template = None, target_format = SurfaceFormat.RGB444P, target_space = c.AVCOL_SPC_RGB, target_range = c.AVCOL_RANGE_JPEG):
        source = source_template.__cuda_array_interface__
        self.source_shape = source['shape']
        self.source_strides = source['strides']
        self.source_typestr = source['typestr']

        self.size = shape2size(source_format, self.source_shape)
        self.target_shape = size2shape(target_format, self.size)

        target = target_template.__cuda_array_interface__
        assert self.target_shape == target['shape']
        self.target_strides = target['strides']
        self.target_typestr = target['typestr']        
        
        if self.source_typestr[1:] in ['f4', 'u2'] and self.target_typestr[1:] in ['f4', 'u2']:
            mtype = 'float'
        else:
            # todo: use low precision 
            mtype = 'float'

        if source['typestr'] == '|u1':
            stype = 'uint8_t'
            if source_range == c.AVCOL_RANGE_MPEG:
                # partial range
                normalize_yuv = f'''
                    {mtype} Y_ = ({mtype})(Y - 16) / ({mtype})220;
                    {mtype} U_ = ({mtype})(U - 128) / ({mtype})225;
                    {mtype} V_ = ({mtype})(V - 128) / ({mtype})225;
                '''
            elif source_range == c.AVCOL_RANGE_JPEG:
                # full range
                normalize_yuv = f'''
                    {mtype} Y_ = ({mtype})(Y) / ({mtype})256.0;
                    {mtype} U_ = ({mtype})(U - 128) / ({mtype})256.0;
                    {mtype} V_ = ({mtype})(V - 128) / ({mtype})256.0;
                '''
            else:
                raise Exception(f"Unsupported source range {source_range}")
        elif source['typestr'] == '<u2':
            stype = 'uint16_t'
            if source_range == c.AVCOL_RANGE_MPEG:
                # partial range
                normalize_yuv = f'''
                    {mtype} Y_ = ({mtype})(Y - (16<<8)) / ({mtype})(220 << 8);
                    {mtype} U_ = ({mtype})(U - (128<<8)) / ({mtype})(225 << 8);
                    {mtype} V_ = ({mtype})(V - (128<<8)) / ({mtype})(225 << 8);
                '''
            elif source_range == c.AVCOL_RANGE_JPEG:
                # full range
                normalize_yuv = f'''
                    {mtype} Y_ = ({mtype})(Y) / ({mtype})(256<<8);
                    {mtype} U_ = ({mtype})(U - (128<<8)) / ({mtype})(256<<8);
                    {mtype} V_ = ({mtype})(V - (128<<8)) / ({mtype})(256<<8);
                '''
            else:
                raise Exception(f"Unsupported source range {source_range}")
        else:
            raise Exception(f"Unsupported source type {source['typestr']}")

        def src(*indices):
            return Converter.idx('src', stype, source['strides'], *indices)

        if source_format is SurfaceFormat.YUV420P:
            self.height, self.width = source['shape'] 
            self.height = self.height // 3 * 2
            load_yuv = f'''
            {stype} Y = {src('x','y')};
            {stype} U = {src(str(self.height) + ' + x/2', 'y/2*2')};
            {stype} V = {src(str(self.height) + ' + x/2', 'y/2*2 + 1')};
            '''
        elif source_format in SurfaceFormat.YUV444P:
            self.height, self.width = source['shape']
            self.height = self.height // 3
            load_yuv = f'''
            {stype} Y = {src('0', 'x', 'y')};
            {stype} U = {src('1', 'x', 'y')};
            {stype} V = {src('2', 'x', 'y')};
            '''
        else:
            raise Exception(f"Unsupported source format {source_format}")    

        assert target_space == c.AVCOL_SPC_RGB, 'only support RGB space as target'
        assert target_range == c.AVCOL_RANGE_JPEG, 'only support JPEG range as target'

        if target['typestr'] == '<u2':
            ttype = 'uint16_t'
            normalize_rgb = f'''
            {ttype} R = min(max(R_ * ({mtype})65536, .5), 65536 - .5);
            {ttype} G = min(max(G_ * ({mtype})65536, .5), 65536 - .5);
            {ttype} B = min(max(B_ * ({mtype})65536, .5), 65536 - .5);
            '''
        elif target['typestr'] == '|u1':
            ttype = 'uint8_t'
            normalize_rgb = f'''
            {ttype} R = min(max(R_ * ({mtype})256, .5), 256 - .5);
            {ttype} G = min(max(G_ * ({mtype})256, .5), 256 - .5);
            {ttype} B = min(max(B_ * ({mtype})256, .5), 256 - .5);
            '''
        elif target['typestr'] == '<f4':
            ttype = 'float'
            normalize_rgb = f'''
            {ttype} R = min(max(R_, 0.0), 1.0);
            {ttype} G = min(max(G_, 0.0), 1.0);
            {ttype} B = min(max(B_, 0.0), 1.0);
            '''
        else:
            raise Exception(f"Unsupported target type {target['typestr']}")

        def dst(*indices):
            return Converter.idx('dst', ttype, target['strides'], *indices)

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

        if target_format is SurfaceFormat.RGB444P:
            store_rgb = f'''
            {dst('0', 'x', 'y')} = R;
            {dst('1', 'x', 'y')} = G;
            {dst('2', 'x', 'y')} = B;
            '''
        else:
            raise Exception(f"Unsupported target format {target_format}")

        self.mod =  SourceModule(f"""
        #include <stdint.h>
    __global__ void convert(uint8_t *src, uint8_t *dst) {{
        int x = blockIdx.x * blockDim.x + threadIdx.x; // row
        if (x >= {self.height}) return;
        int y = blockIdx.y * blockDim.y + threadIdx.y; // column
        if (y >= {self.width}) return;
        {load_yuv}
        {normalize_yuv}
        {yuv_to_rgb}
        {normalize_rgb}
        {store_rgb}
    }}
    """)
        self.convert = self.mod.get_function("convert")

    def __call__(self, source, target, check = True, block = (32, 32, 1), **kwargs):
        if check:
            assert source.__cuda_array_interface__['shape'] == self.source_shape
            assert target.__cuda_array_interface__['shape'] == self.target_shape
            assert source.__cuda_array_interface__['typestr'] == self.source_typestr
            assert target.__cuda_array_interface__['typestr'] == self.target_typestr
            assert source.__cuda_array_interface__['strides'] == self.source_strides
            assert target.__cuda_array_interface__['strides'] == self.target_strides

        grid = ((self.height - 1) // block[0] + 1, (self.width - 1) // block[1] + 1)
        self.convert(np.ulonglong(source.__cuda_array_interface__['data'][0]), np.ulonglong(target.__cuda_array_interface__['data'][0]), block = block, grid = grid, **kwargs)