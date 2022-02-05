from pycuda.compiler import SourceModule
import numpy as np
from .common import *

# copied from http://www.ffmpeg.org/doxygen/trunk/pixfmt_8h_source.html#l00523
class Space(Enum):
    RGB         = 0
    BT709       = 1 
    UNSPECIFIED = 2
    RESERVED    = 3 
    FCC         = 4 
    BT470BG     = 5 
    SMPTE170M   = 6 
    SMPTE240M   = 7 
    YCGCO       = 8 
    YCOCG       = YCGCO
    BT2020_NCL  = 9  
    BT2020_CL   = 10 
    SMPTE2085   = 11
    CHROMA_DERIVED_NCL = 12
    CHROMA_DERIVED_CL = 13
    ICTCP       = 14
    NB = auto()

class Range(Enum):
    UNSPECIFIED = 0
    MPEG = 1
    JPEG = 2
    NB = auto()


class Converter:
    """
    Color converter from YUV to RGB running on CUDA
    each converter instance is parameterized by source and target surface format (shape, strides, atom type) for best performance
    CUDA kernel is compiled once on converter initialization and used repeatedly for each invocation
    """    
    @staticmethod
    def idx(base, ctype, strides, indices):
        code = f'((uint8_t *)({base}))'
        # index is string 
        # stide is integer
        assert len(indices) <= len(strides)
        for index, stride in zip(indices, strides):
            code += f'+ ({index}) * {stride}'
        return f'(*({ctype} *)({code}))'

    @staticmethod
    def shape2strides(shape, typestr):
        if len(shape) == 0:
            return (int(typestr[2:]),)
        else:
            strides = Converter.shape2strides(shape[1:], typestr)
            return (strides[0] * shape[0], *strides)
            
    @staticmethod
    def typestr2ctype(typestr):
        if typestr == '<f4':
            return 'float'
        elif typestr == '|u1':
            return 'uint8_t'
        elif typestr == '<u2':
            return 'uint16_t'
        else:
            raise ValueError(f'unsupported typestr {typestr}')
        
    def __init__(self, source_template, source_format, source_space, source_range, 
    target_template, target_format = SurfaceFormat.RGB444P, target_space = Space.RGB, target_range = Range.JPEG):
        """
        Args:
            source_template : a surface whose CUDA array interface will be used as template for sources
            source_format (SurfaceFormat): the source format
            source_space (Color.Space): color space of source surface
            source_range (Color.Range): color range of source surface
            target_template : a surface whose CUDA array interface will be used as template for targets
            target_format (SurfaceFormat, optional): the target format. Defaults to SurfaceFormat.RGB444P.
            target_space (Color.Space, optional): color space of target surface. Defaults to Space.RGB.
            target_range (Color.Range, optional): color range of target surface. Defaults to Range.JPEG.
        """    
        source = source_template.__cuda_array_interface__
        self.source_shape = source['shape']
        self.source_strides = source['strides']
        self.source_typestr = source['typestr']
        if self.source_strides is None:
            self.source_strides = self.shape2strides(self.source_shape, source['typestr'])[1:]        

        self.size = shape2size(source_format, self.source_shape)
        self.target_shape = size2shape(target_format, self.size)

        target = target_template.__cuda_array_interface__
        assert self.target_shape == target['shape'], f'{self.target_shape} != {target["shape"]}'
        self.target_strides = target['strides']
        self.target_typestr = target['typestr']
        if self.target_strides is None:
            self.target_strides = self.shape2strides(self.target_shape, target['typestr'])[1:]

        stype = Converter.typestr2ctype(self.source_typestr)
        ttype = Converter.typestr2ctype(self.target_typestr)
        
        mtype = 'float'

        if stype == 'uint8_t':
            if source_range == Range.MPEG:
                # partial range
                normalize_yuv = f'''
                    {mtype} Y_ = ({mtype})(Y - 16) / 220;
                    {mtype} U_ = ({mtype})(U - 128) / 225;
                    {mtype} V_ = ({mtype})(V - 128) / 225;
                '''
            elif source_range == Range.JPEG:
                # full range
                normalize_yuv = f'''
                    {mtype} Y_ = ({mtype})(Y) / 256;
                    {mtype} U_ = ({mtype})(U - 128) / 256;
                    {mtype} V_ = ({mtype})(V - 128) / 256;
                '''
            else:
                raise Exception(f"Unsupported source range {source_range}")
        elif stype == 'uint16_t':
            if source_range == Range.MPEG:
                # partial range
                normalize_yuv = f'''
                    {mtype} Y_ = ({mtype})(Y - (16<<8)) / (220 << 8);
                    {mtype} U_ = ({mtype})(U - (128<<8)) / (225 << 8);
                    {mtype} V_ = ({mtype})(V - (128<<8)) / (225 << 8);
                '''
            elif source_range == Range.JPEG:
                # full range
                normalize_yuv = f'''
                    {mtype} Y_ = ({mtype})(Y) / (256<<8);
                    {mtype} U_ = ({mtype})(U - (128<<8)) / (256<<8);
                    {mtype} V_ = ({mtype})(V - (128<<8)) / (256<<8);
                '''
            else:
                raise Exception(f"Unsupported source range {source_range}")
        else:
            raise Exception(f"Unsupported source type {source['typestr']}")

        def src(*indices):
            return Converter.idx('src', stype, self.source_strides, indices)

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

        assert target_space == Space.RGB, 'only support RGB space as target'
        assert target_range == Range.JPEG, 'only support JPEG range as target'

        if ttype == 'uint16_t':
            normalize_rgb = f'''
            {ttype} R = min(max(R_ * 65536, .5), 65536 - .5);
            {ttype} G = min(max(G_ * 65536, .5), 65536 - .5);
            {ttype} B = min(max(B_ * 65536, .5), 65536 - .5);
            '''
        elif ttype == 'uint8_t':
            normalize_rgb = f'''
            {ttype} R = min(max(R_ * 256, .5), 256 - .5);
            {ttype} G = min(max(G_ * 256, .5), 256 - .5);
            {ttype} B = min(max(B_ * 256, .5), 256 - .5);
            '''
        elif ttype == 'float':
            normalize_rgb = f'''
            {ttype} R = min(max(R_, 0.0), 1.0);
            {ttype} G = min(max(G_, 0.0), 1.0);
            {ttype} B = min(max(B_, 0.0), 1.0);
            '''
        else:
            raise Exception(f"Unsupported target type {target['typestr']}")

        def dst(*indices):
            return Converter.idx('dst', ttype, self.target_strides, indices)

        if source_space in [Space.BT470BG, Space.SMPTE170M]:
            m = [[1, 0, 1.402], [1, -0.34414, -0.71414], [1, 1.772, 0]]
        elif source_space == Space.BT709:
            m = [[1, 0, 1.5748], [1, -0.1873, -0.4681], [1, 1.8556, 0]]
        elif source_space == Space.BT2020_NCL:
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
        """perform color convertion on a surface

        Args:
            source: source surface with CUDA Array Interface
            target: target surface with CUDA Array Interface
            check (bool, optional): Check if the surfaces formats are compatible with converter. Defaults to True.
            block (tuple, optional): The block size of this CUDA computation. Defaults to (32, 32, 1).
        """        
        if check:
            assert source.__cuda_array_interface__['shape'] == self.source_shape
            assert target.__cuda_array_interface__['shape'] == self.target_shape
            assert source.__cuda_array_interface__['typestr'] == self.source_typestr
            assert target.__cuda_array_interface__['typestr'] == self.target_typestr
            # assert source.__cuda_array_interface__['strides'] == self.source_strides
            # assert target.__cuda_array_interface__['strides'] == self.target_strides

        grid = ((self.height - 1) // block[0] + 1, (self.width - 1) // block[1] + 1)
        self.convert(np.ulonglong(source.__cuda_array_interface__['data'][0]), np.ulonglong(target.__cuda_array_interface__['data'][0]), block = block, grid = grid, **kwargs)