import numpy as np
from ..core.common import *
from ..ffmpeg.include.libavutil import AVColorRange,AVColorSpace, AVPixelFormat
from ..ffmpeg.libavutil import *
import cupy

class Converter:
    """
    Color converter from YUV to RGB running on CUDA
    each converter instance is parameterized by source and target surface format (shape, strides, atom type) for best performance
    CUDA kernel is compiled once on converter initialization and used repeatedly for each invocation
    """    

    @staticmethod
    def idx(base, ctype, strides, indices):
        code = f'((unsigned char *)({base}))'
        # index is string 
        # stide is integer
        assert len(indices) <= len(strides)
        for index, stride in zip(indices, strides, strict=True):
            code += f'+ ({index}) * ({stride})'
        return f'(*({ctype} *)({code}))'

    @staticmethod
    # generate strides from shape
    # assuming contiguous memory
    def shape2strides(shape, atom_size):
        if len(shape) == 0:
            return (atom_size,)
        else:
            strides = Converter.shape2strides(shape[1:], atom_size)
            return (strides[0] * shape[0], *strides)

    def strides(i):
        s = i['strides']
        if s is None:
            # in thise case, that implicitly assumed contiguous memory
            return Converter.shape2strides(i['shape'], int(i['typestr'][2:]))[1:]
        else:
            return s
            

    @staticmethod
    def typestr2ctype(typestr):
        if typestr == '<f4':
            return 'float'
        elif typestr == '|u1':
            return 'unsigned char'
        elif typestr == '<u2':
            return 'unsigned short'
        else:
            raise ValueError(f'unsupported typestr {typestr}')
        
    def __init__(self, 
        source_template, 
        source_format : AVPixelFormat, 
        source_space : AVColorSpace, 
        source_range : AVColorRange, 
        target_template, 
        target_format : AVPixelFormat = AVPixelFormat.RGB24, 
        target_space: AVColorSpace = AVColorSpace.RGB, 
        target_range: AVColorRange = AVColorRange.JPEG
        ):
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
        self.source_strides = Converter.strides(source)
        self.source_typestr = source['typestr']  
        assert self.source_typestr == typestr(source_format)
        self.size = shape2size(source_format, self.source_shape)
        h,w = self.size

        target = target_template.__cuda_array_interface__
        self.target_shape = target['shape']
        self.target_strides = Converter.strides(target)
        self.target_typestr = target['typestr']
        assert self.target_typestr == typestr(target_format)
        assert self.size == shape2size(target_format, self.target_shape), 'source and target size must be the same'


        # intermedia type
        mtype = 'float'

        def yuv():
            stype = Converter.typestr2ctype(self.source_typestr)
            def src(*indices):
                return Converter.idx('src', stype, self.source_strides, indices)

            if source_format in [AVPixelFormat.YUV420P, AVPixelFormat.YUV420P16LE]:
                load_yuv = f'''
                {stype} Y = {src('x','y')};
                {stype} U = {src(str(h) + ' + x/2', 'y/2*2')};
                {stype} V = {src(str(h) + ' + x/2', 'y/2*2 + 1')};
                '''
            elif source_format in [AVPixelFormat.YUV444P, AVPixelFormat.YUV444P16LE]:
                load_yuv = f'''
                {stype} Y = {src('0', 'x', 'y')};
                {stype} U = {src('1', 'x', 'y')};
                {stype} V = {src('2', 'x', 'y')};
                '''
            else:
                raise Exception(f"Unsupported source format {source_format}")    

            if stype == 'unsigned char':
                bit8 = 0
            elif stype == 'unsigned short':
                bit8 = 8
            else:
                raise Exception(f"Unsupported source type {stype}")

            if source_range == AVColorRange.MPEG:
                # partial range
                normalize_yuv = f'''
                    {mtype} Y_ = ({mtype})(Y - (16<<{bit8})) / (220 << {bit8});
                    {mtype} U_ = ({mtype})(U - (128<<{bit8})) / (225 << {bit8});
                    {mtype} V_ = ({mtype})(V - (128<<{bit8})) / (225 << {bit8});
                '''
            elif source_range == AVColorRange.JPEG:
                # full range
                normalize_yuv = f'''
                    {mtype} Y_ = ({mtype})(Y) / (256<<{bit8});
                    {mtype} U_ = ({mtype})(U - (128<<{bit8})) / (256<<{bit8});
                    {mtype} V_ = ({mtype})(V - (128<<{bit8})) / (256<<{bit8});
                '''
            else:
                raise Exception(f"Unsupported source range {source_range}")

            return load_yuv + normalize_yuv



        def yuv2rgb():
            assert target_space == AVColorSpace.RGB, 'only support RGB space as target'
    
            if source_space in [AVColorSpace.BT470BG, AVColorSpace.SMPTE170M]:
                m = [[1, 0, 1.402], [1, -0.34414, -0.71414], [1, 1.772, 0]]
            elif source_space == AVColorSpace.BT709:
                m = [[1, 0, 1.5748], [1, -0.1873, -0.4681], [1, 1.8556, 0]]
            elif source_space == AVColorSpace.BT2020_NCL:
                # https://gist.github.com/yohhoy/dafa5a47dade85d8b40625261af3776a
                m = [[1, 0, 1.4746], [1, -0.1646, -0.5714], [1, 1.8814, 0]]
            else:
                raise Exception(f"Unsupported color space {source_space}")

            yuv_to_rgb = f'''
            {mtype} R_ = {m[0][0]} * Y_ + {m[0][1]} * U_ + {m[0][2]} * V_;
            {mtype} G_ = {m[1][0]} * Y_ + {m[1][1]} * U_ + {m[1][2]} * V_;
            {mtype} B_ = {m[2][0]} * Y_ + {m[2][1]} * U_ + {m[2][2]} * V_;
            '''        
            return yuv_to_rgb

        def rgb():
            assert target_range == AVColorRange.JPEG, 'only support JPEG range as target'            
            ttype = Converter.typestr2ctype(self.target_typestr)
            def dst(*indices):
                return Converter.idx('dst', ttype, self.target_strides, indices)            

            if ttype == 'unsigned char':
                normalize_rgb = f'''
                {ttype} R = min(max(R_ * 256, .5), 256 - .5);
                {ttype} G = min(max(G_ * 256, .5), 256 - .5);
                {ttype} B = min(max(B_ * 256, .5), 256 - .5);
                '''
            else:
                raise Exception(f"Unsupported target type {target['typestr']}")

            if target_format == AVPixelFormat.RGB24:
                store_rgb = f'''
                {dst('x', 'y', '0')} = R;
                {dst('x', 'y', '1')} = G;
                {dst('x', 'y', '2')} = B;
                '''
            else:
                raise Exception(f"Unsupported target format {target_format}")
            return normalize_rgb + store_rgb

        self.convert =  cupy.RawKernel(f"""
            extern "C" __global__        
            void convert(unsigned char *src, unsigned char *dst) {{
                //printf("%p %p\\n", src, dst);
            int x = blockIdx.x * blockDim.x + threadIdx.x; // row
            if (x >= {h}) return;
            int y = blockIdx.y * blockDim.y + threadIdx.y; // column
            if (y >= {w}) return;
            {yuv()}
            {yuv2rgb()}
            {rgb()}
            }}
        """, 'convert')

    def __call__(self, source, target, check = True, block = (32, 32, 1)):
        """perform color convertion on a surface

        Args:
            source: source surface with CUDA Array Interface
            target: target surface with CUDA Array Interface
            check (bool, optional): Check if the surfaces formats are compatible with converter. Defaults to True.
            block (tuple, optional): The block size of this CUDA computation. Defaults to (32, 32, 1).
        """      
        
        s = source.__cuda_array_interface__
        t = target.__cuda_array_interface__
        print(s, t)          
        if check:
            assert s['shape'] == self.source_shape
            assert t['shape'] == self.target_shape
            assert s['typestr'] == self.source_typestr
            assert t['typestr'] == self.target_typestr
            assert Converter.strides(s) == self.source_strides
            assert Converter.strides(t) == self.target_strides

        grid = ((self.size[0] - 1) // block[0] + 1, (self.size[1] - 1) // block[1] + 1)


        self.convert(grid, block, (s['data'][0], t['data'][0]))
