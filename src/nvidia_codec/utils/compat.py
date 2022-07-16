def extract_stream_ptr(stream):
    """extract cuda stream raw pointer from several known wrappers

    Args:
        stream : cuda stream wrapper

    Raises:
        Exception: the cuda stream wrapper is not supported

    Returns:
        [int]: the raw cuda stream pointer, casted to int
    """    
    if stream is None:
        return 2 # default to per-thread default stream
    elif isinstance(stream, int):
        return int(stream)
    elif hasattr(stream, 'ptr'):
        # cupy
        return stream.ptr
    elif hasattr(stream, 'handle'):
        # pycuda
        return stream.handle
    elif hasattr(stream, '_as_parameter_'):
        # torch
        return stream._as_parameter_.value
    else:
        raise Exception(f'Unknown stream type {type(stream)}')


from nvidia_codec.ffmpeg.include.libavutil import AVPixelFormat
from ..ffmpeg.libavcodec import AVCodecID
from ..core.cuviddec import cudaVideoCodec, cudaVideoSurfaceFormat

# convert ffmpeg things to cuda things
def av2cuda(x):
    if isinstance(x, AVCodecID):
        if x == AVCodecID.HEVC:
            return cudaVideoCodec.HEVC
        elif x == AVCodecID.H264:
            return cudaVideoCodec.H264
        elif x == AVCodecID.VP9:
            return cudaVideoCodec.VP9
        else:
            raise Exception(f'unknown codec : {x}')
    else:
        raise Exception(f'unknown object to adapt: {x}')

def cuda2av(x):
    if isinstance(x, cudaVideoSurfaceFormat):
        if x == cudaVideoSurfaceFormat.NV12:
            return AVPixelFormat.YUV420P
        elif x == cudaVideoSurfaceFormat.YUV444:
            return AVPixelFormat.YUV444P
        elif x == cudaVideoSurfaceFormat.P016:
            return AVPixelFormat.YUV420P16LE
        elif x == cudaVideoSurfaceFormat.YUV444_16Bit:
            return AVPixelFormat.YUV444P16LE
        else:
            raise Exception(f'unknown surface format : {x}')
    else:
        raise Exception(f'unknown object to adapt: {x}')