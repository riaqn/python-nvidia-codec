from enum import Enum, auto
class SurfaceFormat(Enum):
    RGB444P = auto()
    YUV420P = auto()
    YUV444P = auto()

def shape2size(format, shape):
    if format in [SurfaceFormat.RGB444P, SurfaceFormat.YUV444P]:
        c, h, w = shape
        assert c == 3
        return {'width' : w, 'height' : h}
    elif format is SurfaceFormat.YUV420P:
        h, w = shape
        assert h % 3 == 0
        return {'width' : w, 'height' : h // 3 * 2}
    else:
        raise ValueError(f'Unknown format {format}')

def size2shape(format, size):
    if format in [SurfaceFormat.RGB444P, SurfaceFormat.YUV444P]:
        return (3, size['height'], size['width'])
    elif format is SurfaceFormat.YUV420P:
        assert size['height'] % 2 == 0
        return (size['height'] // 2 * 3, size['width'])
    else:
        raise ValueError(f'Unknown format {format}')

def convert_shape(source_format, target_format, source_shape):
    return size2shape(target_format, shape2size(source_format, source_shape))

def get_stream_ptr(stream):
    if stream is None:
        return 2 # default to per-thread default stream
    elif type(stream) is int:
        return stream
    elif hasattr(stream, 'ptr'):
        # cupy
        return stream.ptr
    elif hasattr(stream, 'handle'):
        # pycuda
        return stream.handle
    else:
        raise Exception(f'Unknown stream type {type(stream)}')

