"""Compatibility utilities for converting between FFmpeg and CUDA types.

This module provides conversion functions between FFmpeg codec/format
identifiers and their NVIDIA CUDA equivalents, as well as utilities
for extracting raw CUDA stream pointers from various Python wrappers.
"""

def extract_stream_ptr(stream):
    """Extract raw CUDA stream pointer from various Python wrappers.

    Supports CUDA stream objects from PyTorch, CuPy, PyCUDA, or raw integers.

    Args:
        stream: CUDA stream object, or None for per-thread default stream.
            Supported types:
            - None: Returns per-thread default stream (2)
            - int: Returned as-is
            - CuPy stream (has .ptr attribute)
            - PyCUDA stream (has .handle attribute)
            - PyTorch stream (has ._as_parameter_ attribute)

    Returns:
        int: Raw CUDA stream pointer.

    Raises:
        Exception: If the stream type is not recognized.
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
        return stream.cuda_stream
    else:
        raise Exception(f'Unknown stream type {type(stream)}')


from ..ffmpeg.include.libavutil import AVPixelFormat
from ..ffmpeg.libavcodec import AVCodecID
from ..core.cuviddec import cudaVideoCodec, cudaVideoSurfaceFormat

def av2cuda(x):
    """Convert FFmpeg codec ID to NVIDIA CUDA video codec.

    Args:
        x: FFmpeg AVCodecID enum value.

    Returns:
        cudaVideoCodec: Corresponding NVIDIA codec enum.

    Raises:
        Exception: If the codec is not supported by NVDEC.

    Supported codecs:
        - H.264 (AVC)
        - HEVC (H.265)
        - VP9
        - AV1 (requires Ampere or newer GPU)
        - MPEG4
        - VC1 / WMV3
    """
    if isinstance(x, AVCodecID):
        if x == AVCodecID.HEVC:
            return cudaVideoCodec.HEVC
        elif x == AVCodecID.H264:
            return cudaVideoCodec.H264
        elif x == AVCodecID.VP9:
            return cudaVideoCodec.VP9
        elif x == AVCodecID.AV1:
            return cudaVideoCodec.AV1
        elif x == AVCodecID.MPEG4:
            return cudaVideoCodec.MPEG4
        elif x == AVCodecID.VC1 or x == AVCodecID.WMV3:
            return cudaVideoCodec.VC1
        else:
            raise Exception(f'unknown codec : {x}')
    else:
        raise Exception(f'unknown object to adapt: {x}')

def cuda2av(x):
    """Convert NVIDIA CUDA surface format to FFmpeg pixel format.

    Args:
        x: cudaVideoSurfaceFormat enum value.

    Returns:
        AVPixelFormat: Corresponding FFmpeg pixel format.

    Raises:
        Exception: If the surface format is not supported.

    Supported formats:
        - NV12 -> YUV420P (8-bit 4:2:0)
        - P016 -> YUV420P16LE (16-bit 4:2:0)
        - YUV444 -> YUV444P (8-bit 4:4:4)
        - YUV444_16Bit -> YUV444P16LE (16-bit 4:4:4)
    """
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