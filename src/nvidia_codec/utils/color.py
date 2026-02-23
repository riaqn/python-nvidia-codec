"""Color space conversion utilities for decoded video surfaces.

This module handles conversion from YUV color spaces (as output by NVDEC)
to RGB format using PyTorch operations on the GPU.

Supported input formats:
    - NV12 (8-bit 4:2:0)
    - P016 (16-bit 4:2:0)
    - YUV444 (8-bit 4:4:4)
    - YUV444_16Bit (16-bit 4:4:4)

Supported color spaces:
    - BT.470BG / SMPTE 170M (SD video, PAL/NTSC)
    - BT.709 (HD video)
    - BT.2020 NCL (UHD/4K video)

Supported color ranges:
    - MPEG (limited range: Y 16-235, UV 16-240)
    - JPEG (full range: 0-255)
"""
import torch
from ..ffmpeg.include.libavutil import AVColorRange, AVColorSpace
from ..core.cuviddec import cudaVideoSurfaceFormat
from ..core.decode import Surface

def convert(surface : Surface,
        source_space : AVColorSpace,
        source_range : AVColorRange,
        target_dtype : torch.dtype):
    """Convert a decoded YUV surface to an RGB tensor.

    Performs color space conversion from YUV to RGB on the GPU using PyTorch.
    The conversion accounts for the video's color space (color matrix) and
    color range (limited vs full range).

    Args:
        surface: Decoded Surface object from the NVDEC decoder.
        source_space: Color space of the input (e.g., AVColorSpace.BT709).
        source_range: Color range of the input (AVColorRange.MPEG for limited,
            AVColorRange.JPEG for full range).
        target_dtype: PyTorch dtype for the output tensor:
            - torch.uint8: Output values in [0, 255]
            - torch.float32: Output values normalized to [0, 1]
            - torch.float16: Output values normalized to [0, 1]

    Returns:
        torch.Tensor of shape [3, H, W] representing RGB image on GPU.
        Channel order is R, G, B. Values are clamped to valid range.

    Raises:
        ValueError: If surface format, color range, or target dtype is unsupported.
        Exception: If color space is unsupported.
    """

    width = surface.width
    height = surface.height

    format = surface.format
    
    YUV = torch.as_tensor(surface, device='cuda')
    h,w = YUV.shape        

    if format in (cudaVideoSurfaceFormat.NV12, cudaVideoSurfaceFormat.P016):   
        assert width == w
        assert height % 2 == 0
        assert width % 2 == 0
        assert height // 2 * 3 == h
        assert YUV.dtype == torch.uint8

        Y = YUV[:height,:]
        UV = YUV[height:,:]
        U = UV[:,0::2]
        V = UV[:,1::2]    
    elif format in (cudaVideoSurfaceFormat.YUV444, cudaVideoSurfaceFormat.YUV444_16Bit):
        assert width == w
        assert height * 3 == h
        assert YUV.dtype == torch.int16
        Y = YUV[:height,:]
        U = YUV[height:height * 2,:]
        V = YUV[height*2:,:]
    else:
        raise ValueError(f'unsupported surface format {format}')

    if format in (cudaVideoSurfaceFormat.NV12, cudaVideoSurfaceFormat.YUV444):
        bit8 = 0
    elif format in (cudaVideoSurfaceFormat.P016, cudaVideoSurfaceFormat.YUV444_16Bit):
        bit8 = 8
    else:
        raise ValueError(f'unsupported surface format {format}')

    #normalize
    if source_range == AVColorRange.MPEG:
        y = (Y.type(torch.float) - (16 << bit8)) / (220 << bit8)
        u = (U.type(torch.float) - (128 << bit8)) / (225 << bit8)
        v = (V.type(torch.float) - (128 << bit8)) / (225 << bit8)
    elif source_range == AVColorRange.JPEG:
        y = Y.type(torch.float) / (256 << bit8)
        u = (U.type(torch.float) - (128 << bit8)) / (256 << bit8)
        v = (V.type(torch.float) - (128 << bit8)) / (256 << bit8)
    else:
        raise ValueError(f'unsupported source range {source_range}')

    if source_space in [AVColorSpace.BT470BG, AVColorSpace.SMPTE170M]:
        m = [[1, 0, 1.402], [1, -0.34414, -0.71414], [1, 1.772, 0]]
    elif source_space == AVColorSpace.BT709:
        m = [[1, 0, 1.5748], [1, -0.1873, -0.4681], [1, 1.8556, 0]]
    elif source_space == AVColorSpace.BT2020_NCL:
        # https://gist.github.com/yohhoy/dafa5a47dade85d8b40625261af3776a
        m = [[1, 0, 1.4746], [1, -0.1646, -0.5714], [1, 1.8814, 0]]
    else:
        raise Exception(f"Unsupported color space {source_space}")

    m = torch.as_tensor(m, device='cuda')

    y = torch.tensordot(m[:,0], y, ([], [])) # will be shape 3, h,w
    u = torch.tensordot(m[:,1], u, ([], []))
    v = torch.tensordot(m[:,2], v, ([], []))

    if format in (cudaVideoSurfaceFormat.NV12, cudaVideoSurfaceFormat.P016):
        # u,v is subsampled
        u = u.repeat_interleave(repeats=2, dim=1).repeat_interleave(repeats=2, dim=2)
        v = v.repeat_interleave(repeats=2, dim=1).repeat_interleave(repeats=2, dim=2)
        # print(u)
    elif format in (cudaVideoSurfaceFormat.YUV444, cudaVideoSurfaceFormat.YUV444_16Bit):
        pass
    else:
        raise ValueError(f'unsupported surface format {surface}')

    # print(torch.mean(y), torch.mean(u), torch.mean(v))
    rgb = y + u + v

    if target_dtype == torch.uint8:
        rgb = torch.clamp(rgb * 256, 0.5, 255.5)
    elif target_dtype in (torch.float, torch.float16, torch.float32, torch.float64):
        rgb = torch.clamp(rgb, 0, 1)
    else:
        raise ValueError(f'unsupported target dtype {target_dtype}')

    rgb = rgb.type(target_dtype)
    return rgb