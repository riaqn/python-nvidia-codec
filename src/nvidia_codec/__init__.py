"""NVIDIA hardware video decoding for Python.

This package provides a Pythonic interface to NVIDIA's NVDEC hardware video
decoder, allowing fast GPU-accelerated video decoding with PyTorch integration.

Quick Start:
    from nvidia_codec.utils import Player, Screenshoter
    import torch

    # Stream all frames from a video
    player = Player('/path/to/video.mp4')
    for time, frame in player.frames(torch.float32):
        process(frame)  # frame is [C, H, W] tensor on GPU

    # Extract a single frame
    ss = Screenshoter('/path/to/video.mp4')
    time, frame = ss.screenshot(timedelta(seconds=30), torch.uint8)
    ss.free()

Requirements:
    - NVIDIA GPU with NVDEC support
    - NVIDIA driver with libnvcuvid.so
    - FFmpeg 8.x (for demuxing)
    - PyTorch with CUDA support

Supported Codecs:
    - H.264 (AVC)
    - HEVC (H.265)
    - VP9
    - AV1 (requires Ampere or newer GPU)
    - MPEG4
    - VC1 / WMV3

Exceptions:
    CodecNotSupportedError: Raised when the GPU doesn't support the video codec.
    NoFrameError: Raised when frame extraction fails.
"""

class CodecNotSupportedError(Exception):
    """Raised when the codec is not supported by the GPU's NVDEC."""
    pass

class NoFrameError(Exception):
    """Raised when no frame could be extracted from the video."""
    pass
