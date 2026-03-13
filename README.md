# python-nvidia-codec

A Pythonic library for NVIDIA Video Codec (NVDEC).

> The project is still in active development; expect breaking changes.

## Requirements

- Python 3.10+
- NVIDIA GPU with NVDEC support
- NVIDIA driver with `libnvcuvid.so`
- FFmpeg 8.x (for demuxing)

## Installation

```bash
pip install nvidia-codec
```

Or from source:
```bash
git clone https://github.com/riaqn/python-nvidia-codec.git
cd python-nvidia-codec
pip install -e .
```

## Usage

### Decode all frames

```python
from nvidia_codec.utils import Player
import torch

# Simplest: decode at native resolution
player = Player('/path/to/video.mp4')

for timestamp, frame in player.frames(torch.uint8):
    # frame is a torch tensor on GPU (CHW format)
    print(f'{timestamp}: {frame.shape}')
```

### Screenshot (single frame extraction)

```python
from nvidia_codec.utils import Player
from datetime import timedelta
import torch

with Player('/path/to/video.mp4') as player:
    print(f'Duration: {player.duration}')

    # Get frame at 10 seconds (fast keyframe seek by default)
    timestamp, frame = player.screenshot(timedelta(seconds=10), torch.uint8)

    # For precise frame timing, use accurate=True (slower)
    timestamp, frame = player.screenshot(timedelta(seconds=10), torch.uint8, accurate=True)
```

### Advanced: crop, scale, and letterbox in hardware

All post-processing is done by the NVDEC hardware during decoding — no extra GPU passes needed.

```python
from nvidia_codec.utils import Player
import torch

player = Player(
    '/path/to/video.mp4',
    # Crop 100px from left/right, 50px from top/bottom
    cropping=lambda h, w: {'left': 100, 'top': 50, 'right': w - 100, 'bottom': h - 50},
    # Output buffer dimensions
    target_size=lambda h, w: (384, 384),
    # Place frame centered vertically; scaled to fit this rect, black bars outside
    target_rect=lambda h, w: {'left': 0, 'top': 36, 'right': 384, 'bottom': 348},
)

for timestamp, frame in player.frames(torch.float32):
    # frame is [3, 384, 384] with letterboxing
    process(frame)
```

## Supported Codecs

Codec support depends on your GPU:
- H.264 (AVC)
- H.265 (HEVC)
- VP9
- AV1 (Ampere+)
- VC1/WMV3 (older GPUs only, removed in Turing)

## Why another Python library for NVIDIA Codec?

### Comparison to Video-Processing-Framework (VPF)

**Methodologies**: VPF is written fully in C++ and uses `pybind` to expose Python interfaces. PNC is written fully in Python and uses `ctypes` to access NVIDIA C interfaces. Our code tends to be more concise, less duplicative and easier to read and write. It also allows better interoperability with other Python libraries.

**Performance**: Preliminary tests show little to no difference in performance, because the heavy lifting is done on the GPU anyway. Both libraries can saturate the GPU decoder. PNC uses more CPU than VPF as expected from Python vs. C++, but still negligible (less than 10% of Ryzen 3100 single core for 8K×4K HEVC).

**Resource Management**:
- In VPF, `Surface` given to user is not owned by the user. It will be overwritten by new frames which is counter-intuitive. `Picture` is not exposed to user at all - they are always mapped (post-processed and copied) to `Surface` so the picture can be ready for new frames. The latter is inefficient when only a subset of `Pictures` are needed (e.g., screenshots).
- VPF allocates the bare minimum of resources needed for most decoding tasks. PNC allows the user to specify the amount of resources to be allocated for advanced applications. Users own the resources and decide when and whether to deal with them.
- Managing resources is not painful: similar to `pycuda`, we shift the burden of managing host/device resources to the Python garbage collector. Resources (such as `Picture` and `Surface`) are automatically freed when the user drops the reference.

## Roadmap

- [x] Decoding
- Color Conversion
  - Source Format: NV12, P016, YUV444, YUV444_16Bit
  - Target Format: RGB24, RGB48, RGB444P, RGB444P16
  - Color Ranges: MPEG (limited), JPEG (full)
  - Color Space: BT.601, BT.709, BT.2020
- [x] Built-in cropping and scaling via NV decoder
- [x] Thread-safe and thread-friendly
- [ ] Encoder
- [x] PyTorch integration

## Acknowledgements

- Many thanks to @rarzumanyan for all the help and explanations!
- The blog posts from [myMusing](https://mymusing.co/tag/color/) are very helpful.
