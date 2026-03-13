"""High-level video player classes using NVIDIA hardware decoding.

This module provides user-friendly classes for decoding video files using
NVIDIA's NVDEC hardware decoder. The decoded frames are returned as PyTorch
tensors on the GPU.

Example usage:
    # Stream all frames at native resolution (simplest usage)
    from nvidia_codec.utils import Player
    player = Player('/path/to/video.mp4')
    for time, frame in player.frames(torch.float32):
        # frame is a torch.Tensor of shape [C, H, W] on GPU
        process(frame)

    # Extract a single frame at a specific timestamp
    with Player('/path/to/video.mp4') as player:
        time, frame = player.screenshot(timedelta(seconds=30), torch.uint8)

    # Advanced: crop, scale, and letterbox in hardware
    player = Player(
        '/path/to/video.mp4',
        cropping=lambda h, w: {'left': 100, 'top': 50, 'right': w - 100, 'bottom': h - 50},
        target_size=lambda h, w: (384, 384),
        target_rect=lambda h, w: {'left': 0, 'top': 36, 'right': 384, 'bottom': 348},
    )
"""
from datetime import timedelta
from fractions import Fraction
from queue import Queue
import numpy as np

from ..ffmpeg.libavcodec import BSFContext
from ..ffmpeg.libavformat import FormatContext, AVMediaType, AVCodecID
from ..ffmpeg.include.libavcodec import AV_PKT_FLAG_KEY
from ..ffmpeg.include.libavutil  import AV_NOPTS_VALUE, AV_TIME_BASE, AVColorRange, AVColorSpace
from ..ffmpeg.libavutil import dict_get
from .compat import av2cuda, cuda2av, extract_stream_ptr
from ..core.decode import BaseDecoder
from .color import convert
from .. import NoFrameError

import torch

import logging

log = logging.getLogger(__name__)

class BasePlayer:
    """Base class for video players using NVIDIA hardware decoding.

    This class handles video file opening, stream selection, seeking, and
    provides the foundation for frame decoding. It automatically selects
    the highest bitrate video stream if multiple are present.

    Supported codecs: H.264, HEVC, VP9, AV1, MPEG4, VC1/WMV3
    (actual support depends on GPU capabilities)

    Attributes:
        url: Path or URL of the video file.
        width: Original video width in pixels.
        height: Original video height in pixels.
        target_width: Output buffer width in pixels (if target_size was specified).
        target_height: Output buffer height in pixels (if target_size was specified).
        duration: Video duration as a timedelta.
    """

    def __init__(self, url, num_surfaces, target_size = None, cropping = None, target_rect = None, device = None):
        """Initialize the player with a video file.

        Args:
            url: Path or URL to the video file.
            num_surfaces: Number of output surfaces to allocate. More surfaces
                allow holding multiple decoded frames simultaneously.
            target_size: Function (height, width) -> (new_height, new_width) defining
                the output buffer dimensions. Default is native resolution.
            cropping: Function (height, width) -> {'left', 'top', 'right', 'bottom'}
                defining the source crop rectangle. Default is no cropping (full frame).
            target_rect: Function (target_height, target_width) -> {'left', 'top', 'right', 'bottom'}
                defining where within the output buffer the frame is placed. The
                source (or cropped region) is scaled to fit this rectangle. Area
                outside is filled with black (letterboxing). Default fills the
                entire target buffer.
            device: CUDA device ID to use for decoding. If None, uses the
                current device from torch.cuda.current_device().

        Raises:
            AssertionError: If the file has no video stream.
            CodecNotSupportedError: If the video codec is not supported by the GPU.
        """
        self.url = url
        self.fc = FormatContext(url)
        l = filter(lambda s: s.codecpar.contents.codec_type == AVMediaType.VIDEO, self.fc.streams)

        l = sorted(l, key = lambda s: s.codecpar.contents.bit_rate, reverse = True)

        assert len(l) > 0, 'file has no video stream'
        self.stream = l[0]         
        if len(l) > 1:
            log.warning(f'{url} has multiple video streams, picking the highest bitrate @ {self.stream.codecpar.contents.bit_rate}')

        self._start_time = self.stream.start_time
        if self._start_time == AV_NOPTS_VALUE:
            self._start_time = self.fc.av.start_time
            if self._start_time == AV_NOPTS_VALUE:
                start_time = self.fc.infer_start_time()
                self._start_time = int(start_time / self._time_base)
            else:
                self._start_time = int(self._start_time / AV_TIME_BASE / self._time_base)                

        self._duration = self._infer_duration()
        if self._duration is None:
            log.warning('cannot infer duration')

        self._prepend_extradata = False

        codec_id = self.stream.codecpar.contents.codec_id
        if codec_id == AVCodecID.HEVC:
            f = 'hevc_mp4toannexb'
        elif codec_id == AVCodecID.H264:
            f = 'h264_mp4toannexb'
        else:
            f = None
            # raise Exception(f'unsupported codec {codec_id}')                

        self.bsf = BSFContext(f, self.stream.codecpar.contents, self.stream.time_base)

        self.device = torch.cuda.current_device() if device is None else device
        def decide(p):
            d = {
                'num_pictures': p['min_num_pictures'], # to be safe
                'num_surfaces': num_surfaces,
            }
            if target_size is not None:
                d['target_size'] = target_size
            if cropping is not None:
                d['cropping'] = cropping
            if target_rect is not None:
                d['target_rect'] = target_rect
            return d

        # Extract extradata (sequence header) for codecs like VC1/WMV3 that need it
        extradata = None
        codecpar = self.stream.codecpar.contents
        if codecpar.extradata_size > 0 and codecpar.extradata:
            extradata = bytes(codecpar.extradata[:codecpar.extradata_size])

        self.decoder = BaseDecoder(
            av2cuda(codecpar.codec_id),
            decide = decide,
            device = self.device,
            extradata = extradata,
            coded_width = codecpar.width,
            coded_height = codecpar.height
        )

    @property
    def _time_base(self):
        """Stream time base as a Fraction (internal use)."""
        return Fraction(self.stream.time_base.num, self.stream.time_base.den)

    def _infer_duration(self):
        """Try to determine video duration from various sources.

        Returns duration in stream time_base units, or None if unknown.
        """
        tb = self._time_base

        # 1. stream duration (from container header)
        d = self.stream.duration
        if d != AV_NOPTS_VALUE:
            return d

        # 2. probe the file to populate stream duration
        self.fc.find_stream_info()
        d = self.stream.duration
        if d != AV_NOPTS_VALUE:
            return d

        # 3. stream metadata tag (e.g. MKV DURATION-eng)
        for key in ('DURATION', 'DURATION-eng'):
            tag = dict_get(self.stream.metadata, key)
            if tag:
                h, m, s = tag.split(':')
                td = timedelta(hours=int(h), minutes=int(m), seconds=float(s))
                return int(td.total_seconds() / tb)

        # 4. container duration (AV_TIME_BASE units)
        d = self.fc.av.duration
        if d > 0:
            return int(d / AV_TIME_BASE / tb)

        return None

    @property
    def width(self):
        """Original video width in pixels."""
        return self.decoder.width

    @property
    def height(self):
        """Original video height in pixels."""
        return self.decoder.height

    @property
    def target_width(self):
        """Output width in pixels after scaling."""
        return self.decoder.target_width

    @property
    def target_height(self):
        """Output height in pixels after scaling."""
        return self.decoder.target_height

    @property
    def duration(self):
        """Video duration as a timedelta, or None if unknown."""
        if self._duration is None:
            return None
        return timedelta(seconds = float(self._duration * self._time_base))

    def color_space(self, default = AVColorSpace.UNSPECIFIED):
        """Get the video's color space (e.g., BT.709, BT.601).

        Args:
            default: Color space to return if the video doesn't specify one.

        Returns:
            AVColorSpace enum value.
        """
        r = self.stream.codecpar.contents.color_space
        if r == AVColorSpace.UNSPECIFIED:
            log.debug(f'color space is unspecified, using {default}')
            return default
        else:
            log.debug(f'color space is {r}')
            return r
    
    def color_range(self, default = AVColorRange.UNSPECIFIED):
        """Get the video's color range (limited/full).

        Args:
            default: Color range to return if the video doesn't specify one.

        Returns:
            AVColorRange enum value (MPEG for limited, JPEG for full).
        """
        r = self.stream.codecpar.contents.color_range
        if r == AVColorRange.UNSPECIFIED:
            log.debug(f'color range is unspecified, using {default}')
            return default
        else:
            log.debug(f'color range is {r}')
            return r

    def free(self):
        """Release decoder resources."""
        self.decoder.free()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.free()

    def seek(self, target : timedelta):
        """Seek to a target timestamp in the video.

        Seeks to the nearest keyframe at or before the target timestamp.
        Also flushes the decoder state.

        Args:
            target: Target timestamp as a timedelta.
        """
        target_pts = self.time2pts(target)
        log.debug(f'target_pts: {target_pts}')
        self.fc.seek_file(self.stream, target_pts, max_ts = target_pts)

        self.bsf.flush()
        self.decoder.flush()
        self._prepend_extradata = True

    def time2pts(self, time: timedelta):
        """Convert a timedelta to presentation timestamp (PTS)."""
        return int(time.total_seconds() / self._time_base) + self._start_time

    def pts2time(self, pts : int):
        """Convert a presentation timestamp (PTS) to timedelta."""
        if pts == AV_NOPTS_VALUE:
            return None
        return timedelta(seconds = float((int(pts) - int(self._start_time)) * self._time_base))

    def convert(self, surface, dtype):
        """Convert a decoded surface to an RGB tensor.

        Args:
            surface: Decoded Surface object from the NVDEC decoder.
            dtype: PyTorch dtype for the output tensor (e.g., torch.float32, torch.uint8).

        Returns:
            torch.Tensor of shape [3, H, W] (RGB) on the GPU.
        """
        return convert(surface, self.color_space(AVColorSpace.BT470BG), self.color_range(AVColorRange.MPEG), dtype)

    def decode(self, on_recv, keyframes_only=False):
        """Decode packets and invoke callback for each decoded frame.

        This is a low-level method used internally.
        For most use cases, use Player.frames() or Player.screenshot() instead.

        Args:
            on_recv: Callback function (picture, time, accumulator) -> result.
                Called for each decoded frame. picture is a Picture object
                (or None at end of stream), time is a timedelta, and accumulator
                is the return value from the previous on_recv call.
            keyframes_only: If True, only decode keyframe (I-frame) packets.

        Returns:
            The final return value from on_recv.
        """
        packets = self.fc.read_packets(self.stream)
        if keyframes_only:
            packets = (pkt for pkt in packets if pkt.av.flags & AV_PKT_FLAG_KEY)
        it = self.bsf.filter(packets, flush=False, reuse=True)

        def on_recv_(pic, pts, ret):
            return on_recv(pic, self.pts2time(pts), ret)

        for pkt in it:
            pts = pkt.av.pts if pkt.av.pts != AV_NOPTS_VALUE else pkt.av.dts
            arr = np.ctypeslib.as_array(pkt.av.data, (pkt.av.size,))
            if self._prepend_extradata:
                par_out = self.bsf.av.par_out.contents
                if par_out.extradata_size > 0 and par_out.extradata:
                    extradata = bytes(par_out.extradata[:par_out.extradata_size])
                    arr = np.concatenate([np.frombuffer(extradata, dtype=np.uint8), arr])
                self._prepend_extradata = False
            ret = self.decoder.send(arr, on_recv_, pts)
            if ret is not None:
                return ret
        return self.decoder.send(None, on_recv_, 0)

class Player(BasePlayer):
    """Video player for decoding frames from a video file.

    Supports both streaming all frames and extracting individual frames
    at specific timestamps, using NVIDIA hardware decoding.

    Example:
        # Simplest: native resolution, all frames
        player = Player('/path/to/video.mp4')
        for time, frame in player.frames(torch.float32):
            # time is a timedelta, frame is [C, H, W] tensor on GPU
            process(frame)

        # Extract a frame at a specific timestamp
        with Player('video.mp4') as player:
            time, frame = player.screenshot(timedelta(seconds=30), torch.uint8)

        # Advanced: crop center, scale to 384x384, letterbox with black bars
        player = Player(
            '/path/to/video.mp4',
            cropping=lambda h, w: {'left': 100, 'top': 50, 'right': w - 100, 'bottom': h - 50},
            target_size=lambda h, w: (384, 384),
            target_rect=lambda h, w: {'left': 0, 'top': 36, 'right': 384, 'bottom': 348},
        )
    """

    def __init__(self, url, target_size = None, cropping = None, target_rect = None, device = None):
        """Initialize the player with a video file.

        Args:
            url: Path or URL to the video file.
            target_size: Function (height, width) -> (new_height, new_width) defining
                the output buffer dimensions. Default is native resolution.
            cropping: Function (height, width) -> {'left', 'top', 'right', 'bottom'}
                defining the source crop rectangle. Default is no cropping (full frame).
            target_rect: Function (target_height, target_width) -> {'left', 'top', 'right', 'bottom'}
                defining where within the output buffer the frame is placed. The
                source (or cropped region) is scaled to fit this rectangle. Area
                outside is filled with black (letterboxing). Default fills the
                entire target buffer.
            device: CUDA device ID. If None, uses current device.
        """
        super().__init__(url, num_surfaces=2, target_size=target_size, cropping=cropping, target_rect=target_rect, device=device)

    def frames(self, dtype : torch.dtype, keyframes_only=False):
        """Iterate over frames in the video.

        Args:
            dtype: PyTorch dtype for output tensors. Use torch.float32 for
                normalized [0, 1] values or torch.uint8 for [0, 255] values.
            keyframes_only: If True, only decode keyframe (I-frame) packets.

        Yields:
            Tuple of (time, frame) where:
                - time: timedelta of the frame's presentation timestamp
                - frame: torch.Tensor of shape [3, H, W] (RGB) on GPU
        """
        def on_recv(pic, time, frames):
            if pic is None:
                return frames
            stream = extract_stream_ptr(torch.cuda.current_stream(self.device))
            surface = pic.map(stream)
            pic.free()
            frame = self.convert(surface, dtype)
            surface.free()
            if frames is None:
                frames = []
            frames.append((time, frame))
            return frames

        while True:
            frames = self.decode(on_recv, keyframes_only=keyframes_only)
            if frames is None:
                break
            yield from frames

    def screenshot(self, target : timedelta, dtype : torch.dtype, accurate : bool = False):
        """Extract a frame at the specified timestamp.

        Args:
            target: Target timestamp as a timedelta.
            dtype: PyTorch dtype for the output tensor. Use torch.float32 for
                normalized [0, 1] values or torch.uint8 for [0, 255] values.
            accurate: If False (default), returns the first frame after seeking
                (faster, may be slightly after target). If True, decodes until
                finding the frame closest to but not after target (slower but
                more precise).

        Returns:
            Tuple of (time, frame) where:
                - time: timedelta of the actual frame timestamp (may differ from target)
                - frame: torch.Tensor of shape [3, H, W] (RGB) on GPU

        Raises:
            NoFrameError: If no frame could be extracted at the target timestamp.
        """
        self.seek(target)

        last = None
        for time, frame in self.frames(dtype):
            if not accurate:
                return time, frame
            if time > target:
                break
            last = (time, frame)

        if last is not None:
            return last
        raise NoFrameError(f"No frame found at {target} in {self.url}")

