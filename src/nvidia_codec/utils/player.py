"""High-level video player classes using NVIDIA hardware decoding.

This module provides user-friendly classes for decoding video files using
NVIDIA's NVDEC hardware decoder. The decoded frames are returned as PyTorch
tensors on the GPU.

Example usage:
    # Stream all frames from a video
    from nvidia_codec.utils import Player
    player = Player('/path/to/video.mp4')
    for time, frame in player.frames(torch.float32):
        # frame is a torch.Tensor of shape [C, H, W] on GPU
        process(frame)

    # Extract a single frame at a specific timestamp
    from nvidia_codec.utils import Screenshoter
    ss = Screenshoter('/path/to/video.mp4')
    time, frame = ss.screenshot(timedelta(seconds=30), torch.uint8)
    ss.free()
"""
from datetime import timedelta
from fractions import Fraction
from queue import Queue
import numpy as np

from ..ffmpeg.libavcodec import BSFContext
from ..ffmpeg.libavformat import FormatContext, AVMediaType, AVCodecID
from ..ffmpeg.include.libavutil  import AV_NOPTS_VALUE, AV_TIME_BASE, AVColorRange, AVColorSpace
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
        target_width: Output width after scaling (if target_size was specified).
        target_height: Output height after scaling (if target_size was specified).
        duration: Video duration as a timedelta.
    """

    def __init__(self, url, num_surfaces, target_size = lambda h,w: (h,w),  device = None):
        """Initialize the player with a video file.

        Args:
            url: Path or URL to the video file.
            num_surfaces: Number of output surfaces to allocate. More surfaces
                allow holding multiple decoded frames simultaneously.
            target_size: Function (height, width) -> (new_height, new_width) for
                scaling the output. The decoder performs scaling in hardware.
                Default is no scaling (identity function).
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

        self._duration = self.stream.duration
        if self._duration == AV_NOPTS_VALUE:
            # if stream duration is unknown,
            # get the whole file duration
            self._duration = self.fc.av.duration
            if self._duration == AV_NOPTS_VALUE:
                log.warning('cannot infer duration')
                self._duration = None
            else:
                # remember to convert to stream' time base
                self._duration = int(self._duration / AV_TIME_BASE / self._time_base)

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
            return {
                'num_pictures': p['min_num_pictures'], # to be safe
                'num_surfaces': num_surfaces,
                # will use default surface_format
                # will use default cropping (no cropping)
                'target_size': target_size,
                # will use default target rect (no margin)
            }

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
        """Video duration as a timedelta."""
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

    def time2pts(self, time: timedelta):
        """Convert a timedelta to presentation timestamp (PTS)."""
        return int(time.total_seconds() / self._time_base) + self._start_time

    def pts2time(self, pts : int):
        """Convert a presentation timestamp (PTS) to timedelta."""
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

    def decode(self, on_recv):
        """Decode packets and invoke callback for each decoded frame.

        This is a low-level method used internally by Player and Screenshoter.
        For most use cases, use Player.frames() or Screenshoter.screenshot() instead.

        Args:
            on_recv: Callback function (picture, time, accumulator) -> result.
                Called for each decoded frame. picture is a Picture object
                (or None at end of stream), time is a timedelta, and accumulator
                is the return value from the previous on_recv call.

        Returns:
            The final return value from on_recv.
        """
        it = self.bsf.filter(self.fc.read_packets(self.stream), flush = False, reuse = True)

        def on_recv_(pic, pts, ret):
            return on_recv(pic, self.pts2time(pts), ret)
        
        for pkt in it:
            pts = pkt.av.pts
            arr = np.ctypeslib.as_array(pkt.av.data, (pkt.av.size,))
            ret = self.decoder.send(arr, on_recv_, pts)
            if ret is not None:
                return ret
        return self.decoder.send(None, on_recv_, 0)

class Player(BasePlayer):
    """Video player for streaming all frames from a video file.

    This class provides an iterator interface to decode and yield all frames
    from a video file. Frames are decoded using NVIDIA hardware and returned
    as PyTorch tensors on the GPU.

    Example:
        player = Player('/path/to/video.mp4')
        for time, frame in player.frames(torch.float32):
            # time is a timedelta, frame is [C, H, W] tensor on GPU
            process(frame)

        # With scaling to 720p
        player = Player('video.mp4', target_size=lambda h,w: (720, 1280))
    """

    def __init__(self, url, target_size = lambda h,w: (h,w), device = None):
        """Initialize the player with a video file.

        Args:
            url: Path or URL to the video file.
            target_size: Function (height, width) -> (new_height, new_width) for
                scaling. Default is no scaling.
            device: CUDA device ID. If None, uses current device.
        """
        super().__init__(url, num_surfaces=1, target_size=target_size, device=device)

    def frames(self, dtype : torch.dtype):
        """Iterate over all frames in the video.

        Yields frames from the beginning of the video to the end. Each frame
        is converted from YUV to RGB and returned as a PyTorch tensor.

        Args:
            dtype: PyTorch dtype for output tensors. Use torch.float32 for
                normalized [0, 1] values or torch.uint8 for [0, 255] values.

        Yields:
            Tuple of (time, frame) where:
                - time: timedelta of the frame's presentation timestamp
                - frame: torch.Tensor of shape [3, H, W] (RGB) on GPU
        """
        def on_recv(pic, time, frames):
            if pic is None:
                return frames
            stream = extract_stream_ptr(torch.cuda.current_stream())
            surface = pic.map(stream)
            pic.free()
            frame = self.convert(surface, dtype)
            surface.free()
            if frames is None:
                frames = []
            frames.append((time, frame))
            return frames

        while True:
            frames = self.decode(on_recv)
            if frames is None:
                break
            yield from frames

class Screenshoter(BasePlayer):
    """Video frame extractor for capturing frames at specific timestamps.

    This class is optimized for extracting individual frames from a video
    at specified timestamps, using NVIDIA hardware decoding. It's more
    efficient than Player when you only need specific frames rather than
    streaming the entire video.

    Example:
        ss = Screenshoter('/path/to/video.mp4')
        time, frame = ss.screenshot(timedelta(seconds=30), torch.uint8)
        # frame is [C, H, W] tensor on GPU
        ss.free()  # Release decoder resources

        # With accurate seeking (slower but exact timestamp)
        time, frame = ss.screenshot(timedelta(seconds=30), torch.float32, accurate=True)
    """

    def __init__(self, url, target_size = lambda h,w : (h,w), device = None):
        """Initialize the screenshoter with a video file.

        Args:
            url: Path or URL to the video file.
            target_size: Function (height, width) -> (new_height, new_width) for
                scaling. Default is no scaling.
            device: CUDA device ID. If None, uses current device.
        """
        super().__init__(url, 2, target_size, device=device)

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

        if not accurate:
            def on_recv(pic, time, frame):
                if frame is not None:
                    return frame
                if pic is None:
                    raise NoFrameError(f"No frame found at {target} in {self.url}")
                stream = extract_stream_ptr(torch.cuda.current_stream())
                surface = pic.map(stream)
                pic.free()
                frame = self.convert(surface, dtype)
                surface.free()
                return time, frame

            return self.decode(on_recv)

        last = None

        def on_recv(pic, time, frame):
            nonlocal last
            if frame is not None:
                return frame
            if pic is None:
                # End of stream - return last frame if we have one
                if last is not None:
                    time, surface = last
                    frame = self.convert(surface, dtype)
                    surface.free()
                    return time, frame
                raise NoFrameError(f"No frame found at {target} in {self.url}")
            stream = extract_stream_ptr(torch.cuda.current_stream())
            if target < time:
                if last is not None:
                    pic.free() # current pic not needed
                    time, surface = last
                else:
                    # first frame is already past target, use it
                    surface = pic.map(stream)
                    pic.free()
                frame = self.convert(surface, dtype)
                surface.free()
                return time, frame
            else:
                if last is not None: # free the last surface
                    last[1].free()
                surface = pic.map(stream)
                pic.free()
                last = (time, surface)
                return None

        frame = self.decode(on_recv)
        return frame

    def free(self):
        """Release decoder resources.

        Call this method when done using the Screenshoter to free GPU memory.
        After calling free(), the Screenshoter should not be used again.
        """
        self.decoder.free()