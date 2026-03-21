"""High-level video player classes using NVIDIA hardware decoding.

This module provides user-friendly classes for decoding video files using
NVIDIA's NVDEC hardware decoder. The decoded frames are returned as PyTorch
tensors on the GPU.

Architecture:
    Parser      — opens a file, returns VideoTrack objects (no GPU needed)
    VideoTrackPlayer — takes a VideoTrack, decodes frames on GPU
    Player      — convenience: Parser + pick best track + VideoTrackPlayer

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

    # Just probe metadata (no GPU)
    parser = Parser('/path/to/video.mp4')
    for track in parser.video_tracks:
        print(track.width, track.height, track.mime_codec, track.duration)
"""
from datetime import timedelta
from fractions import Fraction
import numpy as np

from ..ffmpeg.libavcodec import BSFContext
from ..ffmpeg.libavformat import FormatContext, AVMediaType, AVCodecID
from ..ffmpeg.include.libavformat import AVINDEX_KEYFRAME
from ..ffmpeg.include.libavcodec import AV_PKT_FLAG_KEY
from ..ffmpeg.include.libavutil  import AV_NOPTS_VALUE, AV_TIME_BASE, AVColorRange, AVColorSpace, AVDISCARD_NONE, AVDISCARD_NONKEY
from ..ffmpeg.libavutil import dict_get
from .compat import av2cuda, cuda2av, extract_stream_ptr
from ..core.decode import BaseDecoder
from .color import convert
from .. import NoFrameError

import torch
import logging

log = logging.getLogger(__name__)


class VideoTrack:
    """Metadata for a single video track in a container. No GPU needed.

    Attributes:
        stream: The underlying AVStream (for seeking/reading).
        fc: The parent FormatContext.
        codec_id: AVCodecID enum value.
        mime_codec: MIME codec string (e.g. 'avc1.640028', 'hev1.1.6.L93.B0') or None.
        width: Video width in pixels.
        height: Video height in pixels.
        bit_rate: Stream bitrate.
        duration: Video duration as timedelta, or None.
        extradata: Codec extradata bytes, or None.
        color_space: AVColorSpace enum value.
        color_range: AVColorRange enum value.
    """

    def __init__(self, fc, stream):
        self.fc = fc
        self.stream = stream
        self.index = stream.index
        cp = stream.codecpar.contents
        self.codec_id = cp.codec_id
        self.width = cp.width
        self.height = cp.height
        self.bit_rate = cp.bit_rate
        self._color_space = cp.color_space
        self._color_range = cp.color_range

        # Time base
        self._time_base = Fraction(stream.time_base.num, stream.time_base.den)

        # Eagerly read extradata and mime_codec (may be incomplete before probe)
        self._read_extradata()
        self.mime_codec = self._parse_mime_codec()

        # If mime_codec failed, probe and retry
        if self.mime_codec is None and self.codec_id.value != 0:
            self._probe()
            self._read_extradata()
            self.mime_codec = self._parse_mime_codec()

        # Start time
        self._start_time = stream.start_time
        if self._start_time == AV_NOPTS_VALUE:
            self._start_time = fc.av.start_time
            if self._start_time == AV_NOPTS_VALUE:
                start_time = fc.infer_start_time()
                self._start_time = int(start_time / self._time_base)
            else:
                self._start_time = int(self._start_time / AV_TIME_BASE / self._time_base)

        # Duration (lazy — probes only if needed)
        self._duration_pts = self._infer_duration()

    def _probe(self):
        self.fc.find_stream_info()

    def _read_extradata(self):
        cp = self.stream.codecpar.contents
        if cp.extradata_size > 0 and cp.extradata:
            self.extradata = bytes(cp.extradata[:cp.extradata_size])
        else:
            self.extradata = None

    def _parse_mime_codec(self):
        ed = self.extradata
        if self.codec_id == AVCodecID.H264 and ed and len(ed) >= 4:
            if ed[0] == 1:
                # AVCDecoderConfigurationRecord: version=1, profile, compat, level
                return f'avc1.{ed[1]:02x}{ed[2]:02x}{ed[3]:02x}'
            # Annexb format (e.g. TS files): find SPS NAL (type 0x67)
            for i in range(len(ed) - 4):
                if ed[i:i+3] == b'\x00\x00\x01' and (ed[i+3] & 0x1f) == 7:
                    sps = i + 4
                    if sps + 2 < len(ed):
                        return f'avc1.{ed[sps]:02x}{ed[sps+1]:02x}{ed[sps+2]:02x}'
        if self.codec_id == AVCodecID.HEVC and ed and len(ed) >= 13:
            profile_space = ['', 'A', 'B', 'C'][(ed[1] >> 6) & 0x3]
            tier = 'H' if (ed[1] >> 5) & 0x1 else 'L'
            profile_idc = ed[1] & 0x1f
            level_idc = ed[12]
            return f'hev1.{profile_space}{profile_idc}.4.{tier}{level_idc}'
        if self.codec_id == AVCodecID.VP9:
            return 'vp09.00.10.08'
        if self.codec_id == AVCodecID.AV1 and ed and len(ed) >= 4:
            profile = (ed[1] >> 5) & 0x7
            level = ed[1] & 0x1f
            tier = (ed[2] >> 7) & 0x1
            bit_depth = {'0': 8, '1': 10, '2': 12}.get(str((ed[2] >> 5) & 0x3), 8)
            return f'av01.{profile}.{level:02d}{"H" if tier else "M"}.{bit_depth:02d}'
        if self.codec_id == AVCodecID.MPEG4:
            return 'mp4v.20.9'
        if self.codec_id == AVCodecID.MPEG2:
            return 'mp4v.61'
        if self.codec_id == AVCodecID.MPEG1:
            return 'mp4v.6b'
        if self.codec_id == AVCodecID.VP8:
            return 'vp8'
        if self.codec_id == AVCodecID.WMV3:
            return 'wmv3'
        if self.codec_id == AVCodecID.WMV2:
            return 'wmv2'
        if self.codec_id == AVCodecID.GIF:
            return 'gif'
        if self.codec_id == AVCodecID.MJPEG:
            return 'mjpeg'
        if self.codec_id == AVCodecID.DVVIDEO:
            return 'dvvideo'
        if self.codec_id == AVCodecID.PRORES:
            return 'prores'
        if self.codec_id == AVCodecID.RV40:
            return 'rv40'
        if self.codec_id == AVCodecID.VC1:
            return 'vc1'
        if self.codec_id == AVCodecID.VP6F:
            return 'vp6f'
        if self.codec_id == AVCodecID.PNG:
            return 'png'
        if self.codec_id == AVCodecID.WEBP:
            return 'webp'
        raise ValueError(f'unknown video mime codec for codec_id={self.codec_id}')

    def _infer_duration(self):
        tb = self._time_base
        d = self.stream.duration
        if d != AV_NOPTS_VALUE:
            return d
        self._probe()
        d = self.stream.duration
        if d != AV_NOPTS_VALUE:
            return d
        for key in ('DURATION', 'DURATION-eng'):
            tag = dict_get(self.stream.metadata, key)
            if tag:
                h, m, s = tag.split(':')
                td = timedelta(hours=int(h), minutes=int(m), seconds=float(s))
                return int(td.total_seconds() / tb)
        d = self.fc.av.duration
        if d > 0:
            return int(d / AV_TIME_BASE / tb)
        return None

    @property
    def duration(self):
        """Video duration as a timedelta, or None if unknown."""
        if self._duration_pts is None:
            return None
        return timedelta(seconds=float(self._duration_pts * self._time_base))

    def color_space(self, default=AVColorSpace.UNSPECIFIED):
        r = self._color_space
        return default if r == AVColorSpace.UNSPECIFIED else r

    def color_range(self, default=AVColorRange.UNSPECIFIED):
        r = self._color_range
        return default if r == AVColorRange.UNSPECIFIED else r

    def time2pts(self, time: timedelta):
        return int(time.total_seconds() / self._time_base) + self._start_time

    def pts2time(self, pts: int):
        if pts == AV_NOPTS_VALUE:
            return None
        return timedelta(seconds=float((int(pts) - int(self._start_time)) * self._time_base))


class AudioTrack:
    """Metadata for a single audio track in a container.

    Attributes:
        index: Stream index in the container.
        codec_id: AVCodecID enum value.
        mime_codec: MIME codec string (e.g. 'mp4a.40.2') or None.
        sample_rate: Sample rate in Hz.
        bit_rate: Stream bitrate.
    """

    def __init__(self, stream):
        self.stream = stream
        self.index = stream.index
        cp = stream.codecpar.contents
        self.codec_id = cp.codec_id
        self.sample_rate = cp.sample_rate
        self.bit_rate = cp.bit_rate
        self.mime_codec = self._parse_mime_codec()

    def _parse_mime_codec(self):
        if self.codec_id == AVCodecID.AAC:
            return 'mp4a.40.2'
        if self.codec_id == AVCodecID.MP3:
            return 'mp4a.40.34'
        if self.codec_id == AVCodecID.OPUS:
            return 'opus'
        if self.codec_id == AVCodecID.FLAC:
            return 'flac'
        if self.codec_id == AVCodecID.AC3:
            return 'ac-3'
        if self.codec_id == AVCodecID.EAC3:
            return 'ec-3'
        if self.codec_id == AVCodecID.PCM_S16LE:
            return 'pcm'
        if self.codec_id == AVCodecID.PCM_S16BE:
            return 'pcm'
        if self.codec_id == AVCodecID.WMAV2:
            return 'wmav2'
        if self.codec_id == AVCodecID.COOK:
            return 'cook'
        if self.codec_id == AVCodecID.DTS:
            return 'dts'
        if self.codec_id == AVCodecID.VORBIS:
            return 'vorbis'
        if self.codec_id == AVCodecID.MP2:
            return 'mp4a.40.33'
        if self.codec_id == AVCodecID.WMAPRO:
            return 'wmapro'
        if self.codec_id == AVCodecID.WMALOSSLESS:
            return 'wmalossless'
        if self.codec_id.value in (65560,):  # pcm_bluray
            return 'pcm'
        raise ValueError(f'unknown audio mime codec for codec_id={self.codec_id}')


def parse(url):
    """Open a media file and return a list of tracks (VideoTrack and AudioTrack). No GPU needed.

    Example:
        tracks = parse('/path/to/video.mp4')
        video = [t for t in tracks if isinstance(t, VideoTrack)]
        audio = [t for t in tracks if isinstance(t, AudioTrack)]
    """
    fc = FormatContext(url)
    tracks = []
    for s in fc.streams:
        codec_type = s.codecpar.contents.codec_type
        if codec_type == AVMediaType.VIDEO:
            tracks.append(VideoTrack(fc, s))
        elif codec_type == AVMediaType.AUDIO:
            tracks.append(AudioTrack(s))
    return tracks


class VideoTrackPlayer:
    """GPU-accelerated decoder for a single VideoTrack.

    Handles seeking, packet reading, bitstream filtering, and NVDEC decoding.

    Attributes:
        track: The VideoTrack being played.
        width: Original video width.
        height: Original video height.
        target_width: Output width after scaling.
        target_height: Output height after scaling.
        duration: Video duration as timedelta.
    """

    def __init__(self, track, num_surfaces=2, target_size=None, cropping=None, target_rect=None, device=None):
        self.track = track
        self._prepend_extradata = False

        codec_id = track.codec_id
        if codec_id == AVCodecID.HEVC:
            f = 'hevc_mp4toannexb'
        elif codec_id == AVCodecID.H264:
            f = 'h264_mp4toannexb'
        else:
            f = None

        self.bsf = BSFContext(f, track.stream.codecpar.contents, track.stream.time_base)

        self.device = torch.cuda.current_device() if device is None else device

        def decide(p):
            d = {
                'num_pictures': p['min_num_pictures'],
                'num_surfaces': num_surfaces,
            }
            if target_size is not None:
                d['target_size'] = target_size
            if cropping is not None:
                d['cropping'] = cropping
            if target_rect is not None:
                d['target_rect'] = target_rect
            return d

        self.decoder = BaseDecoder(
            av2cuda(codec_id),
            decide=decide,
            device=self.device,
            extradata=track.extradata,
            coded_width=track.width,
            coded_height=track.height,
        )

    @property
    def width(self):
        return self.track.width

    @property
    def height(self):
        return self.track.height

    @property
    def target_width(self):
        return self.decoder.target_width

    @property
    def target_height(self):
        return self.decoder.target_height

    @property
    def duration(self):
        return self.track.duration

    @property
    def mime_codec(self):
        return self.track.mime_codec

    def convert(self, surface, dtype):
        return convert(surface, self.track.color_space(AVColorSpace.BT470BG), self.track.color_range(AVColorRange.MPEG), dtype)

    def seek(self, target: timedelta):
        target_pts = self.track.time2pts(target)
        log.debug(f'target_pts: {target_pts}')
        self.track.fc.seek_file(self.track.stream, target_pts, max_ts=target_pts)
        self.bsf.flush()
        self.decoder.flush()
        self._prepend_extradata = True
        self._last_decoded_pts = None

    def decode(self, on_recv, keyframes_only=False):
        track = self.track
        if keyframes_only:
            track.stream.discard = AVDISCARD_NONKEY
        else:
            track.stream.discard = AVDISCARD_NONE
        packets = track.fc.read_packets(track.stream)
        if keyframes_only:
            packets = (pkt for pkt in packets if pkt.av.flags & AV_PKT_FLAG_KEY)
        it = self.bsf.filter(packets, flush=False, reuse=True)

        def on_recv_(pic, pts, ret):
            return on_recv(pic, track.pts2time(pts), ret)

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

    def frames(self, dtype: torch.dtype, keyframes_only=False):
        """Iterate over decoded frames.

        Yields:
            Tuple of (time, frame) where time is timedelta, frame is [C, H, W] tensor on GPU.
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

    def screenshot(self, target: timedelta, dtype: torch.dtype, accurate: bool = False):
        """Extract a frame at the specified timestamp.

        Args:
            target: Target timestamp as a timedelta.
            dtype: PyTorch dtype for the output tensor.
            accurate: If True, decode until finding the frame closest to target.

        Returns:
            Tuple of (time, frame).

        Raises:
            NoFrameError: If no frame could be extracted.
        """
        target_pts = self.track.time2pts(target)

        should_seek = True
        if hasattr(self, '_last_decoded_pts') and self._last_decoded_pts is not None:
            if self._last_decoded_pts <= target_pts:
                entry = FormatContext.index_get_entry_from_timestamp(
                    self.track.stream, target_pts, 1
                )
                if entry is not None and entry.timestamp <= self._last_decoded_pts:
                    should_seek = False

        if should_seek:
            self.seek(target)

        last = None
        for time, frame in self.frames(dtype):
            self._last_decoded_pts = self.track.time2pts(time)
            if not accurate:
                return time, frame
            if time > target:
                break
            last = (time, frame)

        if last is not None:
            return last
        raise NoFrameError(f"No frame found at {target} in {self.track.fc}")

    def free(self):
        self.decoder.free()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.free()


class Player(VideoTrackPlayer):
    """Convenience class: opens a file, picks the best video track, decodes.

    This is the simplest way to decode video — equivalent to:
        parser = Parser(url)
        track = parser.best_video_track()
        player = VideoTrackPlayer(track, ...)

    Example:
        player = Player('/path/to/video.mp4')
        for time, frame in player.frames(torch.float32):
            process(frame)
    """

    def __init__(self, url, target_size=None, cropping=None, target_rect=None, device=None):
        tracks = [t for t in parse(url) if isinstance(t, VideoTrack)]
        assert tracks, f'{url} has no video stream'
        track = max(tracks, key=lambda t: t.bit_rate)
        if len(tracks) > 1:
            log.warning(f'{url} has {len(tracks)} video tracks, picking highest bitrate @ {track.bit_rate}')
        super().__init__(track, num_surfaces=2, target_size=target_size, cropping=cropping, target_rect=target_rect, device=device)
        self.url = url
