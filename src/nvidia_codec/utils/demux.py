"""Demuxing: open media files and extract track metadata. No GPU needed.

    tracks = parse('/path/to/video.mp4')
    video = [t for t in tracks if isinstance(t, VideoTrack)]
    audio = [t for t in tracks if isinstance(t, AudioTrack)]
"""

from datetime import timedelta
from fractions import Fraction
import functools

from ..ffmpeg.libavformat import FormatContext, AVMediaType, AVCodecID
from ..ffmpeg.include.libavutil import (
    AV_NOPTS_VALUE,
    AV_TIME_BASE,
    AVColorRange,
    AVColorSpace,
)
from ..ffmpeg.libavutil import dict_get


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

        # Start time
        self._start_time = stream.start_time
        if self._start_time == AV_NOPTS_VALUE:
            self._start_time = fc.av.start_time
            if self._start_time == AV_NOPTS_VALUE:
                start_time = fc.infer_start_time()
                self._start_time = int(start_time / self._time_base)
            else:
                self._start_time = int(
                    self._start_time / AV_TIME_BASE / self._time_base
                )

        # Duration (lazy — probes only if needed)

    def _probe(self):
        """Probe stream info, disabling all other streams to avoid errors from corrupt tracks.
        Guarded: avformat_find_stream_info segfaults on double-call.
        """
        if getattr(self.fc, "_probed", False):
            return
        self.fc.find_stream_info()
        self.fc._probed = True

    @functools.cached_property
    def extradata(self):
        cp = self.stream.codecpar.contents
        if cp.extradata_size > 0 and cp.extradata:
            return bytes(cp.extradata[: cp.extradata_size])
        # Probe and retry
        self._probe()
        if cp.extradata_size > 0 and cp.extradata:
            return bytes(cp.extradata[: cp.extradata_size])
        return None

    @functools.cached_property
    def mime_codec(self):
        ed = self.extradata
        if self.codec_id == AVCodecID.H264 and ed and len(ed) >= 4:
            if ed[0] == 1:
                # AVCDecoderConfigurationRecord: version=1, profile, compat, level
                return f"avc1.{ed[1]:02x}{ed[2]:02x}{ed[3]:02x}"
            # Annexb format (e.g. TS files): find SPS NAL (type 0x67)
            for i in range(len(ed) - 4):
                if ed[i : i + 3] == b"\x00\x00\x01" and (ed[i + 3] & 0x1F) == 7:
                    sps = i + 4
                    if sps + 2 < len(ed):
                        return f"avc1.{ed[sps]:02x}{ed[sps+1]:02x}{ed[sps+2]:02x}"
        if self.codec_id == AVCodecID.HEVC and ed and len(ed) >= 13:
            profile_space = ["", "A", "B", "C"][(ed[1] >> 6) & 0x3]
            tier = "H" if (ed[1] >> 5) & 0x1 else "L"
            profile_idc = ed[1] & 0x1F
            level_idc = ed[12]
            return f"hev1.{profile_space}{profile_idc}.4.{tier}{level_idc}"
        if self.codec_id == AVCodecID.VP9:
            return "vp09.00.10.08"
        if self.codec_id == AVCodecID.AV1 and ed and len(ed) >= 4:
            profile = (ed[1] >> 5) & 0x7
            level = ed[1] & 0x1F
            tier = (ed[2] >> 7) & 0x1
            bit_depth = {"0": 8, "1": 10, "2": 12}.get(str((ed[2] >> 5) & 0x3), 8)
            return f'av01.{profile}.{level:02d}{"H" if tier else "M"}.{bit_depth:02d}'
        if self.codec_id == AVCodecID.MPEG4:
            return "mp4v.20.9"
        if self.codec_id == AVCodecID.MPEG2:
            return "mp4v.61"
        if self.codec_id == AVCodecID.MPEG1:
            return "mp4v.6b"
        if self.codec_id == AVCodecID.VP8:
            return "vp8"
        if self.codec_id == AVCodecID.WMV3:
            return "wmv3"
        if self.codec_id == AVCodecID.WMV2:
            return "wmv2"
        if self.codec_id == AVCodecID.GIF:
            return "gif"
        if self.codec_id == AVCodecID.MJPEG:
            return "mjpeg"
        if self.codec_id == AVCodecID.DVVIDEO:
            return "dvvideo"
        if self.codec_id == AVCodecID.PRORES:
            return "prores"
        if self.codec_id == AVCodecID.RV40:
            return "rv40"
        if self.codec_id == AVCodecID.VC1:
            return "vc1"
        if self.codec_id == AVCodecID.VP6F:
            return "vp6f"
        if self.codec_id == AVCodecID.PNG:
            return "png"
        if self.codec_id == AVCodecID.WEBP:
            return "webp"
        return None

    @functools.cached_property
    def _duration_pts(self):
        tb = self._time_base
        d = self.stream.duration
        if d != AV_NOPTS_VALUE:
            return d
        self._probe()
        d = self.stream.duration
        if d != AV_NOPTS_VALUE:
            return d
        for key in ("DURATION", "DURATION-eng"):
            tag = dict_get(self.stream.metadata, key)
            if tag:
                h, m, s = tag.split(":")
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
        return timedelta(
            seconds=float((int(pts) - int(self._start_time)) * self._time_base)
        )


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

    @functools.cached_property
    def mime_codec(self):
        if self.codec_id == AVCodecID.AAC:
            return "mp4a.40.2"
        if self.codec_id == AVCodecID.MP3:
            return "mp4a.40.34"
        if self.codec_id == AVCodecID.OPUS:
            return "opus"
        if self.codec_id == AVCodecID.FLAC:
            return "flac"
        if self.codec_id == AVCodecID.AC3:
            return "ac-3"
        if self.codec_id == AVCodecID.EAC3:
            return "ec-3"
        if self.codec_id == AVCodecID.PCM_S16LE:
            return "pcm"
        if self.codec_id == AVCodecID.PCM_S16BE:
            return "pcm"
        if self.codec_id == AVCodecID.WMAV2:
            return "wmav2"
        if self.codec_id == AVCodecID.COOK:
            return "cook"
        if self.codec_id == AVCodecID.DTS:
            return "dts"
        if self.codec_id == AVCodecID.VORBIS:
            return "vorbis"
        if self.codec_id == AVCodecID.MP2:
            return "mp4a.40.33"
        if self.codec_id == AVCodecID.WMAPRO:
            return "wmapro"
        if self.codec_id == AVCodecID.WMALOSSLESS:
            return "wmalossless"
        if self.codec_id.value in (65560,):  # pcm_bluray
            return "pcm"
        return None


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
