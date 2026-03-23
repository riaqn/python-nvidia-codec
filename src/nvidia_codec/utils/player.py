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
import functools
import numpy as np
from queue import Queue, ShutDown
import threading

from ..ffmpeg.libavcodec import BSFContext
from ..ffmpeg.libavformat import FormatContext, AVMediaType, AVCodecID
from ..ffmpeg.include.libavformat import AVINDEX_KEYFRAME
from ..ffmpeg.include.libavcodec import AV_PKT_FLAG_KEY
from ..ffmpeg.include.libavutil import (
    AV_NOPTS_VALUE,
    AV_TIME_BASE,
    AVColorRange,
    AVColorSpace,
    AVDISCARD_NONE,
    AVDISCARD_NONKEY,
)
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


class DecodeWorker:
    """Background decode worker. Runs on a dedicated thread.

    Commands via cmd_q:
        (ts_or_none, out_q, keyframes_only) — seek (if ts not None), then decode into out_q
        None — stop

    Pushes (pts, pic) tuples to out_q. Consumer maps pics to surfaces.
    """

    def __init__(self, cmd_q, track, bsf, decoder, prepend_extradata=False):
        self.cmd_q = cmd_q
        self.track = track
        self.bsf = bsf
        self.decoder = decoder
        self.prepend_extradata = prepend_extradata

    def flush(self):
        """Flush bsf and decoder, discarding buffered frames."""
        self.bsf.flush()
        self.decoder.send(None, lambda pic: pic.free() if pic is not None else None)

    def run(self):
        while True:
            cmd = self.cmd_q.get()
            if cmd is None:
                break
            kts, out_q, keyframes_only = cmd
            # procotol of [out_q] between DecodeWorker and the consumer: The
            # decoder put pictures onto the queue; it's the consumer's
            # responsibility to free those pictures promptly. The decoder will
            # send EOF either as None or exception, then go on to the next
            # command; it never shutdown [out_q]. The consumer can shutdown to
            # indicate no interest in further picture. The consumer is still
            # responsible for freeing the remaining pictures in the queue.
            try:
                self.decode(kts, out_q, keyframes_only)
            except ShutDown:
                pass
            except Exception as e:
                out_q.put(e)

    def decode(self, kts, out_q, keyframes_only):
        """Seek then demux and decode. kts=None means seek to beginning."""
        track = self.track
        if kts is None:
            track.fc.seek_file(
                track.stream, track._start_time, max_ts=track._start_time
            )
        else:
            track.fc.seek_file(track.stream, kts, min_ts=kts, max_ts=kts)

        self.flush()
        track.stream.discard = AVDISCARD_NONKEY if keyframes_only else AVDISCARD_NONE
        packets = track.fc.read_packets(track.stream)
        it = self.bsf.filter(packets, flush=False, reuse=True)

        def on_recv(pic):
            if pic is None:
                return
            out_q.put(pic)

        for pkt in it:
            pts = pkt.av.pts if pkt.av.pts != AV_NOPTS_VALUE else pkt.av.dts
            arr = np.ctypeslib.as_array(pkt.av.data, (pkt.av.size,))
            if self.prepend_extradata:
                par_out = self.bsf.av.par_out.contents
                if par_out.extradata_size > 0 and par_out.extradata:
                    extradata = bytes(par_out.extradata[: par_out.extradata_size])
                    arr = np.concatenate(
                        [np.frombuffer(extradata, dtype=np.uint8), arr]
                    )
                self.prepend_extradata = False
            self.decoder.send((pts, arr), on_recv)
            if keyframes_only:
                self.decoder.send(None, on_recv)
                self.flush()
        self.decoder.send(None, on_recv)
        out_q.put(None)


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

    def __init__(
        self,
        track,
        num_surfaces=2,
        target_size=None,
        cropping=None,
        target_rect=None,
        device=None,
    ):

        self.track = track
        self._num_surfaces = num_surfaces

        # The queue that the decoderWorker is producing into.
        self._out_q = None

        # the last few decoded frames, guaranteed to be continuous. When in
        # [keyframes_only] mode, it contains at most one element (the keyframe
        # just decoded) and get cleared before decoding the next keyframe.
        self._decoded_surfaces = []

        codec_id = track.codec_id
        if codec_id == AVCodecID.HEVC:
            f = "hevc_mp4toannexb"
        elif codec_id == AVCodecID.H264:
            f = "h264_mp4toannexb"
        else:
            f = None

        # The decoder thread and the command queue that it reads from.
        # INVARIANT: they must be either both None, or both not None. In the
        # latter case, _cmd_q must be the command queue that _thread is reading
        # from.
        self._thread = None
        self._cmd_q = None

        self.bsf = BSFContext(f, track.stream.codecpar.contents, track.stream.time_base)

        self.device = torch.cuda.current_device() if device is None else device

        def decide(p):
            d = {
                "num_pictures": p["min_num_pictures"],
                "num_surfaces": num_surfaces,
            }
            if target_size is not None:
                d["target_size"] = target_size
            if cropping is not None:
                d["cropping"] = cropping
            if target_rect is not None:
                d["target_rect"] = target_rect
            return d

        self.decoder = BaseDecoder(
            av2cuda(codec_id),
            decide=decide,
            device=self.device,
            extradata=track.extradata,
            coded_width=track.width,
            coded_height=track.height,
        )

        self._cmd_q = Queue()
        self._worker = DecodeWorker(
            self._cmd_q,
            track,
            self.bsf,
            self.decoder,
            prepend_extradata=(f is None),
        )
        self._thread = threading.Thread(target=self._worker.run, daemon=True)
        self._thread.start()

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
        return convert(
            surface,
            self.track.color_space(AVColorSpace.BT470BG),
            self.track.color_range(AVColorRange.MPEG),
            dtype,
        )

    def seek(self, target: timedelta, keyframes_only=False):
        """Seek to keyframe at-or-before target position and start decoding."""
        target_pts = self.track.time2pts(target)
        entry = FormatContext.index_get_entry_from_timestamp(
            self.track.stream, target_pts, 1
        )
        assert entry is not None, f"no keyframe found at or before {target}"
        self._start_decode(entry.timestamp, keyframes_only=keyframes_only)

    def assert_decoder_eof(self):
        self._out_q.shutdown()
        assert self._out_q.empty(), "decoder produced more after EOF"
        self._out_q = None

    def _recv_surface(self):
        """Get next decoded frame from queue, map to surface, store in ring buffer.
        Returns surface or None if stream ended. Raises if decode thread errored.
        """
        if self._out_q is None:
            return None
        pic = self._out_q.get()
        if pic is None:
            self.assert_decoder_eof()
            return None
        if isinstance(pic, Exception):
            self.assert_decoder_eof()
            raise pic
        try:
            if len(self._decoded_surfaces) >= self._num_surfaces:
                self._decoded_surfaces.pop(0).free()
            stream = extract_stream_ptr(torch.cuda.current_stream(self.device))
            surface = pic.map(stream)
            self._decoded_surfaces.append(surface)
            return surface
        finally:
            pic.free()

    def _discard(self):
        """Free all held surfaces and drain/shutdown the output queue."""

        if self._out_q is not None:
            self._out_q.shutdown()
            while True:
                try:
                    pic = self._out_q.get()
                except ShutDown:
                    break
                if pic is None or isinstance(pic, Exception):
                    assert self._out_q.empty(), "decoder produced more after EOF"
                    break
                pic.free()
            self._out_q = None

        while len(self._decoded_surfaces) > 0:
            self._decoded_surfaces.pop().free()

    def _start_decode(self, kts: int, keyframes_only=False):
        """Seek to keyframe timestamp and start decoding. Returns the output queue.

        kts must be a keyframe timestamp from the container index.
        """
        self._discard()
        self._out_q = Queue()
        self._cmd_q.put((kts, self._out_q, keyframes_only))

    def frames(self, dtype: torch.dtype):
        """Iterate over decoded frames from current position.

        Caller must seek() or _start_decode() first.

        Yields:
            Tuple of (timedelta, frame) where frame is [C, H, W] tensor on GPU.
        """
        assert self._out_q is not None, "must seek() before frames()"
        while True:
            surface = self._recv_surface()
            if surface is None:
                break
            yield (self.track.pts2time(surface.pts), self.convert(surface, dtype))

    def should_seek(self, target_pts: int):
        """Return the keyframe DTS to seek to, or None if no seek needed.

        target_pts is in PTS space (from decoded frames or time2pts).
        Index entries use DTS. Due to B-frames, DTS <= PTS for any given frame.
        We use target_pts as an approximation for the index lookup — close enough
        since we just need the nearest keyframe.
        """
        entry = FormatContext.index_get_entry_from_timestamp(
            self.track.stream, target_pts, 1  # find keyframe at-or-before target
        )
        assert entry is not None, f"no keyframe found at or before pts={target_pts}"
        kts = entry.timestamp
        if len(self._decoded_surfaces) > 0:
            last_pts = self._decoded_surfaces[-1].pts
            # We can decode forward if:
            # 1. We haven't passed the target yet (last_pts <= target_pts)
            # 2. The keyframe we'd seek to is at-or-before where we already are
            #    (kts <= last_pts)
            if last_pts <= target_pts and kts <= last_pts:
                return None  # can decode forward
        return kts

    def screenshot_forward(self, target: timedelta, dtype: torch.dtype):
        """Decode forward from current position to the frame closest to target.

        Does NOT seek. Caller must ensure the decoder is positioned before target.

        Returns:
            Tuple of (time, frame) where time is timedelta.

        Raises:
            NoFrameError: If no frame could be extracted.
        """
        target_pts = self.track.time2pts(target)

        while True:
            surface = self._recv_surface()
            if surface is None:
                break
            if surface.pts > target_pts:
                break

        if len(self._decoded_surfaces) == 0:
            raise NoFrameError(f"No frame found at {target} in {self.track.fc}")

        overshot = self._decoded_surfaces[-1].pts > target_pts
        if overshot:
            # IMPORTANT: DO NOT remove these assertions. They catch real bugs.
            assert (
                len(self._decoded_surfaces) >= 2
            ), "overshot target but no previous frame"
            assert (
                self._decoded_surfaces[-2].pts <= target_pts
            ), "previous frame is also past target"
            surface = self._decoded_surfaces[-2]
        else:
            surface = self._decoded_surfaces[-1]
        return (self.track.pts2time(surface.pts), self.convert(surface, dtype))

    def screenshot(self, target: timedelta, dtype: torch.dtype, accurate: bool = False):
        """Extract a frame at the specified timestamp. Seeks if needed.

        Returns:
            Tuple of (time, frame) where time is timedelta.

        Raises:
            NoFrameError: If no frame could be extracted.
        """
        target_pts = self.track.time2pts(target)
        seek_kts = self.should_seek(target_pts)
        if seek_kts is not None:
            self._start_decode(seek_kts)
        if accurate:
            return self.screenshot_forward(target, dtype)
        else:
            if seek_kts is None:
                surface = self._decoded_surfaces[-1]
            else:
                surface = self._recv_surface()
                assert surface is not None, "seeked to keyframe but got no output"
            return (self.track.pts2time(surface.pts), self.convert(surface, dtype))

    def screenshots(self, dtype: torch.dtype, max_interval: timedelta, start_kts=None):
        """Take screenshots throughout the video with interval <= max_interval.

        If the file has a keyframe index, walks it efficiently (seeking + fills).
        Otherwise, falls back to sequential keyframe decode.

        Args:
            start_kts: Starting keyframe timestamp (int, from index). None = beginning of file.

        Yields:
            Tuple of (timedelta, frame, kts_timedelta_or_none).
        """
        track = self.track
        max_gap = int(max_interval.total_seconds() / float(track._time_base))

        timestamp = start_kts

        while True:
            # Seek to this keyframe and decode first frame
            self._start_decode(timestamp)
            surface = self._recv_surface()
            assert surface is not None, f"keyframe at ts={timestamp} produced no output"
            if timestamp is None:
                timestamp = surface.pts
            kf_frame = self.convert(surface, dtype)
            yield (track.pts2time(surface.pts), kf_frame, track.pts2time(timestamp))

            # Find next keyframe entry, or use duration as end marker
            next_entry = FormatContext.index_get_entry_from_timestamp(
                track.stream, timestamp + 1, 0
            )
            is_last = next_entry is None
            if is_last:
                # No more keyframes — treat video end as the boundary
                if track._duration_pts is None:
                    break
                end_ts = track._duration_pts + track._start_time
                gap_ts = end_ts - timestamp
            else:
                gap_ts = next_entry.timestamp - timestamp

            if gap_ts <= max_gap:
                if is_last:
                    # Fill one last frame at the end
                    if track.duration > track.pts2time(surface.pts):
                        t, frame = self.screenshot_forward(track.duration, dtype)
                        yield (t, frame, None)
                    break
                # Dense keyframes — skip ahead to the farthest keyframe within max_interval
                kf = next_entry
                while True:
                    kf_after = FormatContext.index_get_entry_from_timestamp(
                        track.stream, kf.timestamp + 1, 0
                    )
                    if kf_after is None:
                        break
                    if kf_after.timestamp - timestamp > max_gap:
                        break
                    kf = kf_after
                timestamp = kf.timestamp
            else:
                # Sparse gap — fill with evenly spaced screenshots
                end_time = (
                    track.duration if is_last else track.pts2time(next_entry.timestamp)
                )
                n = gap_ts // max_gap + 1
                step_td = (end_time - track.pts2time(timestamp)) / n
                last_yield_time = track.pts2time(surface.pts)
                end_range = n + 1 if is_last else n
                for j in range(1, end_range):
                    fill_time = last_yield_time + step_td * j
                    t, frame = self.screenshot_forward(fill_time, dtype)
                    yield (t, frame, None)
                if is_last:
                    break
                timestamp = next_entry.timestamp

    def free(self):
        self._discard()
        if self._cmd_q is not None:
            self._cmd_q.put(None)
            self._cmd_q.shutdown()
            self._cmd_q = None
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        self.decoder.free()

    def __del__(self):
        self.free()

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

    def __init__(
        self,
        url,
        target_size=None,
        cropping=None,
        target_rect=None,
        device=None,
        track_idx=None,
    ):
        tracks = [t for t in parse(url) if isinstance(t, VideoTrack)]
        if not tracks:
            raise ValueError(f"{url} has no video stream")
        if track_idx is not None:
            track = next((t for t in tracks if t.index == track_idx), None)
            if not track:
                raise ValueError(f"{url} has no video track with index {track_idx}")
        elif len(tracks) == 1:
            track = tracks[0]
        else:
            raise ValueError(f"{url} has {len(tracks)} video tracks, specify track_idx")
        super().__init__(
            track,
            num_surfaces=2,
            target_size=target_size,
            cropping=cropping,
            target_rect=target_rect,
            device=device,
        )
        self.url = url
        self._start_decode(None)
