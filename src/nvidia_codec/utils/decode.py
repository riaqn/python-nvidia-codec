"""GPU-accelerated video decoding using NVIDIA NVDEC."""

import collections
from datetime import timedelta
import numpy as np
from queue import Queue
import threading

from ..ffmpeg.libavcodec import BSFContext
from ..ffmpeg.libavformat import FormatContext, AVCodecID
from ..ffmpeg.include.libavutil import (
    AV_NOPTS_VALUE,
    AVColorRange,
    AVColorSpace,
    AVDISCARD_NONE,
    AVDISCARD_NONKEY,
)
from .compat import av2cuda, extract_stream_ptr
from ..core.decode import BaseDecoder, Picture, Surface
from .color import convert
from .demux import VideoTrack
from .. import NoFrameError

import torch
import logging


log = logging.getLogger(__name__)




class OwnedSurface:
    """Wraps a Surface with semaphore ownership. Releases on free."""

    def __init__(self, surface):
        surface.decoder._surface_sem.acquire()
        self._surface = surface

    def __getattr__(self, name):
        return getattr(self._surface, name)

    def free(self):
        if self._surface is not None:
            self._surface.free()
            self._surface.decoder._surface_sem.release()
            self._surface = None


class OwnedPicture:
    """Wraps a Picture with consumer ownership of the slot via lock.

    Acquires the slot lock on creation. free() releases it.
    map() returns an OwnedSurface.
    """

    def __init__(self, pic):
        self._pic = pic
        pic.decoder._slot_locks[pic.index].acquire()

    @property
    def pts(self):
        return self._pic.pts

    @property
    def index(self):
        return self._pic.index

    def free(self):
        if self._pic is not None:
            self._pic.decoder._slot_locks[self._pic.index].release()
            self._pic = None

    def map(self, stream=0):
        assert self._pic is not None, "Picture already freed"
        return OwnedSurface(self._pic.map(stream))


class Decoder(BaseDecoder):
    """GPU-accelerated decoder for a single VideoTrack.

    Subclasses BaseDecoder to add threading, synchronization, and
    high-level decode methods (screenshot, screenshots).

    Owns all synchronization:
    - Per-slot semaphores for picture slots (pre_decode blocks, OwnedPicture.free releases)
    - Surface semaphore for cuvidMapVideoFrame64 (OwnedSurface manages)

    Args:
        extra_pictures: Callable(min_pics) -> int, returns the number of extra
            picture slots beyond codec minimum. Higher values allow better
            pipelining between decode and consumer threads. Default: 2.
    """

    def __init__(
        self,
        track,
        extra_pictures=lambda min_pics: min_pics,
        target_size=None,
        cropping=None,
        target_rect=None,
        device=None,
    ):
        self.track = track
        self._slot_locks = None  # initialized in decide(), after we know num_pictures
        self._surface_sem = None
        self._slots = {}  # slot_idx -> Picture, tracks what's in each decode buffer
        self._recent = collections.deque()  # recent frames in display order, reset on seek

        codec_id = track.codec_id
        if codec_id == AVCodecID.HEVC:
            f = "hevc_mp4toannexb,dump_extra"
        elif codec_id == AVCodecID.H264:
            f = "h264_mp4toannexb,dump_extra"
        elif codec_id == AVCodecID.MPEG4:
            f = "dump_extra"
        else:
            f = None

        self.bsf = BSFContext(f, track.stream.codecpar.contents, track.stream.time_base)

        def decide(p):
            num_pics = p["min_num_pictures"] + extra_pictures(p["min_num_pictures"])
            self._slot_locks = [threading.Lock() for _ in range(num_pics)]
            self._surface_sem = threading.Semaphore(1)
            self._recent = collections.deque(maxlen=num_pics)
            d = {
                "num_pictures": num_pics,
                "num_surfaces": 1,
            }
            if target_size is not None:
                d["target_size"] = target_size
            if cropping is not None:
                d["cropping"] = cropping
            if target_rect is not None:
                d["target_rect"] = target_rect
            return d

        super().__init__(
            av2cuda(codec_id),
            decide=decide,
            device=torch.cuda.current_device() if device is None else device,
            extradata=track.extradata,
            coded_width=track.width,
            coded_height=track.height,
        )

    def pre_decode(self, idx):
        """NVDEC reclaims slot idx. Clear tracking, wait for consumer if held."""
        self._slots.pop(idx, None)
        with self._slot_locks[idx]:
            pass

    def post_decode(self, pic):
        """NVDEC releases slot for display. Record in tracking."""
        if pic is not None:
            self._slots[pic.index] = pic
            self._recent.append(pic)

    def with_callbacks(self, func, post_decode=None):
        """Run func() with a temporary post_decode extension.

        The override runs AFTER the base post_decode (slot tracking is
        always preserved). Restores on return.
        """
        if post_decode is None:
            return func()
        saved = self.__dict__.get('post_decode')
        override = post_decode
        def chained(pic):
            Decoder.post_decode(self, pic)
            override(pic)
        self.post_decode = chained
        try:
            return func()
        finally:
            if saved is not None:
                self.post_decode = saved
            elif 'post_decode' in self.__dict__:
                del self.__dict__['post_decode']

    def _seek(self, kts):
        """Seek the demuxer to the given timestamp. Flushes NVDEC parser, resets state."""
        self.track.fc.seek_file(self.track.stream, kts, min_ts=kts, max_ts=kts)
        self.with_callbacks(lambda: self.send(None), post_decode=lambda pic: None)
        self._recent.clear()

    def _decode_packets(self, keyframes_only=False):
        """Read packets from current position, filter, decode.

        Calls self.post_decode(pic) for each frame.
        """
        self.track.stream.discard = AVDISCARD_NONKEY if keyframes_only else AVDISCARD_NONE
        packets = self.track.fc.read_packets(self.track.stream)
        it = self.bsf.filter(packets, flush=True, reuse=True)
        self.with_callbacks(lambda: self.send(None), post_decode=lambda pic: None)

        for pkt in it:
            pts = pkt.av.pts if pkt.av.pts != AV_NOPTS_VALUE else pkt.av.dts
            arr = np.ctypeslib.as_array(pkt.av.data, (pkt.av.size,))
            self.send((pts, arr))
            if keyframes_only:
                self.bsf.flush()
                self.send(None)
        self.send(None)

    def _map_and_convert(self, pic, dtype):
        """Map an OwnedPicture to a tensor. Frees pic and surface. Returns (timedelta, frame)."""
        stream = extract_stream_ptr(torch.cuda.current_stream(self.device))
        surface = pic.map(stream)  # OwnedSurface acquires surface_sem
        pic.free()                 # releases picture slot sem
        frame = convert(
            surface,
            self.track.color_space(AVColorSpace.BT470BG),
            self.track.color_range(AVColorRange.MPEG),
            dtype,
        )
        t = self.track.pts2time(surface.pts)
        surface.free()             # releases surface sem
        return (t, frame)

    def screenshot(self, target: timedelta, dtype: torch.dtype, accurate: bool = False):
        """Extract a frame at the specified timestamp. Synchronous (no thread).

        Returns (timedelta, frame).
        """
        target_pts = self.track.time2pts(target)
        kts = self._keyframe_before(target_pts)

        # Check if we can serve from _recent without seeking.
        # Only pictures still in _slots are valid (not overwritten by NVDEC).
        def _valid(p):
            return self._slots.get(p.index) is p

        if self._recent:
            kts_pts = kts if kts is not None else 0
            if not accurate:
                # Best frame: from the right keyframe context, at-or-before target
                candidates = [p for p in self._recent if kts_pts <= p.pts <= target_pts and _valid(p)]
                if candidates:
                    return self._map_and_convert(candidates[-1], dtype)
            else:
                before = [p for p in self._recent if p.pts <= target_pts and p.pts >= kts_pts and _valid(p)]
                after = [p for p in self._recent if p.pts > target_pts]
                if before and after:
                    return self._map_and_convert(before[-1], dtype)
        # Only seek if the keyframe is behind our current position
        current = self._recent[-1].pts if self._recent else None
        if kts is not None and (current is None or kts < current):
            self._seek(kts)

        # Decode packet by packet, map+free in callback (synchronous, no deadlock risk).
        result = [None]
        done = [False]

        def _post(pic):
            if done[0] or pic is None:
                return
            if not accurate:
                result[0] = self._map_and_convert(pic, dtype)
                done[0] = True
            elif pic.pts > target_pts:
                if result[0] is None:
                    result[0] = self._map_and_convert(pic, dtype)
                done[0] = True
            else:
                result[0] = self._map_and_convert(pic, dtype)

        self._pump(lambda: done[0], _post)

        if result[0] is None:
            raise NoFrameError(f"No frame found at {target}")
        return result[0]

    def _pump(self, done, post_decode):
        """Read packets and send them one at a time, calling post_decode for each frame.
        Stops when done() returns True or EOF. Flushes on EOF."""
        for pkt in self.bsf.filter(self.track.fc.read_packets(self.track.stream), reuse=True):
            self.with_callbacks(
                lambda pkt=pkt: self.send((pkt.av.pts if pkt.av.pts != AV_NOPTS_VALUE else pkt.av.dts, np.ctypeslib.as_array(pkt.av.data, (pkt.av.size,)))),
                post_decode=post_decode)
            if done():
                return
        self.with_callbacks(lambda: self.send(None), post_decode=post_decode)

    def _get_next_frame(self, pic_q):
        """Send packets until one frame arrives. Puts OwnedPicture on pic_q immediately
        in the callback. Returns the PTS of the emitted frame, or None at EOF.
        """
        result = [None]
        def _post(pic):
            if result[0] is not None or pic is None:
                return
            result[0] = pic.pts
            pic_q.put((OwnedPicture(pic), None))
        self._pump(lambda: result[0] is not None, _post)
        return result[0]

    def _decode_forward_to(self, target_pts, pic_q):
        """Decode forward (no seek) until we pass target_pts. Queue best frame."""
        done = [False]
        def _post(pic):
            if done[0] or pic is None:
                return
            if pic.pts >= target_pts:
                best = max((p for p in self._recent if p.pts <= target_pts),
                           key=lambda p: p.pts, default=pic)
                pic_q.put((OwnedPicture(best), None))
                done[0] = True
        self._pump(lambda: done[0], _post)
        if not done[0] and self._recent:
            pic_q.put((OwnedPicture(self._recent[-1]), None))

    def _keyframe_before(self, pts):
        """Find the keyframe at-or-before pts. Returns its timestamp, or None."""
        entry = FormatContext.index_get_entry_from_timestamp(
            self.track.stream, pts, 1
        )
        if entry is None:
            return None
        return entry.timestamp

    def _keyframe_after(self, pts):
        """Find the first keyframe after pts. Returns its timestamp, or None."""
        entry = FormatContext.index_get_entry_from_timestamp(
            self.track.stream, pts + 1, 0
        )
        if entry is None or entry.timestamp <= pts:
            return None
        return entry.timestamp

    def _screenshots_decode(self, max_interval, start_kts, pic_q):
        """Decode loop for screenshots.

        Loop: get next frame → put on queue → should_seek to decide whether
        to seek to next keyframe or decode forward through fill points.
        """
        max_gap = int(max_interval.total_seconds() / float(self.track._time_base))

        self._seek(start_kts if start_kts is not None else 0)

        while True:
            # Get next frame — puts on queue immediately in callback
            current_pts = self._get_next_frame(pic_q)
            if current_pts is None:
                raise NoFrameError("unexpected EOF during screenshots")

            # Find farthest keyframe within max_gap — skip dense keyframes
            next_kf = self._keyframe_before(current_pts + max_gap)
            if next_kf is not None and next_kf > current_pts:
                self._seek(next_kf)
                continue

            # No keyframe within max_gap — find the next one beyond
            far_kf = self._keyframe_after(current_pts)
            if far_kf is not None:
                target = far_kf
            else:
                if self.track._duration_pts is None:
                    raise NoFrameError("no keyframes in index and unknown duration")
                target = self.track._duration_pts + self.track._start_time
            assert target > current_pts

            # Fill evenly to the target, then seek if there's a keyframe
            gap = target - current_pts
            n = (gap // max_gap) + 1
            for j in range(1, n + (0 if far_kf is not None else 1)):
                self._decode_forward_to(current_pts + gap * j // n, pic_q)
            if far_kf is not None:
                self._seek(far_kf)
            else:
                break

    def _screenshots_thread(self, max_interval, start_kts, pic_q):
        try:
            self._screenshots_decode(max_interval, start_kts, pic_q)
        except Exception as e:
            pic_q.put((e, None))
        finally:
            pic_q.put(None)

    def screenshots(self, dtype: torch.dtype, max_interval: timedelta, start_kts=None):
        """Take screenshots throughout the video with interval <= max_interval.

        Spawns a worker thread. Consumer (main thread) maps+frees pics,
        releasing picture slots for the decoder.

        Yields:
            (timedelta, frame_or_exception, kts_timedelta_or_none)
        """
        pic_q = Queue()
        thread = threading.Thread(
            target=self._screenshots_thread,
            args=(max_interval, start_kts, pic_q),
            daemon=True,
        )
        thread.start()

        try:
            while True:
                item = pic_q.get()
                if item is None:
                    break
                payload, kts_int = item
                kts_td = self.track.pts2time(kts_int) if kts_int is not None else None
                if isinstance(payload, Exception):
                    t = kts_td if kts_td is not None else timedelta(0)
                    yield (t, payload, kts_td)
                else:
                    try:
                        t, frame = self._map_and_convert(payload, dtype)
                        yield (t, frame, kts_td)
                    except Exception as e:
                        payload.free()
                        yield (kts_td or timedelta(0), e, kts_td)
        finally:
            while True:
                try:
                    item = pic_q.get_nowait()
                except:
                    break
                if item is None:
                    break
                if not isinstance(item[0], Exception):
                    item[0].free()
            thread.join(timeout=5)

    # free(), __del__, __enter__, __exit__ inherited from BaseDecoder
