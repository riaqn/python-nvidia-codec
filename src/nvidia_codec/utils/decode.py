"""GPU-accelerated video decoding using NVIDIA NVDEC."""

import collections
from datetime import timedelta
import numpy as np
from queue import Queue, ShutDown
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
from ..ffmpeg.common import AVException
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

    @property
    def __cuda_array_interface__(self):
        return self._surface.__cuda_array_interface__

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
        num_pictures: Callable(min_pics) -> int, returns the total number of
            picture slots (must be >= min_pics and <= 32). Higher values allow
            better pipelining between decode and consumer threads.
            Default: lambda min_pics: min_pics * 2 (double the minimum).
    """

    def __init__(
        self,
        track,
        num_pictures=lambda min_pics: min_pics + 4,
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
            num_pics = num_pictures(p["min_num_pictures"])
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

    def _is_valid(self, pic):
        """Check if a picture's slot hasn't been overwritten by NVDEC."""
        return self._slots.get(pic.index) is pic

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

    def _seek_to_keyframe(self, kts):
        """Seek to a keyframe timestamp from the index. Flushes NVDEC parser, resets state."""
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
        """Map an OwnedPicture to a tensor. Frees pic and surface.
        Returns (timedelta, frame) on success, (timedelta, exception) on error."""
        try:
            t = self.track.pts2time(pic.pts)
            stream = extract_stream_ptr(torch.cuda.current_stream(self.device))
            surface = pic.map(stream)
        except Exception as e:
            return (t, e)
        finally:
            pic.free()
        try:
            frame = convert(
                surface,
                self.track.color_space(AVColorSpace.BT470BG),
                self.track.color_range(AVColorRange.MPEG),
                dtype,
            )
            return (t, frame)
        except Exception as e:
            return (t, e)
        finally:
            surface.free()

    def screenshot(self, target: timedelta, dtype: torch.dtype, accurate: bool = False):
        """Extract a frame at the specified timestamp. Synchronous (no thread).

        Returns (timedelta, frame).
        """
        target_pts = self.track.time2pts(target)
        kts = self._keyframe_before(target_pts)

        # Check if we can serve from _recent without seeking.
        if len(self._recent) > 0:
            kts_pts = kts if kts is not None else 0
            best = next((p for p in reversed(self._recent) if kts_pts <= p.pts <= target_pts and self._is_valid(p)), None)
            if best is not None:
                if not accurate:
                    return self._map_and_convert(best, dtype)
                elif any(p.pts > target_pts for p in self._recent):
                    return self._map_and_convert(best, dtype)
        # Only seek if the keyframe is behind our current position
        current = self._recent[-1].pts if len(self._recent) > 0 else None
        if kts is not None and (current is None or kts < current):
            self._seek_to_keyframe(kts)

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

    def _screenshot_keyframe(self, pic_q, kts):
        """Seek to keyframe kts (if not None), then decode until one frame arrives.
        Always puts exactly one item on pic_q: either (OwnedPicture, kts) or ((pts, exception), kts).
        Returns the PTS of the emitted frame, or None if nothing was produced.
        """
        result = [None]
        def _post(pic):
            if result[0] is not None or pic is None:
                return
            result[0] = pic.pts
            pic_q.put((OwnedPicture(pic), kts))
        try:
            if kts is not None:
                self._seek_to_keyframe(kts)
            self._pump(lambda: result[0] is not None, _post)
            if result[0] is None:
                if kts is None:
                    raise NoFrameError("no frame at start of video")
                pic_q.put(((kts, NoFrameError(f"no frame at keyframe {kts}")), kts))
        except Exception as e:
            if isinstance(e, ShutDown):
                raise e from None
            if kts is None:
                raise
            pic_q.put(((kts, e), kts))
        return result[0]

    def _decode_forward_to(self, target_pts, pic_q):
        """Decode forward (no seek) until we pass target_pts. Queue best frame.
        Returns True if target was reached, False if EOF."""
        done = [False]
        def _post(pic):
            if done[0] or pic is None:
                return
            if pic.pts >= target_pts:
                best = next((p for p in reversed(self._recent) if p.pts <= target_pts and self._is_valid(p)), pic)
                pic_q.put((OwnedPicture(best), None))
                done[0] = True
        try:
            self._pump(lambda: done[0], _post)
            if not done[0]:
                best = next((p for p in reversed(self._recent) if self._is_valid(p)), None)
                if best is not None:
                    pic_q.put((OwnedPicture(best), None))
                return False
            return True
        except Exception as e:
            if isinstance(e, ShutDown):
                raise e from None
            pic_q.put(((target_pts, e), None))
            return False

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

        # Trigger index population for containers that load it lazily (e.g. MKV Cues).
        # If there's at most one index entry, seek to force the demuxer to load its index.
        if self._keyframe_after(0) is None:
            self.track.fc.seek_file(self.track.stream, -(2**63), min_ts=-(2**63), max_ts=-(2**63))

        kts = start_kts

        while True:
            current_pts = self._screenshot_keyframe(pic_q, kts)
            if current_pts is None:
                # the keyframe failed (kts is never None here — first frame raises directly)
                current_pts = kts
            else:
                if kts is None:
                    # use the pts as kts so we can continue
                    kts = current_pts

            next_kf = self._keyframe_before(kts + max_gap)
            if next_kf is not None and next_kf > kts + max_gap // 2:
                kts = next_kf
                continue

            far_kf = self._keyframe_after(kts)
            if far_kf is not None and far_kf <= kts + max_gap:
                # Contradicts _keyframe_before returning None — index was built by seeking.
                # Treat as no keyframe.
                far_kf = None
            if far_kf is not None:
                assert far_kf > kts, f"far_kf={far_kf} <= kts={kts}"
                gap = far_kf - kts
                n = (gap // max_gap) + 1
                for j in range(1, n):
                    if not self._decode_forward_to(current_pts + gap * j // n, pic_q):
                        break
                kts = far_kf
            else:
                while True:
                    current_pts += max_gap
                    if not self._decode_forward_to(current_pts, pic_q):
                        break
                break

    def _screenshots_thread(self, max_interval, start_kts, pic_q):
        try:
            self._screenshots_decode(max_interval, start_kts, pic_q)
            pic_q.put(None)
        except ShutDown:
            pass  # consumer no longer interested
        except Exception as e:
            pic_q.put(e)

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
                if isinstance(item, Exception):
                    raise item from None
                payload, kts_int = item
                kts_td = self.track.pts2time(kts_int) if kts_int is not None else None
                if isinstance(payload, OwnedPicture):
                    t, frame_or_err = self._map_and_convert(payload, dtype)
                    yield (t, frame_or_err, kts_td)
                else:
                    err_pts, exc = payload
                    yield (self.track.pts2time(err_pts), exc, kts_td)
        finally:
            pic_q.shutdown()
            try:
                while True:
                    item = pic_q.get()
                    if item is None:
                        break
                    if isinstance(item, Exception):
                        raise item from None
                    payload, _ = item
                    if isinstance(payload, OwnedPicture):
                        payload.free()
            except ShutDown:
                pass
            thread.join(timeout=5)

    # free(), __del__, __enter__, __exit__ inherited from BaseDecoder
