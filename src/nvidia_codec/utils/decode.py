"""GPU-accelerated video decoding using NVIDIA NVDEC."""

import collections
from datetime import timedelta
import numpy as np
from queue import Queue, ShutDown
import threading
import weakref

from ..ffmpeg.libavcodec import BSFContext
from ..ffmpeg.libavformat import FormatContext, AVCodecID
from ..ffmpeg.include.libavutil import (
    AV_NOPTS_VALUE,
    AVColorRange,
    AVColorSpace,
)
from .compat import av2cuda, extract_stream_ptr
from ..core.decode import BaseDecoder, Picture, Surface, decide_surface_format
from ..ffmpeg.common import AVException
from .color import convert, to_yuv420p
from .demux import VideoTrack
from .. import NoFrameError

import torch
import logging


log = logging.getLogger(__name__)




class OwnedSurface:
    """Maps a Picture under the decoder's output-surface semaphore.

    The semaphore (sized to ulNumOutputSurfaces) must be held BEFORE
    cuvidMapVideoFrame64 runs — two concurrent maps with one output
    surface would overwrite each other — so the map happens here, not
    in the caller. free() unmaps and releases.
    """

    def __init__(self, decoder, pic, stream):
        decoder._surface_sem.acquire()
        try:
            self._surface = decoder.map(pic, stream)
        except BaseException:
            decoder._surface_sem.release()
            raise

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
    map() returns an OwnedSurface. Holds the decoder explicitly (Picture
    no longer back-references it — see Picture's docstring); an
    OwnedPicture is transient (a candidate or a queued item), never
    stored in the decoder's _slots/_recent, so this ref forms no cycle.
    """

    def __init__(self, decoder, pic):
        self._decoder = decoder
        self._pic = pic
        decoder._slot_locks[pic.index].acquire()

    @property
    def pts(self):
        return self._pic.pts

    @property
    def index(self):
        return self._pic.index

    def free(self):
        if self._pic is not None:
            self._decoder._slot_locks[self._pic.index].release()
            self._pic = None

    def map(self, stream=0):
        assert self._pic is not None, "Picture already freed"
        return OwnedSurface(self._decoder, self._pic, stream)


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
            Default: lambda min_pics: min_pics + 4.
        allow_high: Keep >8-bit sources at 16 bits (P016 / YUV444_16Bit
            surfaces) instead of letting NVDEC round them to 8-bit NV12 /
            YUV444. convert() handles both; to_yuv420p() is 8-bit only.
    """

    def __init__(
        self,
        track,
        num_pictures=lambda min_pics: min_pics + 4,
        target_size=None,
        cropping=None,
        target_rect=None,
        device=None,
        allow_high=False,
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

        # BaseDecoder stores this as self.decide and calls it from the
        # sequence callback. Capture self weakly: a strong capture here
        # plus that storage is another self-cycle that would keep the
        # decoder uncollectable by refcounting (same reasoning as the
        # ctypes callbacks). decide only runs during decode, while self
        # is alive, so the weakref always resolves.
        selfref = weakref.ref(self)
        def decide(p):
            s = selfref()
            num_pics = num_pictures(p["min_num_pictures"])
            s._slot_locks = [threading.Lock() for _ in range(num_pics)]
            s._surface_sem = threading.Semaphore(1)
            s._recent = collections.deque(maxlen=num_pics)
            d = {
                "num_pictures": num_pics,
                "num_surfaces": 1,
            }
            if allow_high:
                d["surface_format"] = decide_surface_format(
                    p["chroma_format"], p["bit_depth"],
                    p["supported_surface_formats"], allow_high=True)
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
        """Seek to a keyframe by its DTS. Flushes NVDEC parser, resets state.

        `kts` is the keyframe's DECODE timestamp (as returned by
        `_keyframe_before` / `_keyframe_after`). After the seek the
        first packet `_pump` reads will be the keyframe's packet;
        the corresponding decoded Picture's `pts` may differ from
        `kts` by up to `codecpar.video_delay` frames (B-frame reorder).
        """
        self.track.fc.seek_file(self.track.stream, kts, min_ts=kts, max_ts=kts)
        self.with_callbacks(lambda: self.send(None), post_decode=lambda pic: None)
        self._recent.clear()

    def _seek_before_pts(self, target_pts):
        """Seek so the next packet is the keyframe with pts <= target_pts, using the
        demuxer's NATIVE backward seek (avformat_seek_file -> read_seek), then flush
        NVDEC and reset state (same teardown as _seek_to_keyframe).

        Why not _keyframe_before() + _seek_to_keyframe()? That reads the IN-MEMORY
        index_entries, which avformat_find_stream_info seeds with a few near-t0 entries
        at open. For a far target, index_get_entry_from_timestamp() then returns that
        STALE near-start keyframe (not None) -> we seek to ~BOF and decode-walk the whole
        file (e.g. 160s to reach a 2835s offset). The native seek instead triggers the
        demuxer's read_seek: for MKV that LAZILY parses the on-disk Cues that probing left
        unloaded (an O(1) index seek, exactly what `ffmpeg -ss` does); index-less
        containers fall back to avformat's generic byte-bisection. Either way: no full
        packet/frame traversal. Only if avformat cannot seek the container at all do we
        fall back to the in-memory-index path (then BOF)."""
        try:
            self.track.fc.seek_file(self.track.stream, target_pts,
                                    min_ts=-(2**63), max_ts=target_pts)
        except AVException:
            kts = self._keyframe_before(target_pts)
            if kts is None:
                kts = self._keyframe_after(0)
            if kts is not None:
                self.track.fc.seek_file(self.track.stream, kts, min_ts=kts, max_ts=kts)
        self.with_callbacks(lambda: self.send(None), post_decode=lambda pic: None)
        self._recent.clear()

    def _map_and_convert(self, pic, dtype, pixel_format='rgb'):
        """Map a Picture (or OwnedPicture) to a tensor. Frees pic and surface.
        Returns (timedelta, frame) on success, (timedelta, exception) on error.

        pixel_format='rgb' (default) → (3,H,W) RGB via convert() (color matrix).
        pixel_format='yuv420p' → 1D planar I420 uint8 via to_yuv420p() (no color
        math, half the bytes) — for piping straight to an encoder."""
        t = None
        try:
            t = self.track.pts2time(pic.pts)
            stream = extract_stream_ptr(torch.cuda.current_stream(self.device))
            # Both OwnedPicture (owns its slot) and bare Picture map under
            # the output-surface semaphore; .free() on either is safe.
            if isinstance(pic, OwnedPicture):
                surface = pic.map(stream)
            else:
                surface = OwnedSurface(self, pic, stream)
        except Exception as e:
            return (t, e)
        finally:
            pic.free()
        try:
            if pixel_format == 'yuv420p':
                frame = to_yuv420p(surface)
            else:
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

        # Serve from _recent without seeking only when the current contiguous decode run
        # has already reached PAST target — then the largest cached frame ≤ target is the
        # genuine on-screen frame. Requiring a frame > target avoids returning a stale
        # pre-jump frame (and is what previously made far-forward targets decode-walk:
        # the old code trusted _keyframe_before, which returns a stale near-t0 keyframe
        # for far targets before Cues are parsed).
        if len(self._recent) > 0 and any(p.pts > target_pts for p in self._recent):
            best = next((p for p in reversed(self._recent)
                         if p.pts <= target_pts and self._is_valid(p)), None)
            if best is not None:
                return self._map_and_convert(best, dtype)

        # Cache miss → native seek to the keyframe ≤ target, then decode forward. The
        # demuxer's read_seek is O(1) (lazy-parses MKV Cues; byte-bisects otherwise), so
        # this never BOF-walks. See _seek_before_pts.
        self._seek_before_pts(target_pts)

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

    def _pump(self, done, post_decode, out_q=None):
        """Read packets and send them one at a time, calling post_decode for each frame.
        Stops when done() returns True or EOF. Flushes on EOF.

        `out_q` is the output queue this worker produces into. Shutting
        that queue is the cancel signal — the consumer (or the generator's
        teardown) calls out_q.shutdown() when it's no longer interested,
        and the per-packet `is_shutdown` check below stops the worker.
        This is what unwinds a worker stuck in a long pre-window /
        beginning-of-file seek walk, where it decodes packet after packet
        without ever calling out_q.put (so the put-side ShutDown never
        fires). We reuse the queue's own shutdown state rather than a
        separate cancel flag: one primitive, already shared between the
        two threads. Synchronous callers (screenshot) pass no queue."""
        for pkt in self.bsf.filter(self.track.fc.read_packets(self.track.stream), reuse=True):
            if out_q is not None and out_q.is_shutdown:
                raise ShutDown
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
            pic_q.put((OwnedPicture(self, pic), kts))
        try:
            if kts is not None:
                self._seek_to_keyframe(kts)
            self._pump(lambda: result[0] is not None, _post, pic_q)
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
        """Decode forward (no seek) to target_pts; queue one frame, or raise.

        Contract:
        - Success: queue the most recent frame from _recent with pts <= target.
          Returns None.
        - Skip (first post-target frame found, but _recent has no usable frame
          with pts <= target): raises NoFrameError. Nothing queued.
        - EOF: raises EOFError. Nothing queued.
        - Decoder exceptions propagate directly. Nothing queued.
        """
        # outcome: None=not done; True=frame queued; pts int=skip at that pts.
        outcome = [None]
        def _post(pic):
            if outcome[0] is not None or pic is None:
                return
            if pic.pts >= target_pts:
                best = next((p for p in reversed(self._recent) if p.pts <= target_pts and self._is_valid(p)), None)
                if best is None:
                    outcome[0] = pic.pts
                else:
                    pic_q.put((OwnedPicture(self, best), None))
                    outcome[0] = True
        self._pump(lambda: outcome[0] is not None, _post, pic_q)
        if outcome[0] is True:
            return
        if outcome[0] is not None:
            raise NoFrameError(
                f"gap: target {target_pts}, first decoded pts {outcome[0]}"
            )
        raise EOFError(f"EOF at target {target_pts}")

    def _keyframe_before(self, dts):
        """Find the keyframe at-or-before `dts` (in DECODE timestamp).

        Returns the keyframe's DTS, or None.

        DTS, NOT PTS. The container's stream index — and ffmpeg's
        underlying `avformat_seek_file` — operate on DTS (decode order
        is monotonic; display order isn't, with B-frames). For frames
        with B-frame reordering, the keyframe at DTS=K may DISPLAY
        (PTS) somewhat later, by up to `codecpar.video_delay` frames.
        Callers wanting a PTS-based "before" must compensate for this
        offset; this method alone does not.
        """
        entry = FormatContext.index_get_entry_from_timestamp(
            self.track.stream, dts, 1
        )
        if entry is None:
            return None
        return entry.timestamp

    def _keyframe_after(self, dts):
        """Find the first keyframe after `dts` (in DECODE timestamp).

        Returns the keyframe's DTS, or None. Same DTS-vs-PTS caveat as
        `_keyframe_before` — see that docstring.
        """
        entry = FormatContext.index_get_entry_from_timestamp(
            self.track.stream, dts + 1, 0
        )
        if entry is None or entry.timestamp <= dts:
            return None
        return entry.timestamp

    def _screenshots_decode(self, max_interval, start_kts, pic_q):
        """Decode loop for screenshots.

        Loop: get next frame → put on queue → should_seek to decide whether
        to seek to next keyframe or decode forward through fill points.
        """
        max_gap = int(max_interval.total_seconds() / float(self.track._time_base))

        # Force the demuxer to load its full index (e.g. lazy MKV Cues) before the
        # keyframe walk below relies on _keyframe_before/_keyframe_after. The old
        # `if self._keyframe_after(0) is None` guard skipped this whenever probing had
        # seeded a few near-t0 index_entries — leaving the walk on a STALE partial index
        # (far keyframes missing). Seeking unconditionally triggers matroska_parse_cues
        # (full index); it's an O(1) seek and the loop's first _seek_to_keyframe
        # repositions. Index-less containers that can't seek here just proceed.
        try:
            self.track.fc.seek_file(self.track.stream, -(2**63), min_ts=-(2**63), max_ts=-(2**63))
        except AVException:
            pass

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
                    target = current_pts + gap * j // n
                    try:
                        self._decode_forward_to(target, pic_q)
                    except ShutDown:
                        raise
                    except Exception as e:
                        # EOF or skip mid-gap → record frame-level error.
                        # No useful "last frame" substitute here because
                        # these are intermediate gap-fill samples.
                        pic_q.put(((target, e), None))
                        break
                kts = far_kf
            else:
                while True:
                    current_pts += max_gap
                    try:
                        self._decode_forward_to(current_pts, pic_q)
                    except ShutDown:
                        raise
                    except EOFError:
                        # End of file reached during final tail sampling.
                        # Emit the last recent frame as a best-effort last
                        # sample and stop.
                        best = next((p for p in reversed(self._recent) if self._is_valid(p)), None)
                        if best is not None:
                            pic_q.put((OwnedPicture(self, best), None))
                        break
                    except Exception as e:
                        pic_q.put(((current_pts, e), None))
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
                try:
                    item = pic_q.get()
                except ShutDown:
                    break   # someone shut our queue — stop cleanly
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

    def _frames_thread(self, dtype, end_pts, q, start_pts, pixel_format='rgb'):
        # _Stop: reached end_pts — the requested [start, end] window is
        # fully decoded, finish normally. Cancellation is signalled by
        # the consumer shutting `q`; _pump sees q.is_shutdown and raises
        # ShutDown, caught below.
        class _Stop(Exception):
            pass

        # If start_pts is given, the first yielded pic must be the one
        # with the LARGEST pts <= start_pts (the frame "on screen" at
        # `start`). Phase 1: walk forward keeping a running best
        # candidate (held in an OwnedPicture so NVDEC can't reuse its
        # slot). When a pic with pts > start_pts arrives, the candidate
        # is the answer — emit it, then enter phase 2 and emit every
        # subsequent pic normally.
        candidate = [None]   # OwnedPicture or None
        emitting = [start_pts is None]

        def _emit(pic):
            # Capture pic.pts (int track pts) BEFORE _map_and_convert
            # frees the pic — the queued item carries the raw int so
            # callers get bit-exact pts (no float round-trip loss).
            pts_int = pic.pts
            _, frame_or_err = self._map_and_convert(pic, dtype, pixel_format)
            evt = None
            if not isinstance(frame_or_err, Exception):
                evt = torch.cuda.Event()
                evt.record(stream=torch.cuda.current_stream(self.device))
            q.put((pts_int, frame_or_err, evt))   # blocks if queue full

        def _post(pic):
            if pic is None:
                return
            if not emitting[0]:
                if pic.pts <= start_pts:
                    # Better candidate. Free the previous one (releases
                    # its slot back to NVDEC) and own this one.
                    if candidate[0] is not None:
                        candidate[0].free()
                    candidate[0] = OwnedPicture(self, pic)
                    return
                # First pic past start_pts: emit the candidate (or this
                # pic, if no earlier frame existed) and switch to phase 2.
                if candidate[0] is not None:
                    _emit(candidate[0])    # _map_and_convert calls .free()
                    candidate[0] = None
                emitting[0] = True
                # Fall through to emit this pic too.
            if end_pts is not None and pic.pts > end_pts:
                raise _Stop
            _emit(pic)
        try:
            self._pump(lambda: False, _post, q)
            # EOF before any pic crossed start_pts: emit the final
            # candidate if we ever found one.
            if not emitting[0] and candidate[0] is not None:
                _emit(candidate[0])
                candidate[0] = None
        except _Stop:
            pass
        except ShutDown:
            pass
        except Exception as e:
            try:
                q.put(e)
            except ShutDown:
                pass
        finally:
            if candidate[0] is not None:
                candidate[0].free()
                candidate[0] = None
            try:
                q.put(None)
            except ShutDown:
                pass

    def frames(self, dtype: torch.dtype, start: timedelta = None,
               end: timedelta = None, start_pts: int = None,
               end_pts: int = None, queue_size: int = 16,
               pixel_format: str = 'rgb'):
        """Yield (int_pts, frame_or_exception) for every decoded frame in
        display order. Decodes ALL frames (B-frames included).

        POSITION: without `start` / `start_pts` this does NOT rewind. It
        decodes from wherever the demuxer currently is — the beginning on
        a fresh Decoder, but after a `screenshot()`, `screenshots()` or an
        exhausted `frames()` it resumes from there (and yields nothing if
        that was EOF). Pass `start=timedelta(0)` to decode from the start
        of the file regardless of what ran before.

        The yielded `int_pts` is the raw track pts (same units as
        `pic.pts`, including the demuxer's `start_time` offset). Pass
        through `track.pts2time(pts)` if you need a `timedelta`. Yielding
        the integer directly avoids the lossy round-trip through float
        seconds (`pts2time` is float-valued; `time2pts` truncates the
        inverse) that breaks boundary-frame alignment when callers
        re-feed timestamps back to the decoder.

        `start` / `start_pts` (mutually exclusive, optional): the first
        yielded frame is the one with the LARGEST pts ≤ start (the
        frame "on screen" at this point). Pass `start_pts` (int) when
        you have a raw track pts to anchor to — no rounding through
        seconds. `start` (timedelta) is a convenience that calls
        `time2pts` internally.

        `end` / `end_pts` (mutually exclusive, optional): stop right
        after the first frame past this point. Same `int` vs `timedelta`
        choice as `start`.

        Streams via a bounded queue: the producer thread maps + frees each
        picture inline (so the slot is released back to NVDEC immediately),
        then puts the converted CUDA tensor on a queue of size `queue_size`.
        A bounded queue gives backpressure — the producer blocks when the
        consumer is slow, capping VRAM use to roughly `queue_size` frames.
        A CUDA Event is recorded per frame; the consumer must synchronize
        on it before reading the tensor (the generator does this for you).
        """
        if end_pts is None and end is not None:
            end_pts = self.track.time2pts(end)
        if start_pts is None and start is not None:
            start_pts = self.track.time2pts(start)
        if start_pts is not None:
            # Native demuxer seek to the keyframe <= start_pts. Delegating to
            # read_seek (rather than the in-memory index via _keyframe_before)
            # both parses deferred MKV Cues lazily and avoids the stale-near-t0
            # keyframe that made far seeks decode-walk from BOF. See
            # _seek_before_pts for the full rationale.
            self._seek_before_pts(start_pts)

        q = Queue(maxsize=queue_size)
        thread = threading.Thread(
            target=self._frames_thread,
            args=(dtype, end_pts, q, start_pts, pixel_format),
            daemon=True,
        )
        thread.start()
        try:
            while True:
                try:
                    item = q.get()
                except ShutDown:
                    break   # someone shut our queue — stop cleanly
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item from None
                pts_int, frame_or_err, evt = item
                if evt is not None:
                    evt.synchronize()
                yield pts_int, frame_or_err
        finally:
            # The consumer is done (exhausted / break / close / error).
            # Shutting q is the cancel signal: a producer blocked in
            # q.put wakes with ShutDown, and a producer mid seek-walk
            # sees q.is_shutdown on its next _pump packet and unwinds.
            # Then join. The timeout is a backstop only — a producer
            # truly stuck inside a single libnvcuvid C call can't be
            # interrupted, but the seek walk never reaches that state.
            try:
                q.shutdown()
            except Exception:
                pass
            thread.join(timeout=5)

    # free(), __del__, __enter__, __exit__ inherited from BaseDecoder.
    # NOTE: do NOT call free() explicitly while a frames()/screenshots()
    # producer may be running — cuvidDestroyDecoder would race the
    # producer's in-flight cuvidDecodePicture and SEGV inside libnvcuvid.
    # The safe path is to stop the worker first (shut its queue — which
    # the generator's own teardown does on close/drop) and then drop all
    # refs; the producer holds a ref to this Decoder via its bound
    # method, so __del__ (and thus free()) only runs once the producer
    # has exited and the last external ref is gone.
