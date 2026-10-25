"""Smoke test: synthesize a short H.264 clip with ffmpeg, decode it with NVDEC.

Requires an NVIDIA GPU, libnvcuvid, FFmpeg 9 libraries and the `ffmpeg` CLI.
Run with:  python -m pytest tests/  (or simply: python tests/test_smoke.py)
"""
from datetime import timedelta
import os
import shutil
import subprocess
import tempfile

import pytest
import torch

from nvidia_codec.core.cuviddec import cudaVideoSurfaceFormat
from nvidia_codec.utils import Player, parse, VideoTrack

W, H, FPS, SECONDS = 320, 240, 25, 2

pytestmark = pytest.mark.skipif(
    not (shutil.which("ffmpeg") and torch.cuda.is_available()),
    reason="needs ffmpeg CLI and a CUDA GPU",
)


@pytest.fixture(scope="module")
def clip():
    d = tempfile.mkdtemp(prefix="nvc-smoke-")
    path = os.path.join(d, "clip.mp4")
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
         f"-i", f"testsrc=size={W}x{H}:rate={FPS}:duration={SECONDS}",
         "-pix_fmt", "yuv420p", "-c:v", "libx264", "-g", "12", path],
        check=True, timeout=60,
    )
    yield path
    shutil.rmtree(d, ignore_errors=True)


def test_parse(clip):
    tracks = [t for t in parse(clip) if isinstance(t, VideoTrack)]
    assert len(tracks) == 1
    t = tracks[0]
    assert (t.width, t.height) == (W, H)
    assert t.mime_codec.startswith("avc1.")
    assert abs(t.duration.total_seconds() - SECONDS) < 0.2


def test_frames_rgb(clip):
    with Player(clip) as p:
        n = 0
        last = -1
        for pts, frame in p.frames(torch.uint8):
            assert not isinstance(frame, Exception), frame
            assert frame.shape == (3, H, W) and frame.dtype == torch.uint8
            assert frame.is_cuda
            assert pts > last
            last = pts
            n += 1
    assert n == FPS * SECONDS


def test_frames_yuv420p_window(clip):
    with Player(clip) as p:
        out = list(p.frames(torch.uint8, start=timedelta(seconds=0.5),
                            end=timedelta(seconds=1.0), pixel_format="yuv420p"))
    assert 0 < len(out) <= FPS // 2 + 2
    for _, frame in out:
        assert frame.shape == (H * W * 3 // 2,)


def test_screenshot(clip):
    with Player(clip, target_size=lambda h, w: (h // 2, w // 2)) as p:
        t, frame = p.screenshot(timedelta(seconds=1), torch.float32, accurate=True)
        assert frame.shape == (3, H // 2, W // 2)
        assert abs(t.total_seconds() - 1.0) < 1 / FPS + 1e-3
        assert 0.0 <= frame.min() and frame.max() <= 1.0


def test_screenshots(clip):
    with Player(clip) as p:
        shots = [s for s in p.screenshots(torch.uint8, timedelta(seconds=0.5))]
    assert len(shots) >= 3
    for _, frame, _ in shots:
        assert not isinstance(frame, Exception), frame
        assert frame.shape == (3, H, W)


def test_duration_property(clip):
    with Player(clip) as p:
        assert abs(p.duration.total_seconds() - SECONDS) < 0.2


def test_cover_art_skipped(clip):
    """A PNG attached_pic stream must not be selected or count as ambiguity."""
    d = tempfile.mkdtemp(prefix="nvc-smoke-cover-")
    try:
        png = os.path.join(d, "cover.png")
        out = os.path.join(d, "cover.mp4")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
                        "-i", "color=c=red:size=64x64:duration=1", "-frames:v", "1", png],
                       check=True, timeout=60)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", clip, "-i", png,
                        "-map", "0", "-map", "1", "-c", "copy",
                        "-disposition:v:1", "attached_pic", out], check=True, timeout=60)
        vids = [t for t in parse(out) if isinstance(t, VideoTrack)]
        assert [t.attached_pic for t in vids] == [False, True]
        with Player(out) as p:
            assert p.track.index == 0
            n = sum(1 for _ in p.frames(torch.uint8))
        assert n == FPS * SECONDS
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_mpegts_start_time_and_first_gop(clip):
    """MPEG-TS leaves start_time unset until probed. Probing must not swallow
    the IDR (which used to drop the first GOP) and start_time must be real."""
    d = tempfile.mkdtemp(prefix="nvc-smoke-ts-")
    try:
        out = os.path.join(d, "clip.ts")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", clip, "-c", "copy", out],
                       check=True, timeout=60)
        with Player(out) as p:
            assert p.track._start_time > 0           # ffmpeg's TS muxer starts at 1.4 s
            frames = list(p.frames(torch.uint8))
            assert len(frames) == FPS * SECONDS
            first_t = p.track.pts2time(frames[0][0])
            assert abs(first_t.total_seconds()) < 1 / FPS
            # TS seeks by dts with no keyframe awareness; the decoder must still
            # produce the frame actually on screen at 1.0 s, not the next IDR.
            t, _ = p.screenshot(timedelta(seconds=1), torch.uint8, accurate=True)
            assert abs(t.total_seconds() - 1.0) < 1 / FPS + 1e-3
            pts0, _ = next(iter(p.frames(torch.uint8, start=timedelta(seconds=1))))
            assert abs(p.track.pts2time(pts0).total_seconds() - 1.0) < 1 / FPS + 1e-3
            # DTS-vs-PTS window: with B-frames the GOP-2 keyframe (pts 0.96 s) has
            # dts ~0.88 s. A target in between must still yield the previous GOP's
            # frame, not the keyframe (which displays after the target).
            for target in (0.90, 0.94):
                expect = int(target * FPS + 1e-6) / FPS
                t, _ = p.screenshot(timedelta(seconds=target), torch.uint8, accurate=True)
                assert abs(t.total_seconds() - expect) < 1e-3, (target, t)
                pts0, _ = next(iter(p.frames(torch.uint8, start=timedelta(seconds=target))))
                assert abs(p.track.pts2time(pts0).total_seconds() - expect) < 1e-3, (target, pts0)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_time2pts_exact_on_frame_boundaries(clip):
    """time2pts must not floor a target that is exactly on a frame boundary to
    the previous frame (float 1.16 * 12800 = 14847.999...)."""
    with Player(clip) as p:
        tb = p.track._time_base
        for i in range(FPS * SECONDS):
            td = timedelta(seconds=i / FPS)
            expect = int(round(float(td.total_seconds() / tb))) + p.track._start_time
            assert p.track.time2pts(td) == expect, i
        # between two frames the pts lands strictly between their pts values
        a, b = p.track.time2pts(timedelta(seconds=1.16)), p.track.time2pts(timedelta(seconds=1.20))
        assert a < p.track.time2pts(timedelta(seconds=1.17)) < b


def test_pixel_format_enum_values():
    from nvidia_codec.ffmpeg.include.libavutil import AVPixelFormat
    assert AVPixelFormat.YUV420P == 0 and AVPixelFormat.RGB24 == 2
    assert isinstance(AVPixelFormat.YUV444P, int)


def _has_encoder(name):
    out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                         capture_output=True, text=True, timeout=30).stdout
    return f" {name} " in out


@pytest.fixture(scope="module")
def clip10():
    if not _has_encoder("libx265"):
        pytest.skip("ffmpeg lacks libx265")
    d = tempfile.mkdtemp(prefix="nvc-smoke10-")
    path = os.path.join(d, "clip10.mp4")
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
         "-i", f"testsrc=size={W}x{H}:rate={FPS}:duration=1",
         "-pix_fmt", "yuv420p10le", "-c:v", "libx265",
         "-x265-params", "log-level=none", "-g", "12", path],
        check=True, timeout=120,
    )
    yield path
    shutil.rmtree(d, ignore_errors=True)


def test_10bit_allow_high_matches_8bit(clip10):
    """P016 (allow_high) and NV12 paths must agree on the same 10-bit source."""
    with Player(clip10) as p:
        _, lo = p.screenshot(timedelta(seconds=0.5), torch.float32, accurate=True)
        assert p.surface_format == cudaVideoSurfaceFormat.NV12
    with Player(clip10, allow_high=True) as p:
        _, hi = p.screenshot(timedelta(seconds=0.5), torch.float32, accurate=True)
        assert p.surface_format == cudaVideoSurfaceFormat.P016
        assert hi.shape == (3, H, W)
        assert 0.0 <= hi.min() and hi.max() <= 1.0
        # yuv420p export is 8-bit only: the error is reported in-band
        _, r = next(iter(p.frames(torch.uint8, start=timedelta(0),
                                  pixel_format="yuv420p")))
        assert isinstance(r, ValueError)
    # 10-bit rounds to 8-bit at most 1/255 off, plus matrix rounding
    assert (hi - lo).abs().max() < 0.02


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
