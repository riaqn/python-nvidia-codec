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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
