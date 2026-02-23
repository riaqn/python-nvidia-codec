"""High-level utilities for video decoding.

This module exports the main user-facing classes:
    - Player: Stream all frames from a video file
    - Screenshoter: Extract individual frames at specific timestamps
    - extract_stream_ptr: Helper for CUDA stream interoperability
"""
from .compat import extract_stream_ptr
from .player import Screenshoter, Player