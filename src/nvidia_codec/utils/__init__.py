"""High-level utilities for video decoding.

This module exports the main user-facing classes:
    - Player: Decode video frames (streaming and screenshot)
    - extract_stream_ptr: Helper for CUDA stream interoperability
"""
from .compat import extract_stream_ptr
from .player import Player
