"""High-level utilities for video decoding.

This module exports the main user-facing classes:
    - parse: Probe video file metadata without GPU
    - VideoTrack: Metadata for a single video track
    - TrackPlayer: GPU-accelerated decoder for a single track
    - Player: Convenience — parse + best track + TrackPlayer
    - extract_stream_ptr: Helper for CUDA stream interoperability
"""
from .compat import extract_stream_ptr
from .player import parse, VideoTrack, TrackPlayer, Player
