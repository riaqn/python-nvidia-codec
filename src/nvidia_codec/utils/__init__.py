"""High-level utilities for video decoding.

This module exports the main user-facing classes:
    - parse: Probe video file metadata without GPU (returns ParseResult)
    - VideoTrack: Metadata for a single video track
    - AudioTrack: Metadata for a single audio track
    - VideoTrackPlayer: GPU-accelerated decoder for a single video track
    - Player: Convenience — parse + best track + VideoTrackPlayer
    - extract_stream_ptr: Helper for CUDA stream interoperability
"""
from .compat import extract_stream_ptr
from .player import parse, VideoTrack, AudioTrack, VideoTrackPlayer, Player
