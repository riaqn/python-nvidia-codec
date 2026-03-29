"""High-level utilities for video decoding.

    - parse, VideoTrack, AudioTrack: Probe metadata without GPU
    - Decoder: GPU-accelerated decoder for a single video track
    - Player: Convenience — parse + best track + Decoder
"""
from .compat import extract_stream_ptr
from .demux import parse, VideoTrack, AudioTrack
from .decode import Decoder
from .player import Player
