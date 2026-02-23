"""Low-level NVIDIA CUVID bindings and decoder implementation.

This package provides direct bindings to NVIDIA's CUVID API for hardware
video decoding. Most users should use nvidia_codec.utils instead.

Modules:
    decode: BaseDecoder, Picture, Surface classes
    cuda: CUDA device and error handling utilities
    cuviddec: ctypes definitions for CUVID decoder structures
    nvcuvid: ctypes definitions for CUVID parser structures
"""
