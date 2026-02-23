class CodecNotSupportedError(Exception):
    """Raised when the codec is not supported by the GPU's NVDEC."""
    pass

class NoFrameError(Exception):
    """Raised when no frame could be extracted from the video."""
    pass
