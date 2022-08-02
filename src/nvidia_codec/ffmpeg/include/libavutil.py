from ctypes import *

from ...common import TypedCEnumeration

MKTAG = lambda a, b, c, d: (ord(a)) | (ord(b) << 8) | (ord(c) << 16) | (c_int(ord(d) << 24).value)
FFERRTAG = lambda a, b, c, d: -MKTAG(a, b, c, d)
AVERROR_EOF = FFERRTAG('E', 'O', 'F', ' ')
AVERROR = lambda errnum: -errnum

AV_NOPTS_VALUE= -(2**63)
AV_TIME_BASE = 1000000

class AVRational(Structure):
    _fields_ = [
        ('num', c_int),
        ('den', c_int)
    ]


class AVDictionaryEntry(Structure):
    _fields_ = [
        ('key', c_char_p),
        ('value', c_char_p)
    ]

class AVDictionary(Structure):
    _fields_ = [
        ('count', c_int),
        ('elems', POINTER(AVDictionaryEntry))
    ]

class AVColorSpace(TypedCEnumeration(c_int)):
    RGB         = 0
    BT709       = 1 
    UNSPECIFIED = 2
    RESERVED    = 3 
    FCC         = 4 
    BT470BG     = 5 
    SMPTE170M   = 6 
    SMPTE240M   = 7 
    YCGCO       = 8 
    YCOCG       = YCGCO
    BT2020_NCL  = 9  
    BT2020_CL   = 10 
    SMPTE2085   = 11
    CHROMA_DERIVED_NCL = 12
    CHROMA_DERIVED_CL = 13
    ICTCP       = 14
    # NB = 15

class AVColorRange(TypedCEnumeration(c_int)):
    UNSPECIFIED = 0
    MPEG = 1 # limited range
    JPEG = 2 # full range
    # NB = 3

class AVPixelFormat(TypedCEnumeration(c_int)):
    NONE = -1,
    RGB24 = 0,
    YUV420P = 1,
    YUV444P = 5,
    YUV420P16LE = 45,
    YUV444P16LE = 49
    # other omitted