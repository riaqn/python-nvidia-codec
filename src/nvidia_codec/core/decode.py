"""Low-level NVDEC decoder — thin wrapper around CUVID API.

This module provides the core decoding interface. It does NOT manage
threading or slot ownership; that is the caller's responsibility
(see nvidia_codec.utils.player for the higher-level interface).

Callbacks:
    pre_decode(pic_idx): Called before cuvidDecodePicture. The caller
        should block here if the picture slot is in use.
    post_decode(pic): Called when a decoded frame is ready for display
        (the "display callback"). pic is a Picture or None (EOS).
"""

from . import cuda
from .nvcuvid import *
from .common import *
from .. import CodecNotSupportedError
import numpy as np
from ctypes import *
import weakref

import logging

log = logging.getLogger(__name__)

nvcuvid = cdll.LoadLibrary("libnvcuvid.so")


class Surface:
    """Post-processed decoded frame accessible as a CUDA array.

    Do not instantiate directly; use Picture.map() instead.
    """

    def __init__(self, decoder, index, params, stream, pts):
        self.decoder = decoder
        self.pts = pts

        self.params = CUVIDPROCPARAMS()
        memmove(byref(self.params), byref(params), sizeof(CUVIDPROCPARAMS))

        self.params.output_stream = stream

        self.c_devptr = c_ulonglong()
        self.c_pitch = c_uint()

        with cuda.Device(self.decoder.device):
            cuda.check(
                nvcuvid.cuvidMapVideoFrame64(
                    self.decoder.cuvid_decoder,
                    c_int(index),
                    byref(self.c_devptr),
                    byref(self.c_pitch),
                    byref(self.params),
                )
            )

    @property
    def format(self) -> cudaVideoSurfaceFormat:
        return self.decoder.surface_format

    @property
    def height(self):
        return self.decoder.target_height

    @property
    def width(self):
        return self.decoder.target_width

    @property
    def size(self):
        return {"width": self.width, "height": self.height}

    def free(self):
        if self.c_devptr and self.decoder.cuvid_decoder:
            with cuda.Device(self.decoder.device):
                cuda.check(
                    nvcuvid.cuvidUnmapVideoFrame64(
                        self.decoder.cuvid_decoder, self.c_devptr
                    )
                )
                self.c_devptr = c_ulonglong()

    def __del__(self):
        self.free()

    @property
    def shape(self):
        return self.__cuda_array_interface__["shape"]

    @property
    def __cuda_array_interface__(self):
        format = self.format
        if format == cudaVideoSurfaceFormat.NV12:
            assert self.height % 2 == 0
            shape = (self.height // 2 * 3, self.width)
            typestr = "|u1"
            strides = (self.c_pitch.value, 1)
        elif format == cudaVideoSurfaceFormat.P016:
            assert self.height % 2 == 0
            shape = (self.height // 2 * 3, self.width)
            typestr = "<u2"
            strides = (self.c_pitch.value, 2)
        elif format == cudaVideoSurfaceFormat.YUV444:
            shape = (3, self.height, self.width)
            typestr = "|u1"
            strides = (self.c_pitch.value * self.height, self.c_pitch.value, 1)
        elif format == cudaVideoSurfaceFormat.YUV444_16Bit:
            shape = (3, self.height, self.width)
            typestr = "<u2"
            strides = (self.c_pitch.value * self.height, self.c_pitch.value, 2)
        else:
            raise ValueError(f"unsupported format {format}")
        return {
            "shape": shape,
            "typestr": typestr,
            "strides": strides,
            "version": 3,
            "data": (self.c_devptr.value, False),
            "stream": self.params.output_stream,
        }


class Picture:
    """Decoded frame before post-processing.

    A pure data object — index, proc params, pts. Deliberately holds NO
    reference to the decoder: the decoder tracks live Pictures in
    _slots/_recent, and a back-reference here would form a cycle that
    keeps the decoder (and its large NVDEC context) alive past the
    refcount-zero point, deferring cleanup to the cyclic GC. To turn a
    Picture into a Surface use Decoder.map(pic) (the decoder is always
    in scope where mapping happens).

    Slot ownership is managed by the caller via pre_decode/pic_release
    callbacks.
    """

    def __init__(self, index, proc_params, pts):
        self.index = index
        self.params = proc_params
        self.pts = pts

    def free(self):
        """No-op. Slot ownership is managed by the caller (player.py)."""
        pass


def decide_surface_format(
    chroma_format, bit_depth, supported_surface_formats, allow_high=False
):
    """Select the best surface format for the given video parameters."""
    if chroma_format in [
        cudaVideoChromaFormat.YUV420,
        cudaVideoChromaFormat.MONOCHROME,
    ]:
        f = (
            cudaVideoSurfaceFormat.P016
            if bit_depth > 8 and allow_high
            else cudaVideoSurfaceFormat.NV12
        )
    elif chroma_format == cudaVideoChromaFormat.YUV444:
        f = (
            cudaVideoSurfaceFormat.YUV444_16Bit
            if bit_depth > 8 and allow_high
            else cudaVideoSurfaceFormat.YUV444
        )
    elif chroma_format == cudaVideoChromaFormat.YUV422:
        f = cudaVideoSurfaceFormat.NV12
    else:
        raise Exception(f"unexpected chroma format {chroma_format}")

    if f not in supported_surface_formats:
        if cudaVideoSurfaceFormat.NV12 in supported_surface_formats:
            f = cudaVideoSurfaceFormat.NV12
        elif cudaVideoSurfaceFormat.P016 in supported_surface_formats and allow_high:
            f = cudaVideoSurfaceFormat.P016
        elif cudaVideoSurfaceFormat.YUV444 in supported_surface_formats:
            f = cudaVideoSurfaceFormat.YUV444
        elif (
            cudaVideoSurfaceFormat.YUV444_16Bit in supported_surface_formats
            and allow_high
        ):
            f = cudaVideoSurfaceFormat.YUV444_16Bit
        else:
            raise Exception("No supported surface format")

    return f


class BaseDecoder:
    """NVDEC hardware decoder — thin wrapper around CUVID API.

    Threading and slot ownership are NOT managed here. The caller provides
    callbacks (pre_decode, pic_release, surface_acquire, surface_release)
    to handle synchronization.
    """

    def catch_exception(self, func, return_on_error=0):
        # `func` is a bound method of self, and the wrapper records errors
        # on self.exception — both strong refs to self. The wrapper is
        # wrapped in a CFUNCTYPE and stored on self (self.handleXxxCallback),
        # so capturing self strongly here forms a self-cycle that makes the
        # decoder uncollectable by refcounting (only the cyclic GC can take
        # it — which under streaming load lags and balloons VRAM). Hold
        # self weakly instead; the callback only ever fires during a
        # send()/cuvidParseVideoData while a caller keeps the decoder
        # alive, so the weakref always resolves then. The None branch is
        # just belt-and-suspenders for a callback racing finalization.
        selfref = weakref.ref(self)
        fn = func.__func__
        def wrapper(*args, **kwargs):
            s = selfref()
            if s is None:
                return return_on_error
            try:
                return fn(s, *args, **kwargs)
            except BaseException as e:
                log.debug(f"callback exception logged {e}")
                s.exception = e
                log.debug(f"returning error code {return_on_error}")
                return return_on_error

        return wrapper

    def handleVideoSequence(self, pUserData, pVideoFormat):
        log.debug("sequence")
        vf = pVideoFormat.contents

        if self.cuvid_decoder:
            essential = [
                "codec",
                "coded_width",
                "coded_height",
                "chroma_format",
                "bit_depth_luma_minus8",
            ]
            mismatches = [
                (f, getattr(self.video_format, f), getattr(vf, f))
                for f in essential
                if getattr(self.video_format, f) != getattr(vf, f)
            ]
            if not mismatches:
                return self.decode_create_info.ulNumDecodeSurfaces
            else:
                diff = ", ".join(f"{k}: {old} -> {new}" for k, old, new in mismatches)
                raise Exception(
                    f"decoder already initialized with different format: {diff}"
                )
        memmove(byref(self.video_format), byref(vf), sizeof(CUVIDEOFORMAT))

        caps = CUVIDDECODECAPS(
            eCodecType=vf.codec,
            eChromaFormat=vf.chroma_format,
            nBitDepthMinus8=vf.bit_depth_luma_minus8,
        )

        with cuda.Device(self.device):
            cuda.check(nvcuvid.cuvidGetDecoderCaps(byref(caps)))
        if caps.bIsSupported != 1:
            raise CodecNotSupportedError(
                f"Codec not supported: {cudaVideoCodec(vf.codec)}"
            )
        assert vf.coded_width <= caps.nMaxWidth, "width too large"
        assert vf.coded_height <= caps.nMaxHeight, "height too large"
        assert vf.coded_width >= caps.nMinWidth, "width too small"
        assert vf.coded_height >= caps.nMinHeight, "height too small"
        assert (vf.coded_width >> 4) * (
            vf.coded_height >> 4
        ) <= caps.nMaxMBCount, "too many macroblocks"

        supported_surface_formats = []
        for surface_format in range(4):
            if caps.nOutputFormatMask & (1 << surface_format):
                supported_surface_formats.append(cudaVideoSurfaceFormat(surface_format))

        p = {
            "chroma_format": vf.chroma_format,
            "bit_depth": vf.bit_depth_luma_minus8 + 8,
            "supported_surface_formats": supported_surface_formats,
            "min_num_pictures": vf.min_num_decode_surfaces,
        }

        decision = {
            "num_pictures": p["min_num_pictures"] + 1,
            "num_surfaces": 1 + 1,
            "surface_format": decide_surface_format(
                p["chroma_format"], p["bit_depth"], p["supported_surface_formats"]
            ),
            "cropping": lambda height, width: {
                "left": 0,
                "right": width,
                "top": 0,
                "bottom": height,
            },
            "target_size": lambda cropped_height, cropped_width: (
                cropped_height,
                cropped_width,
            ),
            "target_rect": lambda target_height, target_width: {
                "left": 0,
                "right": target_width,
                "top": 0,
                "bottom": target_height,
            },
        }

        decision |= self.decide(p)
        c = decision["cropping"](self.height, self.width)
        target_height, target_width = decision["target_size"](
            c["bottom"] - c["top"], c["right"] - c["left"]
        )
        tr = decision["target_rect"](target_height, target_width)

        assert (
            decision["num_pictures"] <= 32
        ), f"number of pictures {decision['num_pictures']} > 32 max"
        assert (
            decision["surface_format"] in supported_surface_formats
        ), f"surface format {decision['surface_format']} not supported for codec {caps.eCodecType} chroma {caps.eChromaFormat} depth {caps.nBitDepthMinus8 + 8}"
        assert decision["num_surfaces"] >= 0, "number of surfaces must be non-negative"
        da = vf.display_area
        display_area = SRECT(
            left=da.left + c["left"],
            top=da.top + c["top"],
            right=da.left + c["right"],
            bottom=da.top + c["bottom"],
        )
        target_rect = SRECT(
            left=tr["left"], top=tr["top"], right=tr["right"], bottom=tr["bottom"]
        )

        self.decode_create_info = CUVIDDECODECREATEINFO(
            ulWidth=vf.coded_width,
            ulHeight=vf.coded_height,
            ulNumDecodeSurfaces=decision["num_pictures"],
            CodecType=vf.codec,
            ChromaFormat=vf.chroma_format,
            ulCreationFlags=cudaVideoCreateFlags.PreferCUVID,
            bitDepthMinus8=vf.bit_depth_luma_minus8,
            ulIntraDecodeOnly=0,
            ulMaxWidth=vf.coded_width,
            ulMaxHeight=vf.coded_height,
            display_area=display_area,
            OutputFormat=decision["surface_format"],
            DeinterlaceMode=(
                cudaVideoDeinterlaceMode.Weave
                if vf.progressive_sequence
                else cudaVideoDeinterlaceMode.Adaptive
            ),
            ulTargetWidth=target_width,
            ulTargetHeight=target_height,
            ulNumOutputSurfaces=decision["num_surfaces"],
            vidLock=None,
            target_rect=target_rect,
            enableHistogram=0,
        )

        with cuda.Device(self.device):
            cuda.check(
                nvcuvid.cuvidCreateDecoder(
                    byref(self.cuvid_decoder), byref(self.decode_create_info)
                )
            )

        log.debug("sequence successful")
        return decision["num_pictures"]

    @property
    def codec(self):
        return self.decode_create_info.CodecType

    @property
    def height(self):
        return (
            self.video_format.display_area.bottom - self.video_format.display_area.top
        )

    @property
    def width(self):
        return (
            self.video_format.display_area.right - self.video_format.display_area.left
        )

    @property
    def target_width(self):
        return self.decode_create_info.ulTargetWidth

    @property
    def target_height(self):
        return self.decode_create_info.ulTargetHeight

    @property
    def surface_format(self):
        return self.decode_create_info.OutputFormat

    def pre_decode(self, idx):
        """NVDEC is reclaiming picture slot idx. After this returns,
        slot ownership is transferred back to NVDEC."""
        raise NotImplementedError

    def post_decode(self, pic):
        """NVDEC is releasing picture slot for display.
        pic is a Picture or None (EOS)."""
        raise NotImplementedError

    def handlePictureDecode(self, pUserData, pPicParams):
        log.debug("decode")
        pp = pPicParams.contents
        log.debug(f"decode picture index: {pp.CurrPicIdx}")

        self.pre_decode(pp.CurrPicIdx)

        with cuda.Device(self.device):
            cuda.check(nvcuvid.cuvidDecodePicture(self.cuvid_decoder, byref(pp)))

        return 1

    def handlePictureDisplay(self, pUserData, pDispInfo):
        log.debug("display")
        if not bool(pDispInfo):
            self.post_decode(None)
            return 1
        di = pDispInfo.contents

        params = CUVIDPROCPARAMS(
            progressive_frame=di.progressive_frame,
            second_field=di.repeat_first_field + 1,
            top_field_first=di.top_field_first,
            unpaired_field=di.repeat_first_field < 0,
        )

        picture = Picture(di.picture_index, params, di.timestamp)
        self.post_decode(picture)
        return 1

    def map(self, pic, stream: int = 0):
        """Map a decoded Picture to a Surface (post-processed frame).
        Lives on the decoder (not Picture.map) so Picture needn't hold a
        decoder back-reference — see Picture's docstring."""
        return Surface(self, pic.index, pic.params, stream, pic.pts)

    def handleOperatingPoint(self, pUserData, pOPInfo):
        opi = pOPInfo.contents
        if opi.codec == cudaVideoCodec.AV1:
            if opi.av1.operating_points_cnt > 1:
                if self.operating_point >= opi.av1.operating_points_cnt:
                    self.operating_point = 0
                return self.operating_point | (1 << 10 if self.disp_all_layers else 0)
        return -1

    def __init__(
        self,
        codec: cudaVideoCodec,
        decide=lambda p: {},
        device=None,
        extradata=None,
        coded_width=0,
        coded_height=0,
    ):
        self.dirty = False
        self.decide = decide
        self.exception = None
        self.cuvid_parser = CUvideoparser()
        self.cuvid_decoder = CUvideodecoder()


        self.operating_point = 0
        self.disp_all_layers = False

        self.video_format = CUVIDEOFORMAT()

        self.handleVideoSequenceCallback = PFNVIDSEQUENCECALLBACK(
            self.catch_exception(self.handleVideoSequence)
        )
        self.handlePictureDecodeCallback = PFNVIDDECODECALLBACK(
            self.catch_exception(self.handlePictureDecode)
        )
        self.handlePictureDisplayCallback = PFNVIDDISPLAYCALLBACK(
            self.catch_exception(self.handlePictureDisplay)
        )
        self.handleOperatingPointCallback = PFNVIDOPPOINTCALLBACK(
            self.catch_exception(self.handleOperatingPoint, -1)
        )

        self.ext_video_info = None
        pExtVideoInfo = None
        if extradata is not None and len(extradata) > 0:
            self.ext_video_info = CUVIDEOFORMATEX()
            self.ext_video_info.format.codec = codec
            self.ext_video_info.format.coded_width = coded_width
            self.ext_video_info.format.coded_height = coded_height
            self.ext_video_info.format.chroma_format = 1
            copy_len = min(len(extradata), 1024)
            memmove(self.ext_video_info.raw_seqhdr_data, extradata, copy_len)
            self.ext_video_info.format.seqhdr_data_length = copy_len
            pExtVideoInfo = pointer(self.ext_video_info)
            log.debug(
                f"Passing {copy_len} bytes of extradata to parser, size={coded_width}x{coded_height}"
            )

        p = CUVIDPARSERPARAMS(
            CodecType=codec,
            ulMaxNumDecodeSurfaces=0,
            ulErrorThreshold=0,
            ulMaxDisplayDelay=0,
            pUserData=None,
            pfnSequenceCallback=self.handleVideoSequenceCallback,
            pfnDecodePicture=self.handlePictureDecodeCallback,
            pfnDisplayPicture=self.handlePictureDisplayCallback,
            pfnGetOperatingPoint=self.handleOperatingPointCallback,
            pExtVideoInfo=pExtVideoInfo,
        )

        self.device = cuda.get_current_device(device)

        caps = CUVIDDECODECAPS(
            eCodecType=codec, eChromaFormat=1, nBitDepthMinus8=0
        )
        with cuda.Device(self.device):
            cuda.check(nvcuvid.cuvidGetDecoderCaps(byref(caps)))
        if caps.bIsSupported != 1:
            raise CodecNotSupportedError(
                f"Codec not supported: {cudaVideoCodec(codec)}"
            )

        cuda.check(
            nvcuvid.cuvidCreateVideoParser(byref(self.cuvid_parser), byref(p))
        )

    def send(self, packet):
        """Send a compressed packet to the decoder.

        Calls self.pre_decode(idx) before each decode and
        self.post_decode(pic) for each displayed frame.

        Args:
            packet: (pts, data) tuple or None for end-of-stream.
        """
        if packet is None:
            p = CUVIDSOURCEDATAPACKET(
                flags=(
                    CUvideopacketflags.ENDOFSTREAM | CUvideopacketflags.NOTIFY_EOS
                ).value,
                payload_size=0,
                payload=None,
                timestamp=0,
            )
        else:
            pts, data = packet
            p = CUVIDSOURCEDATAPACKET(
                flags=CUvideopacketflags.TIMESTAMP.value,
                payload_size=data.shape[0],
                payload=data.ctypes.data_as(POINTER(c_uint8)),
                timestamp=pts,
            )
        # self.exception relays an error out of a ctypes callback (which
        # can't propagate through the C parser). Clear it as we consume
        # it: a retained exception keeps its __traceback__ alive, whose
        # frames reference self, pinning the decoder. Raise it WITHOUT
        # binding it to a local here — a `e = self.exception; raise e`
        # would put `e` in this frame, and the raised exception's
        # traceback includes this same frame → e ↔ frame cycle, defeating
        # the point. `try: raise self.exception finally: clear` keeps the
        # exception out of this frame's locals.
        self.exception = None
        try:
            with cuda.Device(self.device):
                cuda.check(nvcuvid.cuvidParseVideoData(self.cuvid_parser, byref(p)))
        except cuda.CUError:
            if self.exception is None:
                raise
            # else: a callback captured a more specific exception — it's
            # raised (and cleared) below in preference to the CUError.
        if self.exception is not None:
            try:
                raise self.exception
            finally:
                self.exception = None

    def free(self):
        # __del__ can call this during interpreter shutdown, by which
        # point module globals (nvcuvid, cuda, the ctypes types) may be
        # torn down to None — calling through them then raises
        # "NoneType is not callable". The OS reclaims the GPU context on
        # process exit regardless, so just bail if our entry points are
        # gone. (Now reachable because the decoder is refcount-collectable
        # — its self-reference cycles were removed — so __del__ actually
        # runs at shutdown instead of the object leaking.)
        if nvcuvid is None or cuda is None:
            return
        if getattr(self, 'cuvid_parser', None):
            cuda.check(nvcuvid.cuvidDestroyVideoParser(self.cuvid_parser))
            self.cuvid_parser = CUvideoparser()
        if getattr(self, 'cuvid_decoder', None):
            with cuda.Device(self.device):
                cuda.check(nvcuvid.cuvidDestroyDecoder(self.cuvid_decoder))
                self.cuvid_decoder = CUvideodecoder()

    def __del__(self):
        try:
            self.free()
        except Exception:
            pass   # best-effort during finalization / shutdown

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.free()
