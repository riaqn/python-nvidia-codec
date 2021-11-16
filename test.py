import pycuda.driver as cuda
from .decoder import Decoder
import logging
import av
from .pyav import StreamTranslate
import faulthandler
import time
import tqdm
import numpy as np
from PIL import Image

faulthandler.enable()

log = logging.getLogger(__name__)

logging.basicConfig(level=logging.WARNING)

def test(path):
    ctx = cuda.Device(0).retain_primary_context()
    ctx.push()
    custr = cuda.Stream()
    ctx.pop()

    idx = 0

    def display_callback(picture):
        nonlocal idx
        idx += 1
        pts = picture.timestamp
        log.info(f"display_callback: {pts}")
        
        if idx % 60 > 0: # only save for every 60 frames
            return

        surface = picture.map(custr)
        
        # log.warning(time.time())
        #custr.synchronize()
        # log.warning(time.time())
        y, uv = surface.planes()
        # print(y)
        arr_y = cuda.pagelocked_empty((y.height, y.width), dtype=y.dtype)# mem_flags=cuda.host_alloc_flags.DEVICEMAP)
        # print(arr_y.shape, arr_y.dtype)
        memcpy = cuda.Memcpy2D()
        memcpy.set_src_device(y.devptr)
        memcpy.src_pitch = y.pitch
        memcpy.set_dst_host(arr_y)
        memcpy.dst_pitch = y.width * np.dtype(y.dtype).itemsize
        memcpy.width_in_bytes = y.width * np.dtype(y.dtype).itemsize
        memcpy.height = y.height
        # print(memcpy.src_pitch, memcpy.dst_pitch, memcpy.width_in_bytes, memcpy.height)
        memcpy(custr)
        # log.warning(time.time())
        custr.synchronize()
        # log.warning(time.time())

        #custr.synchronize()
        Image.fromarray(arr_y).save(f"dump.jpg")

        # memcpy.set_src_device(uv.devptr)
        # memcpy.src_pitch = uv.pitch
        # arr_uv = cuda.pagedlocked_empty((uv.height, uv.width), uv.dtype)
        # memcpy.set_dst_host(arr_uv)
        # memcpy.dst_pitch = uv.width * uv.dtype.itemsize
        # memcpy.width_in_bytes = uv.width * uv.dtype.itemsize
        # memcpy.height = uv.height
        # memcpy(custr)

        # arr_u = np.empty((y.height, y.width), dtype=uv.dtype)
        # arr_v = np.empty((y.height, y.width), dtype=uv.dtype)
        # arr_u[:, ::2] = arr_uv[:, ::2]
        # arr_v[:, 1::2] = arr_uv[:, 1::2]
        

    decoder = Decoder(ctx, display_callback=display_callback, extra_pictures=8)

    container = av.open(path)
    stream = container.streams.video[0]
    trans = StreamTranslate(stream)
    container.seek(int(600/stream.time_base), stream=stream)
    for packet in tqdm.tqdm(container.demux(stream)):
        for pts, bs in trans.translate_packet(packet):
            decoder.send(bs, pts)
            del bs

if __name__ == '__main__':
    import sys
    test(sys.argv[1])