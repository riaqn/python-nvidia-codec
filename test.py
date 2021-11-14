import pycuda.driver as cuda
from .decoder import Decoder
import logging
import av
from .pyav import StreamTranslate
import faulthandler
import itertools
import tqdm

faulthandler.enable()

log = logging.getLogger(__name__)

logging.basicConfig(level=logging.WARNING)

def test(path):
    ctx = cuda.Device(0).retain_primary_context()
    ctx.push()
    stream = cuda.Stream()
    ctx.pop()

    def display_callback(picture):
        log.info(f"display_callback: {picture.timestamp}")

    decoder = Decoder(ctx, display_callback=display_callback)

    container = av.open(path)
    stream = container.streams.video[0]
    trans = StreamTranslate(stream)
    for packet in tqdm.tqdm(container.demux(stream)):
        for pts, bs in trans.translate_packet(packet):
            decoder.send(bs, pts)
            del bs

if __name__ == '__main__':
    import sys
    test(sys.argv[1])