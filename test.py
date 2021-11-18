import pycuda.driver as cuda
from .decode import Decoder
import logging
import av
from .pyav import StreamTranslate
import faulthandler
from tqdm import tqdm

faulthandler.enable()

log = logging.getLogger(__name__)

logging.basicConfig(level=logging.WARNING)

def test(path):
    cuda.init()    
    ctx = cuda.Device(0).retain_primary_context()

    decoder = Decoder(ctx, extra_pictures=24)

    container = av.open(path)
    stream = container.streams.video[0]
    trans = StreamTranslate(stream)
    # container.seek(int(600/stream.time_base), stream=stream)
    bar = tqdm(decoder.decode(trans.translate_packets(container.demux(stream), False)))
    for picture, pts in bar:
        bar.set_description(f'{pts} @ {picture.index}')

if __name__ == '__main__':
    import sys
    test(sys.argv[1])