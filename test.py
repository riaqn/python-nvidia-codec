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

def test(deviceID, path):
    cuda.init()    
    ctx = cuda.Device(deviceID).retain_primary_context()


    container = av.open(path)
    stream = container.streams.video[0]
    trans = StreamTranslate(stream)
    decoder = Decoder(ctx, trans.translate_codec(), extra_pictures=8)

    # container.seek(int(600/stream.time_base), stream=stream)
    bar = tqdm(decoder.decode(trans.translate_packets(container.demux(stream), False)))
    for picture, pts in bar:
        bar.set_description(f'{pts}')

if __name__ == '__main__':
    import sys
    deviceID = int(sys.argv[1])
    path = sys.argv[2]
    test(deviceID, path)