import pycuda.driver as cuda
from ..decode import Decoder
import logging
import av
from ..pyav import PyAVStreamAdaptor
import faulthandler
from tqdm import tqdm

import cppyy
cppyy.include('libavutil/avutil.h')
c = cppyy.gbl

faulthandler.enable()

log = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)

'''
Simply decode a video file without any further action
'''
def test(deviceID, path):
    cuda.init()    
    ctx = cuda.Device(deviceID).retain_primary_context()
    ctx.push()

    container = av.open(path)
    stream = container.streams.video[0]
    trans = PyAVStreamAdaptor(stream)
    decoder = Decoder(ctx, trans.translate_codec())

    # container.seek(int(600/stream.time_base), stream=stream)
    bar = tqdm(decoder.decode(trans.translate_packets(container.demux(stream), False)))
    for i, (picture, pts) in enumerate(bar):
        bar.set_description(f'{pts}')
        del picture # drop reference to picture to free up slot

    ctx.pop()

if __name__ == '__main__':
    import sys
    deviceID = int(sys.argv[1])
    path = sys.argv[2]
    test(deviceID, path)