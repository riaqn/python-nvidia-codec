from datetime import timedelta
import pycuda.driver as cuda
from nvidia_codec.decode import Decoder
import av
from nvidia_codec.pyav import PyAVStreamAdaptor
# import faulthandler
from tqdm import tqdm

# faulthandler.enable()

def test(deviceID, path):
    '''
    Simply decode a video file without any further action
    '''
    cuda.init()    
    ctx = cuda.Device(deviceID).retain_primary_context()
    ctx.push()

    container = av.open(path)
    stream = container.streams.video[0]
    trans = PyAVStreamAdaptor(stream)
    decoder = Decoder(trans.translate_codec())

    bar = tqdm(decoder.decode(trans.translate_packets(container.demux(stream), False)))
    for picture, pts in bar:
        seconds = (pts - stream.start_time) * float(stream.time_base)
        delta = timedelta(seconds=seconds)
        bar.set_description(f'{delta}')

    ctx.pop()

if __name__ == '__main__':
    import sys
    deviceID = int(sys.argv[1])
    path = sys.argv[2]
    test(deviceID, path)