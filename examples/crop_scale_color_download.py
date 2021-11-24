import pycuda.driver as cuda
from types import SimpleNamespace
from nvidia_codec.decode import Decoder, decide_surface_format
import logging
import av
from nvidia_codec.pyav import PyAVStreamAdaptor
from nvidia_codec import color
import faulthandler
from tqdm import tqdm

import cppyy
from PIL import Image
cppyy.include('libavutil/avutil.h')
c = cppyy.gbl

faulthandler.enable()

log = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)

def test(deviceID, path):
    cuda.init()    
    ctx = cuda.Device(deviceID).retain_primary_context()
    ctx.push()
    s = cuda.Stream()

    container = av.open(path)
    stream = container.streams.video[0]
    trans = PyAVStreamAdaptor(stream)
    def decide(p):
        log.info(p)
        # cropping: we only want the left half of the picture
        cropping = {
            'left' : 0,
            'top' : 0,
            'right' : p['size']['width'] // 2,
            'bottom' : p['size']['height']
        }
        # resize to 1/2 in both dimensions
        target_size = {
            'width' : (cropping['right'] - cropping['left'])//2,
            'height' : (cropping['bottom'] - cropping['top'])//2
        }

        return {
            'cropping' : cropping,
            'target_size' : target_size,
        }
        
    decoder = Decoder(trans.translate_codec(), decide)

    # container.seek(int(600/stream.time_base), stream=stream)
    bar = tqdm(decoder.decode(trans.translate_packets(container.demux(stream), False)))
    cvt = None
    for i, (picture, pts) in enumerate(bar):
        bar.set_description(f'{pts}')
        if i % 60 == 0:
            surface = picture.map(stream=s)
            if cvt is None:
                # the following two lines rely on some patch
                # to query the color space/range of video stream
                # https://github.com/riaqn/PyAV
                space = stream.colorspace
                range = stream.color_range                
                if space == c.AVCOL_SPC_UNSPECIFIED:
                    space = c.AVCOL_SPC_BT470BG
                if range == c.AVCOL_RANGE_UNSPECIFIED:
                    range = c.AVCOL_RANGE_MPEG                
                log.info(f'color space = {space} color range = {range}')
                cvt = color.Converter(surface, space, range)
            surface_rgb24 = cvt(surface, stream=s)
            arr = surface_rgb24.download(stream = s)
            s.synchronize()
            arr = arr.reshape((surface_rgb24.height, surface_rgb24.width, 3))
            log.debug(arr.shape)
            img = Image.fromarray(arr, mode='RGB')
            img.save('dump.jpg')
            del surface # drop the reference to the surface to free up the slot
        del picture # drop reference to picture to free up slot

    ctx.pop()

if __name__ == '__main__':
    import sys
    deviceID = int(sys.argv[1])
    path = sys.argv[2]
    test(deviceID, path)