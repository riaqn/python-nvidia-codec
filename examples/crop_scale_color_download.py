from datetime import timedelta
from hypothesis import target
import pycuda.driver as cuda
from pycuda.gpuarray  import GPUArray
from types import SimpleNamespace
from nvidia_codec.common import SurfaceFormat, convert_shape, size2shape
from nvidia_codec.decode import Decoder, Surface, decide_surface_format
import logging
import av
from nvidia_codec.pyav import PyAVStreamAdaptor
from nvidia_codec import color
import faulthandler
from tqdm import tqdm
import numpy as np

from PIL import Image

faulthandler.enable()

log = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)

def test(deviceID, path):
    '''
    take a screenshot for every 10 secs
    - only take the left half of the picture as source
    - the output will be 256x224
    - the source is scaled to 80x204, and put to the target rectangle (20,20,100,224) of the output
    - the margin in the output will be left black
    '''
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
        # resize to 1/2 in both dimensions
        cropping = {
            'top': 0,
            'left': 0,
            'right': p['size']['width'] // 2, # we only want the left half
            'bottom': p['size']['height']
        }

        target_size = {
            'width' : 256,
            'height' : 224
        }

        target_rect = {
            'top': 20,
            'left': 20,
            'right': 100,
            'bottom': 224
        }

        return {
            'cropping': cropping,
            'target_size' : target_size,
            'target_rect' : target_rect,
        }
        
    decoder = Decoder(trans.translate_codec(), decide)

    # container.seek(int(600/stream.time_base), stream=stream)
    bar = tqdm(decoder.decode(trans.translate_packets(container.demux(stream), False)))
    cvt = None
    position = timedelta()
    for picture, pts in bar:
        if pts > position.total_seconds() * stream.time_base + stream.start_time:
            bar.set_description(f'{position}')
            position += timedelta(seconds=10) # take a screenshot for every 10 sec
            surface = picture.map(stream=s)
            if cvt is None:
                # the following two lines rely on some patch
                # to query the color space/range of video stream
                # https://github.com/riaqn/PyAV
                space = color.Space(stream.colorspace)
                range = color.Range(stream.color_range)
                if space is color.Space.UNSPECIFIED:
                    space = color.Space.BT470BG
                if range == color.Range.UNSPECIFIED:
                    range = color.Range.MPEG          
                log.info(f'color space = {space} color range = {range}')
                target_format = SurfaceFormat.RGB444P
                target_shape = size2shape(target_format, surface.size)
                # for best performance, we have a fixed gpu array for output
                surface_rgb = GPUArray(shape = target_shape, dtype=np.uint8)
                # for best performance, we have a fixed cpu array for downloading
                array_rgb = np.ndarray(shape = target_shape, dtype=np.uint8)
                cvt = color.Converter(surface, surface.format, space, range, target_template=surface_rgb, target_format=target_format)
            cvt(surface, surface_rgb)
            surface_rgb.get_async(s, array_rgb)
            s.synchronize()
            # the downloaded array is 3*h*w
            # reorganize to h*w*3 for image loading
            # zero cost
            arr = np.moveaxis(array_rgb, 0, -1)
            log.debug(arr.shape)
            img = Image.fromarray(arr, mode='RGB')
            img.save('dump.jpg')
            surface.free() # drop the reference to the surface to free up the slot
        picture.free() # drop reference to picture to free up slot

    ctx.pop()

if __name__ == '__main__':
    import sys
    deviceID = int(sys.argv[1])
    path = sys.argv[2]
    test(deviceID, path)