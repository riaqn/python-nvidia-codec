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
        if i % 600 == 0:
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
            del surface # drop the reference to the surface to free up the slot
        del picture # drop reference to picture to free up slot

    ctx.pop()

if __name__ == '__main__':
    import sys
    deviceID = int(sys.argv[1])
    path = sys.argv[2]
    test(deviceID, path)