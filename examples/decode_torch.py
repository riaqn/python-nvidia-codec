from datetime import timedelta
import torch
import torchvision


import pycuda.driver as cuda_dri
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

import requests
import json

def test(deviceID, path):
    '''
    take a screenshot for every 10 secs, and run image classification on it
    pictures never leaves GPU
    '''

    cuda_dri.init()    
    cuda = torch.device(f'cuda:{deviceID}')


    resp = requests.get('https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json')
    id2label = json.loads(resp.text)

    net = torchvision.models.efficientnet_b3(pretrained=True).to(cuda)
    # must be 256x224    

    ctx = cuda_dri.Device(deviceID).retain_primary_context()
    ctx.push()
    s = cuda_dri.Stream()

    container = av.open(path)
    stream = container.streams.video[0]
    trans = PyAVStreamAdaptor(stream)
    def decide(p):
        log.info(p)
        width = 320
        height = 300
        target_size = {
            'width' : width,
            'height' : height
        } # size required by efficientnet-b0

        ratio = p['size']['width'] / p['size']['height']
        if ratio > width/height:
            # too wide
            target_rect = {
                'top': 0,
                'left': 0,
                'right': width,
                'bottom': int(width/ratio)
            }
        else:
            target_rect = {
                'top': 0,
                'left': 0,
                'right': int(ratio*height),
                'bottom': height
            }

        return {
            'target_size' : target_size,
            'target_rect' : target_rect
        }
        
    decoder = Decoder(trans.translate_codec(), decide)

    # container.seek(int(600/stream.time_base), stream=stream)
    bar = tqdm(decoder.decode(trans.translate_packets(container.demux(stream), False)))
    timestamp = timedelta()
    cvt = None
    for picture, pts in bar:
        time = timedelta(seconds = float((pts - stream.start_time) * stream.time_base))
        bar.set_description(f'{time}')        
        if time > timestamp:
            timestamp += timedelta(seconds=1) # take a screenshot for every 10 sec
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
                surface_rgb = torch.empty(target_shape, dtype=torch.uint8, device=cuda)
                cvt = color.Converter(surface, surface.format, space, range, target_template=surface_rgb, target_format=target_format)
            cvt(surface, surface_rgb, stream=s)
            s.synchronize()        
            array_rgb = surface_rgb.cpu().numpy()            

            arr =  np.moveaxis(array_rgb, 0, -1)
            img = Image.fromarray(arr, mode='RGB')
            img.save(f'{time}.jpg')

            surface_float = surface_rgb.type(torch.float) / 256.0
            # print('surface_float mean', torch.mean(surface_float))

            # print(torch.mean(surface_float), torch.std(surface_float))

            norm = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]).to(cuda)

            surface_norm = norm(surface_float)
            # print('surface_norm',torch.mean(surface_norm))

            features = net(torch.unsqueeze(surface_norm, dim=0))
            idx = torch.argmax(features).item()
            # print(idx)
            # print(f'{features.shape}, {len(id2label)}')
            print(f'{id2label[str(idx)][1]} {features[0][idx].item()}')

            # the downloaded array is 3*h*w
            # reorganize to h*w*3 for image loading
            # zero cost

            surface.free() # drop the reference to the surface to free up the slot
        picture.free() # drop reference to picture to free up slot

    ctx.pop()

if __name__ == '__main__':
    import sys
    deviceID = int(sys.argv[1])
    path = sys.argv[2]
    test(deviceID, path)