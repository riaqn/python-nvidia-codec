from datetime import timedelta
import math
import torch
import torchvision

import logging
import faulthandler
from tqdm import tqdm
import numpy as np

from nvidia_codec.utils import Screenshot, extract_stream_ptr

from PIL import Image

faulthandler.enable()

log = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)

import requests
import json

def test(device_idx, path):
    '''
    take a screenshot for every 10 secs, and run image classification on it
    pictures never leaves GPU
    '''

    with torch.cuda.device(device_idx):
        stream = torch.cuda.Stream()
  
        resp = requests.get('https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json')
        id2label = json.loads(resp.text)

        pixels = 384 * 384

        def target_size(h, w):
            aspect = h /w 
            h = int(math.sqrt(pixels * aspect) // 2) * 2
            w = int(math.sqrt(pixels / aspect) // 2) * 2
            return h,w

        s = Screenshot(path, target_size, '<f4')

        net = torchvision.models.convnext_tiny(weights = torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT).cuda()        
        net.eval()

        time = timedelta(seconds = 10)
        while time < s.duration:
            with torch.cuda.stream(stream):
                screen = s.shoot(time, target='torch', stream=extract_stream_ptr(stream))

                inp = screen[None, :]
                mean = inp.mean((0,2,3))    
                std = inp.std((0,2,3))
                norm = torchvision.transforms.Normalize(mean=mean,
                                                        std=std).cuda()
                inp = norm(inp)
                out = net(inp)
                idx = torch.argmax(out).cpu().numpy()

                # Image.fromarray((screen.moveaxis(0, -1).cpu().numpy() * 255).astype(np.uint8)).save(f'test-{time}.png')            
            print(f'{time} {id2label[str(idx)][1]} {out[0][idx].item()}')
            time += timedelta(seconds = 10)

if __name__ == '__main__':
    import sys
    device_idx = int(sys.argv[1])
    path = sys.argv[2]
    test(device_idx, path)