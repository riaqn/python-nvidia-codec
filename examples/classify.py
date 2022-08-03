from datetime import timedelta
import math
import torch
import torchvision

import logging
import faulthandler
from tqdm import tqdm
import numpy as np

from nvidia_codec.utils import Screenshoter

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

        player = Screenshoter(path, target_size)

        net = torchvision.models.convnext_tiny(weights = torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT).cuda()        
        net.eval()

        time = timedelta(seconds=0)
        while True:
            time += timedelta(seconds = 10)
            if time > player.duration:
                break
            with torch.cuda.stream(stream):
                time_act, screen = player.screenshot(time, torch.float)
                assert abs(time - time_act).total_seconds() < 1

                inp = screen[None, :]
                mean = inp.mean((0,2,3))    
                std = inp.std((0,2,3))

                norm = torchvision.transforms.Normalize(mean=mean,
                                                        std=std).cuda()
                try:
                    inp = norm(inp)
                except ValueError:
                    log.warning(f'skip {time}: black screen')
                    continue
                out = net(inp)
                idx = torch.argmax(out).cpu().numpy()

                # Image.fromarray((screen.moveaxis(0, -1).cpu().numpy() * 255).astype(np.uint8)).save(f'test-{time}.png')            
            print(f'{time} {id2label[str(idx)][1]} {out[0][idx].item()}')
   
if __name__ == '__main__':
    import sys
    device_idx = int(sys.argv[1])
    path = sys.argv[2]
    test(device_idx, path)