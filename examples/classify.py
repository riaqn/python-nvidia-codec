from datetime import timedelta
import math
import torch
import torchvision
from tqdm import tqdm
import numpy as np
import sys

from nvidia_codec.utils import Screenshoter
from nvidia_codec import NoFrameError

import requests
import json

def test(path):
    '''
    Take a screenshot every 10 seconds and run image classification.
    Frames never leave GPU until final label lookup.
    '''
    resp = requests.get('https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json')
    id2label = json.loads(resp.text)

    pixels = 384 * 384

    def target_size(h, w):
        aspect = h / w
        h = int(math.sqrt(pixels * aspect) // 2) * 2
        w = int(math.sqrt(pixels / aspect) // 2) * 2
        return h, w

    player = Screenshoter(path, target_size)

    if player.duration is None:
        print(f'Cannot determine duration for {path}')
        player.free()
        return

    net = torchvision.models.convnext_tiny(
        weights=torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT
    ).cuda()
    net.eval()

    time = timedelta(seconds=0)
    while True:
        time += timedelta(seconds=10)
        if time > player.duration:
            break
        try:
            time_act, screen = player.screenshot(time, torch.float)
        except NoFrameError:
            print(f'{time}: no frame')
            continue

        inp = screen[None, :]
        mean = inp.mean((0, 2, 3))
        std = inp.std((0, 2, 3))

        norm = torchvision.transforms.Normalize(mean=mean, std=std).cuda()
        try:
            inp = norm(inp)
        except ValueError:
            print(f'{time}: black screen')
            continue
        out = net(inp)
        idx = torch.argmax(out).cpu().numpy()
        print(f'{time_act or time} {id2label[str(idx)][1]} {out[0][idx].item():.2f}')

    player.free()

if __name__ == '__main__':
    path = sys.argv[1]
    test(path)
