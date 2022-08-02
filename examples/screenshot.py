
from nvidia_codec.utils import Player

from datetime import timedelta
from PIL import Image
import numpy

import sys
import torch

_, device, path, seconds, out = sys.argv
player = Player(path, lambda h,w: (h//4*2,w//4*2))

time, tensor = player.screenshoot(timedelta(seconds=int(seconds)), torch.uint8)
arr = tensor.cpu().numpy()
arr =  numpy.moveaxis(arr, 0, -1)

Image.fromarray(arr).save(out)