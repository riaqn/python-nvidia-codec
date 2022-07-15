from numpy import extract
from nvidia_codec.utils import Screenshot, extract_stream_ptr
import torch
from datetime import timedelta
from PIL import Image

import sys

_, device, path, seconds, out = sys.argv
screenshot = Screenshot(path, lambda h,w: (h//4*2,w//4*2), int(device))

arr = screenshot.shoot(timedelta(seconds=int(seconds)))
k = arr.get()
Image.fromarray(k).save(out)