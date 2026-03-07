from nvidia_codec.utils import Screenshoter
from nvidia_codec import NoFrameError
from datetime import timedelta
from PIL import Image
import numpy
import sys
import torch

_, path, seconds, out = sys.argv

player = Screenshoter(path, lambda h, w: (h // 4 * 2, w // 4 * 2))

try:
    time, tensor = player.screenshot(timedelta(seconds=int(seconds)), torch.uint8)
except NoFrameError as e:
    print(f'No frame found: {e}')
    sys.exit(1)

print(f'Got frame at {time} (requested {seconds}s)')
arr = tensor.cpu().numpy()
arr = numpy.moveaxis(arr, 0, -1)
Image.fromarray(arr).save(out)

player.free()
