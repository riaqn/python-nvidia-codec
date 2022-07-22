import itertools
from nvidia_codec.utils.player import Player
import torch

# import faulthandler
from tqdm import tqdm

import numpy as np
# faulthandler.enable()

def test(deviceID, path):
    '''
    Simply decode a video file without any further action
    '''


    player = Player(path, device = deviceID)
    
    bar = tqdm(itertools.count())
    for i in bar:
        time = player.skip_frame()
        # seconds = (pts - stream.start_time) * float(stream.time_base)
        # delta = timedelta(seconds=)
        bar.set_description(f'{time}')

if __name__ == '__main__':
    import sys
    deviceID = int(sys.argv[1])
    path = sys.argv[2]
    test(deviceID, path)