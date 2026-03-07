from nvidia_codec.utils import Player
import torch
from tqdm import tqdm
import sys

def test(path):
    '''
    Decode all frames of a video file and display progress.
    '''
    player = Player(path)
    print(f'Duration: {player.duration}')

    for i, (time, frame) in enumerate(tqdm(player.frames(torch.float))):
        pass  # frame is [3, H, W] tensor on GPU

    print(f'Decoded {i+1} frames')

if __name__ == '__main__':
    path = sys.argv[1]
    test(path)
