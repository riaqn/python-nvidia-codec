from numpy import extract
from nvidia_codec.utils import Screenshot, extract_stream_ptr
import torch
from datetime import timedelta
from PIL import Image

import cupy


# url = '/mnt/Downloads/movies/Everything.Everywhere.All.At.Once.2022.2160p.WEB-DL.DDP5.1.x265.10bit-EVO.60fps.mkv'
#url = '/mnt/Downloads/movies/L.A.Confidential.1997.BluRay.1080p.TrueHD.5.1.x265.10bit-BeiTai/L.A.Confidential.1997.BluRay.1080p.TrueHD.5.1.x265.10bit-BeiTai.mkv'
url = '/mnt/Downloads/movies/现代启示录.Apocalypse.Now.1979.BluRay.2160p.x265.10bit.HDR.3Audio.mUHD-FRDS/Apocalypse.Now.1979.BluRay.2160p.x265.10bit.HDR.3Audio.mUHD-FRDS.mkv'
screenshot = Screenshot(url)
print(screenshot.start_time, screenshot.duration)
cuda = torch.device('cuda:1')
stream = torch.cuda.Stream()


arr = torch.empty((screenshot.height, screenshot.width, 3), dtype=torch.uint8, device=cuda)
with torch.cuda.device(cuda):
    with cupy.cuda.Device(1):
        with torch.cuda.stream(stream):
            with cupy.cuda.ExternalStream(extract_stream_ptr(stream)):
                screenshot.shoot(timedelta(minutes=60), arr)
                print(arr.shape)
                k = arr.cpu().numpy()
                Image.fromarray(k).show()
