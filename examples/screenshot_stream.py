from numpy import extract
from nvidia_codec.utils import Screenshot, extract_stream_ptr
import torch
from datetime import timedelta
from PIL import Image


# url = '/mnt/Downloads/movies/Everything.Everywhere.All.At.Once.2022.2160p.WEB-DL.DDP5.1.x265.10bit-EVO.60fps.mkv'
url = '/mnt/Downloads/movies/L.A.Confidential.1997.BluRay.1080p.TrueHD.5.1.x265.10bit-BeiTai/L.A.Confidential.1997.BluRay.1080p.TrueHD.5.1.x265.10bit-BeiTai.mkv'
# url = '/mnt/Downloads/movies/现代启示录.Apocalypse.Now.1979.BluRay.2160p.x265.10bit.HDR.3Audio.mUHD-FRDS/Apocalypse.Now.1979.BluRay.2160p.x265.10bit.HDR.3Audio.mUHD-FRDS.mkv'

cuda_device = 1


with torch.cuda.device(1):
    screenshot = Screenshot(url)    
    print(screenshot.start_time, screenshot.duration)    
    arr = torch.empty((screenshot.height, screenshot.width, 3), dtype=torch.uint8, device='cuda')
    stream = torch.cuda.Stream()        
    print(f'using stream {extract_stream_ptr(stream)}')
    # print(arr.stride())
    with torch.cuda.stream(stream):
        screenshot.shoot(timedelta(seconds=3602), arr, extract_stream_ptr(stream))

        # print(arr.shape)
        k = arr.cpu().numpy()

    stream.synchronize()
    Image.fromarray(k).show()