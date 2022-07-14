from datetime import timedelta
from nvidia_codec.core.decode import Decoder
from nvidia_codec.ffmpeg.libavformat import FormatContext
from nvidia_codec.ffmpeg.libavcodec import AVMediaType, AVCodecID, BitStreamFilter, BSFContext

from nvidia_codec.utils.compat import av2cuda
import nvidia_codec.core.cuda  as cuda
# import faulthandler
from tqdm import tqdm

import numpy as np
# faulthandler.enable()

def test(deviceID, path):
    '''
    Simply decode a video file without any further action
    '''
    # cuda.check(cuda.lib.cuInit(0))
    ctx = FormatContext(path)
    [stream] = list(filter(lambda s: s.codecpar.contents.codec_type == AVMediaType.VIDEO, ctx.streams))

    codec_id = stream.codecpar.contents.codec_id

    if codec_id == AVCodecID.HEVC:
            f = BitStreamFilter('hevc_mp4toannexb')
    elif codec_id == AVCodecID.H264:
            f = BitStreamFilter('h264_mp4toannexb') 
    else:
            raise Exception(f'unsupported codec {codec_id}')                

    bsf = BSFContext(f, stream.codecpar.contents, stream.time_base)

    with cuda.Device(deviceID):
        decoder = Decoder(av2cuda(codec_id))

    def adapt(pkt):
        pts = pkt.av.pts
        # log.warning(pts)
        arr = np.ctypeslib.as_array(pkt.av.data, (pkt.av.size,))
        return arr, pts            

    bar = tqdm(decoder.decode(map(adapt, bsf.filter(ctx.read_frames(stream), True))))
    for picture, pts in bar:
        # seconds = (pts - stream.start_time) * float(stream.time_base)
        # delta = timedelta(seconds=)
        bar.set_description(f'{pts}')

if __name__ == '__main__':
    import sys
    deviceID = int(sys.argv[1])
    path = sys.argv[2]
    test(deviceID, path)