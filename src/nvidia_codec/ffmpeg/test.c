#include <libavcodec/codec_id.h>
#include <libavutil/error.h>
#include <libavformat/avformat.h>

int main(void) {

    printf("MPEG1 %d\n", AV_CODEC_ID_MPEG1VIDEO);
    printf("MPEG2 %d\n", AV_CODEC_ID_MPEG2VIDEO);
    printf("MPEG4 %d\n", AV_CODEC_ID_MPEG4);
    printf("VC1 %d\n", AV_CODEC_ID_VC1);
    printf("H264 = %d\n", AV_CODEC_ID_H264);    
    printf("JPEG %d\n", AV_CODEC_ID_JPEG2000);    
    printf("HEVC = %d\n", AV_CODEC_ID_HEVC);
    printf("VP8 %d\n", AV_CODEC_ID_VP8);
    printf("VP9 %d\n", AV_CODEC_ID_VP9);
    printf("AV1 %d\n", AV_CODEC_ID_AV1);
}