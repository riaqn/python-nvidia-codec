#include <libavcodec/codec_id.h>
#include <libavutil/error.h>
#include <libavformat/avformat.h>

int main(void) {
    printf("HEVC = %d\n", AV_CODEC_ID_HEVC);
    printf("H264 = %d\n", AV_CODEC_ID_H264);
    printf("yuv444P = %d\n", AV_PIX_FMT_YUV444P);
    printf("yuv420P = %d\n", AV_PIX_FMT_YUV420P);
    printf("yuv422P = %d\n", AV_PIX_FMT_YUV422P);
    printf("yuv444P16 = %d\n", AV_PIX_FMT_YUV444P16LE);
    printf("yuv420P16 = %d\n", AV_PIX_FMT_YUV420P16);
    printf("%d\n", offsetof(AVStream, start_time));
}