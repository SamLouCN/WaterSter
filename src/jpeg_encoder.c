/* C ABI over the supplied X5 JPU codec library. One owner per context. */
#include "hb_media_codec.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    media_codec_context_t codec;
    int width, height, started;
} JpegEncoder;

void jpeg_encoder_close(JpegEncoder *e) {
    if (!e) return;
    if (e->started) hb_mm_mc_stop(&e->codec);
    hb_mm_mc_release(&e->codec);
    free(e);
}

JpegEncoder *jpeg_encoder_open(int width, int height, int quality, int *error) {
    if (!error) return NULL;
    *error = -1;
    if (width <= 0 || height <= 0 || width % 16 || height % 8 ||
        width > 8192 || height > 8192 || quality < 1 || quality > 100) return NULL;
    JpegEncoder *e = calloc(1, sizeof(*e));
    if (!e) return NULL;
    e->width = width;
    e->height = height;
    e->codec.codec_id = MEDIA_CODEC_ID_JPEG;
    e->codec.encoder = 1;
    mc_video_codec_enc_params_t *p = &e->codec.video_enc_params;
    p->width = width;
    p->height = height;
    p->pix_fmt = MC_PIXEL_FORMAT_YUV420P;
    p->frame_buf_count = 3;
    p->bitstream_buf_count = 3;
    p->bitstream_buf_size = ((size_t)width * height * 3 + 65536 + 4095) & ~(size_t)4095;
    p->rot_degree = MC_CCW_0;
    p->mir_direction = MC_DIRECTION_NONE;
    p->jpeg_enc_config.quality_factor = quality;
    int rc = hb_mm_mc_initialize(&e->codec);
    if (rc) { *error = rc; free(e); return NULL; }
    rc = hb_mm_mc_configure(&e->codec);
    if (!rc) {
        mc_av_codec_startup_params_t start = {0};
        rc = hb_mm_mc_start(&e->codec, &start);
        if (!rc) e->started = 1;
    }
    if (rc) { *error = rc; jpeg_encoder_close(e); return NULL; }
    *error = 0;
    return e;
}

/* Copy packed I420 into DMA buffers and copy the JPEG before returning the
 * output buffer. Any error is fatal to this context: close it. */
int jpeg_encoder_encode(JpegEncoder *e, const uint8_t *i420, size_t size,
                        uint8_t *jpeg, size_t capacity, size_t *written) {
    if (!e || !i420 || !jpeg || !written ||
        size != (size_t)e->width * e->height * 3 / 2) return -1;
    *written = 0;
    media_codec_buffer_t input = {0}, output = {0};
    int rc = hb_mm_mc_dequeue_input_buffer(&e->codec, &input, 1000);
    if (rc) return rc;
    mc_video_frame_buffer_info_t *f = &input.vframe_buf;
    int stride = f->stride > 0 ? f->stride : e->width;
    /* Planar 4:2:0 chroma is half the luma stride. */
    int cstride = stride / 2;
    if (stride < e->width || stride % 2 || !f->vir_ptr[0] ||
        !f->vir_ptr[1] || !f->vir_ptr[2]) return -2;
    for (int plane = 0; plane < 3; ++plane) {
        int width = plane ? e->width / 2 : e->width;
        int height = plane ? e->height / 2 : e->height;
        int pitch = plane ? cstride : stride;
        if (f->compSize[plane] && f->compSize[plane] < (size_t)pitch * height) return -3;
        const uint8_t *src = i420 + (plane ? (size_t)e->width * e->height +
                                      (plane - 1) * (size_t)e->width * e->height / 4 : 0);
        for (int row = 0; row < height; ++row)
            memcpy(f->vir_ptr[plane] + (size_t)row * pitch, src + (size_t)row * width, width);
    }
    f->width = e->width;
    f->height = e->height;
    f->pix_fmt = MC_PIXEL_FORMAT_YUV420P;
    f->size = size;
    rc = hb_mm_mc_queue_input_buffer(&e->codec, &input, 1000);
    if (rc) return rc;
    rc = hb_mm_mc_dequeue_output_buffer(&e->codec, &output, NULL, 1000);
    if (rc) return rc;
    if (!output.vstream_buf.vir_ptr || !output.vstream_buf.size) rc = -4;
    else if (output.vstream_buf.size > capacity) rc = -5;
    else {
        *written = output.vstream_buf.size;
        memcpy(jpeg, output.vstream_buf.vir_ptr, *written);
    }
    int release_rc = hb_mm_mc_queue_output_buffer(&e->codec, &output, 1000);
    return rc ? rc : release_rc;
}
