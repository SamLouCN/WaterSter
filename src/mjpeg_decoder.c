/* Small C ABI bridge to the X5 SDK. No camera or image encoding happens here. */
#include "hb_media_codec.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    media_codec_context_t codec;
    int width, height, started;
    size_t input_capacity;
} FrontDecoder;

void front_decoder_close(FrontDecoder *d) {
    if (!d) return;
    if (d->started) hb_mm_mc_stop(&d->codec);
    hb_mm_mc_release(&d->codec);
    free(d);
}

FrontDecoder *front_decoder_open(int width, int height, int *error) {
    *error = -1;
    if (width <= 0 || height <= 0 || width % 2 || height % 2) return NULL;
    FrontDecoder *d = calloc(1, sizeof(*d));
    if (!d) return NULL;
    d->width = width;
    d->height = height;
    d->input_capacity = ((size_t)width * height * 3 + 4095) & ~(size_t)4095;
    d->codec.codec_id = MEDIA_CODEC_ID_JPEG;
    d->codec.encoder = 0;
    mc_video_codec_dec_params_t *p = &d->codec.video_dec_params;
    p->feed_mode = MC_FEEDING_MODE_FRAME_SIZE;
    p->pix_fmt = MC_PIXEL_FORMAT_NV12;
    p->frame_buf_count = 3;
    p->bitstream_buf_count = 3;
    p->bitstream_buf_size = d->input_capacity;
    p->jpeg_dec_config.rot_degree = MC_CCW_0;
    p->jpeg_dec_config.mir_direction = MC_DIRECTION_NONE;
    int rc = hb_mm_mc_initialize(&d->codec);
    if (rc) { *error = rc; free(d); return NULL; }
    rc = hb_mm_mc_configure(&d->codec);
    if (!rc) {
        mc_av_codec_startup_params_t start = {0};
        rc = hb_mm_mc_start(&d->codec, &start);
        if (!rc) d->started = 1;
    }
    if (rc) { *error = rc; front_decoder_close(d); return NULL; }
    *error = 0;
    return d;
}

/* Copies NV12 into caller-owned packed storage and returns the SDK buffer. */
int front_decoder_decode(FrontDecoder *d, const uint8_t *jpeg, size_t size,
                         uint8_t *nv12, size_t capacity) {
    if (!d || !jpeg || !nv12 || size > d->input_capacity || size == 0 ||
        capacity < (size_t)d->width * d->height * 3 / 2) return -1;
    media_codec_buffer_t input = {0}, output = {0};
    int rc = hb_mm_mc_dequeue_input_buffer(&d->codec, &input, 1000);
    if (rc) return rc;
    memcpy(input.vstream_buf.vir_ptr, jpeg, size);
    input.vstream_buf.size = size;
    rc = hb_mm_mc_queue_input_buffer(&d->codec, &input, 1000);
    if (rc) return rc;
    rc = hb_mm_mc_dequeue_output_buffer(&d->codec, &output, NULL, 1000);
    if (rc) return rc;
    mc_video_frame_buffer_info_t *f = &output.vframe_buf;
    if (f->width != d->width || f->height != d->height ||
        f->pix_fmt != MC_PIXEL_FORMAT_NV12 || !f->vir_ptr[0] || !f->vir_ptr[1]) {
        rc = -2;
    } else {
        int stride = f->stride > 0 ? f->stride : d->width;
        if (stride < d->width) rc = -3;
        else {
            for (int y = 0; y < d->height; ++y)
                memcpy(nv12 + (size_t)y*d->width,
                       f->vir_ptr[0] + (size_t)y*stride, d->width);
            uint8_t *uv = nv12 + (size_t)d->width*d->height;
            for (int y = 0; y < d->height/2; ++y)
                memcpy(uv + (size_t)y*d->width,
                       f->vir_ptr[1] + (size_t)y*stride, d->width);
        }
    }
    int release_rc = hb_mm_mc_queue_output_buffer(&d->codec, &output, 1000);
    return rc ? rc : release_rc;
}
