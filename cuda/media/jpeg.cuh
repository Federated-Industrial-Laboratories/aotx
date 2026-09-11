/* Purpose: Share bounded JPEG byte and coefficient helpers between device stages.
 * Owns: Constant coefficient order; mutable values belong to each image job.
 * Launch shape: Device functions called by one thread per image stream.
 * Lifetime: The loaded module and each finite scan step. */
#ifndef AOTX_MEDIA_JPEG_CUH
#define AOTX_MEDIA_JPEG_CUH
#include "media/image.cuh"

static __device__ __constant__ unsigned char aotx_jpeg_order[64] = {
     0, 1, 8,16, 9, 2, 3,10,17,24,32,25,18,11, 4, 5,
    12,19,26,33,40,48,41,34,27,20,13, 6, 7,14,21,28,35,
    42,49,56,57,50,43,36,29,22,15,23,30,37,44,51,58,59,
    52,45,38,31,39,46,53,60,61,54,47,55,62,63
};
static __device__ __forceinline__ uint32_t aotx_jpeg_u16(const unsigned char *p) {
    return ((uint32_t)p[0] << 8) | p[1];
}
static __device__ __forceinline__ uint32_t aotx_jpeg_ceil(uint32_t n, uint32_t d) {
    return n / d + (n % d != 0);
}
static __device__ __forceinline__ bool aotx_jpeg_extent(aotx_image_job *j, uint64_t n) {
    if (j->cursor > j->bytes || n > j->bytes - j->cursor) {
        aotx_image_refuse(j, AOTX_IMAGE_INVALID); return false;
    }
    return true;
}
static __device__ __forceinline__ bool aotx_jpeg_dimensions(aotx_image_job *j) {
    uint64_t pixels = (uint64_t)j->width * j->height;
    if (!j->width || !j->height) { aotx_image_refuse(j, AOTX_IMAGE_INVALID); return false; }
    if (j->width > j->dimension_limit || j->height > j->dimension_limit ||
        pixels > j->pixel_limit || pixels > j->rgb_bytes / 3) {
        aotx_image_refuse(j, AOTX_IMAGE_LIMIT); return false;
    }
    return true;
}
#endif
