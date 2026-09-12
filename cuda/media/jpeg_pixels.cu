/* Purpose: Reconstruct JPEG component planes and canonical RGB8 pixels on the device.
 * Owns: One shared separable IDCT tile per block; output belongs to each image job.
 * Launch shape: Parallel component blocks and pixel rows across all admitted images.
 * Lifetime: The final decode step of each image. */
#include "media/image.cuh"
#include <math.h>

/* C[x][u] is cos((2*x+1)*u*pi/16), with the DC coefficient divided by sqrt(2). */
static __device__ __constant__ float aotx_image_cos[8][8] = {
    {0.7071067812f, 0.9807852804f, 0.9238795325f, 0.8314696123f, 0.7071067812f, 0.5555702330f, 0.3826834324f, 0.1950903220f},
    {0.7071067812f, 0.8314696123f, 0.3826834324f, -0.1950903220f, -0.7071067812f, -0.9807852804f, -0.9238795325f, -0.5555702330f},
    {0.7071067812f, 0.5555702330f, -0.3826834324f, -0.9807852804f, -0.7071067812f, 0.1950903220f, 0.9238795325f, 0.8314696123f},
    {0.7071067812f, 0.1950903220f, -0.9238795325f, -0.5555702330f, 0.7071067812f, 0.8314696123f, -0.3826834324f, -0.9807852804f},
    {0.7071067812f, -0.1950903220f, -0.9238795325f, 0.5555702330f, 0.7071067812f, -0.8314696123f, -0.3826834324f, 0.9807852804f},
    {0.7071067812f, -0.5555702330f, -0.3826834324f, 0.9807852804f, -0.7071067812f, -0.1950903220f, 0.9238795325f, -0.8314696123f},
    {0.7071067812f, -0.8314696123f, 0.3826834324f, 0.1950903220f, -0.7071067812f, 0.9807852804f, -0.9238795325f, 0.5555702330f},
    {0.7071067812f, -0.9807852804f, 0.9238795325f, -0.8314696123f, 0.7071067812f, -0.5555702330f, 0.3826834324f, -0.1950903220f}
};
static __device__ __forceinline__ unsigned char aotx_image_byte(int value) {
    return (unsigned char)max(0, min(255, value));
}
__global__ void aotx_image_idct(aotx_image_job *jobs, uint32_t count) {
    __shared__ float rows[64];
    uint32_t x = threadIdx.x & 7u, y = threadIdx.x >> 3;
    for (uint32_t at = blockIdx.y; at < count; at += gridDim.y) {
        aotx_image_job *j = jobs + at;
        if (j->phase != AOTX_IMAGE_IDCT) continue;
        for (uint32_t b = blockIdx.x; b < j->blocks; b += gridDim.x) {
            uint32_t ci = 0;
            while (ci + 1 < j->components && b >= j->channel[ci + 1].base) ++ci;
            const aotx_jpeg_component *c = j->channel + ci;
            const int32_t *coeff = j->coefficients + (uint64_t)b * 64;
            float sum = 0;
            for (uint32_t u = 0; u < 8; ++u)
                sum += (float)coeff[y * 8 + u] * c->quant_values[y * 8 + u] * aotx_image_cos[x][u];
            rows[y * 8 + x] = sum;
            __syncthreads();
            sum = 0;
            for (uint32_t v = 0; v < 8; ++v) sum += rows[v * 8 + x] * aotx_image_cos[y][v];
            uint32_t local = b - c->base;
            uint64_t pixel = (uint64_t)c->base * 64 +
                ((uint64_t)(local / c->stride) * 8 + y) * c->stride * 8 + (local % c->stride) * 8 + x;
            float sample = fminf(255.0f, fmaxf(0.0f, floorf(sum * 0.25f + 128.5f)));
            j->planes[pixel] = (unsigned char)sample;
            __syncthreads();
        }
    }
}
static __device__ __forceinline__ int aotx_image_sample(const aotx_image_job *j,
                                                        const aotx_jpeg_component *c, int x, int y) {
    x = max(0, min((int)c->width - 1, x)); y = max(0, min((int)c->height - 1, y));
    return j->planes[(uint64_t)c->base * 64 + (uint64_t)y * c->stride * 8 + x];
}
static __device__ int aotx_image_chroma(const aotx_image_job *j, uint32_t channel,
                                        uint32_t x, uint32_t y) {
    const aotx_jpeg_component *c = j->channel + channel;
    if (c->h == j->hmax && c->v == j->vmax) return aotx_image_sample(j, c, x, y);
    int cx = (int)(x / 2), cy = (int)(c->v == j->vmax ? y : y / 2);
    if (c->width <= 2) return aotx_image_sample(j, c, cx, cy);
    int nx = cx + ((x & 1u) ? 1 : -1);
    if (c->v == j->vmax) {
        int sum = 3 * aotx_image_sample(j, c, cx, cy) + aotx_image_sample(j, c, nx, cy);
        return (sum + ((x & 1u) ? 2 : 1)) >> 2;
    }
    int ny = cy + ((y & 1u) ? 1 : -1);
    int sum = 9 * aotx_image_sample(j, c, cx, cy) + 3 * aotx_image_sample(j, c, nx, cy) +
              3 * aotx_image_sample(j, c, cx, ny) + aotx_image_sample(j, c, nx, ny);
    return (sum + ((x & 1u) ? 7 : 8)) >> 4;
}
__global__ void aotx_image_pixels(aotx_image_job *jobs, uint32_t count) {
    for (uint32_t at = blockIdx.y; at < count; at += gridDim.y) {
        aotx_image_job *j = jobs + at;
        if (j->phase != AOTX_IMAGE_IDCT && j->phase != AOTX_IMAGE_RAW) continue;
        uint64_t pixels = (uint64_t)j->width * j->height;
        for (uint64_t p = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
             p < pixels; p += (uint64_t)blockDim.x * gridDim.x) {
            if (j->phase == AOTX_IMAGE_RAW) {
                for (uint32_t k = 0; k < 3; ++k) j->rgb[p * 3 + k] = j->source[p * 3 + k];
                continue;
            }
            uint32_t x = (uint32_t)(p % j->width), y = (uint32_t)(p / j->width);
            int luma = aotx_image_sample(j, j->channel, x, y), cb = 0, cr = 0;
            if (j->components == 3) {
                cb = aotx_image_chroma(j, 1, x, y) - 128;
                cr = aotx_image_chroma(j, 2, x, y) - 128;
            }
            j->rgb[p * 3] = aotx_image_byte(luma + ((91881 * cr + 32768) >> 16));
            j->rgb[p * 3 + 1] = aotx_image_byte(luma + ((-22554 * cb - 46802 * cr + 32768) >> 16));
            j->rgb[p * 3 + 2] = aotx_image_byte(luma + ((116130 * cb + 32768) >> 16));
        }
    }
}
