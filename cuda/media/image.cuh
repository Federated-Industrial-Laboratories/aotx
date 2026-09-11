/* Purpose: Decode bounded batches of JPEG or RGB image bytes on the device.
 * Owns: Per-image scan state; callers own source, coefficient and pixel buffers.
 * Launch shape: One thread per entropy stream and parallel blocks per pixel plane.
 * Lifetime: One immutable image generation, from start through completion or refusal. */
#ifndef AOTX_MEDIA_IMAGE_CUH
#define AOTX_MEDIA_IMAGE_CUH
#include <stdint.h>
#include <cuda_runtime.h>

#define AOTX_IMAGE_JPEG 1u
#define AOTX_IMAGE_RGB8 2u
#define AOTX_IMAGE_NEW 0u
#define AOTX_IMAGE_HEADER 1u
#define AOTX_IMAGE_ZERO 2u
#define AOTX_IMAGE_SCAN 3u
#define AOTX_IMAGE_IDCT 4u
#define AOTX_IMAGE_READY 5u
#define AOTX_IMAGE_REFUSED 6u
#define AOTX_IMAGE_RAW 7u
#define AOTX_IMAGE_INVALID 1u
#define AOTX_IMAGE_UNSUPPORTED 2u
#define AOTX_IMAGE_LIMIT 3u
#define AOTX_IMAGE_CANCELLED 4u

typedef struct aotx_jpeg_huffman {
    uint16_t count[17], first[17], start[17];
    unsigned char symbol[256];
    uint32_t valid;
} aotx_jpeg_huffman;
typedef struct aotx_jpeg_component {
    uint32_t id, h, v, quant, width, height, cols, rows, stride, base;
    int32_t predictor;
    uint32_t quant_bound;
    uint16_t quant_values[64];
    unsigned char approximation[64];
} aotx_jpeg_component;
typedef struct aotx_image_job {
    const unsigned char *source;
    uint64_t bytes, cursor;
    int32_t *coefficients;
    unsigned char *planes, *rgb;
    uint64_t coefficient_count, plane_bytes, rgb_bytes, pixel_limit;
    uint32_t dimension_limit, format, width, height, phase, status, cancel;
    uint32_t components, hmax, vmax, mcu_cols, mcu_rows, blocks;
    uint32_t progressive, jfif, frame, scans, restart_interval;
    uint32_t scan_count, component[3], dc_table[3], ac_table[3];
    uint32_t ss, se, ah, al, unit, units, eob, restart_next;
    uint32_t bit_byte, bit_count, quant_valid[4];
    uint16_t quant[4][64];
    aotx_jpeg_huffman huffman[2][4];
    aotx_jpeg_component channel[3];
} aotx_image_job;

static __device__ __forceinline__ void aotx_image_refuse(aotx_image_job *job, uint32_t status) {
    job->status = status; job->phase = AOTX_IMAGE_REFUSED;
}
__global__ void aotx_image_step(aotx_image_job *jobs, uint32_t count, uint32_t quantum);
__global__ void aotx_image_zero(aotx_image_job *jobs, uint32_t count);
__global__ void aotx_image_idct(aotx_image_job *jobs, uint32_t count);
__global__ void aotx_image_pixels(aotx_image_job *jobs, uint32_t count);
__global__ void aotx_image_finish(aotx_image_job *jobs, uint32_t count);
void aotx_image_capture(void *stream, aotx_image_job *jobs, uint32_t count, uint32_t quantum);
#endif
