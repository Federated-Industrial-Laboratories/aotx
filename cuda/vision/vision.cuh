/* Purpose: Execute batches of image encoders in finite graph steps.
 * Owns: The device caller owns the job, weights, source, output and work spans.
 * Launch shape: One independent job per grid plane; rows run in parallel.
 * Lifetime: From job admission until completion or refusal. */
#ifndef AOTX_VISION_CUH
#define AOTX_VISION_CUH
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "vision/format.h"

enum aotx_vision_phase {
    AOTX_VISION_NEW, AOTX_VISION_RESIZE_H, AOTX_VISION_RESIZE_V,
    AOTX_VISION_PATCH, AOTX_VISION_PREPARE, AOTX_VISION_ATTEND,
    AOTX_VISION_BLOCK, AOTX_VISION_MERGE, AOTX_VISION_READY, AOTX_VISION_REFUSED
};
enum aotx_vision_status {
    AOTX_VISION_INVALID = 1, AOTX_VISION_LIMIT, AOTX_VISION_CANCELLED,
    AOTX_VISION_NONFINITE
};
struct aotx_vision_job {
    const unsigned char *source;
    unsigned long long source_bytes;
    unsigned width, height, max_pixels, patch_capacity;
    float *horizontal;
    unsigned long long horizontal_values;
    unsigned char *rgb;
    unsigned long long rgb_bytes;
    float *residual; /* patch_capacity * 768 values */
    float *product;  /* patch_capacity * 3072 values */
    half *input;     /* patch_capacity * 3072 values */
    half *input_low; /* patch_capacity * 3072 residual values */
    float *qkv;      /* patch_capacity * 2304 values */
    float *features;
    unsigned feature_capacity; /* rows of 1024 values */
    unsigned phase, status, cancel;
    unsigned resized_width, resized_height, patches, rows, layer, query;
};

/* All weight offsets have passed the disk reader before this call. */
void aotx_vision_capture(cudaStream_t on, aotx_vision_job *jobs, unsigned count,
                         const unsigned char *weights, const aotx_vision_desc *desc,
                         unsigned patch_capacity, unsigned query_quantum);
__global__ void aotx_vision_step(aotx_vision_job *, unsigned);
__global__ void aotx_vision_finish(aotx_vision_job *, unsigned, unsigned);
__global__ void aotx_vision_resize(aotx_vision_job *, unsigned, unsigned);
__global__ void aotx_vision_patch(aotx_vision_job *, unsigned);
__global__ void aotx_vision_product(aotx_vision_job *, unsigned, const unsigned char *,
                                    const aotx_vision_desc *, unsigned, unsigned);
__global__ void aotx_vision_position(aotx_vision_job *, unsigned, const unsigned char *,
                                     const aotx_vision_desc *, unsigned);
__global__ void aotx_vision_norm(aotx_vision_job *, unsigned, const unsigned char *,
                                 const aotx_vision_desc *, unsigned);
__global__ void aotx_vision_rope(aotx_vision_job *, unsigned, const unsigned char *,
                                 const aotx_vision_desc *);
__global__ void aotx_vision_attention(aotx_vision_job *, unsigned, unsigned);
__global__ void aotx_vision_activation(aotx_vision_job *, unsigned, const unsigned char *,
                                       const aotx_vision_desc *, unsigned);
__global__ void aotx_vision_residual(aotx_vision_job *, unsigned, const unsigned char *,
                                     const aotx_vision_desc *, unsigned);

__device__ __forceinline__ static void aotx_vision_xy(unsigned p, unsigned width,
                                                     unsigned &y, unsigned &x)
{
    unsigned group = p / 4u, columns = width / 32u;
    y = (group / columns) * 2u + (p % 4u) / 2u;
    x = (group % columns) * 2u + p % 2u;
}
__device__ __forceinline__ static float aotx_vision_sum(float v)
{
    for (unsigned step = 16; step; step >>= 1)
        v += __shfl_xor_sync(0xffffffffu, v, step);
    return v;
}
__device__ __forceinline__ static float aotx_vision_finite(aotx_vision_job &j, float v)
{
    if (!isfinite(v)) { atomicCAS(&j.status, 0u, AOTX_VISION_NONFINITE); return 0.0f; }
    return v;
}
/* Two half terms retain activation precision while the products use tensor cores. */
__device__ __forceinline__ static void aotx_vision_input(aotx_vision_job &j, unsigned at, float v)
{
    v = aotx_vision_finite(j, v);
    half high = __float2half(v);
    if (!isfinite(__half2float(high))) {
        atomicCAS(&j.status, 0u, AOTX_VISION_NONFINITE); high = __float2half(0.0f); v = 0.0f;
    }
    j.input[at] = high;
    j.input_low[at] = __float2half(v - __half2float(high));
}
#endif
