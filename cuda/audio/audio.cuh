/* Purpose: Execute batches of sound encoders in finite graph steps.
 * Owns: Callers own source, sample, spectral, weight and feature storage.
 * Launch shape: Independent jobs in grid planes; samples and rows run in parallel.
 * Lifetime: From admission until completion or refusal. */
#ifndef AOTX_AUDIO_CUH
#define AOTX_AUDIO_CUH
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cufft.h>
#include "audio/format.h"
enum aotx_audio_phase {
    AOTX_AUDIO_NEW, AOTX_AUDIO_DECODE, AOTX_AUDIO_RESAMPLE, AOTX_AUDIO_MEL,
    AOTX_AUDIO_FIRST, AOTX_AUDIO_SECOND, AOTX_AUDIO_PREPARE, AOTX_AUDIO_ATTEND,
    AOTX_AUDIO_BLOCK, AOTX_AUDIO_POOL, AOTX_AUDIO_READY, AOTX_AUDIO_REFUSED
};
enum aotx_audio_status {
    AOTX_AUDIO_INVALID = 1, AOTX_AUDIO_LIMIT, AOTX_AUDIO_CANCELLED,
    AOTX_AUDIO_NONFINITE, AOTX_AUDIO_NO_SIGNAL
};
struct aotx_audio_job {
    const unsigned char *source;
    unsigned long long source_bytes, data_offset, data_bytes;
    unsigned format, encoding, rate, channels, source_frames, source_capacity;
    unsigned samples, frames, keys, rows, feature_capacity;
    float *decoded, *samples_out, *fft_input, *mel;
    cufftComplex *spectrum;
    float *conv, *residual, *product, *qkv, *features;
    float *input;
    unsigned phase, status, cancel, layer, query, peak;
    float log_peak;
};
struct aotx_audio_coefficients {
    float *resample441, *resample48, *mel;
};
void aotx_audio_capture(cudaStream_t, aotx_audio_job *, unsigned,
    const unsigned char *, const aotx_audio_desc *, const aotx_audio_coefficients *,
    const cufftHandle *, float *const *, cufftComplex *const *, unsigned);
__global__ void aotx_audio_coefficients_make(aotx_audio_coefficients);
__global__ void aotx_audio_step(aotx_audio_job *, unsigned);
__global__ void aotx_audio_finish(aotx_audio_job *, unsigned, unsigned);
__global__ void aotx_audio_decode(aotx_audio_job *, unsigned);
__global__ void aotx_audio_resample(aotx_audio_job *, unsigned, aotx_audio_coefficients);
__global__ void aotx_audio_window(aotx_audio_job *, unsigned);
__global__ void aotx_audio_mel(aotx_audio_job *, unsigned, aotx_audio_coefficients);
__global__ void aotx_audio_log(aotx_audio_job *, unsigned);
__global__ void aotx_audio_floor(aotx_audio_job *, unsigned);
__global__ void aotx_audio_convolution(aotx_audio_job *, unsigned, unsigned);
__global__ void aotx_audio_product(aotx_audio_job *, unsigned, const unsigned char *,
    const aotx_audio_desc *, unsigned);
__global__ void aotx_audio_element(aotx_audio_job *, unsigned, const unsigned char *,
    const aotx_audio_desc *, unsigned);
__global__ void aotx_audio_norm(aotx_audio_job *, unsigned, const unsigned char *,
    const aotx_audio_desc *, unsigned);
__global__ void aotx_audio_attention(aotx_audio_job *, unsigned, unsigned);
__device__ bool aotx_audio_header(aotx_audio_job &);
__device__ __forceinline__ static float aotx_audio_sum(float v)
{
    for (unsigned n = 16; n; n >>= 1) v += __shfl_xor_sync(0xffffffffu, v, n);
    return v;
}
__device__ __forceinline__ static float aotx_audio_finite(aotx_audio_job &j, float v)
{
    if (!isfinite(v)) { atomicCAS(&j.status, 0u, AOTX_AUDIO_NONFINITE); return 0.0f; }
    return v;
}
__device__ __forceinline__ static void aotx_audio_input(aotx_audio_job &j, unsigned at, float v)
{
    j.input[at] = aotx_audio_finite(j, v);
}
#endif
