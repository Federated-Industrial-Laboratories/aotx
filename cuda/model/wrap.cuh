/* Purpose: Render model turn spans and check them at load.
 * Owns: One bounded wrap table for each model role.
 * Launch shape: One thread for each prompt in a batch.
 * Lifetime: From model load to model release. */
#ifndef AOTX_MODEL_WRAP_CUH
#define AOTX_MODEL_WRAP_CUH

#include "model/model.cuh"
#include "disk/modelfile/wrap.h"

extern __device__ aotx_wrap aotx_model_wrap[AOTX_MODEL_ROLES];

__device__ __forceinline__ const aotx_wrap *aotx_wrap_active(void)
{
    unsigned int role = aotx_model[AOTX_MODEL_LANGUAGE].layers != 0u
                      ? AOTX_MODEL_LANGUAGE : AOTX_MODEL_LANGUAGE_Q4;
    return &aotx_model_wrap[role];
}

__device__ __forceinline__ unsigned int aotx_wrap_run(
    unsigned char *out, unsigned int at, unsigned int capacity,
    const unsigned char *bytes, unsigned int length)
{
    if (at > capacity || length > capacity - at) return capacity + 1u;
    for (unsigned int i = 0u; i < length; ++i) out[at++] = bytes[i];
    return at;
}

__device__ __forceinline__ unsigned int aotx_wrap_put(
    unsigned char *out, unsigned int at, unsigned int capacity,
    const aotx_wrap *wrap, unsigned int span)
{
    return aotx_wrap_run(out, at, capacity, wrap->bytes + wrap->offset[span],
                         wrap->length[span]);
}

__device__ __forceinline__ unsigned int aotx_wrap_prefix(
    unsigned char *out, unsigned int at, unsigned int capacity, const aotx_wrap *wrap)
{
    return aotx_wrap_run(out, at, capacity,
                         wrap->bytes + wrap->offset[AOTX_WRAP_SYSTEM_HEAD],
                         wrap->prefix_length);
}

__device__ __forceinline__ unsigned int aotx_wrap_generation(
    unsigned char *out, unsigned int at, unsigned int capacity, const aotx_wrap *wrap)
{
    at = aotx_wrap_put(out, at, capacity, wrap, AOTX_WRAP_GENERATION_HEAD);
    at = aotx_wrap_put(out, at, capacity, wrap, AOTX_WRAP_THINK_OPEN);
    return aotx_wrap_put(out, at, capacity, wrap, AOTX_WRAP_THINK_CLOSE);
}

__device__ __forceinline__ int aotx_wrap_end(unsigned int role, unsigned int token)
{
    const aotx_wrap *wrap = &aotx_model_wrap[role];
    for (unsigned int i = 0u; i < wrap->end_count; ++i)
        if (wrap->end_ids[i] == token) return 1;
    return 0;
}

__device__ __forceinline__ int aotx_wrap_think_open(unsigned int role, unsigned int token)
{
    const aotx_wrap *wrap = &aotx_model_wrap[role];
    return wrap->length[AOTX_WRAP_THINK_OPEN] != 0u && token == wrap->think_open_id;
}

__device__ __forceinline__ int aotx_wrap_think_close(unsigned int role, unsigned int token)
{
    const aotx_wrap *wrap = &aotx_model_wrap[role];
    return wrap->length[AOTX_WRAP_THINK_CLOSE] != 0u && token == wrap->think_close_id;
}

/* The check compares the device table with the disk table, then runs a short prefill.
 * Failure disables prompt use, but leaves the descriptor and weights loaded. */
int aotx_model_wrap_check(unsigned int role, const aotx_wrap *expected, const char *file);

#endif
