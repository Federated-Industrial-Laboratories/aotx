/* Purpose: Read the selected tool call form during device work.
 * Owns: One bounded call form for each model role.
 * Launch shape: One thread for each reply or prompt in a batch.
 * Lifetime: From model load to model release. */
#ifndef AOTX_MODEL_CALL_FORMAT_CUH
#define AOTX_MODEL_CALL_FORMAT_CUH

#include "model/wrap.cuh"
#include "disk/modelfile/call_format.h"

extern __device__ aotx_call_format aotx_model_call_format[AOTX_MODEL_ROLES];

__device__ __forceinline__ const aotx_call_format *aotx_call_format_active(void)
{
    unsigned int role = aotx_model[AOTX_MODEL_LANGUAGE].layers != 0u
                      ? AOTX_MODEL_LANGUAGE : AOTX_MODEL_LANGUAGE_Q4;
    return &aotx_model_call_format[role];
}

__device__ __forceinline__ unsigned int aotx_call_format_put(
    unsigned char *out, unsigned int at, unsigned int capacity,
    const aotx_call_format *format, unsigned int span)
{
    if (format->kind >= AOTX_CALL_FORMAT_KINDS || span >= AOTX_CALL_FORMAT_SPANS
        || format->offset[span] > AOTX_CALL_FORMAT_BYTES
        || format->length[span] > AOTX_CALL_FORMAT_SPAN_BYTES
        || format->length[span] > AOTX_CALL_FORMAT_BYTES - format->offset[span]) {
        return capacity + 1u;
    }
    if (at > capacity || format->length[span] > capacity - at) return capacity + 1u;
    if (out == 0) return at + format->length[span];
    return aotx_wrap_run(out, at, capacity, format->bytes + format->offset[span],
                         format->length[span]);
}

#endif
