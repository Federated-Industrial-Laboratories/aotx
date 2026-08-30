/* Purpose: Check the region bound of a model file before its tensors move.
 * Owns: Nothing; the model file owns its metadata.
 * Launch shape: Host glue only; no device work runs.
 * Lifetime: One model placement check. */
#include <stdlib.h>

#include "boot/boot.cuh"
#include "mem/mem.cuh"

extern "C" {
#include "disk/modelfile/modelfile.h"
}

#define AOTX_WEIGHTS_PLAN_ALIGN 256ull

static int aotx_weights_plan_readable(unsigned int type)
{
    return type == AOTX_TENSOR_F32 || type == AOTX_TENSOR_F16
        || type == AOTX_TENSOR_Q4_0 || type == AOTX_TENSOR_Q8_0;
}

int aotx_model_weights_fits(struct aotx_modelfile *file, unsigned long long cursor,
                            unsigned long long *end)
{
    unsigned long long count = aotx_modelfile_tensor_count(file);
    for (unsigned long long i = 0ull; i < count; ++i) {
        aotx_tensor_info info;
        if (aotx_modelfile_tensor(file, i, &info) != 0) {
            return 1;
        }
        if (!aotx_weights_plan_readable(info.type)) {
            continue;
        }
        unsigned long long at = (cursor + AOTX_WEIGHTS_PLAN_ALIGN - 1ull)
                              / AOTX_WEIGHTS_PLAN_ALIGN * AOTX_WEIGHTS_PLAN_ALIGN;
        if (at > AOTX_MEM_WEIGHTS_BYTES || info.bytes > AOTX_MEM_WEIGHTS_BYTES - at) {
            return 1;
        }
        cursor = at + info.bytes;
    }
    *end = cursor;
    return 0;
}
