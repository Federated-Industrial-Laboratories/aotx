/* Purpose: Check fitted controls against the current resident model and turn format.
 * Owns: The exact identity attached to each loaded control table.
 * Launch shape: One thread checks an identity before a batch uses its control rows.
 * Lifetime: From registration to model release; a model change invalidates the match. */
#ifndef AOTX_MODEL_CONTROL_CUH
#define AOTX_MODEL_CONTROL_CUH
#include "disk/runtime/control.h"
#include "model/load.cuh"
#include "model/wrap.cuh"
__device__ __forceinline__ int aotx_control_matches(const aotx_control_identity *identity,
    unsigned role) {
    if (role >= AOTX_MODEL_ROLES || !aotx_model_load.resident[role].active || !aotx_model_wrap[role].usable) return 0;
    for (unsigned i = 0; i < 32; ++i)
        if (identity->model[i] != aotx_model_load.resident[role].body.digest[i]) return 0;
    const unsigned char *left = (const unsigned char *)&identity->wrap;
    const unsigned char *right = (const unsigned char *)&aotx_model_wrap[role];
    for (unsigned i = 0; i < offsetof(aotx_wrap, usable); ++i) if (left[i] != right[i]) return 0;
    return 1;
}
unsigned aotx_control_current(aotx_control_identity *identity);
__global__ void aotx_control_values_check(const float *, unsigned long long, unsigned *);
int aotx_control_values(const float *values, unsigned long long count);
int aotx_control_check(const char *store, const char *name, unsigned kind, FILE *asset, unsigned *positions = 0);
int aotx_control_save(const char *path, unsigned kind, unsigned positions = AOTX_CONTROL_ALL);
#endif
