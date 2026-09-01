/* Purpose: Hold and change one sampling row for each agent.
 * Owns: The sampler table in device memory.
 * Launch shape: Device functions; the command and agent paths call them for one row.
 * Lifetime: The whole run; a new agent starts from the settings defaults. */
#ifndef AOTX_MODEL_SAMPLER_CUH
#define AOTX_MODEL_SAMPLER_CUH

#include "model/forward.cuh"
#include "profile/profile.cuh"

#define AOTX_SAMPLER_TOOK    0u
#define AOTX_SAMPLER_UNKNOWN 1u
#define AOTX_SAMPLER_VALUE   2u
#define AOTX_SAMPLER_RANGE   3u

typedef struct aotx_sampler_table {
    aotx_model_how row[AOTX_SLOTS];
    unsigned int changed[AOTX_SLOTS];
    unsigned int refused;
} aotx_sampler_table;

extern __device__ aotx_sampler_table aotx_sampler;

/* Reset one row to the defaults that the settings file gave the device. */
__device__ void aotx_sampler_reset(unsigned int agent);

/* Set one field from command text. The return is AOTX_SAMPLER_*. */
__device__ unsigned int aotx_sampler_set(unsigned int agent, const char *key,
                                         unsigned int key_len, const char *value,
                                         unsigned int value_len);

#endif
