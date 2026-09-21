/* Purpose: Check exact model and processor qualification for automatic memory.
 * Owns: Immutable qualification entries and the shared admission predicate.
 * Launch shape: One bounded check per model role or source row.
 * Lifetime: New inference only; saved choices retain their replay checks. */
#ifndef AOTX_COGNITIVE_INTAKE_CAPABILITY_CUH
#define AOTX_COGNITIVE_INTAKE_CAPABILITY_CUH
#include "cognitive/codec.cuh"
#include "cognitive/intake.cuh"
#include "disk/modelfile/wrap.h"
#include "model/load.cuh"
#include "model/wrap.cuh"
struct aotx_intake_capability {
    unsigned char model[32], statement[32], source[32], profile[32];
    aotx_wrap wrapper;
};
__device__ __forceinline__ bool aotx_intake_capability_match(const aotx_intake_capability *entry,
    const unsigned char *model, const unsigned char *statement, const unsigned char *source,
    const unsigned char *profile, const aotx_wrap *wrapper) {
    return !aotx_cog_zero(entry->model, 32) && wrapper->usable &&
        aotx_cog_equal(entry->model, model, 32) && aotx_cog_equal(entry->statement, statement, 32) &&
        aotx_cog_equal(entry->source, source, 32) && aotx_cog_equal(entry->profile, profile, 32) &&
        aotx_cog_equal((const unsigned char *)&entry->wrapper, (const unsigned char *)wrapper, sizeof(*wrapper));
}
#define AOTX_INTAKE_CAPABILITIES 8u
extern __constant__ aotx_intake_capability aotx_intake_capabilities[AOTX_INTAKE_CAPABILITIES];
static __device__ const unsigned char aotx_intake_empty_identity[32] = {};
__device__ __forceinline__ bool aotx_intake_qualified(unsigned role, bool sources = true) {
    if (role >= AOTX_MODEL_ROLES || !aotx_model_is_language(role) || aotx_model_load.pending_count ||
        !aotx_model_load.resident[role].active || !aotx_wrap_active(role)->usable) return false;
    for (unsigned i = 0; i < AOTX_INTAKE_CAPABILITIES; ++i)
        if (aotx_intake_capability_match(aotx_intake_capabilities + i,
            aotx_model_load.resident[role].body.digest,
            sources ? aotx_intake_statement_processor : aotx_intake_empty_identity,
            sources ? aotx_intake_source_processor : aotx_intake_processor,
            sources ? aotx_source_profile_digest : aotx_intake_empty_identity, aotx_wrap_active(role))) return true;
    return false;
}
#endif
