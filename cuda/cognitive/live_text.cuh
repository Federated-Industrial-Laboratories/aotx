/* Purpose: Validate recorded text preparation against its original request.
 * Owns: The fixed processor identity and inline replay checks.
 * Launch shape: One thread for each recorded query row.
 * Lifetime: Exact recorded choices in a compatible runtime. */
#ifndef AOTX_COGNITIVE_LIVE_TEXT_CUH
#define AOTX_COGNITIVE_LIVE_TEXT_CUH
#include "cognitive/live_validate.cuh"
#include "model/load.cuh"
extern __device__ const unsigned char aotx_live_processor[32];
__device__ __forceinline__ uint32_t aotx_live_text_recorded(const unsigned char *raw, const unsigned char *q) {
    /* A recorded vector cannot replace input, authority, limits, pins or unused bytes. */
    if (!aotx_cog_equal(raw, q, 64) || !aotx_cog_equal(raw + 132, q + 132, 28) ||
        !aotx_cog_equal(raw + 4256, q + 4256, AOTX_RECALL_QUERY - 4256)) return AOTX_COG_REFERENCE;
    const aotx_model_resident_row *r = aotx_model_load.resident + AOTX_MODEL_EMBEDDING;
    if (!r->active || r->slot != AOTX_MODEL_EMBEDDING ||
        !aotx_cog_equal(q + 64, r->body.digest, 32) || !aotx_cog_equal(q + 96, aotx_live_processor, 32) ||
        aotx_cog_u32(q + 128) != aotx_model[AOTX_MODEL_EMBEDDING].hidden) return AOTX_COG_LAYOUT;
    return aotx_recall_query_check(q);
}
#endif
