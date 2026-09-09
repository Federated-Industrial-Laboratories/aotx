/* Purpose: Add the binding's bounded working set to an explicit query.
 * Owns: Request scratch only; the persistent set changes on retained decisions.
 * Launch shape: Serial admission of at most 64 queries and eight references each.
 * Lifetime: One admitted query batch. */
#ifndef AOTX_COGNITIVE_LIVE_FOCUS_CUH
#define AOTX_COGNITIVE_LIVE_FOCUS_CUH
#include "cognitive/live_validate.cuh"

__device__ inline uint32_t aotx_live_focus_query(const unsigned char *prefix, unsigned char *q) {
    if (!aotx_cog_u32(prefix + 4)) return AOTX_COG_OK;
    const aotx_live_binding *b = aotx_live_bindings + aotx_cog_u32(prefix);
    uint32_t count = aotx_cog_u32(q + 144);
    for (uint32_t i = 0; i < b->focus_count; ++i) {
        bool seen = false;
        for (uint32_t j = 0; j < count; ++j)
            if (aotx_cog_equal(q + 4448 + j * 24, b->focus[i], 24)) seen = true;
        if (seen) continue;
        if (count == AOTX_RECALL_PINS) return AOTX_COG_CAPACITY;
        for (uint32_t j = 0; j < 24; ++j) q[4448 + count * 24 + j] = b->focus[i][j];
        ++count;
    }
    aotx_cog_put(q + 144, count, 4);
    return AOTX_COG_OK;
}
#endif
