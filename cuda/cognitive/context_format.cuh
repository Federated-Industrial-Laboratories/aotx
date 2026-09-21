/* Purpose: Validate explicit task, participant and appraisal query controls.
 * Owns: The optional query extension; no source interpretation or state changes.
 * Launch shape: One helper call per prepared query.
 * Lifetime: One request and its exact recorded replay. */
#ifndef AOTX_COGNITIVE_CONTEXT_FORMAT_CUH
#define AOTX_COGNITIVE_CONTEXT_FORMAT_CUH
#include "cognitive/recall.h"
#include "cognitive/memory_schema.cuh"

__device__ inline uint32_t aotx_context_flags(const unsigned char *q) {
    return aotx_cog_u32(q + AOTX_RECALL_EXTENSION + 12);
}
__device__ inline bool aotx_context_sources(const unsigned char *q) {
    return aotx_cog_u32(q + AOTX_RECALL_EXTENSION + 8) == 2;
}
__device__ inline uint32_t aotx_context_check(const unsigned char *q) {
    const unsigned char *p = q + AOTX_RECALL_EXTENSION;
    if (aotx_cog_zero(p, AOTX_RECALL_QUERY - AOTX_RECALL_EXTENSION)) return AOTX_COG_OK;
    uint32_t flags = aotx_cog_u32(p + 12), count = aotx_cog_u32(p + 32);
    bool sources = aotx_context_sources(q);
    if (!aotx_cog_equal(p, (const unsigned char *)(sources ? "AOTXCTX2" : "AOTXCTX1"), 8) ||
        aotx_cog_u32(p + 8) != (sources ? 2u : 1u) ||
        (!sources && !flags) || flags & ~3u || count > AOTX_RECALL_SUBJECTS ||
        aotx_cog_u32(p + 44) != (sources ? 2u : 1u) ||
        aotx_cog_u32(p + 36) > AOTX_COG_SCALE || aotx_cog_u32(p + 40) > AOTX_COG_SCALE ||
        (!(flags & AOTX_RECALL_APPRAISE) && !aotx_cog_zero(p + 36, 8)) ||
        (flags & AOTX_RECALL_TASKS ? aotx_cog_zero(p + 16, 16) : !aotx_cog_zero(p + 16, 20)) ||
        !aotx_cog_zero(p + 48 + count * 16, (AOTX_RECALL_SUBJECTS - count) * 16)) return AOTX_COG_FORMAT;
    if (sources ? !aotx_cog_zero(q + AOTX_RECALL_ACTOR + 16, AOTX_RECALL_QUERY - AOTX_RECALL_ACTOR - 16) :
        !aotx_cog_zero(q + AOTX_RECALL_ACTOR, AOTX_RECALL_QUERY - AOTX_RECALL_ACTOR)) return AOTX_COG_FORMAT;
    for (uint32_t i = 0; i < count; ++i) {
        if (aotx_cog_zero(p + 48 + i * 16, 16)) return AOTX_COG_REFERENCE;
        for (uint32_t j = 0; j < i; ++j)
            if (aotx_cog_equal(p + 48 + i * 16, p + 48 + j * 16)) return AOTX_COG_REFERENCE;
    }
    return AOTX_COG_OK;
}
#endif
