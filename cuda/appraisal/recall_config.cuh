/* Purpose: Apply current appraisal recall defaults to admitted query batches.
 * Owns: Query extension bytes; stored configuration remains unchanged.
 * Launch shape: One pass over a complete validated live request batch.
 * Lifetime: Admission through exact recorded query recovery. */
#ifndef AOTX_APPRAISAL_RECALL_CONFIG_CUH
#define AOTX_APPRAISAL_RECALL_CONFIG_CUH
#include "appraisal/recall.cuh"

__device__ inline uint32_t aotx_appraisal_recall_config(const aotx_cognitive_store *s) {
    uint32_t best = UINT32_MAX;
    for (uint32_t j = 0; j < s->count; ++j) {
        const unsigned char *r = s->objects[j];
        if (aotx_cog_u16(r + AOTX_CO_KIND) != AOTX_COG_POLICY ||
            aotx_cog_u64(r + AOTX_CO_BYTES) != AOTX_APPRAISAL_CONFIG_BYTES ||
            !aotx_recall_magic(s->payload + aotx_cog_u64(r + AOTX_CO_OFFSET), "AOTXAPC1")) continue;
        if (best == UINT32_MAX || aotx_cog_u64(r + AOTX_CO_UPDATED) > aotx_cog_u64(s->objects[best] + AOTX_CO_UPDATED) ||
            (aotx_cog_u64(r + AOTX_CO_UPDATED) == aotx_cog_u64(s->objects[best] + AOTX_CO_UPDATED) &&
                aotx_recall_before(r, s->objects[best]))) best = j;
    }
    return best;
}
__device__ inline void aotx_appraisal_recall_defaults(const aotx_cognitive_store *s,
    uint32_t config, unsigned char *q) {
    uint32_t flags = aotx_context_flags(q);
    if (config == UINT32_MAX || flags & AOTX_RECALL_APPRAISE) return;
    const unsigned char *r = s->objects[config], *p = s->payload + aotx_cog_u64(r + AOTX_CO_OFFSET);
    if (aotx_appraisal_config_schema(r, p, aotx_cog_u64(r + AOTX_CO_BYTES)) ||
        !aotx_appraisal_contract(p + 40) || !(aotx_cog_u32(p + 12) & AOTX_APPRAISAL_RECALL) ||
        !aotx_appraisal_recall_current(s, q, config)) return;
    unsigned char *c = q + AOTX_RECALL_EXTENSION;
    if (!flags && !aotx_context_sources(q)) {
        for (uint32_t j = 0; j < 8; ++j) c[j] = "AOTXCTX1"[j];
        aotx_cog_put(c + 8, 1, 4); aotx_cog_put(c + 44, 1, 4);
    }
    aotx_cog_put(c + 12, flags | AOTX_RECALL_APPRAISE, 4);
    aotx_cog_put(c + 36, aotx_cog_u32(p + 28), 4);
    aotx_cog_put(c + 40, aotx_cog_u32(p + 32), 4);
}
#endif
