/* Purpose: Give distinct optional sources a turn before repeated representations.
 * Owns: Read-only source counts and complete working-text preference.
 * Launch shape: One selection thread per query after parallel candidate validation.
 * Lifetime: One search; recorded selection replay does not rank again. */
#ifndef AOTX_COGNITIVE_RECALL_DIVERSITY_CUH
#define AOTX_COGNITIVE_RECALL_DIVERSITY_CUH
#include "cognitive/recall_labels.cuh"

__device__ inline uint32_t aotx_recall_source_count(const aotx_recall_result *out,
    const aotx_recall_scratch *scratch, uint32_t source) {
    uint32_t count = 0;
    for (uint32_t j = 0; j < out->count; ++j)
        if (out->reason[j] != AOTX_RECALL_ASSESSMENT && scratch->sources[out->index[j]] == source) ++count;
    return count;
}
__device__ inline bool aotx_recall_complete_working(const aotx_cognitive_store *s,
    uint32_t index, uint32_t source) {
    const unsigned char *r = s->objects[index], *event = s->objects[source];
    uint64_t bytes = aotx_cog_u64(r + AOTX_CO_BYTES);
    return aotx_cog_u16(event + AOTX_CO_KIND) == AOTX_COG_EVENT &&
        bytes == aotx_cog_u64(event + AOTX_CO_BYTES) &&
        aotx_cog_equal(s->payload + aotx_cog_u64(r + AOTX_CO_OFFSET),
            s->payload + aotx_cog_u64(event + AOTX_CO_OFFSET), (uint32_t)bytes);
}
__device__ inline uint32_t aotx_recall_working(const aotx_cognitive_store *s,
    const unsigned char *q, const aotx_recall_result *out, const aotx_recall_scratch *scratch,
    uint32_t best, uint32_t used) {
    uint32_t preferred = UINT32_MAX, cap = aotx_cog_u32(q + 136);
    for (uint32_t j = 0; j < s->count; ++j) {
        const unsigned char *r = s->objects[j];
        if (scratch->states[j] != AOTX_COG_OK || scratch->sources[j] != scratch->sources[best] ||
            aotx_cog_u16(r + AOTX_CO_KIND) != AOTX_COG_WORKING ||
            aotx_cog_u64(r + AOTX_CO_BYTES) < 32 ||
            !aotx_recall_magic(s->payload + aotx_cog_u64(r + AOTX_CO_OFFSET), "AOTXMEM1") ||
            !aotx_recall_complete_working(s, j, scratch->sources[j])) continue;
        bool present = false;
        for (uint32_t k = 0; k < out->count; ++k) if (out->index[k] == j) present = true;
        /* Compact requests check size after the full source group is known. */
        if (present || (!aotx_context_compact(q) &&
            aotx_recall_one(s, j, AOTX_RECALL_SEMANTIC, 0, used, cap, true) > cap)) continue;
        if (preferred == UINT32_MAX || aotx_recall_before(r, s->objects[preferred])) preferred = j;
    }
    return preferred == UINT32_MAX ? best : preferred;
}
#endif
