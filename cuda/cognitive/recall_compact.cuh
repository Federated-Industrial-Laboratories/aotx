/* Purpose: Count and render one copy of an exact selected source body.
 * Owns: Bounded selection sizing; all source and evidence references remain.
 * Launch shape: One selection or rendering thread per query.
 * Lifetime: One query with rendering revision three or four. */
#ifndef AOTX_COGNITIVE_RECALL_COMPACT_CUH
#define AOTX_COGNITIVE_RECALL_COMPACT_CUH
#include "cognitive/recall_groups.cuh"

__device__ inline bool aotx_recall_body_reference(const aotx_cognitive_store *s,
    const unsigned char *q, uint32_t index, const uint32_t *indices, uint32_t count) {
    if (!aotx_context_compact(q)) return false;
    const unsigned char *r = s->objects[index];
    if (aotx_cog_cold(r) || aotx_cog_u16(r + AOTX_CO_KIND) != AOTX_COG_WORKING) return false;
    uint64_t bytes = aotx_cog_u64(r + AOTX_CO_BYTES);
    const unsigned char *p = s->payload + aotx_cog_u64(r + AOTX_CO_OFFSET);
    if (bytes <= 52 || !aotx_recall_magic(p, "AOTXMEM1")) return false;
    for (uint32_t j = 0; j < count; ++j) {
        const unsigned char *event = s->objects[indices[j]];
        if (aotx_cog_cold(event) || aotx_cog_u16(event + AOTX_CO_KIND) != AOTX_COG_EVENT ||
            !aotx_cog_equal(r + AOTX_CO_SOURCE, event + AOTX_CO_ID) ||
            aotx_cog_u64(r + AOTX_CO_SOURCE_VERSION) != aotx_cog_u64(event + AOTX_CO_VERSION) ||
            bytes != aotx_cog_u64(event + AOTX_CO_BYTES)) continue;
        return aotx_cog_equal(p, s->payload + aotx_cog_u64(event + AOTX_CO_OFFSET), (uint32_t)bytes);
    }
    return false;
}
__device__ inline uint32_t aotx_recall_compact_reason(const aotx_cognitive_store *s,
    uint32_t index, uint32_t reason, const uint32_t *indices, uint32_t count) {
    if (reason == AOTX_RECALL_REQUIRED || reason == AOTX_RECALL_FOCUS || reason == AOTX_RECALL_OBLIGATION) return reason;
    const unsigned char *r = s->objects[index];
    if (aotx_cog_u16(r + AOTX_CO_KIND) == AOTX_COG_APPRAISAL || aotx_appraisal_recall_kind(s, r))
        return AOTX_RECALL_ASSESSMENT;
    for (uint32_t j = 0; j < count; ++j)
        if (aotx_recall_assesses(s, r, s->objects[indices[j]])) return AOTX_RECALL_SIGNIFICANT;
    return AOTX_RECALL_SEMANTIC;
}
/* Count the full prospective selection because a new source can replace an earlier body. */
__device__ inline uint32_t aotx_recall_compact_size(const aotx_cognitive_store *s,
    const unsigned char *q, const uint32_t *indices, const uint32_t *reasons, uint32_t count) {
    uint32_t cap = aotx_cog_u32(q + 136), at = aotx_recall_context_label(q, 0, 0, cap);
    uint32_t groups[AOTX_RECALL_LIMIT];
    at = aotx_recall_group_labels(s, q, indices, count, groups, 0, at, cap);
    for (uint32_t j = 0; j < count; ++j) {
        uint32_t reason = aotx_recall_compact_reason(s, indices[j], reasons[j], indices, count);
        at = aotx_recall_one(s, indices[j], reason, 0, at, cap, true,
            aotx_recall_body_reference(s, q, indices[j], indices, count), groups[j]);
    }
    return at;
}
/* A complete optional group enters together after its exact rendered size fits. */
__device__ inline uint32_t aotx_recall_compact_add(const aotx_cognitive_store *s,
    const unsigned char *q, const uint32_t *group, const uint32_t *group_reasons,
    uint32_t total, uint32_t *used, aotx_recall_result *out) {
    uint32_t indices[AOTX_RECALL_LIMIT], reasons[AOTX_RECALL_LIMIT], count = out->count;
    for (uint32_t j = 0; j < count; ++j) { indices[j] = out->index[j]; reasons[j] = out->reason[j]; }
    for (uint32_t j = 0; j < total; ++j) {
        bool present = false;
        for (uint32_t k = 0; k < count; ++k) if (indices[k] == group[j]) present = true;
        if (present) continue;
        if (count >= aotx_cog_u32(q + 132) || count == AOTX_RECALL_LIMIT) return AOTX_COG_CAPACITY;
        if (!aotx_recall_applicable(s, q, s->objects[group[j]])) return AOTX_COG_DENIED;
        indices[count] = group[j]; reasons[count++] = group_reasons[j];
    }
    uint32_t after = aotx_recall_compact_size(s, q, indices, reasons, count);
    if (after > aotx_cog_u32(q + 136)) return AOTX_COG_CAPACITY;
    for (uint32_t j = 0; j < count; ++j) {
        const unsigned char *r = s->objects[indices[j]];
        unsigned char *entry = out->selection + 16 + j * 32;
        for (uint32_t k = 0; k < 16; ++k) entry[k] = r[AOTX_CO_ID + k];
        aotx_cog_put(entry + 16, aotx_cog_u64(r + AOTX_CO_VERSION), 8); aotx_cog_put(entry + 24, 1, 4);
        out->index[j] = indices[j];
        out->reason[j] = aotx_recall_compact_reason(s, indices[j], reasons[j], indices, count);
    }
    out->count = count; *used = after;
    return AOTX_COG_OK;
}
#endif
