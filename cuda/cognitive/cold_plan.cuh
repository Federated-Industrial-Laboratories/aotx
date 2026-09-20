/* Purpose: Check residency batches and compute their complete resident layout.
 * Owns: Device selection, scope checks and fit decisions; no disk operations.
 * Launch shape: One block selects a bounded object batch before parallel copies.
 * Lifetime: One quiescent residency operation. */
#ifndef AOTX_COGNITIVE_COLD_PLAN_CUH
#define AOTX_COGNITIVE_COLD_PLAN_CUH
#include "cognitive/cold.h"
#include "cognitive/validate.cuh"
#include "cognitive/lookup.cuh"

__device__ inline void aotx_cold_mark_ref(const aotx_cognitive_store *s, uint32_t *marks,
    const unsigned char *id, uint64_t version) {
    int i = aotx_cog_find(s, id, version);
    if (i >= 0) marks[i] = 1;
}
__device__ inline void aotx_cold_dependencies(const aotx_cognitive_store *s,
    const unsigned char *r, uint32_t *marks) {
    const uint32_t refs[3] = {AOTX_CO_SOURCE, AOTX_CO_SUPERSEDES, AOTX_CO_EMBEDDING};
    for (uint32_t k = 0; k < 3; ++k)
        aotx_cold_mark_ref(s, marks, r + refs[k], aotx_cog_u64(r + refs[k] + 16));
    if (aotx_cog_cold(r) || !aotx_cog_u64(r + AOTX_CO_BYTES)) return;
    const unsigned char *p = s->payload + aotx_cog_u64(r + AOTX_CO_OFFSET);
    for (uint32_t k = 0; k < 2; ++k) {
        const unsigned char *ref = aotx_appraisal_reference(r, p, aotx_cog_u64(r + AOTX_CO_BYTES), k);
        if (ref) aotx_cold_mark_ref(s, marks, ref, aotx_cog_u64(ref + 16));
    }
    if (aotx_cog_u16(r + AOTX_CO_KIND) == AOTX_COG_SELECTION)
        for (uint32_t k = 0; k < aotx_cog_u32(p + 4); ++k)
            aotx_cold_mark_ref(s, marks, p + 16 + k * 32, aotx_cog_u64(p + 32 + k * 32));
    if (aotx_cog_u16(r + AOTX_CO_KIND) == AOTX_COG_EVENT &&
        aotx_cog_u64(r + AOTX_CO_BYTES) == 16 + AOTX_RECALL_QUERY &&
        aotx_cog_equal(p, (const unsigned char *)"AOTXQUE1", 8)) {
        const unsigned char *q = p + 16;
        for (uint32_t group = 0; group < 2; ++group)
            for (uint32_t k = 0; k < aotx_cog_u32(q + 140 + group * 4) && k < AOTX_RECALL_PINS; ++k) {
                const unsigned char *ref = q + 4256 + group * 192 + k * 24;
                aotx_cold_mark_ref(s, marks, ref, aotx_cog_u64(ref + 16));
            }
    }
}
/* A selected source cannot remove bytes required to validate a resident object. */
__device__ inline uint32_t aotx_cold_closure(const aotx_cognitive_store *s,
    const uint32_t *selected, uint32_t *needed) {
    for (uint32_t i = 0; i < s->count; ++i) needed[i] = 0;
    for (uint32_t i = 0; i < s->count; ++i)
        if (!selected[i] && !aotx_cog_cold(s->objects[i])) aotx_cold_dependencies(s, s->objects[i], needed);
    for (uint32_t i = 0; i < s->count; ++i) if (needed[i] && selected[i]) return AOTX_COG_REFERENCE;
    return 0;
}
__device__ inline uint32_t aotx_cold_select(const aotx_cognitive_store *s,
    const unsigned char *input, uint32_t bytes, uint32_t *selected, uint32_t *need, uint32_t *done) {
    if (bytes < AOTX_COLD_HEADER || !aotx_cog_equal(input, (const unsigned char *)"AOTXTIR1", 8) ||
        aotx_cog_u32(input + 8) != 1 || aotx_cog_u32(input + 20) != AOTX_COLD_ROW ||
        !aotx_cog_zero(input + 48, 16)) return AOTX_COG_FORMAT;
    uint32_t mode = aotx_cog_u32(input + 12), count = aotx_cog_u32(input + 16);
    if (mode < AOTX_COLD_OFFLOAD || mode > AOTX_COLD_ENABLE || count > AOTX_COLD_BATCH ||
        bytes != AOTX_COLD_HEADER + count * AOTX_COLD_ROW ||
        ((mode <= AOTX_COLD_FETCH) != (count != 0))) return AOTX_COG_FORMAT;
    if (!aotx_cog_equal(input + 24, s->lineage)) return AOTX_COG_SOURCE;
    if (aotx_cog_u64(input + 40) != s->sequence) return AOTX_COG_STALE;
    if (!s->tiered && (mode == AOTX_COLD_OFFLOAD || mode == AOTX_COLD_FETCH)) return AOTX_COG_DENIED;
    for (uint32_t i = 0; i < s->count; ++i) selected[i] = mode == AOTX_COLD_GPU && aotx_cog_cold(s->objects[i]);
    for (uint32_t k = 0; k < count; ++k) {
        const unsigned char *p = input + AOTX_COLD_HEADER + k * AOTX_COLD_ROW;
        if (!aotx_cog_zero(p + 56, 8) || aotx_cog_zero(p + 24, 16)) return AOTX_COG_FORMAT;
        aotx_cognitive_query q = {};
        for (uint32_t j = 0; j < 16; ++j) { q.id[j] = p[j]; q.principal[j] = p[24 + j]; q.room[j] = p[40 + j]; }
        q.version = aotx_cog_u64(p + 16);
        int i = aotx_cog_find(s, q.id, q.version);
        if (i < 0) return AOTX_COG_MISSING;
        int current = aotx_cog_latest(s, q.id);
        if (current < 0 || !aotx_cog_visible(s->objects[current], &q, true, s->sequence)) return AOTX_COG_DENIED;
        if (current != i || aotx_cog_superseded(s, s->objects[i])) return AOTX_COG_STALE;
        for (uint32_t prior = 0; prior < k; ++prior)
            if (aotx_cog_equal(input + AOTX_COLD_HEADER + prior * AOTX_COLD_ROW, p, 24)) return AOTX_COG_FORMAT;
        if (mode == AOTX_COLD_OFFLOAD && (!aotx_cog_offloadable(s, s->objects[i]) || aotx_cog_cold(s->objects[i])))
            return AOTX_COG_DENIED;
        if (mode == AOTX_COLD_FETCH && !aotx_cog_cold(s->objects[i])) return AOTX_COG_DENIED;
        uint32_t status = aotx_cog_dependencies_scratch(s, &q, i, true, s->sequence, need, done, false);
        if (status) return status;
        selected[i] = 1;
        if (mode == AOTX_COLD_FETCH)
            for (uint32_t j = 0; j < s->count; ++j)
                if ((done[j / 32] & (1u << (j % 32))) && aotx_cog_cold(s->objects[j])) selected[j] = 1;
    }
    uint64_t total = s->bytes;
    if (mode == AOTX_COLD_FETCH || mode == AOTX_COLD_GPU)
        for (uint32_t i = 0; i < s->count; ++i) if (selected[i]) {
            uint64_t n = aotx_cog_u64(s->objects[i] + AOTX_CO_BYTES);
            if (n > AOTX_COG_PAYLOAD - total) return AOTX_COG_CAPACITY;
            total += n;
        }
    return 0;
}
#endif
