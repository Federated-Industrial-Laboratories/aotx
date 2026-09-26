/* Purpose: Validate exact historical dependencies of task review cues.
 * Owns: Read-only payload checks; current access is checked by recall.
 * Launch shape: One validation call for each admitted object in a batch.
 * Lifetime: A staged store, including retained superseded evidence. */
#ifndef AOTX_REFLECTION_SCHEMA_CUH
#define AOTX_REFLECTION_SCHEMA_CUH
#include "reflection/format.h"
#include "appraisal/schema.cuh"

__device__ inline bool aotx_review_kind(const aotx_cognitive_store *s, const unsigned char *r) {
    return !aotx_cog_cold(r) && aotx_cog_u16(r + AOTX_CO_KIND) == AOTX_COG_REVIEW &&
        aotx_cog_u64(r + AOTX_CO_BYTES) >= 8 && aotx_cog_equal(
            s->payload + aotx_cog_u64(r + AOTX_CO_OFFSET), (const unsigned char *)"AOTXMEM4", 8);
}
__device__ inline uint32_t aotx_review_schema(const aotx_cognitive_store *s,
    const unsigned char *r, const unsigned char *p, uint64_t bytes, uint32_t *indices = 0) {
    if (bytes != AOTX_REVIEW_CUE_BYTES || aotx_cog_u32(p + 8) != 4 ||
        aotx_cog_u32(p + 12) != AOTX_REVIEW_TEXT_BYTES || aotx_cog_u32(p + 32) != 1 ||
        !aotx_cog_zero(p + 36, 28) || aotx_cog_zero(p + 16, 16) ||
        aotx_cog_u16(r + AOTX_CO_KIND) != AOTX_COG_REVIEW ||
        aotx_cog_u32(r + AOTX_CO_SOURCE_KIND) != AOTX_COG_INFERRED ||
        !aotx_cog_equal(p + 64, (const unsigned char *)AOTX_REVIEW_TEXT, AOTX_REVIEW_TEXT_BYTES))
        return AOTX_COG_FORMAT;
    int selection = aotx_cog_find(s, r + AOTX_CO_SOURCE, aotx_cog_u64(r + AOTX_CO_SOURCE_VERSION));
    if (selection < 0) return AOTX_COG_REFERENCE;
    const unsigned char *sr = s->objects[selection];
    if (aotx_cog_cold(sr) || aotx_cog_u16(sr + AOTX_CO_KIND) != AOTX_COG_SELECTION ||
        aotx_cog_u64(sr + AOTX_CO_BYTES) != AOTX_REVIEW_SELECTION_BYTES ||
        aotx_cog_u32(sr + AOTX_CO_SOURCE_KIND) != AOTX_COG_INFERRED) return AOTX_COG_SOURCE;
    const unsigned char *sp = s->payload + aotx_cog_u64(sr + AOTX_CO_OFFSET);
    if (aotx_cog_u32(sp) != 1 || aotx_cog_u32(sp + 4) != AOTX_REVIEW_REFERENCES ||
        !aotx_cog_zero(sp + 8, 8)) return AOTX_COG_FORMAT;
    uint32_t group[AOTX_REVIEW_REFERENCES];
    for (uint32_t j = 0; j < AOTX_REVIEW_REFERENCES; ++j) {
        const unsigned char *ref = sp + 16 + j * 32;
        int found = aotx_cog_find(s, ref, aotx_cog_u64(ref + 16));
        if (found < 0 || aotx_cog_u32(ref + 24) != 1 || aotx_cog_u32(ref + 28)) return AOTX_COG_REFERENCE;
        group[j] = (uint32_t)found;
        if (aotx_cog_cold(s->objects[found])) return AOTX_COG_UNAVAILABLE;
        for (uint32_t k = 0; k < j; ++k) if (group[k] == group[j]) return AOTX_COG_REFERENCE;
        if (indices) indices[j] = group[j];
    }
    const unsigned char *source = s->objects[group[0]], *assessment = s->objects[group[1]],
        *queue = s->objects[group[2]], *relation = s->objects[group[3]], *task = s->objects[group[4]];
    const unsigned char *ap = s->payload + aotx_cog_u64(assessment + AOTX_CO_OFFSET),
        *qp = s->payload + aotx_cog_u64(queue + AOTX_CO_OFFSET),
        *rp = s->payload + aotx_cog_u64(relation + AOTX_CO_OFFSET);
    if (aotx_cog_u16(source + AOTX_CO_KIND) != AOTX_COG_EVENT ||
        aotx_cog_u32(source + AOTX_CO_SOURCE_KIND) == AOTX_COG_INFERRED ||
        aotx_cog_u16(assessment + AOTX_CO_KIND) != AOTX_COG_APPRAISAL ||
        aotx_cog_u64(assessment + AOTX_CO_BYTES) != AOTX_APPRAISAL_ASSESS_BYTES ||
        aotx_cog_u16(relation + AOTX_CO_KIND) != AOTX_COG_RELATIONSHIP ||
        aotx_cog_u64(relation + AOTX_CO_BYTES) != AOTX_APPRAISAL_RELATION_BYTES ||
        aotx_cog_u16(queue + AOTX_CO_KIND) != AOTX_COG_POLICY ||
        aotx_cog_u64(queue + AOTX_CO_BYTES) != AOTX_APPRAISAL_QUEUE_BYTES) return AOTX_COG_SOURCE;
    if (!aotx_cog_equal(ap + 96, queue + AOTX_CO_ID) ||
        aotx_cog_u64(ap + 112) != aotx_cog_u64(queue + AOTX_CO_VERSION) ||
        !aotx_cog_equal(rp + 136, ap + 96, 24) || !aotx_cog_equal(rp + 48, ap + 120, 8) ||
        !aotx_cog_equal(qp + 128, task + AOTX_CO_ID) ||
        aotx_cog_u64(qp + 144) != aotx_cog_u64(task + AOTX_CO_VERSION) ||
        !aotx_cog_equal(p + 16, qp + 40) || !aotx_cog_equal(p + 16, rp + 32) ||
        aotx_cog_u32(qp + 12) != AOTX_APPRAISAL_COMPLETE) return AOTX_COG_REFERENCE;
    for (uint32_t j = 1; j < 4; ++j) {
        const unsigned char *v = s->objects[group[j]];
        if (!aotx_cog_equal(v + AOTX_CO_SOURCE, source + AOTX_CO_ID) ||
            aotx_cog_u64(v + AOTX_CO_SOURCE_VERSION) != aotx_cog_u64(source + AOTX_CO_VERSION) ||
            !aotx_cog_equal(v + AOTX_CO_SUBJECT, r + AOTX_CO_SUBJECT) || !aotx_cog_scope(r, v)) return AOTX_COG_SOURCE;
    }
    return aotx_cog_equal(sr + AOTX_CO_SOURCE, source + AOTX_CO_ID) &&
        aotx_cog_u64(sr + AOTX_CO_SOURCE_VERSION) == aotx_cog_u64(source + AOTX_CO_VERSION)
        ? AOTX_COG_OK : AOTX_COG_REFERENCE;
}
#endif
