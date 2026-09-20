/* Purpose: Resolve contextual obligations and exact appraisal sources.
 * Owns: Read-only task, subject and version predicates.
 * Launch shape: Helpers used by all rows of a recall batch.
 * Lifetime: One immutable store cut through selection and replay. */
#ifndef AOTX_COGNITIVE_RECALL_CONTEXTUAL_CUH
#define AOTX_COGNITIVE_RECALL_CONTEXTUAL_CUH
#include "cognitive/recall_format.cuh"

__device__ inline bool aotx_recall_contextual(const aotx_cognitive_store *s, const unsigned char *r) {
    return !aotx_cog_cold(r) && aotx_cog_u64(r + AOTX_CO_BYTES) >= 8 &&
        aotx_recall_magic(s->payload + aotx_cog_u64(r + AOTX_CO_OFFSET), "AOTXMEM2");
}
__device__ inline bool aotx_recall_applicable(const aotx_cognitive_store *s, const unsigned char *q, const unsigned char *r) {
    if (!aotx_recall_contextual(s, r)) return true;
    const unsigned char *c = q + AOTX_RECALL_EXTENSION, *p = s->payload + aotx_cog_u64(r + AOTX_CO_OFFSET);
    if (!(aotx_context_flags(q) & AOTX_RECALL_TASKS) || !aotx_cog_equal(c + 16, p + 16) ||
        (aotx_cog_u16(r + AOTX_CO_KIND) == AOTX_COG_INTENTION && aotx_cog_u32(r + AOTX_CO_RETENTION) != 2)) return false;
    if (aotx_cog_zero(r + AOTX_CO_SUBJECT, 16)) return true;
    for (uint32_t j = 0; j < aotx_cog_u32(c + 32); ++j)
        if (aotx_cog_equal(r + AOTX_CO_SUBJECT, c + 48 + j * 16)) return true;
    return false;
}
__device__ inline bool aotx_recall_obligatory(const aotx_cognitive_store *s, const unsigned char *q, const unsigned char *r) {
    return aotx_recall_contextual(s, r) && aotx_recall_applicable(s, q, r) &&
        aotx_cog_u32(s->payload + aotx_cog_u64(r + AOTX_CO_OFFSET) + 32) == 1;
}
__device__ inline bool aotx_recall_before(const unsigned char *a, const unsigned char *b) {
    for (uint32_t j = 0; j < 16; ++j) if (a[AOTX_CO_ID + j] != b[AOTX_CO_ID + j])
        return a[AOTX_CO_ID + j] < b[AOTX_CO_ID + j];
    return aotx_cog_u64(a + AOTX_CO_VERSION) < aotx_cog_u64(b + AOTX_CO_VERSION);
}
__device__ inline bool aotx_recall_assesses(const aotx_cognitive_store *s, const unsigned char *r, const unsigned char *a) {
    if (aotx_cog_u16(a + AOTX_CO_KIND) != AOTX_COG_APPRAISAL ||
        !aotx_cog_equal(r + AOTX_CO_SUBJECT, a + AOTX_CO_SUBJECT)) return false;
    if (aotx_cog_equal(r + AOTX_CO_ID, a + AOTX_CO_SOURCE) &&
        aotx_cog_u64(r + AOTX_CO_VERSION) == aotx_cog_u64(a + AOTX_CO_SOURCE_VERSION)) return true;
    if (aotx_cog_u16(r + AOTX_CO_KIND) != AOTX_COG_WORKING ||
        !aotx_cog_equal(r + AOTX_CO_SOURCE, a + AOTX_CO_SOURCE, 24)) return false;
    int source = aotx_cog_find(s, r + AOTX_CO_SOURCE, aotx_cog_u64(r + AOTX_CO_SOURCE_VERSION));
    return source >= 0 && aotx_cog_u16(s->objects[source] + AOTX_CO_KIND) == AOTX_COG_EVENT;
}
__device__ inline uint32_t aotx_recall_intensity(const unsigned char *p) {
    uint32_t benefit = aotx_cog_u32(p + 4), harm = aotx_cog_u32(p + 8);
    if (benefit == AOTX_COG_UNKNOWN) return harm;
    if (harm == AOTX_COG_UNKNOWN) return benefit;
    return benefit > harm ? benefit : harm;
}
#endif
