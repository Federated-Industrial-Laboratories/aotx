/* Purpose: Resolve complete source groups for automatic appraisal recall.
 * Owns: Read-only task, subject, source and current-version checks.
 * Launch shape: Helpers for each candidate or recorded row in a query batch.
 * Lifetime: One admitted store cut; no exposure or memory writes. */
#ifndef AOTX_APPRAISAL_RECALL_CUH
#define AOTX_APPRAISAL_RECALL_CUH
#include "cognitive/recall_contextual.cuh"

__device__ inline uint32_t aotx_appraisal_recall_kind(const aotx_cognitive_store *s, const unsigned char *r) {
    if (aotx_cog_cold(r)) return 0;
    uint64_t n = aotx_cog_u64(r + AOTX_CO_BYTES);
    const unsigned char *p = s->payload + aotx_cog_u64(r + AOTX_CO_OFFSET);
    uint32_t kind = aotx_cog_u16(r + AOTX_CO_KIND);
    if (kind == AOTX_COG_APPRAISAL && n == AOTX_APPRAISAL_ASSESS_BYTES && aotx_cog_u32(p) == 2) return 1;
    if (kind == AOTX_COG_RELATIONSHIP && n == AOTX_APPRAISAL_RELATION_BYTES && aotx_recall_magic(p, "AOTXREL1")) return 2;
    return kind == AOTX_COG_POLICY && n == AOTX_APPRAISAL_QUEUE_BYTES && aotx_recall_magic(p, "AOTXAPQ1") ? 3 : 0;
}
__device__ inline bool aotx_appraisal_recall_current(const aotx_cognitive_store *s,
    const unsigned char *q, uint32_t index) {
    const unsigned char *r = s->objects[index];
    return !aotx_recall_match(s, q, r + AOTX_CO_ID, aotx_cog_u64(r + AOTX_CO_VERSION)).status;
}
/* A registered task and listed participants restrict the whole source group. */
__device__ inline bool aotx_appraisal_recall_context(const unsigned char *q,
    const unsigned char *r, const unsigned char *p) {
    const unsigned char *c = q + AOTX_RECALL_EXTENSION;
    bool tasks = aotx_context_flags(q) & AOTX_RECALL_TASKS;
    if (!aotx_cog_zero(p + 32, 16) && (!tasks || !aotx_cog_equal(p + 32, c + 16))) return false;
    if (!tasks || !aotx_cog_u32(c + 32)) return true;
    for (uint32_t j = 0; j < aotx_cog_u32(c + 32); ++j)
        if (aotx_cog_equal(r + AOTX_CO_SUBJECT, c + 48 + j * 16)) return true;
    return false;
}
/* The four entries are source, assessment, completed queue and relationship. */
static __device__ __noinline__ bool aotx_appraisal_recall_group(const aotx_cognitive_store *s,
    const unsigned char *q, uint32_t assessment, uint32_t *group) {
    const unsigned char *r = s->objects[assessment];
    if (aotx_appraisal_recall_kind(s, r) != 1 || !aotx_appraisal_recall_current(s, q, assessment)) return false;
    const unsigned char *p = s->payload + aotx_cog_u64(r + AOTX_CO_OFFSET);
    int source = aotx_cog_find(s, r + AOTX_CO_SOURCE, aotx_cog_u64(r + AOTX_CO_SOURCE_VERSION));
    int queue = aotx_cog_find(s, p + 96, aotx_cog_u64(p + 112));
    if (source < 0 || queue < 0 || !aotx_appraisal_recall_current(s, q, source) ||
        !aotx_appraisal_recall_current(s, q, queue) || aotx_appraisal_evidence_schema(s, r, p,
            AOTX_APPRAISAL_ASSESS_BYTES, false)) return false;
    const unsigned char *qr = s->objects[queue], *qp = s->payload + aotx_cog_u64(qr + AOTX_CO_OFFSET);
    if (aotx_appraisal_queue_schema(s, qr, qp, aotx_cog_u64(qr + AOTX_CO_BYTES))) return false;
    if (!aotx_cog_zero(qp + 128, 16) && aotx_recall_match(s, q, qp + 128, aotx_cog_u64(qp + 144)).status) return false;
    uint32_t relation = UINT32_MAX;
    for (uint32_t j = 0; j < s->count; ++j) {
        const unsigned char *other = s->objects[j];
        if (aotx_appraisal_recall_kind(s, other) != 2) continue;
        const unsigned char *op = s->payload + aotx_cog_u64(other + AOTX_CO_OFFSET);
        if (!aotx_cog_equal(op + 136, p + 96, 24) || !aotx_appraisal_recall_current(s, q, j)) continue;
        if (relation != UINT32_MAX || !aotx_cog_equal(other + AOTX_CO_SOURCE, r + AOTX_CO_SOURCE, 40) ||
            !aotx_cog_equal(other + AOTX_CO_OWNER, r + AOTX_CO_OWNER, 32) ||
            aotx_cog_u32(other + AOTX_CO_SCOPE) != aotx_cog_u32(r + AOTX_CO_SCOPE) ||
            !aotx_cog_equal(op + 48, p + 120, 8) ||
            aotx_appraisal_evidence_schema(s, other, op, AOTX_APPRAISAL_RELATION_BYTES, true) ||
            !aotx_appraisal_recall_context(q, other, op)) return false;
        relation = j;
    }
    if (relation == UINT32_MAX) return false;
    group[0] = source; group[1] = assessment; group[2] = queue; group[3] = relation;
    return true;
}
/* Separate known magnitudes can increase priority once; unknown is not a value. */
__device__ inline uint32_t aotx_appraisal_recall_intensity(const aotx_cognitive_store *s,
    const unsigned char *q, uint32_t index) {
    const unsigned char *r = s->objects[index], *p = s->payload + aotx_cog_u64(r + AOTX_CO_OFFSET);
    uint32_t value = aotx_recall_intensity(p);
    if (aotx_appraisal_recall_kind(s, r) != 1) return value;
    uint32_t group[4];
    if (!aotx_appraisal_recall_group(s, q, index, group)) return AOTX_COG_UNKNOWN;
    p = s->payload + aotx_cog_u64(s->objects[group[3]] + AOTX_CO_OFFSET);
    for (uint32_t at = 16; at < 32; at += 4) {
        uint32_t next = aotx_cog_u32(p + at);
        if (next != AOTX_COG_UNKNOWN && (value == AOTX_COG_UNKNOWN || next > value)) value = next;
    }
    return value;
}
#endif
