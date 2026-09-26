/* Purpose: Select current task evidence for internal review batches.
 * Owns: Read-only eligibility and complete required recall groups.
 * Launch shape: One source or cue per batch row.
 * Lifetime: One exact store cut and its current access rules. */
#ifndef AOTX_REFLECTION_EVIDENCE_CUH
#define AOTX_REFLECTION_EVIDENCE_CUH
#include "reflection/schema.cuh"
#include "appraisal/recall.cuh"

__device__ inline bool aotx_review_evidence(const aotx_cognitive_store *s,
    const unsigned char *q, uint32_t assessment, uint32_t *group) {
    if (assessment >= s->count || !aotx_appraisal_recall_group(s, q, assessment, group)) return false;
    const unsigned char *r = s->objects[assessment],
        *p = s->payload + aotx_cog_u64(r + AOTX_CO_OFFSET),
        *source = s->objects[group[0]], *queue = s->objects[group[2]],
        *qp = s->payload + aotx_cog_u64(queue + AOTX_CO_OFFSET);
    uint32_t benefit = aotx_cog_u32(p + 4), harm = aotx_cog_u32(p + 8);
    if (!((benefit && benefit != AOTX_COG_UNKNOWN) || (harm && harm != AOTX_COG_UNKNOWN)) ||
        aotx_cog_u32(source + AOTX_CO_SOURCE_KIND) == AOTX_COG_INFERRED ||
        aotx_cog_u16(source + AOTX_CO_KIND) != AOTX_COG_EVENT ||
        aotx_cog_zero(qp + 40, 16) || aotx_cog_zero(qp + 128, 16) ||
        !aotx_cog_u32(p + 124) || aotx_appraisal_contract(p + 32) != 2) return false;
    aotx_cognitive_match task = aotx_recall_match(s, q, qp + 128, aotx_cog_u64(qp + 144));
    if (task.status) return false;
    group[4] = task.index;
    return true;
}
__device__ inline bool aotx_review_group(const aotx_cognitive_store *s,
    const unsigned char *q, uint32_t cue, uint32_t *group) {
    const unsigned char *r = s->objects[cue];
    if (!aotx_review_kind(s, r) || !aotx_recall_obligatory(s, q, r) ||
        aotx_recall_match(s, q, r + AOTX_CO_ID, aotx_cog_u64(r + AOTX_CO_VERSION)).status) return false;
    const unsigned char *p = s->payload + aotx_cog_u64(r + AOTX_CO_OFFSET);
    if (aotx_review_schema(s, r, p, aotx_cog_u64(r + AOTX_CO_BYTES), group)) return false;
    uint32_t checked[AOTX_REVIEW_REFERENCES];
    if (!aotx_review_evidence(s, q, group[1], checked)) return false;
    for (uint32_t j = 0; j < AOTX_REVIEW_REFERENCES; ++j) if (group[j] != checked[j]) return false;
    return true;
}
#endif
