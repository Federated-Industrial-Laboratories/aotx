/* Purpose: Apply bounded appraisal priority inside optional semantic recall.
 * Owns: Per-query scores and exact assessment indices; no learned-state mutation.
 * Launch shape: One thread per candidate after the shared appraisal scan.
 * Lifetime: One search; selected source versions enter its recorded result. */
#ifndef AOTX_COGNITIVE_RECALL_APPRAISAL_CUH
#define AOTX_COGNITIVE_RECALL_APPRAISAL_CUH
#include "appraisal/recall.cuh"
#define AOTX_RECALL_CUE_STATE 256u
#define AOTX_RECALL_APPRAISAL_STATE 257u

__device__ inline void aotx_recall_appraisal_score(const aotx_cognitive_store *s,
    const unsigned char *q, uint32_t index, aotx_recall_scratch *scratch) {
    scratch->appraisals[index] = UINT32_MAX;
    if (!(aotx_context_flags(q) & AOTX_RECALL_APPRAISE) || scratch->states[index] != AOTX_COG_OK) return;
    const unsigned char *p = q + AOTX_RECALL_EXTENSION;
    if (scratch->scores[index] < (double)aotx_cog_u32(p + 36) / AOTX_COG_SCALE) {
        scratch->appraisals[index] = UINT32_MAX - 1; return;
    }
    if (!aotx_cog_u32(p + 40)) return;
    uint32_t best = UINT32_MAX;
    for (uint32_t j = 0; j < s->count; ++j)
        if (scratch->states[j] == AOTX_RECALL_APPRAISAL_STATE &&
            aotx_recall_assesses(s, s->objects[index], s->objects[j]) &&
            (best == UINT32_MAX || scratch->scores[j] > scratch->scores[best] ||
             (scratch->scores[j] == scratch->scores[best] && aotx_recall_before(s->objects[j], s->objects[best])))) best = j;
    if (best == UINT32_MAX) return;
    scratch->appraisals[index] = best;
    scratch->scores[index] += ((double)aotx_cog_u32(p + 40) / AOTX_COG_SCALE) *
        (scratch->scores[best] / AOTX_COG_SCALE);
}
#endif
