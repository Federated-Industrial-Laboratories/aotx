/* Purpose: Assemble bounded context from exact validated memory references.
 * Owns: Context bytes and source labels; no conversation history or model state.
 * Launch shape: One rendering thread for each request in a batch.
 * Lifetime: One quiescent selection or replay. */
#ifndef AOTX_COGNITIVE_RECALL_CONTEXT_CUH
#define AOTX_COGNITIVE_RECALL_CONTEXT_CUH
#include "cognitive/recall_labels.cuh"
#include "cognitive/recall_required.cuh"

__device__ inline uint32_t aotx_recall_reason(const unsigned char *q, const unsigned char *entry) {
    for (uint32_t group = 0; group < 2; ++group) {
        uint32_t n = aotx_cog_u32(q + (group ? 144 : 140));
        const unsigned char *refs = q + (group ? 4448 : 4256);
        for (uint32_t j = 0; j < n; ++j)
            if (aotx_cog_equal(entry, refs + j * 24) && aotx_cog_u64(entry + 16) == aotx_cog_u64(refs + j * 24 + 16))
                return group ? AOTX_RECALL_FOCUS : AOTX_RECALL_REQUIRED;
    }
    return AOTX_RECALL_SEMANTIC;
}
__device__ inline uint32_t aotx_recall_render(const aotx_cognitive_store *s,
    const unsigned char *q, aotx_recall_result *out) {
    uint32_t at = 0, cap = aotx_cog_u32(q + 136), required = 0;
    uint32_t status = aotx_recall_selection_check(s, q, out, &required);
    if (status) return status;
    for (uint32_t j = 0; j < out->count; ++j) {
        const unsigned char *entry = out->selection + 16 + j * 32, *r = s->objects[out->index[j]];
        uint32_t reason = aotx_recall_reason(q, entry);
        if (reason != AOTX_RECALL_REQUIRED && aotx_recall_obligatory(s, q, r)) reason = AOTX_RECALL_OBLIGATION;
        if (j >= required) {
            if (aotx_cog_u16(r + AOTX_CO_KIND) == AOTX_COG_APPRAISAL || aotx_appraisal_recall_kind(s, r)) reason = AOTX_RECALL_ASSESSMENT;
            else if (aotx_recall_pair(s, out, out->index[j])) reason = AOTX_RECALL_SIGNIFICANT;
        }
        out->reason[j] = reason;
        at = aotx_recall_one(s, out->index[j], reason, out->context, at, cap, aotx_context_sources(q));
        if (at > cap) return AOTX_COG_CAPACITY;
    }
    at = aotx_recall_context_label(q, out->context, at, cap);
    if (at > cap) return AOTX_COG_CAPACITY;
    at = aotx_recall_word(out->context, at, AOTX_RECALL_CONTEXT, "[input]\n");
    at = aotx_recall_run(out->context, at, AOTX_RECALL_CONTEXT, q + 4640, aotx_cog_u32(q + 148));
    if (at > AOTX_RECALL_CONTEXT) return AOTX_COG_CAPACITY;
    out->context_bytes = at;
    return AOTX_COG_OK;
}
__device__ inline void aotx_recall_clear(aotx_recall_result *out) {
    for (uint32_t j = threadIdx.x; j < sizeof(*out); j += blockDim.x) ((unsigned char *)out)[j] = 0;
    __syncthreads();
}
/* A refusal exposes status and request identity, with no partially selected content. */
__device__ inline void aotx_recall_refuse(aotx_recall_result *out, uint32_t status) {
    out->status = status; out->count = 0; out->context_bytes = 0;
    for (uint32_t j = 0; j < AOTX_RECALL_CONTEXT; ++j) out->context[j] = 0;
    for (uint32_t j = 0; j < AOTX_RECALL_SELECTION; ++j) out->selection[j] = 0;
    for (uint32_t j = 0; j < AOTX_RECALL_LIMIT; ++j) out->index[j] = out->reason[j] = 0;
}
#endif
