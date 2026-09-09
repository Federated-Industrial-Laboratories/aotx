/* Purpose: Assemble bounded context from exact validated memory references.
 * Owns: Context bytes and source labels; no conversation history or model state.
 * Launch shape: One rendering thread for each request in a batch.
 * Lifetime: One quiescent selection or replay. */
#ifndef AOTX_COGNITIVE_RECALL_CONTEXT_CUH
#define AOTX_COGNITIVE_RECALL_CONTEXT_CUH
#include "cognitive/recall_format.cuh"

__device__ inline uint32_t aotx_recall_run(unsigned char *out, uint32_t at,
    uint32_t cap, const unsigned char *p, uint32_t n) {
    if (at > cap || n > cap - at) return cap + 1;
    if (out) for (uint32_t j = 0; j < n; ++j) out[at + j] = p[j];
    return at + n;
}
__device__ inline uint32_t aotx_recall_word(unsigned char *out, uint32_t at,
    uint32_t cap, const char *p) {
    uint32_t n = 0; while (p[n]) ++n;
    return aotx_recall_run(out, at, cap, (const unsigned char *)p, n);
}
__device__ inline uint32_t aotx_recall_number(unsigned char *out, uint32_t at,
    uint32_t cap, uint64_t value) {
    unsigned char digits[20]; uint32_t n = 0;
    do { digits[n++] = '0' + value % 10; value /= 10; } while (value);
    while (n) at = aotx_recall_run(out, at, cap, digits + --n, 1);
    return at;
}
__device__ inline uint32_t aotx_recall_one(const aotx_cognitive_store *s,
    uint32_t index, uint32_t reason, unsigned char *out, uint32_t at, uint32_t cap) {
    const unsigned char *r = s->objects[index];
    const unsigned char *p = s->payload + aotx_cog_u64(r + AOTX_CO_OFFSET);
    at = aotx_recall_word(out, at, cap, "[memory id=");
    const char *hex = "0123456789abcdef";
    for (uint32_t j = 0; j < 16; ++j) {
        unsigned char pair[2] = {(unsigned char)hex[r[AOTX_CO_ID + j] >> 4], (unsigned char)hex[r[AOTX_CO_ID + j] & 15]};
        at = aotx_recall_run(out, at, cap, pair, 2);
    }
    at = aotx_recall_word(out, at, cap, " version=");
    at = aotx_recall_number(out, at, cap, aotx_cog_u64(r + AOTX_CO_VERSION));
    at = aotx_recall_word(out, at, cap, " source=");
    at = aotx_recall_number(out, at, cap, aotx_cog_u32(r + AOTX_CO_SOURCE_KIND));
    at = aotx_recall_word(out, at, cap, " evidence=");
    at = aotx_recall_number(out, at, cap, aotx_cog_u32(r + AOTX_CO_EVIDENCE));
    at = aotx_recall_word(out, at, cap, " reason=");
    at = aotx_recall_number(out, at, cap, reason);
    at = aotx_recall_word(out, at, cap, "]\n");
    at = aotx_recall_run(out, at, cap, p + 32, aotx_cog_u32(p + 12));
    return aotx_recall_word(out, at, cap, "\n");
}
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
    uint32_t at = 0, cap = aotx_cog_u32(q + 136);
    for (uint32_t j = 0; j < out->count; ++j) {
        const unsigned char *entry = out->selection + 16 + j * 32;
        aotx_cognitive_match m = aotx_recall_match(s, q, entry, aotx_cog_u64(entry + 16));
        if (m.status) return m.status;
        if (aotx_recall_text(s, s->objects[m.index]) <= 0) return AOTX_COG_FORMAT;
        out->index[j] = m.index; out->reason[j] = aotx_recall_reason(q, entry);
        at = aotx_recall_one(s, m.index, out->reason[j], out->context, at, cap);
        if (at > cap) return AOTX_COG_CAPACITY;
    }
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
