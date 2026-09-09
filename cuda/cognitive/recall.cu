/* Purpose: Select required, focused and semantic memory in bounded device batches.
 * Owns: Per-request scores and ordered selections; no persistent mutation.
 * Launch shape: One 64-thread block per request, with candidates spread across threads.
 * Lifetime: One quiescent admitted store and request batch. */
#include "cognitive/recall_context.cuh"
#include <math.h>

/* A missing vector is distinct from a malformed vector or an incompatible space. */
static __device__ uint32_t aotx_recall_score(const aotx_cognitive_store *s,
    const unsigned char *q, const unsigned char *r, double *score) {
    if (aotx_cog_zero(r + AOTX_CO_EMBEDDING, 16)) return AOTX_COG_MISSING;
    int index = aotx_cog_find(s, r + AOTX_CO_EMBEDDING, aotx_cog_u64(r + AOTX_CO_EMBED_VERSION));
    if (index < 0) return AOTX_COG_REFERENCE;
    const unsigned char *v = s->objects[index];
    const unsigned char *p = s->payload + aotx_cog_u64(v + AOTX_CO_OFFSET);
    uint64_t bytes = aotx_cog_u64(v + AOTX_CO_BYTES);
    if (bytes < 128 || !aotx_recall_magic(p, "AOTXVEC1") || aotx_cog_u32(p + 8) != 1 ||
        aotx_cog_u32(p + 16) != 4 || aotx_cog_u32(p + 20) != 1 || !aotx_cog_zero(p + 120, 8) ||
        aotx_cog_zero(p + 24, 32) || aotx_cog_zero(p + 56, 32) || aotx_cog_zero(p + 88, 32)) return AOTX_COG_LAYOUT;
    uint32_t width = aotx_cog_u32(p + 12);
    if (!width || width > AOTX_RECALL_WIDTH || bytes != 128 + width * 4) return AOTX_COG_LAYOUT;
    if (width != aotx_cog_u32(q + 128) || !aotx_cog_equal(p + 24, q + 64, 32) ||
        !aotx_cog_equal(p + 56, q + 96, 32)) return AOTX_COG_SOURCE;
    double dot = 0, norm = 0, query_norm = 0;
    for (uint32_t j = 0; j < width; ++j) {
        if (!aotx_recall_finite(p + 128 + j * 4)) return AOTX_COG_LAYOUT;
        double x = aotx_recall_float(p + 128 + j * 4), y = aotx_recall_float(q + 160 + j * 4);
        dot += x * y; norm += x * x; query_norm += y * y;
    }
    if (!(norm > 0)) return AOTX_COG_LAYOUT;
    *score = dot / (sqrt(norm) * sqrt(query_norm));
    return AOTX_COG_OK;
}
static __device__ bool aotx_recall_before(const unsigned char *a, const unsigned char *b) {
    for (uint32_t j = 0; j < 16; ++j) if (a[AOTX_CO_ID + j] != b[AOTX_CO_ID + j])
        return a[AOTX_CO_ID + j] < b[AOTX_CO_ID + j];
    return aotx_cog_u64(a + AOTX_CO_VERSION) < aotx_cog_u64(b + AOTX_CO_VERSION);
}
static __device__ bool aotx_recall_has(const aotx_recall_result *out, uint32_t index) {
    for (uint32_t j = 0; j < out->count; ++j) if (out->index[j] == index) return true;
    return false;
}
static __device__ uint32_t aotx_recall_add(const aotx_cognitive_store *s, const unsigned char *q,
    uint32_t index, uint32_t reason, uint32_t *used, aotx_recall_result *out) {
    if (aotx_recall_has(out, index)) return AOTX_COG_OK;
    uint32_t cap = aotx_cog_u32(q + 136);
    uint32_t after = aotx_recall_one(s, index, reason, 0, *used, cap);
    if (out->count == aotx_cog_u32(q + 132) || after > cap) return AOTX_COG_CAPACITY;
    const unsigned char *r = s->objects[index];
    unsigned char *entry = out->selection + 16 + out->count * 32;
    for (uint32_t j = 0; j < 16; ++j) entry[j] = r[AOTX_CO_ID + j];
    aotx_cog_put(entry + 16, aotx_cog_u64(r + AOTX_CO_VERSION), 8);
    aotx_cog_put(entry + 24, 1, 4);
    out->index[out->count] = index; out->reason[out->count++] = reason; *used = after;
    return AOTX_COG_OK;
}

__global__ void aotx_recall_search(const aotx_cognitive_store *live,
    const unsigned char *requests, uint64_t bytes, aotx_recall_result *results, uint32_t count) {
    uint32_t n = blockIdx.x;
    if (n >= count || count > AOTX_RECALL_BATCH) return;
    aotx_recall_result *out = results + n;
    aotx_recall_clear(out);
    __shared__ double scores[AOTX_COG_OBJECTS];
    __shared__ uint32_t states[AOTX_COG_OBJECTS];
    const unsigned char *q = requests + AOTX_RECALL_HEADER + (uint64_t)n * AOTX_RECALL_QUERY;
    if (!threadIdx.x) {
        out->status = aotx_recall_envelope(live, requests, bytes, count);
        if (!out->status) out->status = aotx_recall_query_check(q);
        if (!out->status) {
            out->cut = live->sequence; out->searches = 1;
            for (uint32_t j = 0; j < 16; ++j) { out->request_id[j] = q[j]; out->selection_id[j] = q[48 + j]; }
        }
    }
    __syncthreads();
    if (out->status) return;
    for (uint32_t j = threadIdx.x; j < live->count; j += blockDim.x) {
        const unsigned char *r = live->objects[j]; states[j] = AOTX_COG_MISSING; scores[j] = -2;
        if (aotx_recall_match(live, q, r + AOTX_CO_ID, aotx_cog_u64(r + AOTX_CO_VERSION)).status) continue;
        int length = aotx_recall_text(live, r);
        if (length < 0) { states[j] = AOTX_COG_FORMAT; continue; }
        if (length > 0) states[j] = aotx_recall_score(live, q, r, scores + j);
    }
    __syncthreads();
    if (threadIdx.x) return;
    uint32_t compatible = 0, incompatible = 0, used = 0;
    for (uint32_t j = 0; j < live->count; ++j) {
        if (states[j] == AOTX_COG_OK) ++compatible;
        else if (states[j] == AOTX_COG_SOURCE) ++incompatible;
        else if (states[j] != AOTX_COG_MISSING) { aotx_recall_refuse(out, states[j]); return; }
    }
    if (incompatible && !compatible) { aotx_recall_refuse(out, AOTX_COG_SOURCE); return; }
    for (uint32_t group = 0; group < 2; ++group) {
        uint32_t pins = aotx_cog_u32(q + (group ? 144 : 140));
        const unsigned char *refs = q + (group ? 4448 : 4256);
        for (uint32_t j = 0; j < pins; ++j) {
            const unsigned char *ref = refs + j * 24;
            aotx_cognitive_match m = aotx_recall_match(live, q, ref, aotx_cog_u64(ref + 16));
            uint32_t status = m.status;
            if (!status && aotx_recall_text(live, live->objects[m.index]) <= 0) status = AOTX_COG_FORMAT;
            if (!status) status = aotx_recall_add(live, q, m.index, group ? AOTX_RECALL_FOCUS : AOTX_RECALL_REQUIRED, &used, out);
            if (status) { aotx_recall_refuse(out, status); return; }
        }
    }
    for (uint32_t pass = 0; pass < live->count && out->count < aotx_cog_u32(q + 132); ++pass) {
        uint32_t best = UINT32_MAX;
        for (uint32_t j = 0; j < live->count; ++j)
            if (states[j] == AOTX_COG_OK && !aotx_recall_has(out, j) && (best == UINT32_MAX || scores[j] > scores[best] ||
                (scores[j] == scores[best] && aotx_recall_before(live->objects[j], live->objects[best])))) best = j;
        if (best == UINT32_MAX) break;
        states[best] = AOTX_COG_MISSING;
        aotx_recall_add(live, q, best, AOTX_RECALL_SEMANTIC, &used, out);
    }
    aotx_cog_put(out->selection, 1, 4); aotx_cog_put(out->selection + 4, out->count, 4);
    uint32_t status = aotx_recall_render(live, q, out);
    if (status) aotx_recall_refuse(out, status);
}
