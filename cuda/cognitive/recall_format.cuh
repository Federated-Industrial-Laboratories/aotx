/* Purpose: Validate prepared text, vector and query bytes on the device.
 * Owns: Bounded schema checks; no file IO or state mutation.
 * Launch shape: Helpers for each request in a batch.
 * Lifetime: One quiescent recall operation. */
#ifndef AOTX_COGNITIVE_RECALL_FORMAT_CUH
#define AOTX_COGNITIVE_RECALL_FORMAT_CUH
#include "cognitive/recall.cuh"
#include "cognitive/lookup.cuh"
#include "cognitive/context_format.cuh"
#include "cognitive/intake_schema.cuh"

__device__ inline bool aotx_recall_magic(const unsigned char *p, const char *s) {
    return aotx_cog_equal(p, (const unsigned char *)s, 8);
}
__device__ inline int aotx_recall_source(const aotx_cognitive_store *s, const unsigned char *r) {
    if (aotx_cog_u16(r + AOTX_CO_KIND) != AOTX_COG_SELECTION) return -1;
    int i = aotx_cog_find(s, r + AOTX_CO_SOURCE, aotx_cog_u64(r + AOTX_CO_SOURCE_VERSION));
    if (i < 0) return -1;
    const unsigned char *event = s->objects[i];
    if (aotx_cog_cold(event)) return -1;
    if (aotx_cog_u16(event + AOTX_CO_KIND) != AOTX_COG_EVENT ||
        aotx_cog_u64(event + AOTX_CO_BYTES) != 16 + AOTX_RECALL_QUERY) return -1;
    const unsigned char *p = s->payload + aotx_cog_u64(event + AOTX_CO_OFFSET);
    return aotx_recall_magic(p, "AOTXQUE1") ? i : -1;
}
__device__ inline bool aotx_recall_kind(uint32_t kind) {
    return kind == AOTX_COG_EVENT || kind == AOTX_COG_ASSERTION || kind == AOTX_COG_CUE ||
           kind == AOTX_COG_INTENTION || kind == AOTX_COG_WORKING || kind == AOTX_COG_IDENTITY || kind == AOTX_COG_APPRAISAL || kind == AOTX_COG_RELATIONSHIP || kind == AOTX_COG_POLICY;
}
/* Zero means an unrelated payload; malformed recognized text returns minus one. */
__device__ inline int aotx_recall_text(const aotx_cognitive_store *s, const unsigned char *r) {
    if (aotx_cog_cold(r)) return 0;
    uint64_t n = aotx_cog_u64(r + AOTX_CO_BYTES);
    if (!aotx_recall_kind(aotx_cog_u16(r + AOTX_CO_KIND)) || n < 8) return 0;
    const unsigned char *p = s->payload + aotx_cog_u64(r + AOTX_CO_OFFSET);
    if (aotx_cog_u16(r + AOTX_CO_KIND) == AOTX_COG_APPRAISAL) return 1;
    if (aotx_cog_u16(r + AOTX_CO_KIND) == AOTX_COG_RELATIONSHIP)
        return n == AOTX_APPRAISAL_RELATION_BYTES && aotx_recall_magic(p, "AOTXREL1") ? 1 : 0;
    if (aotx_cog_u16(r + AOTX_CO_KIND) == AOTX_COG_POLICY)
        return n == AOTX_APPRAISAL_QUEUE_BYTES && aotx_recall_magic(p, "AOTXAPQ1") &&
            aotx_cog_u32(p + 12) == AOTX_APPRAISAL_COMPLETE ? 1 : 0;
    if (aotx_recall_magic(p, "AOTXMEM3")) return aotx_intake_schema(s, r, p, n) ? -1 : (int)aotx_cog_u32(p + 12);
    if (aotx_recall_magic(p, "AOTXMEM2")) return aotx_memory_schema(r, p, n) ? -1 : (int)aotx_cog_u32(p + 12);
    if (!aotx_recall_magic(p, "AOTXMEM1")) return 0;
    if (n < 32 || aotx_cog_u32(p + 8) != 1 || !aotx_cog_zero(p + 16, 16)) return -1;
    uint32_t bytes = aotx_cog_u32(p + 12);
    return bytes && bytes <= AOTX_RECALL_TEXT && n == 32 + bytes && aotx_recall_utf8(p + 32, bytes)
        ? (int)bytes : -1;
}
__device__ inline double aotx_recall_float(const unsigned char *p) {
    return (double)__uint_as_float(aotx_cog_u32(p));
}
__device__ inline bool aotx_recall_finite(const unsigned char *p) {
    return (aotx_cog_u32(p) & 0x7f800000u) != 0x7f800000u;
}
__device__ inline uint32_t aotx_recall_envelope(const aotx_cognitive_store *s,
    const unsigned char *p, uint64_t bytes, uint32_t count) {
    if (!count || count > AOTX_RECALL_BATCH || bytes != AOTX_RECALL_HEADER + (uint64_t)count * AOTX_RECALL_QUERY)
        return AOTX_COG_FORMAT;
    if (!aotx_recall_magic(p, "AOTXREQ1") || aotx_cog_u32(p + 8) != count || aotx_cog_u32(p + 12) != 1 ||
        aotx_cog_u32(p + 40) != AOTX_RECALL_QUERY || !aotx_cog_zero(p + 44, 20)) return AOTX_COG_FORMAT;
    if (!aotx_cog_equal(p + 16, s->lineage)) return AOTX_COG_SOURCE;
    return aotx_cog_u64(p + 32) == s->sequence ? AOTX_COG_OK : AOTX_COG_STALE;
}
__device__ __forceinline__ uint32_t aotx_recall_query_check(const unsigned char *q, bool raw_text = false) {
    uint32_t width = aotx_cog_u32(q + 128), limit = aotx_cog_u32(q + 132), budget = aotx_cog_u32(q + 136);
    uint32_t required = aotx_cog_u32(q + 140), focus = aotx_cog_u32(q + 144), text = aotx_cog_u32(q + 148);
    uint32_t scope = aotx_cog_u32(q + 152);
    if (aotx_cog_zero(q, 16) || aotx_cog_zero(q + 16, 16) || aotx_cog_zero(q + 48, 16) ||
        aotx_cog_equal(q, q + 48) || (!raw_text && (aotx_cog_zero(q + 64, 32) ||
        aotx_cog_zero(q + 96, 32) || !width)) || width > AOTX_RECALL_WIDTH || !limit || limit > AOTX_RECALL_LIMIT || !budget ||
        budget > AOTX_RECALL_BUDGET || required > AOTX_RECALL_PINS || focus > AOTX_RECALL_PINS ||
        !text || text > AOTX_RECALL_TEXT || scope > AOTX_COG_INSTANCE || !aotx_cog_zero(q + 156, 4) ||
        (scope == AOTX_COG_ROOM ? aotx_cog_zero(q + 32, 16) : !aotx_cog_zero(q + 32, 16))) return AOTX_COG_FORMAT;
    if (!aotx_recall_utf8(q + 4640, text) || !aotx_cog_zero(q + 4640 + text, AOTX_RECALL_TEXT - text) ||
        aotx_context_check(q) ||
        !aotx_cog_zero(q + 160 + width * 4, (AOTX_RECALL_WIDTH - width) * 4)) return AOTX_COG_FORMAT;
    for (uint32_t group = 0; group < 2; ++group) {
        uint32_t n = group ? focus : required;
        const unsigned char *refs = q + (group ? 4448 : 4256);
        if (!aotx_cog_zero(refs + n * 24, (AOTX_RECALL_PINS - n) * 24)) return AOTX_COG_FORMAT;
        for (uint32_t i = 0; i < n; ++i)
            if (aotx_cog_zero(refs + i * 24, 16) || !aotx_cog_u64(refs + i * 24 + 16)) return AOTX_COG_REFERENCE;
    }
    if (raw_text) return !width && aotx_cog_zero(q + 64, 64) ? AOTX_COG_OK : AOTX_COG_FORMAT;
    double norm = 0;
    for (uint32_t i = 0; i < width; ++i) {
        if (!aotx_recall_finite(q + 160 + i * 4)) return AOTX_COG_LAYOUT;
        double x = aotx_recall_float(q + 160 + i * 4); norm += x * x;
    }
    return norm > 0 ? AOTX_COG_OK : AOTX_COG_LAYOUT;
}
__device__ inline aotx_cognitive_match aotx_recall_match_scratch(const aotx_cognitive_store *s,
    const unsigned char *q, const unsigned char *id, uint64_t version, uint64_t cut,
    uint32_t *need, uint32_t *done) {
    aotx_cognitive_query query = {};
    for (uint32_t i = 0; i < 16; ++i) { query.id[i] = id[i]; query.principal[i] = q[16 + i]; query.room[i] = q[32 + i]; }
    query.version = version;
    aotx_cognitive_match m = aotx_cog_resolve_scratch(s, &query, true, cut, need, done);
    if (!m.status) {
        const unsigned char *r = s->objects[m.index];
        uint32_t scope = aotx_cog_u32(r + AOTX_CO_SCOPE);
        if (scope != AOTX_COG_INSTANCE && scope != aotx_cog_u32(q + 152)) m.status = AOTX_COG_SCOPE;
        if (aotx_cog_u32(r + AOTX_CO_EVIDENCE) == 3) m.status = AOTX_COG_STALE;
    }
    return m;
}
__device__ inline aotx_cognitive_match aotx_recall_match(const aotx_cognitive_store *s,
    const unsigned char *q, const unsigned char *id, uint64_t version, uint64_t cut) {
    uint32_t need[AOTX_COG_WORDS], done[AOTX_COG_WORDS];
    return aotx_recall_match_scratch(s, q, id, version, cut, need, done);
}
__device__ inline aotx_cognitive_match aotx_recall_match(const aotx_cognitive_store *s,
    const unsigned char *q, const unsigned char *id, uint64_t version) {
    return aotx_recall_match(s, q, id, version, s->sequence);
}
#endif
