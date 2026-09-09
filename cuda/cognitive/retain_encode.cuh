/* Purpose: Encode accepted input as source, vector and working-memory objects.
 * Owns: A zeroed canonical tail; no live mutation or device hashing.
 * Launch shape: One thread encodes each of up to 64 distinct rows.
 * Lifetime: A staged retention decision and its exact recorded bytes. */
#ifndef AOTX_COGNITIVE_RETAIN_ENCODE_CUH
#define AOTX_COGNITIVE_RETAIN_ENCODE_CUH
#include "cognitive/retain_validate.cuh"

__device__ inline uint32_t aotx_retain_payload_bytes(const unsigned char *r) {
    const unsigned char *q = aotx_live_bindings[aotx_cog_u32(r)].query;
    return 2 * (32 + aotx_cog_u32(q + 148)) + 128 + 4 * aotx_cog_u32(q + 128);
}
__device__ inline void aotx_retain_header(unsigned char *tail, uint32_t count, uint64_t payload) {
    for (uint32_t j = 0; j < 8; ++j) tail[j] = "AOTXLOG1"[j];
    aotx_cog_put(tail + 8, 1, 4); aotx_cog_put(tail + 12, AOTX_COG_HEADER, 4);
    aotx_cog_put(tail + 16, AOTX_COG_OBJECT, 4); aotx_cog_put(tail + 20, count * 3, 4);
    aotx_cog_put(tail + 24, payload, 8); aotx_cog_put(tail + 32, aotx_live_store.sequence + 1, 8);
    aotx_cog_put(tail + 40, aotx_live_store.tick + 1, 8);
    for (uint32_t j = 0; j < 16; ++j) tail[48 + j] = aotx_live_store.lineage[j];
    uint32_t base = AOTX_COG_HEADER + count * 3 * AOTX_COG_OBJECT;
    aotx_cog_put(tail + 64, AOTX_COG_HEADER, 8); aotx_cog_put(tail + 72, base, 8);
    aotx_cog_put(tail + 80, base + payload, 8); aotx_cog_put(tail + 88, 1, 4);
}
static __device__ __noinline__ void aotx_retain_encode(unsigned char *tail, uint32_t i, uint32_t count) {
    const unsigned char *in = aotx_live.retain_rows[i];
    const unsigned char *q = aotx_live_bindings[aotx_cog_u32(in)].query;
    uint32_t offset = 0;
    for (uint32_t j = 0; j < i; ++j) offset += aotx_retain_payload_bytes(aotx_live.retain_rows[j]);
    uint32_t base = AOTX_COG_HEADER + count * 3 * AOTX_COG_OBJECT;
    uint32_t text = aotx_cog_u32(q + 148), width = aotx_cog_u32(q + 128);
    for (uint32_t k = 0; k < 3; ++k) {
        unsigned char *r = tail + AOTX_COG_HEADER + (i * 3 + k) * AOTX_COG_OBJECT;
        unsigned char *p = tail + base + offset;
        uint32_t kind = k == 0 ? AOTX_COG_EVENT : (k == 1 ? AOTX_COG_COMPONENT : AOTX_COG_WORKING);
        uint32_t bytes = k == 1 ? 128 + width * 4 : 32 + text;
        uint64_t seq = aotx_live_store.sequence + i * 3 + k + 1;
        aotx_cog_put(r, 1, 2); aotx_cog_put(r + AOTX_CO_KIND, kind, 2);
        for (uint32_t j = 0; j < 16; ++j) {
            r[AOTX_CO_ID + j] = in[(k == 0 ? 32 : (k == 1 ? 64 : 48)) + j];
            r[AOTX_CO_LINEAGE + j] = aotx_live_store.lineage[j];
            r[AOTX_CO_OWNER + j] = q[16 + j]; r[AOTX_CO_ROOM + j] = q[32 + j];
            r[AOTX_CO_SUBJECT + j] = in[112 + j];
            if (k) r[AOTX_CO_SOURCE + j] = in[32 + j];
            if (k == 2) { r[AOTX_CO_EMBEDDING + j] = in[64 + j]; r[AOTX_CO_SUPERSEDES + j] = in[80 + j]; }
        }
        aotx_cog_put(r + AOTX_CO_VERSION, 1, 8);
        aotx_cog_put(r + AOTX_CO_CREATED, seq, 8); aotx_cog_put(r + AOTX_CO_UPDATED, seq, 8);
        aotx_cog_put(r + AOTX_CO_OFFSET, offset, 8); aotx_cog_put(r + AOTX_CO_BYTES, bytes, 8);
        aotx_cog_put(r + AOTX_CO_SCOPE, aotx_cog_u32(q + 152), 4);
        aotx_cog_put(r + AOTX_CO_SOURCE_KIND, k == 1 ? AOTX_COG_INFERRED : AOTX_COG_REPORTED, 4);
        aotx_cog_put(r + AOTX_CO_IMPORTANCE, aotx_cog_u32(in + 128), 4);
        aotx_cog_put(r + AOTX_CO_RETENTION, aotx_cog_u32(in + 132), 4);
        aotx_cog_put(r + AOTX_CO_EXPIRY, aotx_cog_u64(in + 136), 8);
        aotx_cog_put(r + AOTX_CO_POLICY, aotx_cog_u64(in + 104), 8);
        if (k) aotx_cog_put(r + AOTX_CO_SOURCE_VERSION, 1, 8);
        if (k == 2) {
            aotx_cog_put(r + AOTX_CO_EMBED_VERSION, 1, 8);
            aotx_cog_put(r + AOTX_CO_SUPER_VERSION, aotx_cog_u64(in + 96), 8);
        }
        for (uint32_t j = 0; j < 8; ++j) p[j] = (k == 1 ? "AOTXVEC2" : "AOTXMEM1")[j];
        aotx_cog_put(p + 8, k == 1 ? 2 : 1, 4);
        aotx_cog_put(p + 12, k == 1 ? width : text, 4);
        if (k == 1) {
            aotx_cog_put(p + 16, 4, 4); aotx_cog_put(p + 20, 1, 4);
            for (uint32_t j = 0; j < 64; ++j) p[24 + j] = q[64 + j];
            for (uint32_t j = 0; j < 16; ++j) p[88 + j] = in[32 + j];
            aotx_cog_put(p + 104, 1, 8);
            for (uint32_t j = 0; j < width * 4; ++j) p[128 + j] = q[160 + j];
        } else for (uint32_t j = 0; j < text; ++j) p[32 + j] = q[4640 + j];
        offset += bytes;
    }
}
#endif
