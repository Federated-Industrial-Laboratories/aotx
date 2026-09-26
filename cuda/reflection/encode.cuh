/* Purpose: Encode inferred task cues and exact supporting selections.
 * Owns: Canonical rows in a caller-owned zeroed tail.
 * Launch shape: One independent source per batch row.
 * Lifetime: A recorded review result; no live store mutation. */
#ifndef AOTX_REFLECTION_ENCODE_CUH
#define AOTX_REFLECTION_ENCODE_CUH
#include "reflection/format.h"
#include "cognitive/codec.cuh"

__device__ inline void aotx_review_id(unsigned char *id, uint64_t revision, uint32_t cue) {
    for (uint32_t j = 0; j < 8; ++j) id[j] = (cue ? "RVCUE001" : "RVSEL001")[j];
    aotx_cog_put(id + 8, revision, 8);
}
__device__ inline void aotx_review_tail(const aotx_cognitive_store *s,
    unsigned char *tail, uint32_t count) {
    uint32_t objects = 2 * count, bytes = count * AOTX_REVIEW_PAYLOAD;
    for (uint32_t j = 0; j < 8; ++j) tail[j] = "AOTXLOG1"[j];
    aotx_cog_put(tail + 8, 1, 4); aotx_cog_put(tail + 12, AOTX_COG_HEADER, 4);
    aotx_cog_put(tail + 16, AOTX_COG_OBJECT, 4); aotx_cog_put(tail + 20, objects, 4);
    aotx_cog_put(tail + 24, bytes, 8); aotx_cog_put(tail + 32, s->sequence + 1, 8);
    aotx_cog_put(tail + 40, s->tick + 1, 8);
    for (uint32_t j = 0; j < 16; ++j) tail[48 + j] = s->lineage[j];
    aotx_cog_put(tail + 64, AOTX_COG_HEADER, 8);
    aotx_cog_put(tail + 72, AOTX_COG_HEADER + objects * AOTX_COG_OBJECT, 8);
    aotx_cog_put(tail + 80, AOTX_COG_HEADER + objects * AOTX_COG_OBJECT + bytes, 8);
    aotx_cog_put(tail + 88, 1, 4); aotx_cog_policy_write(tail, s);
}
__device__ inline void aotx_review_encode(const aotx_cognitive_store *s, const uint32_t *group,
    unsigned char *tail, uint32_t row, uint32_t count) {
    const unsigned char *assessment = s->objects[group[1]], *source = s->objects[group[0]],
        *queue = s->objects[group[2]], *qp = s->payload + aotx_cog_u64(queue + AOTX_CO_OFFSET);
    uint64_t revision = aotx_cog_u64(assessment + AOTX_CO_UPDATED);
    uint64_t selected_version = s->pressure_percent ? s->sequence + 2 * row + 1 : 1;
    for (uint32_t k = 0; k < 2; ++k) {
        uint32_t offset = row * AOTX_REVIEW_PAYLOAD + (k ? AOTX_REVIEW_SELECTION_BYTES : 0);
        unsigned char *r = tail + AOTX_COG_HEADER + (2 * row + k) * AOTX_COG_OBJECT,
            *p = tail + AOTX_COG_HEADER + 2 * count * AOTX_COG_OBJECT + offset;
        uint64_t sequence = s->sequence + 2 * row + k + 1;
        aotx_cog_put(r, 1, 2); aotx_cog_put(r + AOTX_CO_KIND, k ? AOTX_COG_REVIEW : AOTX_COG_SELECTION, 2);
        aotx_review_id(r + AOTX_CO_ID, revision, k);
        for (uint32_t j = 0; j < 16; ++j) {
            r[AOTX_CO_LINEAGE + j] = s->lineage[j];
            r[AOTX_CO_OWNER + j] = assessment[AOTX_CO_OWNER + j];
            r[AOTX_CO_ROOM + j] = assessment[AOTX_CO_ROOM + j];
            r[AOTX_CO_SUBJECT + j] = assessment[AOTX_CO_SUBJECT + j];
        }
        if (k) aotx_review_id(r + AOTX_CO_SOURCE, revision, 0);
        else for (uint32_t j = 0; j < 16; ++j) r[AOTX_CO_SOURCE + j] = source[AOTX_CO_ID + j];
        aotx_cog_put(r + AOTX_CO_SOURCE_VERSION, k ? selected_version : aotx_cog_u64(source + AOTX_CO_VERSION), 8);
        aotx_cog_put(r + AOTX_CO_VERSION, s->pressure_percent ? sequence : 1, 8);
        aotx_cog_put(r + AOTX_CO_CREATED, sequence, 8); aotx_cog_put(r + AOTX_CO_UPDATED, sequence, 8);
        aotx_cog_put(r + AOTX_CO_OFFSET, offset, 8);
        aotx_cog_put(r + AOTX_CO_BYTES, k ? AOTX_REVIEW_CUE_BYTES : AOTX_REVIEW_SELECTION_BYTES, 8);
        aotx_cog_put(r + AOTX_CO_SCOPE, aotx_cog_u32(assessment + AOTX_CO_SCOPE), 4);
        aotx_cog_put(r + AOTX_CO_SOURCE_KIND, AOTX_COG_INFERRED, 4);
        aotx_cog_put(r + AOTX_CO_IMPORTANCE, AOTX_COG_UNKNOWN, 4); aotx_cog_put(r + AOTX_CO_POLICY, 1, 8);
        if (k) {
            for (uint32_t j = 0; j < 8; ++j) p[j] = "AOTXMEM4"[j];
            aotx_cog_put(p + 8, 4, 4); aotx_cog_put(p + 12, AOTX_REVIEW_TEXT_BYTES, 4);
            for (uint32_t j = 0; j < 16; ++j) p[16 + j] = qp[40 + j];
            aotx_cog_put(p + 32, 1, 4);
            for (uint32_t j = 0; j < AOTX_REVIEW_TEXT_BYTES; ++j) p[64 + j] = AOTX_REVIEW_TEXT[j];
        } else {
            aotx_cog_put(p, 1, 4); aotx_cog_put(p + 4, AOTX_REVIEW_REFERENCES, 4);
            for (uint32_t j = 0; j < AOTX_REVIEW_REFERENCES; ++j) {
                const unsigned char *e = s->objects[group[j]];
                for (uint32_t b = 0; b < 16; ++b) p[16 + j * 32 + b] = e[AOTX_CO_ID + b];
                aotx_cog_put(p + 32 + j * 32, aotx_cog_u64(e + AOTX_CO_VERSION), 8);
                aotx_cog_put(p + 40 + j * 32, 1, 4);
            }
        }
    }
}
#endif
