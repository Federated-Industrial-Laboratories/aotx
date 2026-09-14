/* Purpose: Check the exact vector space and cosine value for selected evidence.
 * Owns: Read-only vector validation and device arithmetic.
 * Launch shape: One candidate per query thread.
 * Lifetime: One search or recorded selection check. */
#ifndef AOTX_COGNITIVE_RECALL_SCORE_CUH
#define AOTX_COGNITIVE_RECALL_SCORE_CUH
#include "cognitive/recall_format.cuh"
#include "cognitive/text_space.cuh"
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
    if (bytes < 128 || aotx_cog_u32(p + 16) != 4 || aotx_cog_u32(p + 20) != 1 ||
        aotx_cog_zero(p + 24, 32) || aotx_cog_zero(p + 56, 32)) return AOTX_COG_LAYOUT;
    if (aotx_recall_magic(p, "AOTXVEC2") && aotx_cog_u32(p + 8) == 2) {
        int source = aotx_cog_find(s, p + 88, aotx_cog_u64(p + 104));
        if (source < 0 || aotx_cog_u16(s->objects[source] + AOTX_CO_KIND) != AOTX_COG_EVENT ||
            !aotx_cog_zero(p + 112, 16) || !aotx_cog_equal(p + 88, v + AOTX_CO_SOURCE, 24)) return AOTX_COG_LAYOUT;
    } else if (!aotx_recall_magic(p, "AOTXVEC1") || aotx_cog_u32(p + 8) != 1 ||
        !aotx_cog_zero(p + 120, 8) || aotx_cog_zero(p + 88, 32)) return AOTX_COG_LAYOUT;
    uint32_t width = aotx_cog_u32(p + 12);
    if (!width || width > AOTX_RECALL_WIDTH || bytes != 128 + width * 4) return AOTX_COG_LAYOUT;
    if (width != aotx_cog_u32(q + 128) || !aotx_cog_equal(p + 24, q + 64, 32) ||
        !aotx_text_space(p + 56, q + 96)) return AOTX_COG_SOURCE;
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
#endif
