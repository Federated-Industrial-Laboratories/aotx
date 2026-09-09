/* Purpose: Validate exact recorded choices before context rendering.
 * Owns: Result scratch only; no query or store mutation.
 * Launch shape: One thread per live query row.
 * Lifetime: One recorded decision and its replay. */
#ifndef AOTX_COGNITIVE_LIVE_SELECTED_CUH
#define AOTX_COGNITIVE_LIVE_SELECTED_CUH
#include "cognitive/live_text.cuh"

static __device__ __noinline__ uint32_t aotx_live_selected(const unsigned char *q, aotx_recall_result *out) {
    const unsigned char *s = out->selection;
    uint32_t count = aotx_cog_u32(s + 4);
    if (aotx_cog_u32(s) != 1 || count > aotx_cog_u32(q + 132) || count > AOTX_RECALL_LIMIT ||
        !aotx_cog_zero(s + 8, 8) || !aotx_cog_zero(s + 16 + count * 32, (AOTX_RECALL_LIMIT - count) * 32)) return AOTX_COG_FORMAT;
    for (uint32_t i = 0; i < count; ++i) {
        const unsigned char *entry = s + 16 + i * 32;
        if (aotx_cog_u32(entry + 24) != 1 || !aotx_cog_zero(entry + 28, 4)) return AOTX_COG_FORMAT;
        for (uint32_t j = 0; j < i; ++j)
            if (aotx_cog_equal(entry, s + 16 + j * 32)) return AOTX_COG_REFERENCE;
    }
    uint32_t expected = 0;
    for (uint32_t group = 0; group < 2; ++group) {
        uint32_t pins = aotx_cog_u32(q + (group ? 144 : 140));
        const unsigned char *refs = q + (group ? 4448 : 4256);
        for (uint32_t i = 0; i < pins; ++i) {
            const unsigned char *r = refs + i * 24;
            bool seen = false;
            for (uint32_t j = 0; j < expected; ++j)
                if (aotx_cog_equal(r, s + 16 + j * 32, 24)) seen = true;
            if (seen) continue;
            if (expected == count || !aotx_cog_equal(r, s + 16 + expected * 32, 24)) return AOTX_COG_REFERENCE;
            ++expected;
        }
    }
    out->count = count;
    return aotx_recall_render(&aotx_live_store, q, out);
}

#endif
