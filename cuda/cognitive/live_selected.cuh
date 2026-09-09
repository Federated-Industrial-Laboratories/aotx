/* Purpose: Validate exact recorded choices before context rendering.
 * Owns: Result scratch only; no query or store mutation.
 * Launch shape: One thread per live query row.
 * Lifetime: One recorded decision and its replay. */
#ifndef AOTX_COGNITIVE_LIVE_SELECTED_CUH
#define AOTX_COGNITIVE_LIVE_SELECTED_CUH
#include "cognitive/live_text.cuh"

__device__ __forceinline__ uint32_t aotx_live_selected(const unsigned char *q, aotx_recall_result *out) {
    out->count = aotx_cog_u32(out->selection + 4);
    return aotx_recall_render(&aotx_live_store, q, out);
}

#endif
