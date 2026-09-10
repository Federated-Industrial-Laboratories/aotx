/* Purpose: Release completed cognitive conversation caches after their last consumer.
 * Owns: Ephemeral KV release requests; no persistent memory or sequence text mutation.
 * Launch shape: One device thread per conversation slot.
 * Lifetime: The idle interval after a completed language sequence. */
#ifndef AOTX_COGNITIVE_LIVE_CACHE_CUH
#define AOTX_COGNITIVE_LIVE_CACHE_CUH
#include "cognitive/live.cuh"
#include "agent/agent_state.cuh"
#include "model/decode_state.cuh"
#include "tool/tool_state.cuh"
#ifdef AOTX_AFFECT
#include "quality/quality.cuh"
#endif

__device__ inline void aotx_live_cache_release(uint32_t slot) {
    if (!aotx_live_bound(slot) || aotx_live_busy(slot) || aotx_seqs.slot[slot].state != AOTX_SEQ_STATE_DONE ||
        aotx_tool_embed.state[slot] != AOTX_TOOL_EMBED_NONE ||
        (!aotx_kv.count[slot] && !aotx_seq_asked[slot])) return;
#ifdef AOTX_AFFECT
    const aotx_quality_slot *quality = aotx_quality_state + slot;
    if (quality->ended || quality->pending) return;
#endif
    if (aotx_kv_release(slot)) aotx_seq_asked[slot] = 0;
}
#endif
