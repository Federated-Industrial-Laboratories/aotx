/* Purpose: Encode idle bindings and their exact memory image on the device.
 * Owns: The checkpoint image; live state is read only during this node.
 * Launch shape: One 64-thread block; threads stride all configured agent slots.
 * Lifetime: One finite coherent checkpoint copy. */
#include "cognitive/checkpoint.cuh"
#include "cognitive/live_validate.cuh"
#include "cognitive/live_cache.cuh"
#include "model/load.cuh"
#include "media/runtime.cuh"
#include "shared/state.cuh"

__device__ unsigned char aotx_checkpoint_image[AOTX_CP_BYTES];

static __device__ void aotx_cp_bytes(unsigned char *out, const unsigned char *in, uint32_t n) {
    for (uint32_t i = 0; i < n; ++i) out[i] = in[i];
}
__device__ bool aotx_checkpoint_foreign_quiet(bool all_slots) {
    if (!aotx_media_quiet()) return false;
    for (uint32_t i = 0; i < AOTX_SLOTS; ++i) {
        if (!all_slots && !aotx_live_bound(i) && !aotx_runtime_enabled) continue;
        if (((aotx_live_bound(i) || aotx_agents.agent[i].state != AOTX_AGENT_STATE_FREE) && aotx_live_busy(i)) ||
            aotx_say.slot[i].wanted || aotx_tool_embed.state[i] != AOTX_TOOL_EMBED_NONE ||
            (aotx_seqs.slot[i].state != AOTX_SEQ_STATE_FREE && aotx_seqs.slot[i].state != AOTX_SEQ_STATE_DONE)) return false;
#ifdef AOTX_AFFECT
        if (aotx_quality_state[i].ended || aotx_quality_state[i].pending) return false;
#endif
    }
    if (all_slots || aotx_runtime_enabled) {
        if (aotx_model_load.pending_count) return false;
        for (uint32_t i = 0; i < AOTX_TASK_SLOTS; ++i)
            if (aotx_task_used[i] && aotx_agents.task[i].state == AOTX_TASK_PENDING) return false;
        for (uint32_t i = 0; i < AOTX_CATALOG_ARRIVING_MAX; ++i)
            if (aotx_catalog.arriving[i].import) return false;
    }
    return true;
}
__device__ bool aotx_checkpoint_quiet(void) {
    return aotx_shared_quiet() && aotx_checkpoint_foreign_quiet();
}
__device__ bool aotx_checkpoint_idle(bool all_slots) {
    return aotx_live.ready && aotx_live.phase == AOTX_LIVE_IDLE && !aotx_live.received &&
        !aotx_live.fatal && aotx_shared_quiet() && aotx_checkpoint_foreign_quiet(all_slots);
}
static __device__ void aotx_cp_result(unsigned char *out, const aotx_recall_result *r) {
    aotx_cog_put(out, r->status, 4); aotx_cog_put(out + 4, r->count, 4);
    aotx_cog_put(out + 8, r->context_bytes, 4); aotx_cog_put(out + 12, r->searches, 4);
    aotx_cog_put(out + 16, r->cut, 8);
    aotx_cp_bytes(out + 24, r->request_id, 16); aotx_cp_bytes(out + 40, r->selection_id, 16);
    for (uint32_t i = 0; i < AOTX_RECALL_LIMIT; ++i) {
        aotx_cog_put(out + 56 + 4 * i, r->index[i], 4);
        aotx_cog_put(out + 120 + 4 * i, r->reason[i], 4);
    }
    aotx_cp_bytes(out + 184, r->selection, AOTX_RECALL_SELECTION);
    aotx_cp_bytes(out + 184 + AOTX_RECALL_SELECTION, r->context, AOTX_RECALL_CONTEXT);
}
__device__ void aotx_checkpoint_encode(void) {
    unsigned char *image = aotx_checkpoint_image;
    if (!threadIdx.x) {
        aotx_checkpoint.bindings = 0;
        for (uint32_t i = 0; i < AOTX_SLOTS; ++i) aotx_checkpoint.bindings += aotx_live_bound(i);
    }
    __syncthreads();
    uint32_t base = AOTX_CP_HEADER + aotx_checkpoint.bindings * AOTX_CP_ROW;
    for (uint32_t j = threadIdx.x; j < base; j += blockDim.x) image[j] = 0;
    __syncthreads();
    for (uint32_t i = threadIdx.x; i < AOTX_SLOTS; i += blockDim.x) {
        if (!aotx_live_bound(i)) continue;
        uint32_t row = 0;
        for (uint32_t j = 0; j < i; ++j) row += aotx_live_bound(j);
        unsigned char *out = image + AOTX_CP_HEADER + row * AOTX_CP_ROW;
        const aotx_live_binding *b = aotx_live_bindings + i;
        aotx_cog_put(out, i, 4); aotx_cog_put(out + 4, b->pages, 4);
        aotx_cog_put(out + 8, b->scope, 4); aotx_cog_put(out + 12, b->context_bytes, 4);
        aotx_cog_put(out + 16, b->ordinal, 8);
        aotx_cp_bytes(out + 24, b->principal, 16); aotx_cp_bytes(out + 40, b->room, 16);
        aotx_cp_bytes(out + 56, b->conversation, 16);
        aotx_cog_put(out + 72, b->focus_count, 4); aotx_cog_put(out + 76, b->auto_retain, 4);
        aotx_cog_put(out + 80, aotx_agents.agent[i].turn, 4);
        aotx_cog_put(out + 84, aotx_agent_gear[i].opens, 4);
        aotx_cp_bytes(out + 128, b->query, AOTX_RECALL_QUERY);
        aotx_cp_result(out + 128 + AOTX_RECALL_QUERY, &b->choice);
        aotx_cp_bytes(out + 128 + AOTX_RECALL_QUERY + AOTX_CP_RESULT,
            &b->focus[0][0], AOTX_RECALL_PINS * 24);
    }
    __syncthreads();
    aotx_cognitive_checkpoint_header_block(&aotx_live_store, image + base, AOTX_COG_IMAGE, &aotx_live.result);
    __syncthreads();
    if (!threadIdx.x) {
        aotx_checkpoint.bytes = base + (uint32_t)aotx_live.result.bytes;
        aotx_cp_bytes(image, (const unsigned char *)"AOTXLCP1", 8);
        aotx_cog_put(image + 8, 1, 4); aotx_cog_put(image + 12, AOTX_CP_ROW, 4);
        aotx_cog_put(image + 16, aotx_checkpoint.bindings, 4);
        aotx_cog_put(image + 24, aotx_live.result.bytes, 8);
        aotx_cp_bytes(image + 32, aotx_live_store.lineage, 16);
        aotx_cog_put(image + 48, aotx_live_store.sequence, 8);
        aotx_cog_put(image + 56, aotx_live_store.tick, 8);
        aotx_cog_put(image + 64, aotx_live.accepted, 8);
        aotx_cog_put(image + 72, aotx_time_tick, 8);
        aotx_checkpoint.captured = aotx_live.accepted;
        aotx_checkpoint.runtime_captured = aotx_runtime_dirty;
    }
}
