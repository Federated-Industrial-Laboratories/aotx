/* Purpose: Admit complete state, binding and request batches for live use.
 * Owns: Staged store mutation and prepared query buffers.
 * Launch shape: One 64-thread block per tick.
 * Lifetime: Explicit cognitive operation within the existing tick graph. */
#include "cognitive/live_focus.cuh"
#include "cognitive/live_retain.cuh"
#include "cognitive/recall_search.cuh"
#include "cognitive/live_cache.cuh"

__device__ aotx_cognitive_store aotx_live_candidate, aotx_live_scratch;
__device__ aotx_recall_scratch aotx_live_search_scratch[AOTX_RECALL_BATCH];

__device__ bool aotx_live_busy(uint32_t slot) {
    return slot >= AOTX_SLOTS || aotx_agents.agent[slot].state != AOTX_AGENT_STATE_IDLE ||
        aotx_agents.agent[slot].task != ~0u || aotx_agent_gear[slot].has_message || aotx_say.slot[slot].wanted;
}
__global__ void aotx_live_stage(void) {
    if (aotx_sched.held) return;
    if (threadIdx.x < AOTX_SLOTS) aotx_live_cache_release(threadIdx.x);
    __syncthreads();
    if (aotx_live.phase != AOTX_LIVE_READY) return;
    uint32_t op = aotx_live.op;
    if (!threadIdx.x) {
        aotx_live.status = 0; aotx_live.auto_mode = aotx_live.auto_count = 0;
        if (op == AOTX_LIVE_LOAD || op == AOTX_LIVE_UPDATE) {
            if (op == AOTX_LIVE_LOAD ? aotx_live.ready : !aotx_live.ready) aotx_live.status = AOTX_COG_DENIED;
            for (uint32_t j = 0; j < AOTX_SLOTS; ++j)
                if (aotx_live_bound(j) && aotx_live_busy(j)) aotx_live.status = AOTX_COG_DENIED;
            if (op == AOTX_LIVE_LOAD) {
                uint64_t bytes = aotx_live.total, checkpoint = aotx_cog_u64(aotx_live.input), tail = aotx_cog_u64(aotx_live.input + 8);
                if (bytes < 16 || checkpoint < AOTX_COG_HEADER || checkpoint > AOTX_COG_IMAGE ||
                    tail > AOTX_COG_IMAGE || (tail && tail < AOTX_COG_HEADER) ||
                    bytes != 16 + checkpoint + tail) aotx_live.status = AOTX_COG_FORMAT;
            }
        } else if (op == AOTX_LIVE_BIND) aotx_live.status = aotx_live_bind_check();
        else if (op == AOTX_LIVE_QUERY || op == AOTX_LIVE_TEXT)
            aotx_live.status = aotx_live_query_check(op == AOTX_LIVE_TEXT);
        else if (op == AOTX_LIVE_RETAIN) aotx_live.status = aotx_live_retain_check();
        else aotx_live.status = AOTX_COG_FORMAT;
    }
    __syncthreads();
    if (!aotx_live.status && op == AOTX_LIVE_LOAD) {
        uint64_t checkpoint = aotx_cog_u64(aotx_live.input), tail = aotx_cog_u64(aotx_live.input + 8);
        aotx_cognitive_restore_block(&aotx_live_candidate, &aotx_live_scratch,
            aotx_live.input + 16, checkpoint, &aotx_live.result);
        __syncthreads();
        if (!aotx_live.result.status && tail)
            aotx_cognitive_apply_block(&aotx_live_candidate, &aotx_live_scratch,
                aotx_live.input + 16 + checkpoint, tail, &aotx_live.result);
        __syncthreads();
        if (!aotx_live.result.status)
            for (uint32_t j = threadIdx.x; j < sizeof(aotx_live_store); j += blockDim.x)
                ((unsigned char *)&aotx_live_store)[j] = ((unsigned char *)&aotx_live_candidate)[j];
        __syncthreads();
        if (!threadIdx.x) {
            aotx_live.status = aotx_live.result.status;
            if (!aotx_live.status) aotx_live.ready = 1;
        }
    } else if (!aotx_live.status && op == AOTX_LIVE_UPDATE) {
        aotx_cognitive_apply_block(&aotx_live_store, &aotx_live_scratch,
            aotx_live.input, aotx_live.total, &aotx_live.result);
        __syncthreads();
        if (!threadIdx.x) aotx_live.status = aotx_live.result.status;
    }
    __syncthreads();
    if (threadIdx.x) return;
    aotx_live.count = 0;
    if (op == AOTX_LIVE_QUERY || op == AOTX_LIVE_TEXT || op == AOTX_LIVE_RETAIN) {
        aotx_live.text_mode = op == AOTX_LIVE_RETAIN ? 2 : op == AOTX_LIVE_TEXT;
        aotx_live.request_seq = aotx_live.source_seq;
        for (uint32_t j = 0; j < 16; ++j) aotx_live.query_id[j] = aotx_live.transfer_id[j];
        if (!aotx_live.status) {
            aotx_live.count = aotx_cog_u32(aotx_live.input + 8);
            aotx_live_make_header(aotx_live.requests, "AOTXREQ1", aotx_live.count, AOTX_RECALL_QUERY);
            for (uint32_t i = 0; i < aotx_live.count; ++i) {
                if (op == AOTX_LIVE_RETAIN) {
                    for (uint32_t j = 0; j < AOTX_LIVE_RETAIN_ROW; ++j)
                        aotx_live.retain_rows[i][j] = aotx_live.input[64 + i * AOTX_LIVE_RETAIN_ROW + j];
                    continue;
                }
                const unsigned char *r = aotx_live.input + 64 + i * AOTX_LIVE_QUERY_ROW;
                for (uint32_t j = 0; j < 64; ++j) aotx_live.prefixes[i][j] = r[j];
                for (uint32_t j = 0; j < AOTX_RECALL_QUERY; ++j) aotx_live.requests[64 + i * AOTX_RECALL_QUERY + j] = r[64 + j];
                uint32_t status = aotx_live_focus_query(r, aotx_live.requests + 64 + i * AOTX_RECALL_QUERY);
                if (status && !aotx_live.status) aotx_live.status = status;
            }
        }
        aotx_live.phase = aotx_seam.replaying ? AOTX_LIVE_WAIT : AOTX_LIVE_SEARCH;
        if (!aotx_seam.replaying && aotx_live.text_mode == 1 && !aotx_live.status) aotx_live_text_begin();
    } else {
        if (!aotx_live.status && op == AOTX_LIVE_BIND) {
            aotx_live.count = aotx_cog_u32(aotx_live.input + 8);
            for (uint32_t i = 0; i < aotx_live.count; ++i) {
                const unsigned char *r = aotx_live.input + 64 + i * AOTX_LIVE_BIND_ROW;
                aotx_live_binding *b = aotx_live_bindings + aotx_cog_u32(r);
                b->auto_retain = aotx_cog_u32(r + 60);
                b->active = 1; b->pages = aotx_cog_u32(r + 56); b->scope = aotx_cog_u32(r + 4);
                for (uint32_t j = 0; j < 16; ++j) { b->principal[j] = r[8 + j]; b->room[j] = r[24 + j]; b->conversation[j] = r[40 + j]; }
            }
        }
        aotx_live_note(op, aotx_live.status, aotx_live.count);
        if (aotx_live.status) ++aotx_live.refused; else ++aotx_live.accepted;
        aotx_live.phase = AOTX_LIVE_IDLE;
    }
    aotx_live.received = 0;
}
__global__ void aotx_live_search(void) {
    if (aotx_sched.held || aotx_live.phase != AOTX_LIVE_SEARCH || aotx_live.status || aotx_live.text_mode == 2) return;
    aotx_recall_search_block(&aotx_live_store, aotx_live.requests,
        64 + (uint64_t)aotx_live.count * AOTX_RECALL_QUERY, aotx_live.results, aotx_live_search_scratch, aotx_live.count);
}
