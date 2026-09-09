/* Purpose: Validate live binding and request envelopes before publication.
 * Owns: No state; reads the resident store and current agent lifetimes.
 * Launch shape: Serial batch admission on the device.
 * Lifetime: One complete transfer. */
#ifndef AOTX_COGNITIVE_LIVE_VALIDATE_CUH
#define AOTX_COGNITIVE_LIVE_VALIDATE_CUH
#include "cognitive/live.cuh"
#include "cognitive/recall_context.cuh"
#include "agent/agent_state.cuh"
#include "agent/transcript.cuh"
#include "cli/prompt.cuh"
#include "sched/sched.cuh"

__device__ inline uint32_t aotx_live_header(const unsigned char *p, uint32_t bytes,
    const char *magic, uint32_t row, bool choice = false) {
    if (bytes < AOTX_LIVE_HEADER) return AOTX_COG_FORMAT;
    uint32_t n = aotx_cog_u32(p + 8);
    if ((!n && !choice) || n > AOTX_RECALL_BATCH || bytes != AOTX_LIVE_HEADER + n * row ||
        !aotx_recall_magic(p, magic) || aotx_cog_u32(p + 12) != 1 || aotx_cog_u32(p + 40) != row ||
        !aotx_cog_zero(p + (choice ? 48 : 44), choice ? 16 : 20)) return AOTX_COG_FORMAT;
    if (!aotx_live.ready || !aotx_cog_equal(p + 16, aotx_live_store.lineage)) return AOTX_COG_SOURCE;
    return aotx_cog_u64(p + 32) == aotx_live_store.sequence ? AOTX_COG_OK : AOTX_COG_STALE;
}
__device__ inline void aotx_live_make_header(unsigned char *p, const char *magic,
    uint32_t count, uint32_t row) {
    for (uint32_t j = 0; j < AOTX_LIVE_HEADER; ++j) p[j] = 0;
    for (uint32_t j = 0; j < 8; ++j) p[j] = magic[j];
    aotx_cog_put(p + 8, count, 4); aotx_cog_put(p + 12, 1, 4);
    for (uint32_t j = 0; j < 16; ++j) p[16 + j] = aotx_live_store.lineage[j];
    aotx_cog_put(p + 32, aotx_live_store.sequence, 8); aotx_cog_put(p + 40, row, 4);
}
__device__ inline uint32_t aotx_live_bind_check(void) {
    const unsigned char *p = aotx_live.input;
    uint32_t status = aotx_live_header(p, aotx_live.total, "AOTXBND1", AOTX_LIVE_BIND_ROW);
    if (status) return status;
    uint32_t count = aotx_cog_u32(p + 8);
    for (uint32_t i = 0; i < count; ++i) {
        const unsigned char *r = p + 64 + i * AOTX_LIVE_BIND_ROW;
        uint32_t slot = aotx_cog_u32(r), scope = aotx_cog_u32(r + 4), pages = aotx_cog_u32(r + 56);
        if (slot >= AOTX_SLOTS || scope > AOTX_COG_INSTANCE || !pages || pages > AOTX_KV_PAGES_EACH ||
            !aotx_cog_zero(r + 60, 4) || aotx_cog_zero(r + 8, 16) || aotx_cog_zero(r + 40, 16) ||
            (scope == AOTX_COG_ROOM ? aotx_cog_zero(r + 24, 16) : !aotx_cog_zero(r + 24, 16))) return AOTX_COG_FORMAT;
        if (aotx_live_bound(slot) || aotx_live_busy(slot) || aotx_agents.agent[slot].turn ||
            aotx_transcript[slot].count || aotx_agent_gear[slot].opens) return AOTX_COG_DENIED;
        for (uint32_t j = 0; j < AOTX_TASK_SLOTS; ++j)
            if (aotx_task_used[j] && aotx_agents.task[j].agent == slot &&
                aotx_agents.task[j].state == AOTX_TASK_PENDING) return AOTX_COG_DENIED;
        for (uint32_t j = 0; j < AOTX_SLOTS; ++j)
            if (aotx_live_bound(j) && aotx_cog_equal(r + 40, aotx_live_bindings[j].conversation)) return AOTX_COG_REFERENCE;
        for (uint32_t j = 0; j < i; ++j) {
            const unsigned char *old = p + 64 + j * AOTX_LIVE_BIND_ROW;
            if (aotx_cog_u32(old) == slot || aotx_cog_equal(old + 40, r + 40)) return AOTX_COG_REFERENCE;
        }
    }
    return AOTX_COG_OK;
}
__device__ inline uint32_t aotx_live_query_check(bool text = false) {
    const unsigned char *p = aotx_live.input;
    uint32_t status = aotx_live_header(p, aotx_live.total, text ? "AOTXTXT1" : "AOTXLIV1", AOTX_LIVE_QUERY_ROW);
    if (status) return status;
    uint32_t count = aotx_cog_u32(p + 8);
    for (uint32_t i = 0; i < count; ++i) {
        const unsigned char *r = p + 64 + i * AOTX_LIVE_QUERY_ROW, *q = r + 64;
        uint32_t slot = aotx_cog_u32(r);
        if (slot >= AOTX_SLOTS || aotx_cog_u32(r + 4) > 1 || !aotx_cog_zero(r + 8, 8) || !aotx_cog_zero(r + 40, 24)) return AOTX_COG_FORMAT;
        const aotx_live_binding *b = aotx_live_bindings + slot;
        if (!b->active || aotx_live_busy(slot) || !aotx_cog_equal(r + 16, b->conversation) ||
            !aotx_cog_equal(q + 16, b->principal) || !aotx_cog_equal(q + 32, b->room) ||
            aotx_cog_u32(q + 152) != b->scope) return AOTX_COG_DENIED;
        if (b->ordinal == UINT64_MAX || aotx_cog_u64(r + 32) != b->ordinal + 1) return AOTX_COG_SEQUENCE;
        status = aotx_recall_query_check(q, text);
        if (text && aotx_cog_u32(q + 148) > AOTX_LIVE_TEXT_BYTES) return AOTX_COG_CAPACITY;
        if (status) return status;
        for (uint32_t j = 0; j < i; ++j) {
            const unsigned char *old = p + 64 + j * AOTX_LIVE_QUERY_ROW;
            if (aotx_cog_u32(old) == slot || aotx_cog_equal(old + 64, q) ||
                aotx_cog_equal(old + 112, q + 48) || aotx_cog_equal(old + 64, q + 48) ||
                aotx_cog_equal(old + 112, q)) return AOTX_COG_REFERENCE;
        }
    }
    return AOTX_COG_OK;
}
#endif
