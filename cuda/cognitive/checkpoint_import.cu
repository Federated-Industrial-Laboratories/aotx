/* Purpose: Validate and publish checkpoint bindings with their exact typed store.
 * Owns: Staged bindings and all-or-nothing attachment to fresh idle slots.
 * Launch shape: One 64-thread block strides the complete binding batch.
 * Lifetime: One completed file transfer; failed input changes no live state. */
#include "cognitive/checkpoint.cuh"
#include "cognitive/live_validate.cuh"

static __device__ aotx_live_binding aotx_checkpoint_bindings[AOTX_SLOTS];

static __device__ void aotx_cp_read_bytes(unsigned char *out, const unsigned char *in, uint32_t n) {
    for (uint32_t j = 0; j < n; ++j) out[j] = in[j];
}
static __device__ void aotx_cp_read_result(aotx_recall_result *r, const unsigned char *p) {
    r->status = aotx_cog_u32(p); r->count = aotx_cog_u32(p + 4);
    r->context_bytes = aotx_cog_u32(p + 8); r->searches = aotx_cog_u32(p + 12);
    r->cut = aotx_cog_u64(p + 16);
    aotx_cp_read_bytes(r->request_id, p + 24, 16); aotx_cp_read_bytes(r->selection_id, p + 40, 16);
    for (uint32_t j = 0; j < AOTX_RECALL_LIMIT; ++j) {
        r->index[j] = aotx_cog_u32(p + 56 + 4 * j);
        r->reason[j] = aotx_cog_u32(p + 120 + 4 * j);
    }
    aotx_cp_read_bytes(r->selection, p + 184, AOTX_RECALL_SELECTION);
    aotx_cp_read_bytes(r->context, p + 184 + AOTX_RECALL_SELECTION, AOTX_RECALL_CONTEXT);
}
static __device__ uint32_t aotx_cp_binding(const unsigned char *p, uint32_t row) {
    uint32_t slot = aotx_cog_u32(p), scope = aotx_cog_u32(p + 8), pages = aotx_cog_u32(p + 4);
    if (slot >= AOTX_SLOTS || scope > AOTX_COG_INSTANCE || !pages || pages > AOTX_KV_PAGES_EACH ||
        aotx_cog_u32(p + 72) > AOTX_RECALL_PINS || aotx_cog_u32(p + 76) > 1 ||
        !aotx_cog_zero(p + 88, 40) || aotx_cog_zero(p + 24, 16) || aotx_cog_zero(p + 56, 16) ||
        (scope == AOTX_COG_ROOM ? aotx_cog_zero(p + 40, 16) : !aotx_cog_zero(p + 40, 16))) return AOTX_COG_FORMAT;
    if (aotx_live_bound(slot) || aotx_live_busy(slot) || aotx_agents.agent[slot].turn ||
        aotx_transcript[slot].count || aotx_agent_gear[slot].opens) return AOTX_COG_DENIED;
    for (uint32_t j = 0; j < AOTX_TASK_SLOTS; ++j)
        if (aotx_task_used[j] && aotx_agents.task[j].agent == slot &&
            aotx_agents.task[j].state == AOTX_TASK_PENDING) return AOTX_COG_DENIED;
    for (uint32_t j = 0; j < row; ++j) {
        const unsigned char *old = aotx_live.input + AOTX_CP_HEADER + j * AOTX_CP_ROW;
        if (aotx_cog_u32(old) == slot || aotx_cog_equal(old + 56, p + 56)) return AOTX_COG_REFERENCE;
    }
    aotx_live_binding *b = aotx_checkpoint_bindings + row;
    b->active = 1; b->pages = pages; b->scope = scope; b->context_bytes = aotx_cog_u32(p + 12);
    b->ordinal = aotx_cog_u64(p + 16); b->focus_count = aotx_cog_u32(p + 72); b->auto_retain = aotx_cog_u32(p + 76);
    aotx_cp_read_bytes(b->principal, p + 24, 16); aotx_cp_read_bytes(b->room, p + 40, 16);
    aotx_cp_read_bytes(b->conversation, p + 56, 16);
    aotx_cp_read_bytes(b->query, p + 128, AOTX_RECALL_QUERY);
    const unsigned char *saved = p + 128 + AOTX_RECALL_QUERY;
    aotx_cp_read_result(&b->choice, saved);
    aotx_cp_read_bytes(&b->focus[0][0], saved + AOTX_CP_RESULT, AOTX_RECALL_PINS * 24);
    if (!b->ordinal) return !b->context_bytes && !b->focus_count &&
        aotx_cog_zero(p + 128, AOTX_CP_ROW - 128) ? AOTX_COG_OK : AOTX_COG_FORMAT;
    uint32_t status = aotx_recall_query_check(b->query);
    if (status) return status;
    if (!aotx_cog_equal(b->query + 16, b->principal) || !aotx_cog_equal(b->query + 32, b->room) ||
        aotx_cog_u32(b->query + 152) != b->scope || b->choice.status ||
        b->choice.count > AOTX_RECALL_LIMIT || b->choice.context_bytes > AOTX_RECALL_CONTEXT ||
        b->context_bytes > b->choice.context_bytes ||
        b->context_bytes + 8 + aotx_cog_u32(b->query + 148) != b->choice.context_bytes || b->choice.cut > aotx_live_candidate.sequence ||
        !aotx_cog_equal(b->choice.request_id, b->query) ||
        !aotx_cog_equal(b->choice.selection_id, b->query + 48) ||
        aotx_cog_u32(b->choice.selection) != 1 ||
        aotx_cog_u32(b->choice.selection + 4) != b->choice.count) return AOTX_COG_REFERENCE;
    if (!aotx_cog_zero(b->choice.selection + 8, 8) ||
        !aotx_cog_zero(b->choice.selection + 16 + b->choice.count * 32, (AOTX_RECALL_LIMIT - b->choice.count) * 32) ||
        !aotx_cog_equal(b->choice.context + b->context_bytes, (const unsigned char *)"[input]\n", 8) ||
        !aotx_cog_equal(b->choice.context + b->context_bytes + 8, b->query + 4640, aotx_cog_u32(b->query + 148)))
        return AOTX_COG_FORMAT;
    for (uint32_t j = 0; j < b->choice.count; ++j) {
        const unsigned char *ref = b->choice.selection + 16 + j * 32;
        if (b->choice.index[j] >= aotx_live_candidate.count || aotx_cog_u32(ref + 24) != 1 || aotx_cog_u32(ref + 28) ||
            aotx_cog_find(&aotx_live_candidate, ref, aotx_cog_u64(ref + 16)) != (int)b->choice.index[j]) return AOTX_COG_REFERENCE;
    }
    if (b->choice.cut == aotx_live_candidate.sequence) {
        status = aotx_recall_render(&aotx_live_candidate, b->query, &b->choice);
        if (status) return status;
        for (uint32_t j = 0; j < b->choice.count; ++j)
            if (b->choice.reason[j] != aotx_cog_u32(saved + 120 + j * 4)) return AOTX_COG_REFERENCE;
        if (b->choice.context_bytes != aotx_cog_u32(saved + 8) ||
            b->context_bytes + 8 + aotx_cog_u32(b->query + 148) != b->choice.context_bytes ||
            !aotx_cog_equal(b->choice.context, saved + 184 + AOTX_RECALL_SELECTION, b->choice.context_bytes))
            return AOTX_COG_REFERENCE;
    }
    for (uint32_t j = 0; j < b->focus_count; ++j)
        if (aotx_cog_find(&aotx_live_candidate, b->focus[j], aotx_cog_u64(b->focus[j] + 16)) < 0)
            return AOTX_COG_REFERENCE;
    return AOTX_COG_OK;
}
__device__ void aotx_checkpoint_import(void) {
    __shared__ uint32_t status, count, base;
    __shared__ unsigned char audit[AOTX_BODY_BYTES];
    const unsigned char *p = aotx_live.input;
    if (!threadIdx.x) {
        status = 0; count = 0; base = 0;
        if (aotx_live.ready || aotx_live_admission_pressure()) status = AOTX_COG_DENIED;
        else if (aotx_live.total < AOTX_CP_HEADER || !aotx_recall_magic(p, "AOTXLCP1") ||
            aotx_cog_u32(p + 8) != 1 || aotx_cog_u32(p + 12) != AOTX_CP_ROW ||
            !aotx_cog_zero(p + 20, 4) || !aotx_cog_zero(p + 80, 48) ||
            aotx_cog_u32(p + 16) > AOTX_SLOTS || !aotx_cog_u64(p + 64)) status = AOTX_COG_FORMAT;
        else {
            count = aotx_cog_u32(p + 16); base = AOTX_CP_HEADER + count * AOTX_CP_ROW;
            uint64_t bytes = aotx_cog_u64(p + 24);
            if (bytes < AOTX_COG_HEADER || bytes > AOTX_COG_IMAGE ||
                base + bytes != aotx_live.total) status = AOTX_COG_FORMAT;
        }
    }
    __syncthreads();
    if (!status) {
        aotx_cognitive_restore_block(&aotx_live_candidate, &aotx_live_scratch,
            p + base, aotx_cog_u64(p + 24), &aotx_live.result);
        __syncthreads();
        if (!threadIdx.x) {
            status = aotx_live.result.status;
            if (!status && (!aotx_cog_equal(p + 32, aotx_live_candidate.lineage) ||
                aotx_cog_u64(p + 48) != aotx_live_candidate.sequence ||
                aotx_cog_u64(p + 56) != aotx_live_candidate.tick)) status = AOTX_COG_REFERENCE;
        }
    }
    __syncthreads();
    if (!status) {
        for (uint32_t i = threadIdx.x; i < count; i += blockDim.x) {
            uint32_t error = aotx_cp_binding(p + AOTX_CP_HEADER + i * AOTX_CP_ROW, i);
            if (error) atomicCAS(&status, 0u, error);
        }
    }
    __syncthreads();
    if (!status) {
        for (uint32_t j = threadIdx.x; j < sizeof(aotx_live_store); j += blockDim.x)
            ((unsigned char *)&aotx_live_store)[j] = ((unsigned char *)&aotx_live_candidate)[j];
        for (uint32_t i = threadIdx.x; i < count; i += blockDim.x) {
            const unsigned char *row = p + AOTX_CP_HEADER + i * AOTX_CP_ROW;
            uint32_t slot = aotx_cog_u32(row);
            aotx_live_bindings[slot] = aotx_checkpoint_bindings[i];
            aotx_agents.agent[slot].turn = aotx_cog_u32(row + 80);
            aotx_agent_gear[slot].opens = aotx_cog_u32(row + 84);
        }
    }
    __syncthreads();
    if (!threadIdx.x) {
        if (!status) {
            aotx_live.ready = 1; aotx_live.accepted = aotx_cog_u64(p + 64);
            for (uint32_t start = 0; start < count; start += AOTX_RESUME_ROWS) {
                uint32_t rows = count - start;
                if (rows > AOTX_RESUME_ROWS) rows = AOTX_RESUME_ROWS;
                aotx_cog_put(audit, 1, 4); aotx_cog_put(audit + 4, rows, 4);
                aotx_cp_read_bytes(audit + 8, aotx_live.transfer_id, 16);
                aotx_cog_put(audit + 24, aotx_live.accepted, 8);
                for (uint32_t j = 0; j < rows; ++j) {
                    const unsigned char *row = p + AOTX_CP_HEADER + (start + j) * AOTX_CP_ROW;
                    aotx_cp_read_bytes(audit + 32 + j * 8, row, 4);
                    aotx_cp_read_bytes(audit + 36 + j * 8, row + 80, 4);
                }
                aotx_seam_write(AOTX_WRITER_SYSTEM, AOTX_CLASS_B, AOTX_REC_COGNITIVE_RESUME, 0, audit, 32 + rows * 8);
            }
        }
        else ++aotx_live.refused;
        aotx_live.status = status; aotx_live.received = 0; aotx_live.phase = AOTX_LIVE_IDLE;
        aotx_live_note(AOTX_CP_RESUME, status, status ? 0 : count);
    }
}
