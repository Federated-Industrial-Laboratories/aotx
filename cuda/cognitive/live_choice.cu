/* Purpose: Record exact live choices and restore them without vector search.
 * Owns: The bounded choice transfer and publication of current conversation state.
 * Launch shape: One 64-thread block; one row per thread at publication.
 * Lifetime: From complete query admission through recorded choice delivery. */
#include "cognitive/live_auto.cuh"
#include "shared/bridge.cuh"
#include "cognitive/cold.cuh"
#include "reflection/state.cuh"

static __device__ unsigned char aotx_live_choice_part[AOTX_BODY_BYTES];

__global__ void aotx_live_decide(void) {
    if (aotx_sched.held) return;
    if (aotx_review.active) { aotx_review_decide(); return; }
    if (aotx_cold.active) { aotx_cold_replay(); return; }
    if (aotx_appraisal.active) { aotx_appraisal_decide(); return; }
    if (aotx_live.intake_mode && aotx_live.phase == AOTX_LIVE_SEARCH) {
        if (!threadIdx.x) {
            for (uint32_t j = 0; j < aotx_live.count; ++j)
                if (!aotx_live.status) aotx_live.status = aotx_live.results[j].status;
            if (aotx_live.status) aotx_live.phase = AOTX_INTAKE_DONE;
            else aotx_intake_begin();
        }
        return;
    }
    if (aotx_live.auto_mode) { aotx_live_auto_decide(); return; }
    if (aotx_live.text_mode == 2) { aotx_live_retain_decide(); return; }
    bool replay = aotx_live.phase == AOTX_LIVE_REPLAY;
    bool text = aotx_live.text_mode != 0;
    uint32_t row = text ? AOTX_LIVE_TEXT_CHOICE_ROW : AOTX_LIVE_CHOICE_ROW;
    const char *magic = text ? "AOTXTCH1" : "AOTXCHO1";
    if (aotx_live.phase != AOTX_LIVE_SEARCH && !replay) return;
    if (!replay) {
        if (threadIdx.x) return;
        if (!aotx_live.status) {
            for (uint32_t i = 0; i < aotx_live.count; ++i) {
                aotx_live.searches += aotx_live.results[i].searches;
                if (aotx_live.results[i].status && !aotx_live.status) aotx_live.status = aotx_live.results[i].status;
            }
        }
        uint32_t count = aotx_live.status ? 0 : aotx_live.count;
        aotx_live_make_header(aotx_live.choices, magic, count, row);
        aotx_cog_put(aotx_live.choices + 44, aotx_live.status, 4);
        for (uint32_t i = 0; i < count; ++i) {
            unsigned char *r = aotx_live.choices + 64 + i * row;
            for (uint32_t j = 0; j < 64; ++j) r[j] = aotx_live.prefixes[i][j];
            if (text) for (uint32_t j = 0; j < AOTX_RECALL_QUERY; ++j)
                r[64 + j] = aotx_live.requests[64 + i * AOTX_RECALL_QUERY + j];
            for (uint32_t j = 0; j < AOTX_RECALL_SELECTION; ++j) r[row - AOTX_RECALL_SELECTION + j] = aotx_live.results[i].selection[j];
        }
        aotx_live.choice_bytes = 64 + count * row;
        aotx_live.written = 0; aotx_live.phase = AOTX_LIVE_WRITE;
        return;
    }
    __shared__ uint32_t error, refusal;
    if (!threadIdx.x) {
        const unsigned char *p = aotx_live.input;
        uint32_t count = aotx_cog_u32(p + 8);
        refusal = aotx_cog_u32(p + 44); error = 0;
        if (aotx_live.total < 64 || aotx_live.total > (text ? AOTX_LIVE_TEXT_CHOICES : AOTX_LIVE_CHOICES) ||
            !aotx_recall_magic(p, magic) || aotx_cog_u32(p + 12) != 1 ||
            aotx_cog_u32(p + 40) != row || !aotx_cog_zero(p + 48, 16) ||
            !aotx_cog_equal(aotx_live.transfer_id, aotx_live.query_id) ||
            !aotx_cog_equal(p + 16, aotx_live_store.lineage) ||
            aotx_cog_u64(p + 32) != aotx_live_store.sequence || refusal > AOTX_COG_UNAVAILABLE ||
            count > AOTX_RECALL_BATCH || aotx_live.total != 64 + count * row ||
            (refusal ? count != 0 : count != aotx_live.count || !count || aotx_live.status) ||
            (aotx_live.status && aotx_live.status != refusal)) error = AOTX_COG_REFERENCE;
    }
    __syncthreads();
    if (!error && !refusal && threadIdx.x < aotx_live.count) {
        uint32_t i = threadIdx.x;
        const unsigned char *p = aotx_live.input + 64 + i * row;
        unsigned char *q = aotx_live.requests + 64 + i * AOTX_RECALL_QUERY;
        uint32_t prepared = text ? aotx_live_text_recorded(q, p + 64) : 0;
        if (text && !prepared) for (uint32_t j = 0; j < AOTX_RECALL_QUERY; ++j) q[j] = p[64 + j];
        aotx_recall_result *out = aotx_live.results + i;
        for (uint32_t j = 0; j < sizeof(*out); ++j) ((unsigned char *)out)[j] = 0;
        out->cut = aotx_live_store.sequence;
        for (uint32_t j = 0; j < 16; ++j) { out->request_id[j] = q[j]; out->selection_id[j] = q[48 + j]; }
        for (uint32_t j = 0; j < AOTX_RECALL_SELECTION; ++j) out->selection[j] = p[row - AOTX_RECALL_SELECTION + j];
        uint32_t status = prepared ? prepared : (!aotx_cog_equal(p, aotx_live.prefixes[i], 64)
            ? AOTX_COG_REFERENCE : aotx_live_selected(q, out));
        if (status) atomicCAS(&error, 0u, status);
    }
    __syncthreads();
    if (!threadIdx.x) {
        aotx_live.received = 0;
        if (error) {
            aotx_live.fatal = 1; ++aotx_live.refused;
            aotx_live_note(text ? AOTX_LIVE_TEXT_CHOICE : AOTX_LIVE_CHOICE, error, 0); aotx_live.phase = AOTX_LIVE_IDLE;
        } else {
            aotx_live.status = refusal;
            aotx_live.choice_bytes = aotx_live.written = 0;
            aotx_live.phase = AOTX_LIVE_WRITE;
        }
    }
}

__global__ void aotx_live_commit(void) {
    if (aotx_sched.held || aotx_live.phase != AOTX_LIVE_WRITE) return;
    if (!threadIdx.x && !aotx_seam.replaying) {
        unsigned char *body = aotx_live_choice_part;
        for (uint32_t part = 0; part < AOTX_LIVE_EMIT && aotx_live.written < aotx_live.choice_bytes; ++part) {
            uint32_t left = aotx_live.choice_bytes - aotx_live.written;
            uint32_t bytes = left < AOTX_LIVE_DATA ? left : AOTX_LIVE_DATA;
            aotx_cog_put(body, 1, 4); aotx_cog_put(body + 4, aotx_live_result_op(), 4);
            for (uint32_t j = 0; j < 16; ++j) body[8 + j] = aotx_live.query_id[j];
            aotx_cog_put(body + 24, aotx_live.choice_bytes, 4); aotx_cog_put(body + 28, aotx_live.written, 4);
            for (uint32_t j = 0; j < bytes; ++j) body[32 + j] = aotx_live.choices[aotx_live.written + j];
            uint32_t flags = ((aotx_appraisal.active && aotx_appraisal.recovery) ||
                (aotx_cold.active && aotx_cold.recovery) || (aotx_review.active && aotx_review.recovery)) && !aotx_live.written ? AOTX_FLAG_ADMISSION : 0;
            aotx_seam_write(AOTX_WRITER_SYSTEM, AOTX_CLASS_A, AOTX_LIVE_RECORD, flags, body, 32 + bytes);
            aotx_seam.apply.state_hash = aotx_seam_fnv1a(aotx_seam.apply.state_hash, body, 32 + bytes);
            ++aotx_seam.apply.applied_count; aotx_live.written += bytes;
        }
    }
    __syncthreads();
    if (aotx_live.written != aotx_live.choice_bytes) return;
    if (aotx_review.active) { aotx_review_publish(); return; }
    if (aotx_cold.active) { aotx_cold_publish(); return; }
    if (aotx_appraisal.active) { aotx_appraisal_publish(); return; }
    uint32_t i = threadIdx.x;
    if (!aotx_live.status && aotx_live.auto_mode) aotx_live_auto_publish();
    if (!aotx_live.status && aotx_live.text_mode == 2) aotx_live_retain_publish();
    if (!aotx_live.status && aotx_live.text_mode != 2 && i < aotx_live.count) {
        uint32_t slot = aotx_cog_u32(aotx_live.prefixes[i]);
        aotx_live_binding *b = aotx_live_bindings + slot;
        const unsigned char *q = aotx_live.requests + 64 + i * AOTX_RECALL_QUERY;
        for (uint32_t j = 0; j < AOTX_RECALL_QUERY; ++j) b->query[j] = q[j];
        b->choice = aotx_live.results[i];
        b->context_bytes = b->choice.context_bytes - 8 - aotx_cog_u32(q + 148);
        b->ordinal = aotx_cog_u64(aotx_live.prefixes[i] + 32);
        if (aotx_shared_owns(slot)) aotx_shared_memory_choice(slot, 0);
        else aotx_agent_queue_message(slot, q + 4640, aotx_cog_u32(q + 148), aotx_live.request_seq);
    }
    __syncthreads();
    if (!threadIdx.x) {
        if (aotx_live.status) for (unsigned slot = 0; slot < AOTX_SLOTS; ++slot)
            if (aotx_shared_owns(slot) && aotx_shared_execution_slots[slot].stage == AOTX_SHARED_MEMORY)
                aotx_shared_memory_choice(slot, aotx_live.status);
        aotx_live_note(aotx_live.text_mode == 2 ? AOTX_LIVE_RETAIN :
            (aotx_live.text_mode ? AOTX_LIVE_TEXT : AOTX_LIVE_QUERY), aotx_live.status, aotx_live.status ? 0 : aotx_live.count);
        if (aotx_live.status) ++aotx_live.refused;
        else { aotx_live.accepted += aotx_live.count; if (aotx_seam.replaying) aotx_live.replays += aotx_live.count; }
        aotx_live.phase = AOTX_LIVE_IDLE;
    }
}
