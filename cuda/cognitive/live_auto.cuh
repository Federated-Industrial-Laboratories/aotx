/* Purpose: Record automatic memory and input admission as one atomic decision.
 * Owns: Combined result bytes, candidate store and per-binding focus publication.
 * Launch shape: One 64-thread block; one thread per input or retained row.
 * Lifetime: One query batch through exact journal recovery. */
#ifndef AOTX_COGNITIVE_LIVE_AUTO_CUH
#define AOTX_COGNITIVE_LIVE_AUTO_CUH
#include "cognitive/live_auto_rows.cuh"
#include "cognitive/intake_encode.cuh"
#include "appraisal/appraisal.cuh"

__device__ __forceinline__ void aotx_live_auto_decide(void) {
    bool replay = aotx_live.phase == AOTX_LIVE_REPLAY;
    if (aotx_live.phase != AOTX_LIVE_SEARCH && aotx_live.phase != AOTX_INTAKE_DONE && !replay) return;
    __shared__ uint32_t error, refusal, tail_offset, tail_bytes, objects, row_status[64];
    uint32_t i = threadIdx.x;
    if (!i) {
        error = refusal = 0;
        if (!replay && !aotx_live.status && !aotx_shared_memory_authorized()) aotx_live.status = AOTX_COG_DENIED;
        if (replay) error = aotx_live_auto_header(&refusal);
        else if (!aotx_live.status) for (uint32_t j = 0; j < aotx_live.count; ++j) {
            aotx_live.searches += aotx_live.results[j].searches;
            if (!aotx_live.status) aotx_live.status = aotx_live.results[j].status;
        }
    }
    __syncthreads();
    if (replay && !error && !refusal && i < aotx_live.count) {
        row_status[i] = aotx_live_auto_recorded(i);
        if (!row_status[i] && aotx_live.intake_mode) row_status[i] = aotx_intake_recorded(i);
    }
    __syncthreads();
    if (!i && replay && !error && !refusal)
        for (uint32_t j = 0; j < aotx_live.count && !error; ++j) error = row_status[j];
    __syncthreads();
    if (replay && (error || refusal)) {
        if (!i) {
            if (error) aotx_live_auto_fatal(error);
            else {
                aotx_live.status = refusal; aotx_live.received = aotx_live.choice_bytes = aotx_live.written = 0;
                aotx_live.phase = AOTX_LIVE_WRITE;
            }
        }
        return;
    }
    if (!i) {
        if (!aotx_live.status) aotx_live.status = aotx_live_auto_rows();
        if (!aotx_live.status && aotx_live.intake_mode) aotx_live.status = aotx_intake_prepare_rows();
        if (aotx_live.status) aotx_live.auto_count = 0;
        objects = aotx_live.intake_mode && !aotx_live.status ? aotx_intake.objects : aotx_live.auto_count * 3;
        uint32_t payload = 0;
        for (uint32_t j = 0; j < aotx_live.auto_count; ++j) payload += aotx_retain_payload_bytes(aotx_live.retain_rows[j]);
        if (aotx_live.intake_mode && !aotx_live.status) payload = aotx_intake.payload;
        if (!aotx_live.status) aotx_live.status = aotx_appraisal_queue_prepare(&objects, &payload);
        if (aotx_live.status) { aotx_live.auto_count = 0; aotx_appraisal.enqueue_count = 0; }
        tail_offset = 64 + aotx_live.count * aotx_live_auto_stride();
        tail_bytes = AOTX_COG_HEADER + objects * AOTX_COG_OBJECT + payload;
        aotx_live.choice_bytes = aotx_live.status ? 64 : tail_offset + tail_bytes;
    }
    __syncthreads();
    for (uint32_t j = i; j < aotx_live.choice_bytes; j += blockDim.x) aotx_live.choices[j] = 0;
    __syncthreads();
    if (!aotx_live.status && i < aotx_live.count) {
        unsigned char *r = aotx_live.choices + 64 + i * aotx_live_auto_stride();
        for (uint32_t j = 0; j < 64; ++j) r[j] = aotx_live.prefixes[i][j];
        for (uint32_t j = 0; j < AOTX_RECALL_QUERY; ++j) r[64 + j] = aotx_live.requests[64 + i * AOTX_RECALL_QUERY + j];
        for (uint32_t j = 0; j < AOTX_RECALL_SELECTION; ++j) r[64 + AOTX_RECALL_QUERY + j] = aotx_live.results[i].selection[j];
        if (aotx_live.intake_mode) { aotx_intake_metadata(i); aotx_intake_encode(aotx_live.choices + tail_offset, i); }
    }
    if (i < aotx_live.auto_count) {
        unsigned char *out = aotx_live_auto_retained(i);
        for (uint32_t j = 0; j < AOTX_LIVE_RETAIN_ROW; ++j) out[j] = aotx_live.retain_rows[i][j];
        row_status[i] = aotx_retain_row_check(out, aotx_live.auto_count);
        if (!row_status[i]) row_status[i] = aotx_retain_focus(out);
        if (!row_status[i]) aotx_retain_encode(aotx_live.choices + tail_offset, i, aotx_live.auto_count, objects);
        if (!row_status[i]) aotx_appraisal_queue_encode(aotx_live.choices + tail_offset, i, objects);
    }
    __syncthreads();
    if (!i) {
        for (uint32_t j = 0; j < aotx_live.auto_count && !error; ++j) error = row_status[j];
        if (!error && !aotx_live.status) error = aotx_retain_unique(aotx_live.auto_count);
        if (!error && !aotx_live.status) aotx_retain_header(aotx_live.choices + tail_offset,
            aotx_live.auto_count, tail_bytes - AOTX_COG_HEADER - objects * AOTX_COG_OBJECT, objects);
    }
    __syncthreads();
    for (uint32_t j = i; !error && !aotx_live.status && j < sizeof(aotx_live_store); j += blockDim.x)
        ((unsigned char *)&aotx_live_candidate)[j] = ((const unsigned char *)&aotx_live_store)[j];
    __syncthreads();
    if (!error && !aotx_live.status) {
        aotx_cognitive_apply_block(&aotx_live_candidate, &aotx_live_scratch,
            aotx_live.choices + tail_offset, tail_bytes, &aotx_live.result);
        __syncthreads();
        if (!i) error = aotx_live.result.status;
    }
    __syncthreads();
    if (!error && !aotx_live.status && i < aotx_live.count) {
        row_status[i] = aotx_live.intake_mode ? aotx_intake_context(i) : 0;
        if (!row_status[i]) row_status[i] = aotx_recall_render(&aotx_live_candidate,
            aotx_live.requests + 64 + i * AOTX_RECALL_QUERY, aotx_live.results + i);
        if (!row_status[i]) {
            aotx_live.results[i].cut = aotx_live_candidate.sequence;
            if (aotx_live.intake_mode) {
                unsigned char *selected = aotx_live.choices + 64 + i * aotx_live_auto_stride() +
                    AOTX_LIVE_AUTO_ROW + AOTX_INTAKE_META + AOTX_INTAKE_REPLY;
                for (uint32_t j = 0; j < AOTX_RECALL_SELECTION; ++j) selected[j] = aotx_live.results[i].selection[j];
            }
        }
    }
    __syncthreads();
    if (!i) {
        if (!error && !aotx_live.status) for (uint32_t j = 0; j < aotx_live.count && !error; ++j) error = row_status[j];
        if (error) aotx_live.status = error;
        uint32_t count = aotx_live.status ? 0 : aotx_live.count;
        if (!count) { tail_bytes = 0; aotx_live.choice_bytes = 64; }
        aotx_live_make_header(aotx_live.choices, aotx_live_auto_magic(), count, aotx_live_auto_stride());
        aotx_cog_put(aotx_live.choices + 44, aotx_live.status, 4); aotx_cog_put(aotx_live.choices + 48, tail_bytes, 8);
        error = replay && aotx_live.total != aotx_live.choice_bytes ? AOTX_COG_REFERENCE : 0;
    }
    __syncthreads();
    if (replay && !error) for (uint32_t j = i; j < aotx_live.choice_bytes; j += blockDim.x)
        if (aotx_live.input[j] != aotx_live.choices[j]) atomicExch(&error, AOTX_COG_REFERENCE);
    __syncthreads();
    if (!i) {
        if (error) aotx_live_auto_fatal(error);
        else {
            aotx_live.received = aotx_live.written = 0;
            if (replay) aotx_live.choice_bytes = 0;
            aotx_live.phase = AOTX_LIVE_WRITE;
        }
    }
}
__device__ __forceinline__ void aotx_live_auto_publish(void) {
    for (uint32_t j = threadIdx.x; j < sizeof(aotx_live_store); j += blockDim.x)
        ((unsigned char *)&aotx_live_store)[j] = ((const unsigned char *)&aotx_live_candidate)[j];
    uint32_t i = threadIdx.x;
    if (i < aotx_live.auto_count) {
        const unsigned char *r = aotx_live_auto_retained(i);
        aotx_live_binding *b = aotx_live_bindings + aotx_cog_u32(r);
        b->focus_count = aotx_cog_u32(r + 160);
        for (uint32_t j = 0; j < AOTX_RECALL_PINS * 24; ++j) ((unsigned char *)b->focus)[j] = r[192 + j];
    }
}
#endif
