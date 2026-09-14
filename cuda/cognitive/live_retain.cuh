/* Purpose: Record and publish GPU memory retention as one atomic batch.
 * Owns: Canonical staging and bounded working-set decisions.
 * Launch shape: One 64-thread block within the live decision and commit nodes.
 * Lifetime: Exact recorded mutation from the binding's last accepted input. */
#ifndef AOTX_COGNITIVE_LIVE_RETAIN_CUH
#define AOTX_COGNITIVE_LIVE_RETAIN_CUH
#include "cognitive/retain_encode.cuh"
#include "appraisal/appraisal.cuh"

__device__ __forceinline__ uint32_t aotx_live_retain_check(void) {
    const unsigned char *p = aotx_live.input;
    uint32_t status = aotx_live_header(p, aotx_live.total, "AOTXRTN1", AOTX_LIVE_RETAIN_ROW);
    if (status) return status;
    uint32_t count = aotx_cog_u32(p + 8);
    for (uint32_t i = 0; i < AOTX_SLOTS; ++i)
        if (aotx_live_bound(i) && aotx_live_busy(i)) return AOTX_COG_DENIED;
    if (count * 3 > AOTX_COG_OBJECTS - aotx_live_store.count || aotx_live_store.tick == UINT64_MAX ||
        aotx_live_store.sequence > UINT64_MAX - count * 3) return AOTX_COG_CAPACITY;
    uint64_t bytes = 0;
    for (uint32_t i = 0; i < count; ++i) {
        const unsigned char *r = p + 64 + i * AOTX_LIVE_RETAIN_ROW;
        if (aotx_cog_u32(r) >= AOTX_SLOTS) return AOTX_COG_FORMAT;
        bytes += aotx_retain_payload_bytes(r);
    }
    return bytes > AOTX_COG_PAYLOAD - aotx_live_store.bytes ? AOTX_COG_CAPACITY : AOTX_COG_OK;
}

__device__ __forceinline__ void aotx_live_retain_decide(void) {
    if (aotx_live.phase != AOTX_LIVE_SEARCH && aotx_live.phase != AOTX_LIVE_REPLAY) return;
    __shared__ uint32_t count, tail_offset, tail_bytes, error, objects;
    if (!threadIdx.x) {
        count = aotx_live.status ? 0 : aotx_live.count; error = 0;
    }
    __syncthreads();
    if (threadIdx.x < count) {
        uint32_t status = aotx_retain_row_check(aotx_live.retain_rows[threadIdx.x], count);
        aotx_live.results[threadIdx.x].status = status;
    }
    __syncthreads();
    if (!threadIdx.x) {
        for (uint32_t i = 0; i < count && !error; ++i) error = aotx_live.results[i].status;
        if (!error && count) error = aotx_retain_unique(count);
        if (error) { aotx_live.status = error; count = 0; }
        uint32_t payload = 0;
        for (uint32_t i = 0; i < count; ++i) payload += aotx_retain_payload_bytes(aotx_live.retain_rows[i]);
        objects = count * 3;
        if (count) error = aotx_appraisal_queue_prepare(&objects, &payload);
        if (error) { aotx_live.status = error; count = 0; }
        tail_offset = 64 + count * AOTX_LIVE_RETAINED_ROW;
        tail_bytes = count ? AOTX_COG_HEADER + objects * AOTX_COG_OBJECT + payload : 0;
        aotx_live.choice_bytes = tail_offset + tail_bytes;
    }
    __syncthreads();
    for (uint32_t j = threadIdx.x; j < aotx_live.choice_bytes; j += blockDim.x) aotx_live.choices[j] = 0;
    __syncthreads();
    if (threadIdx.x < count) {
        uint32_t i = threadIdx.x;
        unsigned char *r = aotx_live.choices + 64 + i * AOTX_LIVE_RETAINED_ROW;
        for (uint32_t j = 0; j < AOTX_LIVE_RETAIN_ROW; ++j) r[j] = aotx_live.retain_rows[i][j];
        uint32_t status = aotx_retain_focus(r);
        aotx_live.results[threadIdx.x].status = status;
        aotx_retain_encode(aotx_live.choices + tail_offset, i, count, objects);
        aotx_appraisal_queue_encode(aotx_live.choices + tail_offset, i, objects);
    }
    __syncthreads();
    if (!threadIdx.x && count) {
        for (uint32_t i = 0; i < count && !error; ++i) error = aotx_live.results[i].status;
        aotx_retain_header(aotx_live.choices + tail_offset, count,
            tail_bytes - AOTX_COG_HEADER - objects * AOTX_COG_OBJECT, objects);
    }
    __syncthreads();
    for (uint32_t j = threadIdx.x; count && !error && j < sizeof(aotx_live_store); j += blockDim.x)
        ((unsigned char *)&aotx_live_candidate)[j] = ((const unsigned char *)&aotx_live_store)[j];
    __syncthreads();
    if (count && !error) {
        aotx_cognitive_apply_block(&aotx_live_candidate, &aotx_live_scratch,
            aotx_live.choices + tail_offset, tail_bytes, &aotx_live.result);
        __syncthreads();
        if (!threadIdx.x) error = aotx_live.result.status;
    }
    __syncthreads();
    if (!threadIdx.x) {
        if (error) aotx_live.status = error;
        if (aotx_live.status) { count = 0; tail_bytes = 0; aotx_live.choice_bytes = 64; }
        aotx_live_make_header(aotx_live.choices, "AOTXRCH1", count, AOTX_LIVE_RETAINED_ROW);
        aotx_cog_put(aotx_live.choices + 44, aotx_live.status, 4);
        aotx_cog_put(aotx_live.choices + 48, tail_bytes, 8);
        error = 0;
        if (aotx_seam.replaying && (aotx_live.total != aotx_live.choice_bytes ||
            !aotx_cog_equal(aotx_live.transfer_id, aotx_live.query_id))) error = AOTX_COG_REFERENCE;
    }
    __syncthreads();
    if (aotx_seam.replaying && !error)
        for (uint32_t j = threadIdx.x; j < aotx_live.choice_bytes; j += blockDim.x)
            if (aotx_live.input[j] != aotx_live.choices[j]) atomicExch(&error, AOTX_COG_REFERENCE);
    __syncthreads();
    if (!threadIdx.x) {
        aotx_live.received = 0; aotx_live.written = 0;
        if (error) {
            aotx_live.fatal = 1; ++aotx_live.refused;
            aotx_live_note(AOTX_LIVE_RETAINED, error, 0); aotx_live.phase = AOTX_LIVE_IDLE;
        } else {
            if (aotx_seam.replaying) aotx_live.choice_bytes = 0;
            aotx_live.phase = AOTX_LIVE_WRITE;
        }
    }
}

__device__ __forceinline__ void aotx_live_retain_publish(void) {
    for (uint32_t j = threadIdx.x; j < sizeof(aotx_live_store); j += blockDim.x)
        ((unsigned char *)&aotx_live_store)[j] = ((const unsigned char *)&aotx_live_candidate)[j];
    uint32_t i = threadIdx.x;
    if (i < aotx_live.count) {
        const unsigned char *r = aotx_live.choices + 64 + i * AOTX_LIVE_RETAINED_ROW;
        aotx_live_binding *b = aotx_live_bindings + aotx_cog_u32(r);
        b->focus_count = aotx_cog_u32(r + 160);
        for (uint32_t j = 0; j < AOTX_RECALL_PINS * 24; ++j) ((unsigned char *)b->focus)[j] = r[192 + j];
    }
}

#endif
