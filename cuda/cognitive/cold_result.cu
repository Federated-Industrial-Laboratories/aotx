/* Purpose: Record bounded read results and replay exact residency changes.
 * Owns: Response validation and the portable result image.
 * Launch shape: One 64-thread block copies at most one transport chunk per tick.
 * Lifetime: One asynchronous read or its recorded result. */
#include "cognitive/cold.cuh"
#include "cognitive/codec.cuh"
#include "sched/sched.cuh"

__device__ void aotx_cold_result_header(void) {
    if (threadIdx.x) return;
    unsigned char *p = aotx_live.choices;
    for (uint32_t i = 0; i < AOTX_COLD_HEADER; ++i) p[i] = 0;
    for (uint32_t i = 0; i < 8; ++i) p[i] = "AOTXTIR2"[i];
    aotx_cog_put(p + 8, 1, 4); aotx_cog_put(p + 12, aotx_cold.status, 4);
    aotx_cog_put(p + 16, aotx_cold.mode, 4);
    for (uint32_t i = 0; i < 16; ++i) p[24 + i] = aotx_live_store.lineage[i];
    aotx_cog_put(p + 40, aotx_live_store.sequence, 8);
    if (!aotx_cold.status) { aotx_cog_put(p + 20, aotx_cold.count, 4); aotx_cog_put(p + 48, aotx_cold.bytes, 8); }
    aotx_live.status = aotx_cold.status;
    aotx_live.choice_bytes = AOTX_COLD_HEADER + (aotx_cold.status ? 0 : aotx_cold.count * AOTX_COG_OBJECT + aotx_cold.bytes);
    aotx_live.written = 0; aotx_live.phase = AOTX_LIVE_WRITE;
}
__global__ void aotx_cold_step(void) {
    if (aotx_sched.held || !aotx_cold.active || aotx_seam.replaying) return;
    if (aotx_live.phase != AOTX_COLD_WAIT && aotx_live.phase != AOTX_COLD_BUILD) return;
    aotx_cold_transport *r = aotx_checkpoint.ring ?
        (aotx_cold_transport *)((unsigned char *)aotx_checkpoint.ring + AOTX_CP_COLD_OFFSET) : NULL;
    if (!threadIdx.x && aotx_live.phase == AOTX_COLD_WAIT) {
        if (r && aotx_seam_acquire_sys(&r->response) == aotx_cold.serial) {
            if (r->status || r->bytes != aotx_cold.bytes) aotx_cold.status = AOTX_COG_UNAVAILABLE;
            aotx_live.phase = AOTX_COLD_BUILD;
        } else if (++aotx_cold.ticks >= AOTX_COLD_TICKS) {
            aotx_cold.status = AOTX_COG_UNAVAILABLE; aotx_live.phase = AOTX_COLD_BUILD;
        }
    }
    __syncthreads();
    if (aotx_live.phase != AOTX_COLD_BUILD) return;
    uint32_t base = AOTX_COLD_HEADER + aotx_cold.count * AOTX_COG_OBJECT;
    if (!aotx_cold.status && aotx_cold.count) {
        if (!aotx_cold.copied) {
            uint32_t at = 0;
            for (uint32_t i = 0; i < aotx_live_store.count; ++i) if (aotx_cold.selected[i]) {
                for (uint32_t j = threadIdx.x; j < AOTX_COG_OBJECT; j += blockDim.x)
                    aotx_live.choices[AOTX_COLD_HEADER + at * AOTX_COG_OBJECT + j] = aotx_live_store.objects[i][j];
                ++at;
            }
        }
        uint32_t take = aotx_cold.bytes - aotx_cold.copied;
        if (take > AOTX_COLD_COPY) take = AOTX_COLD_COPY;
        for (uint32_t i = threadIdx.x; i < take; i += blockDim.x)
            aotx_live.choices[base + aotx_cold.copied + i] = r->payload[aotx_cold.copied + i];
        __syncthreads();
        if (!threadIdx.x) aotx_cold.copied += take;
        __syncthreads();
        if (aotx_cold.copied != aotx_cold.bytes) return;
    }
    aotx_cold_candidate(aotx_live.choices + base);
    __syncthreads();
    aotx_cold_result_header();
}
__device__ void aotx_cold_replay(void) {
    if (aotx_live.phase != AOTX_LIVE_REPLAY) return;
    __shared__ uint32_t error;
    if (!threadIdx.x) {
        error = 0;
        const unsigned char *p = aotx_live.input;
        uint32_t status = aotx_cog_u32(p + 12), count = aotx_cog_u32(p + 20);
        uint64_t bytes = aotx_cog_u64(p + 48);
        if (aotx_live.total < AOTX_COLD_HEADER || !aotx_cog_equal(p, (const unsigned char *)"AOTXTIR2", 8) ||
            aotx_cog_u32(p + 8) != 1 || status > AOTX_COG_UNAVAILABLE ||
            aotx_cog_u32(p + 16) != aotx_cold.mode || !aotx_cog_equal(p + 24, aotx_live_store.lineage) ||
            aotx_cog_u64(p + 40) != aotx_live_store.sequence || !aotx_cog_zero(p + 56, 8) ||
            !aotx_cog_equal(aotx_live.transfer_id, aotx_live.query_id) ||
            count != (status ? 0 : aotx_cold.count) || bytes != (status ? 0 : aotx_cold.bytes) ||
            aotx_live.total != AOTX_COLD_HEADER + (uint64_t)count * AOTX_COG_OBJECT + bytes ||
            (aotx_cold.status && aotx_cold.status != status)) error = AOTX_COG_REFERENCE;
        if (!error && !status) {
            uint32_t at = 0;
            for (uint32_t i = 0; i < aotx_live_store.count; ++i) if (aotx_cold.selected[i] && aotx_cold.count) {
                if (!aotx_cog_equal(p + AOTX_COLD_HEADER + at * AOTX_COG_OBJECT, aotx_live_store.objects[i], AOTX_COG_OBJECT))
                    error = AOTX_COG_REFERENCE;
                ++at;
            }
        }
        if (!error) aotx_cold.status = status;
    }
    __syncthreads();
    if (!error) {
        aotx_cold_candidate(aotx_live.input + AOTX_COLD_HEADER + aotx_cold.count * AOTX_COG_OBJECT);
        if (!threadIdx.x && aotx_cold.status != aotx_cog_u32(aotx_live.input + 12)) error = aotx_cold.status;
    }
    __syncthreads();
    if (!threadIdx.x) {
        aotx_live.received = 0;
        if (error) {
            aotx_live.fatal = 1; ++aotx_live.refused; aotx_cold.active = 0;
            aotx_live_note(AOTX_COLD_RESULT, error, 0); aotx_live.phase = AOTX_LIVE_IDLE;
        } else {
            aotx_live.status = aotx_cold.status;
            aotx_live.choice_bytes = aotx_live.written = 0; aotx_live.phase = AOTX_LIVE_WRITE;
        }
    }
}
