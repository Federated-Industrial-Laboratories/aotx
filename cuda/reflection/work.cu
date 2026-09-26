/* Purpose: Build, verify and publish complete source-bound task review results.
 * Owns: Canonical result bytes and an atomic candidate store.
 * Launch shape: One block constructs and validates up to 64 independent rows.
 * Lifetime: One durable request; replay never runs a model or invents evidence. */
#include "reflection/state.cuh"
#include "reflection/build.cuh"
#include "cognitive/checkpoint.cuh"
#include "appraisal/appraisal.cuh"
#include "policy/state.cuh"
#include "sched/sched.cuh"
__device__ void aotx_review_decide(void) {
    bool replay = aotx_live.phase == AOTX_LIVE_REPLAY;
    if (aotx_live.phase != AOTX_REVIEW_BUILD && !replay) return;
    __shared__ uint32_t status, error, tail;
    if (!threadIdx.x) {
        status = aotx_review.status; error = tail = 0;
        if (replay) {
            const unsigned char *p = aotx_live.input;
            if (aotx_live.total < 64 || !aotx_cog_equal(p, (const unsigned char *)"AOTXRVS1", 8) ||
                aotx_cog_u32(p + 8) != 1 || aotx_cog_u32(p + 12) != aotx_review.count ||
                aotx_cog_u32(p + 16) > AOTX_COG_UNAVAILABLE || aotx_cog_u32(p + 20) ||
                aotx_cog_u64(p + 24) != aotx_review.frontier ||
                aotx_cog_u64(p + 32) != aotx_review.after || aotx_cog_u64(p + 40) != aotx_live_store.sequence ||
                aotx_cog_u64(p + 56) != aotx_review.work_wake || !aotx_cog_equal(aotx_live.transfer_id, aotx_live.query_id))
                error = AOTX_COG_FORMAT;
            else status = aotx_cog_u32(p + 16);
        } else if (aotx_review.recovery || !aotx_review.enabled || aotx_policy.paused ||
            aotx_policy.stopped || aotx_appraisal_foreground()) status = AOTX_COG_DENIED;
    }
    __syncthreads();
    if (!error && !status) {
        aotx_review_build_block(&aotx_live_store, aotx_review.queries, aotx_review.indices,
            aotx_review.count, aotx_live.choices + 64, &aotx_live.result);
        __syncthreads();
        if (!threadIdx.x) { status = aotx_live.result.status; tail = status ? 0 : aotx_live.result.bytes; }
    }
    __syncthreads();
    for (uint32_t j = threadIdx.x; j < sizeof(aotx_live_store); j += blockDim.x)
        ((unsigned char *)&aotx_live_candidate)[j] = ((const unsigned char *)&aotx_live_store)[j];
    __syncthreads();
    if (!error && !status) {
        aotx_cognitive_apply_block(&aotx_live_candidate, &aotx_live_scratch,
            aotx_live.choices + 64, tail, &aotx_live.result);
        __syncthreads();
        if (!threadIdx.x) { status = aotx_live.result.status; if (status) tail = 0; }
    }
    __syncthreads();
    if (!threadIdx.x) {
        unsigned char *p = aotx_live.choices;
        for (uint32_t j = 0; j < 64; ++j) p[j] = 0;
        for (uint32_t j = 0; j < 8; ++j) p[j] = "AOTXRVS1"[j];
        aotx_cog_put(p + 8, 1, 4); aotx_cog_put(p + 12, aotx_review.count, 4);
        aotx_cog_put(p + 16, status, 4); aotx_cog_put(p + 24, aotx_review.frontier, 8);
        aotx_cog_put(p + 32, aotx_review.after, 8); aotx_cog_put(p + 40, aotx_live_store.sequence, 8);
        aotx_cog_put(p + 48, tail, 8); aotx_cog_put(p + 56, aotx_review.work_wake, 8);
        aotx_live.choice_bytes = 64 + tail;
        if (replay && aotx_live.total != aotx_live.choice_bytes) error = AOTX_COG_REFERENCE;
    }
    __syncthreads();
    if (replay && !error) for (uint32_t j = threadIdx.x; j < aotx_live.choice_bytes; j += blockDim.x)
        if (aotx_live.input[j] != aotx_live.choices[j]) atomicCAS(&error, 0u, AOTX_COG_REFERENCE);
    __syncthreads();
    if (!threadIdx.x) {
        aotx_review.status = status; aotx_live.status = error;
        if (error) {
            aotx_live.fatal = 1; aotx_review.active = 0; aotx_live.phase = AOTX_LIVE_IDLE;
            aotx_live_note(AOTX_REVIEW_RESULT, error, 0);
        } else {
            aotx_live.received = aotx_live.written = 0;
            if (replay) aotx_live.choice_bytes = 0;
            aotx_live.phase = AOTX_LIVE_WRITE;
        }
    }
}
__device__ void aotx_review_publish(void) {
    if (!aotx_review.status) for (uint32_t j = threadIdx.x; j < sizeof(aotx_live_store); j += blockDim.x)
        ((unsigned char *)&aotx_live_store)[j] = ((const unsigned char *)&aotx_live_candidate)[j];
    __syncthreads();
    if (threadIdx.x) return;
    if (!aotx_review.status) aotx_review.completed += aotx_review.count;
    else if (aotx_review.status == AOTX_COG_DENIED) aotx_review.interrupted += aotx_review.count;
    else aotx_review.refused += aotx_review.count;
    if (aotx_review.status == AOTX_COG_CAPACITY) {
        aotx_review.blocked_sequence = aotx_live_store.sequence; aotx_review.blocked_root = aotx_live_store.root_sequence;
        aotx_review.blocked_bytes = aotx_live_store.bytes; aotx_review.blocked_count = aotx_live_store.count;
    } else aotx_review.frontier = aotx_review.after;
    ++aotx_review.wake; ++aotx_live.accepted;
    aotx_review.elapsed_ns = aotx_time_globaltimer() - aotx_review.started_ns;
    if (!aotx_seam.replaying && aotx_review.elapsed_ns > aotx_review.maximum_ns) aotx_review.maximum_ns = aotx_review.elapsed_ns;
    aotx_live_note(AOTX_REVIEW_RESULT, aotx_review.status, aotx_review.count);
    aotx_review.active = aotx_review.pending = 0; aotx_review.observed = UINT64_MAX;
    aotx_live.phase = AOTX_LIVE_IDLE;
}
