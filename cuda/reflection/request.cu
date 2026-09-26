/* Purpose: Record exact idle review requests before evidence construction.
 * Owns: Bounded request fragments and source revision admission.
 * Launch shape: Ordered metadata thread with one exact reference per batch row.
 * Lifetime: One request through complete publication or recorded interruption. */
#include "reflection/state.cuh"
#include "reflection/evidence.cuh"
#include "cognitive/checkpoint.cuh"
#include "appraisal/appraisal.cuh"
#include "policy/state.cuh"
#include "sched/sched.cuh"
static __device__ unsigned char aotx_review_request_bytes[64 + AOTX_REVIEW_BATCH * 32];
static __device__ unsigned char aotx_review_part[AOTX_BODY_BYTES];
__device__ void aotx_review_auto_request(void) {
    if (aotx_seam.replaying || aotx_sched.held || !aotx_checkpoint_idle(true) ||
        aotx_checkpoint_pressure() || !aotx_policy_quiet() || aotx_appraisal.control_pending ||
        aotx_appraisal_foreground() || aotx_policy.paused || aotx_policy.stopped ||
        !aotx_review_pending() || !aotx_policy_review()) return;
    unsigned char *p = aotx_review_request_bytes;
    uint32_t count = aotx_review.pending, total = 64 + count * 32;
    for (uint32_t j = 0; j < total; ++j) p[j] = 0;
    for (uint32_t j = 0; j < 8; ++j) p[j] = "AOTXRVR1"[j];
    aotx_cog_put(p + 8, 1, 4); aotx_cog_put(p + 12, count, 4);
    aotx_cog_put(p + 16, aotx_live_store.sequence, 8);
    aotx_cog_put(p + 24, aotx_cog_u64(aotx_live_store.objects[aotx_review.indices[count - 1]] + AOTX_CO_UPDATED), 8);
    aotx_cog_put(p + 32, aotx_review.frontier, 8); aotx_cog_put(p + 40, aotx_review.wake, 8);
    for (uint32_t i = 0; i < count; ++i) {
        const unsigned char *r = aotx_live_store.objects[aotx_review.indices[i]];
        for (uint32_t j = 0; j < 16; ++j) p[64 + i * 32 + j] = r[AOTX_CO_ID + j];
        aotx_cog_put(p + 80 + i * 32, aotx_cog_u64(r + AOTX_CO_VERSION), 8);
    }
    unsigned char *part = aotx_review_part;
    for (uint32_t offset = 0; offset < total; offset += AOTX_LIVE_DATA) {
        for (uint32_t j = 0; j < AOTX_LIVE_PART; ++j) part[j] = 0;
        aotx_cog_put(part, 1, 4); aotx_cog_put(part + 4, AOTX_REVIEW_REQUEST, 4);
        for (uint32_t j = 0; j < 8; ++j) part[8 + j] = "AOTXRVR1"[j];
        aotx_cog_put(part + 16, aotx_live.accepted + 1, 8);
        aotx_cog_put(part + 24, total, 4); aotx_cog_put(part + 28, offset, 4);
        uint32_t bytes = min(AOTX_LIVE_DATA, total - offset);
        for (uint32_t j = 0; j < bytes; ++j) part[AOTX_LIVE_PART + j] = p[offset + j];
        uint64_t seq = aotx_seam_write(AOTX_WRITER_SYSTEM, AOTX_CLASS_A, AOTX_LIVE_RECORD,
            AOTX_FLAG_ADMISSION, part, AOTX_LIVE_PART + bytes);
        aotx_seam.apply.state_hash = aotx_seam_fnv1a(aotx_seam.apply.state_hash, part, AOTX_LIVE_PART + bytes);
        ++aotx_seam.apply.applied_count;
        aotx_live_part(part, AOTX_LIVE_PART + bytes, seq, AOTX_FLAG_ADMISSION);
    }
}
__device__ void aotx_review_begin(void) {
    if (threadIdx.x) return;
    const unsigned char *p = aotx_live.input;
    uint32_t count = aotx_cog_u32(p + 12), status = 0;
    if (aotx_live.total < 64 || !aotx_cog_equal(p, (const unsigned char *)"AOTXRVR1", 8) ||
        aotx_cog_u32(p + 8) != 1 || !count || count > AOTX_REVIEW_BATCH ||
        aotx_live.total != 64 + count * 32 || !aotx_cog_zero(p + 48, 16) ||
        aotx_cog_u64(p + 16) != aotx_live_store.sequence ||
        aotx_cog_u64(p + 32) != aotx_review.frontier || aotx_cog_u64(p + 40) != aotx_review.wake ||
        aotx_review.wake == UINT64_MAX || !aotx_review.enabled || aotx_policy.paused || aotx_policy.stopped ||
        aotx_policy.config.abi != AOTX_POLICY_REVIEW_ABI) status = AOTX_COG_FORMAT;
    uint64_t previous = aotx_review.frontier;
    for (uint32_t i = 0; !status && i < count; ++i) {
        const unsigned char *ref = p + 64 + i * 32;
        int index = aotx_cog_find(&aotx_live_store, ref, aotx_cog_u64(ref + 16));
        if (index < 0 || aotx_cog_u64(ref + 24) ||
            !aotx_review_query(&aotx_live_store, index, aotx_review.queries + i * AOTX_RECALL_QUERY)) {
            status = AOTX_COG_REFERENCE; break;
        }
        uint64_t revision = aotx_cog_u64(aotx_live_store.objects[index] + AOTX_CO_UPDATED);
        if (revision <= previous) status = AOTX_COG_REFERENCE;
        previous = revision; aotx_review.indices[i] = index;
    }
    if (!status && previous != aotx_cog_u64(p + 24)) status = AOTX_COG_REFERENCE;
    if (status) {
        aotx_review.status = aotx_live.status = status; ++aotx_live.refused;
        if (aotx_seam.replaying) aotx_live.fatal = 1;
        aotx_live.received = 0; aotx_live.phase = AOTX_LIVE_IDLE;
        aotx_live_note(AOTX_REVIEW_REQUEST, status, 0); return;
    }
    aotx_review.count = aotx_live.count = count; aotx_review.after = previous;
    aotx_review.active = 1; aotx_review.recovery = 0; aotx_review.work_wake = aotx_review.wake;
    aotx_review.status = aotx_live_admission_pressure() ? AOTX_COG_CAPACITY : 0;
    aotx_review.started_ns = aotx_time_globaltimer();
    aotx_live.auto_mode = aotx_live.intake_mode = aotx_live.text_mode = 0;
    aotx_live.request_seq = aotx_live.source_seq;
    for (uint32_t j = 0; j < 16; ++j) aotx_live.query_id[j] = aotx_live.transfer_id[j];
    aotx_live.received = 0; aotx_live.phase = aotx_seam.replaying ? AOTX_LIVE_WAIT : AOTX_REVIEW_BUILD;
    ++aotx_review.calls;
}
