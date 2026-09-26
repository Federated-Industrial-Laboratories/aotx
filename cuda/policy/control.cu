/* Purpose: Apply revision-bound policy operator controls without repeated effects.
 * Owns: Pause, stop and review flags with a monotonic control revision.
 * Launch shape: Serial admission and ordered recorded command replay.
 * Lifetime: One policy runtime and its durable control history. */
#include "policy/control.cuh"
#include "reflection/state.cuh"
#include "cognitive/codec.cuh"
#include "seam/seam.cuh"
__device__ uint32_t aotx_policy_control_check(uint32_t action, uint64_t expected) {
    if (!action || action > AOTX_POLICY_REVIEW_OFF) return AOTX_COG_FORMAT;
    if (!aotx_policy.enabled || expected != aotx_review.control_revision) return AOTX_COG_STALE;
    if (aotx_review.control_revision == UINT64_MAX || aotx_review.wake >= UINT64_MAX - 1) return AOTX_COG_CAPACITY;
    if (action >= AOTX_POLICY_REVIEW_ON && aotx_policy.config.abi != AOTX_POLICY_REVIEW_ABI) return AOTX_COG_LAYOUT;
    if (aotx_policy.fatal && action != AOTX_POLICY_PAUSE && action != AOTX_POLICY_STOP) return AOTX_COG_DENIED;
    return 0;
}
__device__ void aotx_policy_control_apply(uint32_t action) {
    if (action == AOTX_POLICY_PAUSE) aotx_policy.paused = 1;
    if (action == AOTX_POLICY_STOP) { aotx_policy.paused = 1; aotx_policy.stopped = 1; }
    if (action == AOTX_POLICY_RESUME) { aotx_policy.paused = 0; aotx_policy.stopped = 0; }
    if (action == AOTX_POLICY_REVIEW_ON) aotx_review.enabled = 1;
    if (action == AOTX_POLICY_REVIEW_OFF) aotx_review.enabled = 0;
    ++aotx_review.control_revision; ++aotx_review.wake;
    aotx_review.observed = UINT64_MAX;
    aotx_policy.review = aotx_policy.appraise = aotx_policy.maintain = 0;
}
__device__ bool aotx_policy_control_part(const unsigned char *p, uint32_t bytes, uint32_t flags) {
    uint32_t action = aotx_cog_u32(p + 12);
    bool valid = aotx_seam.replaying && (flags & AOTX_FLAG_REPLAYED) && bytes == AOTX_POLICY_CONTROL_BYTES &&
        aotx_cog_equal(p, (const unsigned char *)"AOTXPCT1", 8) && aotx_cog_u32(p + 8) == 1 &&
        aotx_policy.config.abi == AOTX_POLICY_REVIEW_ABI && !aotx_cog_zero(p + 48, 16) &&
        aotx_cog_u64(p + 24) == aotx_review.wake &&
        aotx_cog_u64(p + 32) == aotx_review.control_revision + 1 && aotx_cog_u64(p + 40) == aotx_review.wake + 1 &&
        !aotx_policy_control_check(action, aotx_cog_u64(p + 16));
    if (!valid) { aotx_policy.fatal = 1; return false; }
    aotx_policy_control_apply(action); return true;
}
