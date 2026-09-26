/* Purpose: Expose bounded policy status and revision-bound operator controls.
 * Owns: Scoped request admission and aggregate responses without source text.
 * Launch shape: One ordered batch of service channels per tick.
 * Lifetime: A runtime epoch for requests; journal history for accepted controls. */
#include "service/internal.cuh"
#include "policy/control.cuh"
#include "reflection/state.cuh"
#include "cognitive/checkpoint.cuh"
#include "appraisal/appraisal.cuh"
__device__ void aotx_service_policy(unsigned channel, const aotx_service_grant *g) {
    unsigned char *f = aotx_service.frames + (uint64_t)channel * AOTX_SERVICE_FRAME;
    uint32_t bytes = aotx_service_u32(f + 88);
    bool write = bytes != 0;
    if (!(g->actions & (write ? AOTX_SERVICE_POLICY_MANAGE : AOTX_SERVICE_POLICY_MANAGE | AOTX_SERVICE_TELEMETRY))) {
        aotx_service_answer(channel, 403, 0); return;
    }
    if (aotx_service_nonzero(f + 48, 24) || (!write && aotx_service_get(f + 40, 8)) || (write && bytes != 16)) {
        aotx_service_answer(channel, 400, 0); return;
    }
    if (write) {
        const unsigned char *in = f + AOTX_SERVICE_HEAD;
        uint32_t action = aotx_service_u32(in + 4);
        if (aotx_service_u32(in) != 1 || !action || action > AOTX_POLICY_REVIEW_OFF) {
            aotx_service_answer(channel, 400, 0); return;
        }
        if (aotx_service_get(f + 40, 8) != aotx_service.epoch) { aotx_service_answer(channel, 410, 0); return; }
        if (aotx_sched.held || aotx_checkpoint_pressure()) { aotx_service_answer(channel, 429, 0); return; }
        uint32_t status = aotx_policy.config.abi == AOTX_POLICY_REVIEW_ABI ?
            aotx_policy_control_check(action, aotx_service_get(in + 8, 8)) : AOTX_COG_LAYOUT;
        if (status) { aotx_service_answer(channel, 409, 0); return; }
        unsigned char record[AOTX_POLICY_CONTROL_BYTES] = {};
        for (uint32_t j = 0; j < 8; ++j) record[j] = "AOTXPCT1"[j];
        aotx_service_put(record + 8, 1, 4); aotx_service_put(record + 12, action, 4);
        aotx_service_put(record + 16, aotx_review.control_revision, 8);
        aotx_service_put(record + 24, aotx_review.wake, 8);
        aotx_service_put(record + 32, aotx_review.control_revision + 1, 8);
        aotx_service_put(record + 40, aotx_review.wake + 1, 8);
        aotx_service_bytes(record + 48, g->principal, 16);
        aotx_seam_write(AOTX_WRITER_SYSTEM, AOTX_CLASS_A, AOTX_REC_POLICY_CONTROL, 0, record, sizeof(record));
        aotx_seam.apply.state_hash = aotx_seam_fnv1a(aotx_seam.apply.state_hash, record, sizeof(record));
        ++aotx_seam.apply.applied_count; aotx_policy_control_apply(action);
    }
    unsigned char *p = f + AOTX_SERVICE_HEAD;
    for (uint32_t j = 0; j < 160; ++j) p[j] = 0;
    uint32_t state = !aotx_policy.enabled ? 0 : aotx_policy.fatal ? 5 : aotx_policy.stopped ? 4 :
        aotx_policy.paused ? 3 : aotx_review.active ? 2 : aotx_policy.pending ? 6 : 1;
    uint32_t reason = !aotx_review.enabled ? 5 : aotx_policy.paused || aotx_policy.stopped ? 2 :
        aotx_review.active ? 4 : aotx_review.status == AOTX_COG_CAPACITY ? 3 : aotx_appraisal_foreground() ? 1 : 0;
    uint32_t words[] = {1, aotx_policy.config.abi, aotx_policy.config.mode, state,
        aotx_review.enabled, aotx_review.pending, aotx_review.active ? aotx_review.count : 0, aotx_review.status};
    for (uint32_t j = 0; j < 8; ++j) aotx_service_put(p + j * 4, words[j], 4);
    uint64_t values[] = {aotx_review.control_revision, aotx_review.frontier, aotx_review.completed,
        aotx_review.interrupted, aotx_review.refused, aotx_policy.decision, aotx_checkpoint.generation,
        aotx_review.maximum_ns, aotx_review.elapsed_ns, aotx_review.active ? aotx_live.written : 0,
        aotx_review.active ? aotx_live.choice_bytes : 0};
    for (uint32_t j = 0; j < 11; ++j) aotx_service_put(p + 32 + j * 8, values[j], 8);
    aotx_service_put(p + 120, reason, 4);
    aotx_service_answer(channel, 200, state, 160);
}
