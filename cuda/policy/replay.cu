/* Purpose: Restore complete policy decisions without executing creator code.
 * Owns: Fragment admission and atomic accepted-state replacement.
 * Launch shape: Ordered replay thread over a batch of journal fragments.
 * Lifetime: The exact selected policy revision and state schema. */
#include "policy/state.cuh"
#include "cognitive/codec.cuh"
#include "seam/seam.cuh"
#include <stddef.h>
static_assert(offsetof(aotx_policy_state, event) % 8 == 0, "policy event alignment");

__device__ aotx_policy_history aotx_policy_prior;

static __device__ bool aotx_policy_event_valid(void) {
    const unsigned char *p = aotx_policy.event;
    const aotx_policy_input *in = (const aotx_policy_input *)(p + 64);
    const aotx_policy_output *out = (const aotx_policy_output *)(p + 192);
    const aotx_policy_config *config = &aotx_policy.config;
    const unsigned char *digest = aotx_policy.digest;
    uint64_t decision = aotx_cog_u64(p + 24);
    for (uint32_t i = 0; i < aotx_policy_prior.count; ++i) {
        if (decision > aotx_policy_prior.rows[i].last_decision) continue;
        config = &aotx_policy_prior.rows[i].config; digest = aotx_policy_prior.rows[i].digest; break;
    }
    bool extended = config->abi == AOTX_POLICY_APPRAISAL_ABI;
    uint32_t abi = extended ? AOTX_POLICY_APPRAISAL_ABI : 0;
    if (config->abi != AOTX_POLICY_ABI && !extended) return false;
    if (!aotx_cog_equal(p, (const unsigned char *)"AOTXPD01", 8) || aotx_cog_u32(p + 8) != 1 ||
        aotx_cog_u32(p + 12) != config->state_schema ||
        aotx_cog_u32(p + 16) != config->state_bytes || aotx_cog_u32(p + 20) != abi ||
        !aotx_cog_equal(p + 32, digest, 32) ||
        aotx_cog_u64(p + 24) != aotx_policy.decision + 1 ||
        !in->decision || in->decision != aotx_policy.decision + 1 ||
        in->previous_source != aotx_policy.source || in->previous_root != aotx_policy.root ||
        !in->valid || in->valid != 1 || in->enabled > 1 || in->foreground || in->paused ||
        (!in->enabled && (!extended || !in->reserved1[0])) ||
        in->reserved0 != abi || !in->source || in->root > in->source ||
        !in->object_capacity || !in->byte_capacity || in->objects > in->object_capacity ||
        in->bytes > in->byte_capacity || in->pressure > 100 ||
        (!in->pressure && (!extended || in->enabled)) ||
        in->rule_pressure != config->pressure ||
        in->minimum_move != config->minimum_move || in->backoff != config->backoff ||
        out->action > (extended ? AOTX_POLICY_APPRAISE : AOTX_POLICY_MAINTAIN) ||
        (out->action == AOTX_POLICY_APPRAISE && !in->reserved1[0]) ||
        (extended && out->action == AOTX_POLICY_MAINTAIN && !in->enabled) ||
        (out->status && out->status != AOTX_COG_FORMAT) ||
        (out->status && (out->action || out->reason))) return false;
    if (!extended) for (unsigned i = 0; i < 3; ++i) if (in->reserved1[i]) return false;
    for (unsigned i = 0; i < 6; ++i) if (out->reserved[i]) return false;
    return true;
}
__device__ bool aotx_policy_part(const unsigned char *part, uint32_t bytes, uint32_t flags) {
    if (!aotx_policy.enabled || !aotx_seam.replaying || !(flags & AOTX_FLAG_REPLAYED) ||
        aotx_policy.fatal || bytes <= AOTX_POLICY_PART || bytes > AOTX_BODY_BYTES) {
        aotx_policy.fatal = 1; return false;
    }
    uint32_t total = aotx_cog_u32(part + 4), offset = aotx_cog_u32(part + 8), count = aotx_cog_u32(part + 12);
    uint64_t decision = aotx_cog_u64(part + 16);
    if (aotx_cog_u32(part) != 1 || aotx_cog_u64(part + 24) ||
        total != AOTX_POLICY_HEADER + aotx_policy.config.state_bytes ||
        total > AOTX_POLICY_EVENT_BYTES || count != bytes - AOTX_POLICY_PART ||
        offset > total || count > total - offset || offset != aotx_policy.received ||
        !decision || decision != aotx_policy.decision + 1) {
        aotx_policy.fatal = 1; return false;
    }
    for (uint32_t i = 0; i < count; ++i) aotx_policy.event[offset + i] = part[AOTX_POLICY_PART + i];
    aotx_policy.received += count;
    if (aotx_policy.received != total) return true;
    if (!aotx_policy_event_valid()) { aotx_policy.fatal = 1; return false; }
    const aotx_policy_input *in = (const aotx_policy_input *)(aotx_policy.event + 64);
    const aotx_policy_output *out = (const aotx_policy_output *)(aotx_policy.event + 192);
    for (uint32_t j = 0; j < aotx_policy.config.state_bytes; ++j)
        aotx_policy.current[j] = aotx_policy.event[AOTX_POLICY_HEADER + j];
    aotx_policy.state_hash = aotx_seam_fnv1a(14695981039346656037ull,
        aotx_policy.current, aotx_policy.config.state_bytes);
    aotx_policy.decision = decision; aotx_policy.source = in->source; aotx_policy.root = in->root;
    aotx_policy.work_revision = (uint64_t)in->reserved1[1] | (uint64_t)in->reserved1[2] << 32;
    aotx_policy.observed_objects = in->objects; aotx_policy.observed_bytes = in->bytes;
    aotx_policy.status = out->status;
    if (out->status) aotx_policy.paused = 1;
    aotx_policy.received = 0;
    /* Recorded work admission owns any side effect after the proposal. */
    aotx_policy.maintain = 0; aotx_policy.appraise = 0;
    return true;
}
__device__ bool aotx_policy_restore_end(void) {
    aotx_policy.received = 0; aotx_policy.pending = 0; aotx_policy.maintain = 0; aotx_policy.appraise = 0;
    if (aotx_policy_prior.count && aotx_policy.decision <
        aotx_policy_prior.rows[aotx_policy_prior.count - 1].last_decision) aotx_policy.fatal = 1;
    return !aotx_policy.fatal;
}
