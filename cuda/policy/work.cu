/* Purpose: Observe eligible idle work and publish complete creator decisions.
 * Owns: Candidate state until its complete journal record is available.
 * Launch shape: One preparation block and one bounded publication block.
 * Lifetime: The selected policy; no input is generated during quiet intervals. */
#include "policy/state.cuh"
#include "cognitive/maintenance.cuh"
#include "cognitive/codec.cuh"
#include "sched/sched.cuh"

__device__ aotx_policy_state aotx_policy;

__device__ bool aotx_policy_quiet(void) { return !aotx_policy.pending && !aotx_policy.received; }
__device__ bool aotx_policy_maintenance(void) {
    if (!aotx_policy.enabled) return true;
    bool take = aotx_policy.maintain && !aotx_policy.paused && !aotx_policy.stopped &&
        !aotx_policy.fatal && aotx_policy.source == aotx_live_store.sequence &&
        aotx_policy.root == aotx_live_store.root_sequence;
    aotx_policy.maintain = 0;
    return take;
}
__global__ void aotx_policy_prepare(cudaGraphConditionalHandle condition) {
    if (!threadIdx.x) {
        aotx_policy.input.valid = 0; aotx_policy.launch = 0;
        if (aotx_policy.enabled && !aotx_policy.pending && !aotx_policy.received &&
            !aotx_policy.paused && !aotx_policy.stopped && !aotx_policy.fatal &&
            !aotx_sched.held && !aotx_seam.replaying && aotx_live_store.maintenance &&
            aotx_checkpoint_idle(true) && !aotx_checkpoint_maintenance_pressure() &&
            (aotx_policy.source != aotx_live_store.sequence ||
             aotx_policy.root != aotx_live_store.root_sequence)) {
            aotx_policy_input *in = &aotx_policy.input;
            *in = {};
            in->source = aotx_live_store.sequence; in->root = aotx_live_store.root_sequence;
            in->objects = aotx_live_store.count; in->bytes = aotx_live_store.bytes;
            in->object_capacity = AOTX_COG_OBJECTS; in->byte_capacity = AOTX_COG_PAYLOAD;
            in->previous_source = aotx_policy.source; in->previous_root = aotx_policy.root;
            in->decision = aotx_policy.decision + 1;
            in->enabled = 1; in->pressure = aotx_live_store.pressure_percent;
            in->keep_recent = aotx_live_store.keep_recent; in->max_age = aotx_live_store.max_age;
            in->rule_pressure = aotx_policy.config.pressure;
            in->minimum_move = aotx_policy.config.minimum_move;
            in->backoff = aotx_policy.config.backoff;
            in->valid = in->decision != 0;
            aotx_policy.launch = in->valid; aotx_policy.output = {};
            aotx_policy.started_ns = aotx_time_globaltimer();
        }
        cudaGraphSetConditional(condition, aotx_policy.launch);
    }
    __syncthreads();
    if (aotx_policy.launch) for (uint32_t j = threadIdx.x; j < aotx_policy.config.state_bytes; j += blockDim.x)
        aotx_policy.candidate[j] = aotx_policy.current[j];
}
static __device__ void aotx_policy_event_begin(void) {
    aotx_policy.elapsed_ns = aotx_time_globaltimer() - aotx_policy.started_ns;
    if (aotx_policy.elapsed_ns > aotx_policy.maximum_ns) aotx_policy.maximum_ns = aotx_policy.elapsed_ns;
    ++aotx_policy.calls;
    aotx_policy_output *out = &aotx_policy.output;
    bool invalid = out->action > AOTX_POLICY_MAINTAIN || out->status;
    for (unsigned i = 0; i < 6; ++i) invalid |= out->reserved[i] != 0;
    if (invalid) {
        *out = {}; out->status = AOTX_COG_FORMAT;
        for (uint32_t j = 0; j < aotx_policy.config.state_bytes; ++j)
            aotx_policy.candidate[j] = aotx_policy.current[j];
    }
    unsigned char *p = aotx_policy.event;
    for (uint32_t j = 0; j < AOTX_POLICY_HEADER; ++j) p[j] = 0;
    for (unsigned j = 0; j < 8; ++j) p[j] = "AOTXPD01"[j];
    aotx_cog_put(p + 8, 1, 4); aotx_cog_put(p + 12, aotx_policy.config.state_schema, 4);
    aotx_cog_put(p + 16, aotx_policy.config.state_bytes, 4);
    aotx_cog_put(p + 24, aotx_policy.input.decision, 8);
    for (unsigned j = 0; j < 32; ++j) p[32 + j] = aotx_policy.digest[j];
    for (unsigned j = 0; j < sizeof(aotx_policy.input); ++j)
        p[64 + j] = ((const unsigned char *)&aotx_policy.input)[j];
    for (unsigned j = 0; j < sizeof(aotx_policy.output); ++j)
        p[192 + j] = ((const unsigned char *)&aotx_policy.output)[j];
    aotx_policy.total = AOTX_POLICY_HEADER + aotx_policy.config.state_bytes;
    aotx_policy.pending = 1; aotx_policy.emitted = 0;
}
__global__ void aotx_policy_publish(void) {
    if (aotx_sched.held || aotx_seam.replaying || !aotx_policy.enabled || aotx_policy.fatal) return;
    if (aotx_policy.launch) {
        if (!threadIdx.x) aotx_policy_event_begin();
        __syncthreads();
        for (uint32_t j = threadIdx.x; j < aotx_policy.config.state_bytes; j += blockDim.x)
            aotx_policy.event[AOTX_POLICY_HEADER + j] = aotx_policy.candidate[j];
        __syncthreads();
        if (!threadIdx.x) aotx_policy.launch = 0;
    }
    if (threadIdx.x || !aotx_policy.pending) return;
    __shared__ unsigned char part[AOTX_BODY_BYTES];
    for (unsigned i = 0; i < AOTX_POLICY_EMIT && aotx_policy.emitted < aotx_policy.total; ++i) {
        uint32_t offset = aotx_policy.emitted, bytes = aotx_policy.total - offset;
        if (bytes > AOTX_BODY_BYTES - AOTX_POLICY_PART) bytes = AOTX_BODY_BYTES - AOTX_POLICY_PART;
        for (unsigned j = 0; j < AOTX_POLICY_PART; ++j) part[j] = 0;
        aotx_cog_put(part, 1, 4); aotx_cog_put(part + 4, aotx_policy.total, 4);
        aotx_cog_put(part + 8, offset, 4); aotx_cog_put(part + 12, bytes, 4);
        aotx_cog_put(part + 16, aotx_policy.input.decision, 8);
        for (unsigned j = 0; j < bytes; ++j) part[AOTX_POLICY_PART + j] = aotx_policy.event[offset + j];
        aotx_seam_write(AOTX_WRITER_SYSTEM, AOTX_CLASS_A, AOTX_REC_POLICY, 0, part, AOTX_POLICY_PART + bytes);
        aotx_seam.apply.state_hash = aotx_seam_fnv1a(aotx_seam.apply.state_hash, part, AOTX_POLICY_PART + bytes);
        ++aotx_seam.apply.applied_count;
        aotx_policy.emitted += bytes;
    }
    if (aotx_policy.emitted == aotx_policy.total) {
        for (uint32_t j = 0; j < aotx_policy.config.state_bytes; ++j)
            aotx_policy.current[j] = aotx_policy.candidate[j];
        aotx_policy.state_hash = aotx_seam_fnv1a(14695981039346656037ull,
            aotx_policy.current, aotx_policy.config.state_bytes);
        aotx_policy.decision = aotx_policy.input.decision;
        aotx_policy.source = aotx_policy.input.source; aotx_policy.root = aotx_policy.input.root;
        aotx_policy.status = aotx_policy.output.status;
        if (aotx_policy.status) aotx_policy.paused = 1;
        aotx_policy.pending = 0;
        aotx_policy.maintain = aotx_policy.output.action == AOTX_POLICY_MAINTAIN;
    }
}
