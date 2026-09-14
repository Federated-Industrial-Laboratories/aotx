/* Purpose: Apply recorded appraisal settings through atomic memory admission.
 * Owns: Deferred operator settings and their versioned configuration object.
 * Launch shape: Serial request emission, followed by one block for state publication.
 * Lifetime: Recorded commands, typed memory and exact recovery. */
#include "appraisal/encode.cuh"
#include "cognitive/checkpoint.cuh"
#include "policy/state.cuh"
#include "sched/sched.cuh"

static __device__ unsigned char aotx_appraisal_control_part[AOTX_LIVE_PART + AOTX_APPRAISAL_CONFIG_BYTES];
static __device__ unsigned char aotx_appraisal_control_tail[AOTX_COG_HEADER + AOTX_COG_OBJECT + AOTX_APPRAISAL_CONFIG_BYTES];

__device__ void aotx_appraisal_control_request(void) {
    if (!aotx_appraisal.control_pending || aotx_seam.replaying || aotx_sched.held ||
        !aotx_checkpoint_idle(true) || !aotx_policy_quiet() || aotx_checkpoint_pressure()) return;
    unsigned char *part = aotx_appraisal_control_part;
    for (uint32_t j = 0; j < sizeof(aotx_appraisal_control_part); ++j) part[j] = 0;
    aotx_cog_put(part, 1, 4); aotx_cog_put(part + 4, AOTX_APPRAISAL_CONTROL, 4);
    for (uint32_t j = 0; j < 8; ++j) part[8 + j] = "AOTXAPC1"[j];
    aotx_cog_put(part + 16, aotx_live.accepted + 1, 8);
    aotx_cog_put(part + 24, AOTX_APPRAISAL_CONFIG_BYTES, 4);
    for (uint32_t j = 0; j < AOTX_APPRAISAL_CONFIG_BYTES; ++j)
        part[AOTX_LIVE_PART + j] = aotx_appraisal.control[j];
    uint64_t seq = aotx_seam_write(AOTX_WRITER_SYSTEM, AOTX_CLASS_A, AOTX_LIVE_RECORD,
        AOTX_FLAG_ADMISSION, part, sizeof(aotx_appraisal_control_part));
    aotx_seam.apply.state_hash = aotx_seam_fnv1a(aotx_seam.apply.state_hash, part, sizeof(aotx_appraisal_control_part));
    ++aotx_seam.apply.applied_count;
    aotx_live_part(part, sizeof(aotx_appraisal_control_part), seq, AOTX_FLAG_ADMISSION);
}
__device__ void aotx_appraisal_control(void) {
    __shared__ uint32_t status;
    if (!threadIdx.x) {
        status = 0; aotx_appraisal_refresh();
        if (aotx_live.total != AOTX_APPRAISAL_CONFIG_BYTES ||
            !aotx_appraisal_magic(aotx_live.input, aotx_live.total, "AOTXAPC1")) status = AOTX_COG_FORMAT;
        else if (!aotx_live.ready || !aotx_checkpoint_quiet() || aotx_live_admission_pressure()) status = AOTX_COG_DENIED;
        else if (aotx_live_store.count == AOTX_COG_OBJECTS || AOTX_APPRAISAL_CONFIG_BYTES > AOTX_COG_PAYLOAD - aotx_live_store.bytes ||
            aotx_live_store.sequence == UINT64_MAX || aotx_live_store.tick == UINT64_MAX) status = AOTX_COG_CAPACITY;
        else if (!aotx_cog_equal(aotx_live.input + 40, aotx_appraisal_processor, 32) ||
            ((aotx_cog_u32(aotx_live.input + 12) & AOTX_APPRAISAL_BACKGROUND) &&
            aotx_policy.enabled && aotx_policy.config.abi != AOTX_POLICY_APPRAISAL_ABI)) status = AOTX_COG_LAYOUT;
        if (!status) {
            unsigned char *tail = aotx_appraisal_control_tail;
            for (uint32_t j = 0; j < sizeof(aotx_appraisal_control_tail); ++j) tail[j] = 0;
            aotx_appraisal_tail_header(tail, 1, AOTX_APPRAISAL_CONFIG_BYTES);
            unsigned char *r = tail + AOTX_COG_HEADER;
            uint64_t seq = aotx_live_store.sequence + 1;
            bool existing = aotx_appraisal.config < aotx_live_store.count;
            if (existing) for (uint32_t j = 0; j < AOTX_COG_OBJECT; ++j) r[j] = aotx_live_store.objects[aotx_appraisal.config][j];
            else {
                aotx_cog_put(r, 1, 2); aotx_cog_put(r + AOTX_CO_KIND, AOTX_COG_POLICY, 2);
                for (uint32_t j = 0; j < 8; ++j) r[AOTX_CO_ID + j] = "AOTXAPC1"[j];
                uint64_t suffix = seq;
                do { aotx_cog_put(r + AOTX_CO_ID + 8, suffix++, 8); }
                while (suffix && aotx_cog_latest(&aotx_live_store, r + AOTX_CO_ID) >= 0);
                if (aotx_cog_latest(&aotx_live_store, r + AOTX_CO_ID) >= 0) status = AOTX_COG_CAPACITY;
                for (uint32_t j = 0; j < 16; ++j)
                    r[AOTX_CO_LINEAGE + j] = r[AOTX_CO_OWNER + j] = aotx_live_store.lineage[j];
                aotx_cog_put(r + AOTX_CO_CREATED, seq, 8);
                aotx_cog_put(r + AOTX_CO_SCOPE, AOTX_COG_INSTANCE, 4);
                aotx_cog_put(r + AOTX_CO_SOURCE_KIND, AOTX_COG_AUTHORED, 4);
                aotx_cog_put(r + AOTX_CO_POLICY, 1, 8);
            }
            uint64_t version = aotx_live_store.pressure_percent ? seq : existing ? aotx_cog_u64(r + AOTX_CO_VERSION) + 1 : 1;
            if (!version) status = AOTX_COG_CAPACITY;
            aotx_cog_put(r + AOTX_CO_FLAGS, AOTX_COG_PROTECTED, 4);
            aotx_cog_put(r + AOTX_CO_VERSION, version, 8); aotx_cog_put(r + AOTX_CO_UPDATED, seq, 8);
            aotx_cog_put(r + AOTX_CO_OFFSET, 0, 8); aotx_cog_put(r + AOTX_CO_BYTES, AOTX_APPRAISAL_CONFIG_BYTES, 8);
            for (uint32_t j = 0; j < AOTX_APPRAISAL_CONFIG_BYTES; ++j)
                r[AOTX_COG_OBJECT + j] = aotx_live.input[j];
        }
    }
    __syncthreads();
    for (uint32_t j = threadIdx.x; !status && j < sizeof(aotx_live_store); j += blockDim.x)
        ((unsigned char *)&aotx_live_candidate)[j] = ((const unsigned char *)&aotx_live_store)[j];
    __syncthreads();
    if (!status) {
        aotx_cognitive_apply_block(&aotx_live_candidate, &aotx_live_scratch, aotx_appraisal_control_tail,
            sizeof(aotx_appraisal_control_tail), &aotx_live.result);
        __syncthreads();
        if (!threadIdx.x) status = aotx_live.result.status;
    }
    __syncthreads();
    for (uint32_t j = threadIdx.x; !status && j < sizeof(aotx_live_store); j += blockDim.x)
        ((unsigned char *)&aotx_live_store)[j] = ((const unsigned char *)&aotx_live_candidate)[j];
    __syncthreads();
    if (threadIdx.x) return;
    if (aotx_live.total == AOTX_APPRAISAL_CONFIG_BYTES &&
        aotx_cog_equal(aotx_appraisal.control, aotx_live.input, AOTX_APPRAISAL_CONFIG_BYTES)) aotx_appraisal.control_pending = 0;
    if (status) ++aotx_live.refused; else ++aotx_live.accepted;
    aotx_appraisal.observed = UINT64_MAX; aotx_appraisal.last_status = aotx_live.status = status;
    aotx_live.received = 0; aotx_live.phase = AOTX_LIVE_IDLE;
    aotx_live_note(AOTX_APPRAISAL_CONTROL, status, status ? 0 : 1);
}
