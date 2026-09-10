/* Purpose: Record disk-pressure admission before a direct memory operation can publish.
 * Owns: Correlated admission records and the replay wait for their recorded outcome.
 * Launch shape: The live stage block; inbound decoding uses its serial thread.
 * Lifetime: One complete direct transfer and its authoritative journal decision. */
#include "cognitive/checkpoint.cuh"
#include "cognitive/codec.cuh"
#include "seam/seam.cuh"

static __device__ unsigned char aotx_live_admission_body[96];

__device__ uint32_t aotx_live_record_flags(const volatile unsigned char *p, uint32_t bytes, uint32_t flags) {
    if (aotx_seam.replaying) return flags;
    flags &= ~AOTX_FLAG_ADMISSION;
    if (bytes > AOTX_LIVE_PART) {
        uint32_t op = (uint32_t)p[4] | ((uint32_t)p[5] << 8) | ((uint32_t)p[6] << 16) | ((uint32_t)p[7] << 24);
        if (aotx_live_direct(op)) flags |= AOTX_FLAG_ADMISSION;
    }
    return flags;
}
__device__ bool aotx_live_admission_begin(void) {
    __shared__ uint32_t required;
    if (!threadIdx.x) required = aotx_live_direct(aotx_live.op) && aotx_live.admission == 1;
    __syncthreads();
    if (!required) return true;
    if (!threadIdx.x) {
        if (aotx_seam.replaying) aotx_live.phase = AOTX_LIVE_ADMIT_WAIT;
        else {
            unsigned char *body = aotx_live_admission_body, *p = body + AOTX_LIVE_PART;
            for (uint32_t j = 0; j < 96; ++j) body[j] = 0;
            aotx_cog_put(body, AOTX_LIVE_SCHEMA, 4); aotx_cog_put(body + 4, AOTX_LIVE_ADMISSION, 4);
            for (uint32_t j = 0; j < 16; ++j) body[8 + j] = aotx_live.transfer_id[j];
            aotx_cog_put(body + 24, 64, 4);
            for (uint32_t j = 0; j < 8; ++j) p[j] = "AOTXADM1"[j];
            aotx_cog_put(p + 8, 1, 4); aotx_cog_put(p + 12, aotx_live.op, 4);
            aotx_cog_put(p + 16, aotx_live.total, 4);
            aotx_live.pressure = aotx_checkpoint_pressure();
            aotx_cog_put(p + 20, aotx_live.pressure, 4);
            aotx_cog_put(p + 24, aotx_live.source_seq, 8);
            aotx_cog_put(p + 32, aotx_live.accepted, 8);
            aotx_cog_put(p + 40, aotx_live_store.sequence, 8);
            for (uint32_t j = 0; j < 16; ++j) p[48 + j] = aotx_live_store.lineage[j];
            aotx_seam_write(AOTX_WRITER_SYSTEM, AOTX_CLASS_A, AOTX_LIVE_RECORD, 0, body, 96);
            aotx_seam.apply.state_hash = aotx_seam_fnv1a(aotx_seam.apply.state_hash, body, 96);
            ++aotx_seam.apply.applied_count;
            aotx_live.admission = 2;
        }
    }
    __syncthreads();
    return aotx_live.phase != AOTX_LIVE_ADMIT_WAIT;
}
__device__ bool aotx_live_admission_pressure(void) {
    return aotx_live_direct(aotx_live.op) && aotx_live.admission == 2 ?
        aotx_live.pressure != 0 : aotx_checkpoint_pressure();
}
__device__ bool aotx_live_admission_take(const unsigned char *body, uint32_t bytes) {
    if (!aotx_seam.replaying || aotx_live.phase != AOTX_LIVE_ADMIT_WAIT ||
        aotx_live.admission != 1 || bytes != 96) return false;
    const unsigned char *p = body + AOTX_LIVE_PART;
    if (aotx_cog_u32(body) != AOTX_LIVE_SCHEMA || aotx_cog_u32(body + 24) != 64 || aotx_cog_u32(body + 28) ||
        !aotx_cog_equal(body + 8, aotx_live.transfer_id) || !aotx_cog_equal(p, (const unsigned char *)"AOTXADM1", 8) ||
        aotx_cog_u32(p + 8) != 1 || aotx_cog_u32(p + 12) != aotx_live.op ||
        aotx_cog_u32(p + 16) != aotx_live.total || aotx_cog_u32(p + 20) > 1 ||
        aotx_cog_u64(p + 24) != aotx_live.source_seq || aotx_cog_u64(p + 32) != aotx_live.accepted ||
        aotx_cog_u64(p + 40) != aotx_live_store.sequence || !aotx_cog_equal(p + 48, aotx_live_store.lineage)) return false;
    aotx_live.pressure = aotx_cog_u32(p + 20);
    aotx_live.admission = 2; aotx_live.phase = AOTX_LIVE_READY;
    return true;
}
