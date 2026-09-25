/* Purpose: Record and apply complete appraisal decisions without repeated generation.
 * Owns: Canonical result bytes and candidate publication for an entire source batch.
 * Launch shape: One thread per row; serial whole-batch admission and result status.
 * Lifetime: A leased work batch through ordinary journal and file recovery. */
#include "appraisal/encode.cuh"
#include "appraisal/replay.cuh"
#include "model/load.cuh"
#include "cognitive/checkpoint.cuh"
#include "cli/cli.cuh"

__device__ void aotx_appraisal_decide(void) {
    bool replay = aotx_live.phase == AOTX_LIVE_REPLAY;
    if (aotx_live.phase != AOTX_INTAKE_DONE && !replay) return;
    __shared__ uint32_t error, work_status, objects, payload, tail_at, tail_bytes, version, stride;
    uint32_t row = threadIdx.x, count = aotx_appraisal.count;
    if (!row) {
        version = aotx_appraisal.result_version; stride = version == 1 ? AOTX_APPRAISAL_LEGACY_ROW : AOTX_APPRAISAL_RESULT_ROW;
        error = version != 1 && version != 2 ? AOTX_COG_LAYOUT : 0; work_status = aotx_live.status;
        if (replay) {
            const unsigned char *p = aotx_live.input;
            if (aotx_live.total < 64 || !aotx_appraisal_magic(p, aotx_live.total, "AOTXAPS1") ||
                aotx_cog_u32(p + 8) != version || aotx_cog_u64(p + 16) != aotx_live_store.sequence ||
                aotx_cog_u32(p + 32) > AOTX_COG_UNAVAILABLE || !aotx_cog_zero(p + 36, 28) ||
                !aotx_cog_equal(aotx_live.transfer_id, aotx_live.query_id)) error = AOTX_COG_FORMAT;
            else {
                work_status = aotx_cog_u32(p + 32);
                uint32_t recorded = aotx_cog_u32(p + 12);
                uint64_t tail = aotx_cog_u64(p + 24);
                if ((!recorded && (!work_status || tail || aotx_live.total != 64)) ||
                    (recorded && (recorded != count || tail > AOTX_COG_IMAGE ||
                    aotx_live.total != 64 + (uint64_t)count * stride + tail))) error = AOTX_COG_FORMAT;
            }
        }
    }
    __syncthreads();
    if (replay && !error && aotx_cog_u32(aotx_live.input + 12) && row < count) {
        uint32_t status = aotx_appraisal_replay_row(row, aotx_live.input + 64 + row * stride, version, work_status);
        if (status) atomicCAS(&error, 0u, status);
    }
    __syncthreads();
    if (!error && !work_status && row < count) {
        const aotx_intake_row *r = aotx_intake.rows + row;
        uint32_t status = 0;
        if (version == 2) status = r->phase != 2 || r->second_call != 1 ||
            !aotx_cog_equal(aotx_appraisal.rows[row].first_model, r->model, 32) ? AOTX_COG_FORMAT :
            aotx_appraisal_outcome_parse(row, r->first_reply, r->first_bytes);
        if (!status) status = aotx_appraisal_parse(row);
        if (status) atomicCAS(&error, 0u, status);
    }
    __syncthreads();
    if (!row) {
        if (error && !replay) { work_status = error; error = 0; }
        if (!work_status) for (uint32_t i = 0; i < count; ++i) {
            const aotx_appraisal_row *r = aotx_appraisal.rows + i;
            if (!r->correction) continue;
            for (uint32_t j = 0; j < i; ++j) {
                const aotx_appraisal_row *old = aotx_appraisal.rows + j;
                if (old->correction && old->prior[old->correction - 1] == r->prior[r->correction - 1]) work_status = AOTX_COG_REFERENCE;
            }
        }
        if (!work_status) {
            if (count > UINT64_MAX - aotx_live_store.sequence) work_status = AOTX_COG_CAPACITY;
            else work_status = aotx_appraisal_ids();
        }
        objects = count * (work_status ? 1 : 3);
        payload = count * (AOTX_APPRAISAL_QUEUE_BYTES + (work_status ? 0 : AOTX_APPRAISAL_ASSESS_BYTES + AOTX_APPRAISAL_RELATION_BYTES));
        if (objects > AOTX_COG_OBJECTS - aotx_live_store.count || payload > AOTX_COG_PAYLOAD - aotx_live_store.bytes ||
            objects > UINT64_MAX - aotx_live_store.sequence || aotx_live_store.tick == UINT64_MAX) {
            work_status = AOTX_COG_CAPACITY; objects = payload = 0;
        }
        if (replay && !aotx_cog_u32(aotx_live.input + 12)) { objects = payload = 0; }
        tail_at = 64 + (objects ? count * stride : 0);
        tail_bytes = objects ? AOTX_COG_HEADER + objects * AOTX_COG_OBJECT + payload : 0;
        aotx_live.choice_bytes = tail_at + tail_bytes;
    }
    __syncthreads();
    for (uint32_t j = row; j < aotx_live.choice_bytes; j += blockDim.x) aotx_live.choices[j] = 0;
    __syncthreads();
    if (!row) {
        unsigned char *p = aotx_live.choices;
        for (uint32_t j = 0; j < 8; ++j) p[j] = "AOTXAPS1"[j];
        aotx_cog_put(p + 8, version, 4); aotx_cog_put(p + 12, objects ? count : 0, 4);
        aotx_cog_put(p + 16, aotx_live_store.sequence, 8);
        aotx_cog_put(p + 24, tail_bytes, 8); aotx_cog_put(p + 32, work_status, 4);
        if (objects) aotx_appraisal_tail_header(p + tail_at, objects, payload);
    }
    if (!error && objects && row < count) {
        const aotx_appraisal_row *item = aotx_appraisal.rows + row;
        const unsigned char *queue = aotx_live_store.objects[item->queue];
        const aotx_intake_row *r = aotx_intake.rows + row;
        unsigned char *p = aotx_live.choices + 64 + row * stride;
        for (uint32_t j = 0; j < 16; ++j) p[j] = queue[AOTX_CO_ID + j];
        aotx_cog_put(p + 16, aotx_cog_u64(queue + AOTX_CO_VERSION), 8);
        for (uint32_t j = 0; j < 32; ++j) p[24 + j] = r->model[j];
        if (version == 2) {
            bool live_first = r->phase == 1 && !replay;
            uint32_t first = min(live_first ? r->bytes : r->first_bytes, AOTX_INTAKE_REPLY);
            aotx_cog_put(p + 4160, first, 4); aotx_cog_put(p + 4164, r->second_call, 4);
            aotx_cog_put(p + 4168, r->phase, 4);
            for (uint32_t j = 0; j < 32; ++j) p[4192 + j] = item->first_model[j];
            for (uint32_t j = 0; j < first; ++j) p[4224 + j] = live_first ? r->reply[j] : r->first_reply[j];
        }
        uint32_t bytes = r->phase == 1 ? 0 : min(r->bytes, AOTX_INTAKE_REPLY);
        aotx_cog_put(p + 56, bytes, 4); aotx_cog_put(p + 60, work_status, 4);
        for (uint32_t j = 0; j < bytes; ++j) p[64 + j] = r->reply[j];
        aotx_appraisal_encode(aotx_live.choices + tail_at, row, work_status);
    }
    __syncthreads();
    for (uint32_t j = row; !error && j < sizeof(aotx_live_store); j += blockDim.x)
        ((unsigned char *)&aotx_live_candidate)[j] = ((const unsigned char *)&aotx_live_store)[j];
    __syncthreads();
    if (!error && objects) {
        aotx_cognitive_apply_block(&aotx_live_candidate, &aotx_live_scratch,
            aotx_live.choices + tail_at, tail_bytes, &aotx_live.result);
        __syncthreads();
        if (!row) error = aotx_live.result.status;
    }
    __syncthreads();
    if (!row && replay && aotx_live.total != aotx_live.choice_bytes) error = AOTX_COG_REFERENCE;
    __syncthreads();
    if (replay && !error) for (uint32_t j = row; j < aotx_live.choice_bytes; j += blockDim.x)
        if (aotx_live.input[j] != aotx_live.choices[j]) atomicCAS(&error, 0u, AOTX_COG_REFERENCE);
    __syncthreads();
    if (!row) {
        aotx_appraisal.status = work_status; aotx_live.status = error;
        if (error) {
            aotx_live.fatal = 1; aotx_appraisal.active = 0;
            aotx_live_note(AOTX_APPRAISAL_RESULT, error, 0); aotx_live.phase = AOTX_LIVE_IDLE;
        } else {
            aotx_live.received = aotx_live.written = 0;
            if (replay) aotx_live.choice_bytes = 0;
            aotx_live.phase = AOTX_LIVE_WRITE;
        }
    }
}
__device__ void aotx_appraisal_publish(void) {
    for (uint32_t j = threadIdx.x; j < sizeof(aotx_live_store); j += blockDim.x)
        ((unsigned char *)&aotx_live_store)[j] = ((const unsigned char *)&aotx_live_candidate)[j];
    __syncthreads();
    if (threadIdx.x) return;
    aotx_appraisal.last_status = aotx_appraisal.status;
    if (aotx_appraisal.status == AOTX_COG_CAPACITY && !aotx_cog_u64(aotx_live.choices + 24)) {
        aotx_appraisal.blocked_sequence = aotx_live_store.sequence; aotx_appraisal.blocked_root = aotx_live_store.root_sequence;
        aotx_appraisal.blocked_count = aotx_live_store.count; aotx_appraisal.blocked_bytes = aotx_live_store.bytes;
    }
    if (!aotx_appraisal.status) aotx_appraisal.completed += aotx_appraisal.count;
    else if (aotx_appraisal.status == AOTX_COG_DENIED) aotx_appraisal.interrupted += aotx_appraisal.count;
    else aotx_appraisal.refused += aotx_appraisal.count;
    ++aotx_live.accepted;
    aotx_live_note(AOTX_APPRAISAL_RESULT, aotx_appraisal.status, aotx_appraisal.count);
    aotx_appraisal.active = 0; aotx_live.phase = AOTX_LIVE_IDLE;
    aotx_appraisal.observed = UINT64_MAX;
}
