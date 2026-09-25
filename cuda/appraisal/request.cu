/* Purpose: Admit explicit and policy-selected appraisal batches without user turns.
 * Owns: Exact queue references, idle sequence assignment and recorded work requests.
 * Launch shape: Serial admission with one independent source per leased row.
 * Lifetime: One work request through its recorded result. */
#include "appraisal/appraisal.cuh"
#include "appraisal/schema.cuh"
#include "cognitive/checkpoint.cuh"
#include "cognitive/lookup.cuh"
#include "model/decode_state.cuh"
#include "model/load.cuh"
#include "policy/state.cuh"
#include "sched/sched.cuh"

static __device__ unsigned char aotx_appraisal_request_bytes[64 + AOTX_RECALL_BATCH * 32];
static __device__ unsigned char aotx_appraisal_request_part[AOTX_BODY_BYTES];

__device__ uint32_t aotx_appraisal_request(bool background) {
    aotx_appraisal_refresh();
    if (aotx_seam.replaying || aotx_sched.held || !aotx_checkpoint_idle(true) ||
        aotx_checkpoint_pressure() || !aotx_policy_quiet() || aotx_appraisal.control_pending ||
        aotx_appraisal_foreground() || aotx_policy.paused || aotx_policy.stopped ||
        aotx_appraisal.config >= aotx_live_store.count || !(aotx_appraisal.write_flags & AOTX_APPRAISAL_WRITE)) return AOTX_COG_DENIED;
    const unsigned char *c = aotx_live_store.objects[aotx_appraisal.config];
    const unsigned char *config = aotx_live_store.payload + aotx_cog_u64(c + AOTX_CO_OFFSET);
    uint32_t limit = aotx_cog_u32(config + 36), count = 0, slot = 0;
    unsigned char *request = aotx_appraisal_request_bytes;
    for (uint32_t j = 0; j < sizeof(aotx_appraisal_request_bytes); ++j) request[j] = 0;
    for (uint32_t i = 0; i < aotx_live_store.count && count < limit; ++i) {
        const unsigned char *r = aotx_live_store.objects[i], *p = aotx_live_store.payload + aotx_cog_u64(r + AOTX_CO_OFFSET);
        if (aotx_cog_cold(r)) continue;
        if (!aotx_appraisal_magic(p, aotx_cog_u64(r + AOTX_CO_BYTES), "AOTXAPQ1") ||
            !aotx_cog_equal(p + 64, config + 40, 32) ||
            aotx_cog_u32(p + 12) == AOTX_APPRAISAL_COMPLETE ||
            (background && aotx_cog_u32(p + 12) != AOTX_APPRAISAL_PENDING) ||
            aotx_cog_latest(&aotx_live_store, r + AOTX_CO_ID) != (int)i) continue;
        while (slot < AOTX_SLOTS && (aotx_seqs.slot[slot].state != AOTX_SEQ_STATE_FREE ||
            aotx_kv.count[slot] || ((aotx_agents.agent[slot].state != AOTX_AGENT_STATE_FREE) &&
                aotx_live_busy(slot)))) ++slot;
        if (slot == AOTX_SLOTS) break;
        aotx_cognitive_query q = {};
        for (uint32_t j = 0; j < 16; ++j) {
            q.id[j] = r[AOTX_CO_ID + j]; q.principal[j] = r[AOTX_CO_OWNER + j]; q.room[j] = r[AOTX_CO_ROOM + j];
        }
        q.version = aotx_cog_u64(r + AOTX_CO_VERSION);
        if (aotx_appraisal_resolve(&q)) continue;
        unsigned char *out = request + 64 + count++ * 32;
        for (uint32_t j = 0; j < 16; ++j) out[j] = q.id[j];
        aotx_cog_put(out + 16, q.version, 8); aotx_cog_put(out + 24, slot++, 4);
    }
    if (!count) return aotx_appraisal.pending ? AOTX_COG_CAPACITY : AOTX_COG_MISSING;
    if (background && !aotx_policy_appraisal()) return AOTX_COG_DENIED;
    for (uint32_t j = 0; j < 8; ++j) request[j] = "AOTXAPR1"[j];
    aotx_cog_put(request + 8, 1, 4); aotx_cog_put(request + 12, count, 4);
    aotx_cog_put(request + 16, aotx_live_store.sequence, 8);
    for (uint32_t j = 0; j < 16; ++j) request[24 + j] = c[AOTX_CO_ID + j];
    aotx_cog_put(request + 40, aotx_cog_u64(c + AOTX_CO_VERSION), 8);
    aotx_cog_put(request + 48, background ? 1 : 0, 4);
    unsigned char *part = aotx_appraisal_request_part;
    uint32_t total = 64 + count * 32;
    for (uint32_t offset = 0; offset < total; offset += AOTX_LIVE_DATA) {
        for (uint32_t j = 0; j < AOTX_LIVE_PART; ++j) part[j] = 0;
        aotx_cog_put(part, 1, 4); aotx_cog_put(part + 4, AOTX_APPRAISAL_REQUEST, 4);
        for (uint32_t j = 0; j < 8; ++j) part[8 + j] = "AOTXAPR1"[j];
        aotx_cog_put(part + 16, aotx_live.accepted + 1, 8);
        aotx_cog_put(part + 24, total, 4); aotx_cog_put(part + 28, offset, 4);
        uint32_t bytes = min(AOTX_LIVE_DATA, total - offset);
        for (uint32_t j = 0; j < bytes; ++j) part[AOTX_LIVE_PART + j] = request[offset + j];
        uint64_t seq = aotx_seam_write(AOTX_WRITER_SYSTEM, AOTX_CLASS_A, AOTX_LIVE_RECORD,
            AOTX_FLAG_ADMISSION, part, AOTX_LIVE_PART + bytes);
        aotx_seam.apply.state_hash = aotx_seam_fnv1a(aotx_seam.apply.state_hash, part, AOTX_LIVE_PART + bytes);
        ++aotx_seam.apply.applied_count;
        aotx_live_part(part, AOTX_LIVE_PART + bytes, seq, AOTX_FLAG_ADMISSION);
    }
    return 0;
}
__device__ void aotx_appraisal_auto_request(void) {
    if (aotx_seam.replaying || aotx_appraisal.control_pending) return;
    if (aotx_appraisal.explicit_pending) {
        if (!aotx_checkpoint_idle(true) || !aotx_policy_quiet() || aotx_checkpoint_pressure()) return;
        uint32_t status = aotx_appraisal_request(false);
        aotx_appraisal.explicit_pending = 0;
        aotx_appraisal.last_status = status == AOTX_COG_MISSING ? 0 : status;
        return;
    }
    if (!aotx_appraisal_enabled() || !aotx_appraisal_pending()) return;
    uint32_t status = aotx_appraisal_request(true);
    if (status && status != AOTX_COG_DENIED && status != AOTX_COG_MISSING) aotx_appraisal.last_status = status;
}
__device__ void aotx_appraisal_begin(void) {
    if (threadIdx.x) return;
    const unsigned char *p = aotx_live.input;
    uint32_t count = aotx_cog_u32(p + 12), status = 0;
    aotx_appraisal_refresh();
    if (aotx_live.total < 64 || !aotx_appraisal_magic(p, aotx_live.total, "AOTXAPR1") ||
        aotx_cog_u32(p + 8) != 1 || !count || count > AOTX_RECALL_BATCH ||
        aotx_live.total != 64 + count * 32 || aotx_cog_u32(p + 48) > 1 || !aotx_cog_zero(p + 52, 12)) status = AOTX_COG_FORMAT;
    if (!status && (aotx_appraisal.config >= aotx_live_store.count || aotx_live_admission_pressure() ||
        !aotx_checkpoint_quiet() || aotx_cog_u64(p + 16) != aotx_live_store.sequence)) status = AOTX_COG_DENIED;
    if (!status) {
        const unsigned char *c = aotx_live_store.objects[aotx_appraisal.config];
        const unsigned char *config = aotx_live_store.payload + aotx_cog_u64(c + AOTX_CO_OFFSET);
        if (!aotx_cog_equal(p + 24, c + AOTX_CO_ID) || aotx_cog_u64(p + 40) != aotx_cog_u64(c + AOTX_CO_VERSION) ||
            !(aotx_appraisal.write_flags & AOTX_APPRAISAL_WRITE)) status = AOTX_COG_STALE;
        else if (count > aotx_cog_u32(config + 36)) status = AOTX_COG_CAPACITY;
        if (!status) aotx_appraisal.result_version = aotx_appraisal_contract(config + 40);
    }
    for (uint32_t i = 0; !status && i < count; ++i) {
        const unsigned char *in = p + 64 + i * 32;
        uint32_t slot = aotx_cog_u32(in + 24);
        int queue = aotx_cog_find(&aotx_live_store, in, aotx_cog_u64(in + 16));
        if (queue < 0 || slot >= AOTX_SLOTS || aotx_cog_u32(in + 28) ||
            aotx_cog_latest(&aotx_live_store, in) != queue ||
            (!aotx_seam.replaying && (aotx_seqs.slot[slot].state != AOTX_SEQ_STATE_FREE || aotx_kv.count[slot]))) { status = AOTX_COG_REFERENCE; break; }
        for (uint32_t j = 0; j < i; ++j)
            if (aotx_appraisal.rows[j].queue == (uint32_t)queue || aotx_appraisal.rows[j].slot == slot) status = AOTX_COG_REFERENCE;
        const unsigned char *r = aotx_live_store.objects[queue], *qp = aotx_live_store.payload + aotx_cog_u64(r + AOTX_CO_OFFSET);
        aotx_cognitive_query access = {};
        for (uint32_t j = 0; j < 16; ++j) {
            access.id[j] = r[AOTX_CO_ID + j]; access.principal[j] = r[AOTX_CO_OWNER + j]; access.room[j] = r[AOTX_CO_ROOM + j];
        }
        access.version = aotx_cog_u64(r + AOTX_CO_VERSION);
        if (aotx_appraisal_resolve(&access)) { status = AOTX_COG_DENIED; break; }
        uint32_t length = 0;
        const unsigned char *source = aotx_appraisal_source(&aotx_live_store, r, &length);
        if (!source || !aotx_appraisal_magic(qp, aotx_cog_u64(r + AOTX_CO_BYTES), "AOTXAPQ1") ||
            aotx_appraisal_contract(qp + 64) != aotx_appraisal.result_version ||
            aotx_cog_u32(qp + 12) == AOTX_APPRAISAL_COMPLETE ||
            (aotx_cog_u32(p + 48) && aotx_cog_u32(qp + 12) != AOTX_APPRAISAL_PENDING)) { status = AOTX_COG_SOURCE; break; }
        aotx_appraisal_row *row = aotx_appraisal.rows + i;
        *row = {}; row->queue = (uint32_t)queue; row->slot = slot;
        row->source = (uint32_t)aotx_cog_find(&aotx_live_store, r + AOTX_CO_SOURCE, aotx_cog_u64(r + AOTX_CO_SOURCE_VERSION));
        row->task_source = (uint32_t)aotx_cog_find(&aotx_live_store, qp + 128, aotx_cog_u64(qp + 144));
        for (uint32_t j = 0; j < 16; ++j) row->task[j] = qp[40 + j];
        unsigned char *q = aotx_live.requests + 64 + i * AOTX_RECALL_QUERY;
        for (uint32_t j = 0; j < AOTX_RECALL_QUERY; ++j) q[j] = 0;
        aotx_cog_put(q + 148, length, 4);
        for (uint32_t j = 0; j < length; ++j) q[4640 + j] = source[j];
        for (uint32_t j = 0; j < 64; ++j) aotx_live.prefixes[i][j] = 0;
        aotx_cog_put(aotx_live.prefixes[i], slot, 4);
        for (uint32_t remaining = aotx_live_store.count; remaining && row->prior_count < AOTX_RECALL_LIMIT; --remaining) {
            uint32_t j = remaining - 1;
            const unsigned char *old = aotx_live_store.objects[j];
            if (aotx_cog_u16(old + AOTX_CO_KIND) == AOTX_COG_APPRAISAL &&
                aotx_cog_u64(old + AOTX_CO_BYTES) == AOTX_APPRAISAL_ASSESS_BYTES &&
                aotx_cog_equal(old + AOTX_CO_SUBJECT, r + AOTX_CO_SUBJECT) &&
                aotx_cog_equal(old + AOTX_CO_OWNER, r + AOTX_CO_OWNER, 32) &&
                aotx_cog_u32(old + AOTX_CO_SCOPE) == aotx_cog_u32(r + AOTX_CO_SCOPE) &&
                aotx_cog_latest(&aotx_live_store, old + AOTX_CO_ID) == (int)j &&
                !aotx_cog_superseded(&aotx_live_store, old) &&
                !(aotx_cog_u32(old + AOTX_CO_FLAGS) & (AOTX_COG_PROTECTED | AOTX_COG_TOMBSTONE)) &&
                aotx_cog_u32(old + AOTX_CO_EVIDENCE) != 3 &&
                (!aotx_cog_u64(old + AOTX_CO_EXPIRY) || aotx_cog_u64(old + AOTX_CO_EXPIRY) > aotx_live_store.sequence))
                row->prior[row->prior_count++] = j;
        }
    }
    aotx_appraisal.status = aotx_live.status = status;
    if (status) {
        aotx_appraisal.last_status = status; ++aotx_live.refused; aotx_live.received = 0;
        aotx_live_note(AOTX_APPRAISAL_REQUEST, status, 0); aotx_live.phase = AOTX_LIVE_IDLE; return;
    }
    aotx_appraisal.active = 1; aotx_appraisal.count = aotx_live.count = count;
    aotx_appraisal.recovery = 0;
    for (uint32_t i = 0; i < count; ++i) aotx_intake.rows[i] = {};
    aotx_live.auto_mode = aotx_live.intake_mode = aotx_live.text_mode = 0;
    aotx_live.request_seq = aotx_live.source_seq;
    for (uint32_t j = 0; j < 16; ++j) aotx_live.query_id[j] = aotx_live.transfer_id[j];
    aotx_live.received = 0;
    if (aotx_seam.replaying) aotx_live.phase = AOTX_LIVE_WAIT;
    else { ++aotx_appraisal.calls; aotx_intake_begin(); }
}
