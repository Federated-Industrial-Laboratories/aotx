/* Purpose: Assign automatic retention IDs and construct bounded source rows.
 * Owns: Staged retention metadata; no live store or binding mutation.
 * Launch shape: Serial ID allocation for up to 64 distinct query rows.
 * Lifetime: One combined input and retention decision. */
#ifndef AOTX_COGNITIVE_LIVE_AUTO_ROWS_CUH
#define AOTX_COGNITIVE_LIVE_AUTO_ROWS_CUH
#include "cognitive/live_retain.cuh"
#include "cognitive/live_selected.cuh"
#include "shared/bridge.cuh"

__device__ inline uint32_t aotx_live_auto_stride(void) {
    return aotx_live.intake_mode ? AOTX_LIVE_INTAKE_ROW : AOTX_LIVE_AUTO_ROW;
}
static __device__ __noinline__ uint32_t aotx_live_auto_rows(void) {
    aotx_live.auto_count = 0;
    for (uint32_t i = 0; i < aotx_live.count; ++i) {
        uint32_t slot = aotx_cog_u32(aotx_live.prefixes[i]);
        if (aotx_live_bindings[slot].auto_retain) aotx_live.auto_rows[aotx_live.auto_count++] = i;
    }
    uint32_t count = aotx_live.auto_count;
    if (!count) return AOTX_COG_FORMAT;
    if (count * 3 > AOTX_COG_OBJECTS - aotx_live_store.count || aotx_live_store.tick == UINT64_MAX ||
        aotx_live_store.sequence > UINT64_MAX - count * 3) return AOTX_COG_CAPACITY;
    uint64_t next = aotx_live_store.sequence + 1, payload = 0;
    uint32_t attempts = 0, limit = aotx_live_store.count + 3 * aotx_live.count;
    for (uint32_t i = 0; i < count; ++i) {
        uint32_t row = aotx_live.auto_rows[i];
        const unsigned char *prefix = aotx_live.prefixes[row];
        uint32_t slot = aotx_cog_u32(prefix);
        const unsigned char *q = aotx_live.requests + 64 + row * AOTX_RECALL_QUERY;
        unsigned char *r = aotx_live.retain_rows[i];
        for (uint32_t j = 0; j < AOTX_LIVE_RETAIN_ROW; ++j) r[j] = 0;
        aotx_cog_put(r, aotx_cog_u32(prefix), 4); aotx_cog_put(r + 4, 1, 4);
        aotx_cog_put(r + 24, aotx_cog_u64(prefix + 32), 8);
        aotx_cog_put(r + 104, 1, 8); aotx_cog_put(r + 128, AOTX_COG_UNKNOWN, 4);
        for (uint32_t j = 0; j < 16; ++j) { r[8 + j] = prefix[16 + j]; r[32 + j] = q[j]; r[112 + j] = aotx_shared_actor(slot) ? aotx_shared_actor(slot)[j] : q[16 + j]; }
        for (uint32_t k = 0; k < 2; ++k) {
            unsigned char *id = r + 48 + k * 16;
            for (uint32_t j = 0; j < 8; ++j) id[j] = "AOTXGEN1"[j];
            bool occupied;
            do {
                if (!next || attempts++ >= limit) return AOTX_COG_CAPACITY;
                aotx_cog_put(id + 8, next++, 8);
                occupied = aotx_cog_latest(&aotx_live_store, id) >= 0;
                for (uint32_t j = 0; j < aotx_live.count && !occupied; ++j)
                    occupied = aotx_cog_equal(id, aotx_live.requests + 64 + j * AOTX_RECALL_QUERY);
            } while (occupied);
        }
        payload += aotx_retain_payload_bytes(r);
    }
    return payload > AOTX_COG_PAYLOAD - aotx_live_store.bytes ? AOTX_COG_CAPACITY : AOTX_COG_OK;
}
__device__ inline unsigned char *aotx_live_auto_retained(uint32_t i) {
    return aotx_live.choices + 64 + aotx_live.auto_rows[i] * aotx_live_auto_stride() + AOTX_LIVE_TEXT_CHOICE_ROW;
}
static __device__ __noinline__ uint32_t aotx_live_auto_header(uint32_t *refusal) {
    const unsigned char *p = aotx_live.input;
    if (aotx_live.total < 64) return AOTX_COG_FORMAT;
    uint32_t count = aotx_cog_u32(p + 8); *refusal = aotx_cog_u32(p + 44);
    uint64_t tail = aotx_cog_u64(p + 48);
    if (!aotx_recall_magic(p, aotx_live.intake_mode ? "AOTXICH1" : "AOTXACH1") || aotx_cog_u32(p + 12) != 1 ||
        aotx_cog_u32(p + 40) != aotx_live_auto_stride() || !aotx_cog_zero(p + 56, 8) ||
        !aotx_cog_equal(aotx_live.transfer_id, aotx_live.query_id) ||
        !aotx_cog_equal(p + 16, aotx_live_store.lineage) || aotx_cog_u64(p + 32) != aotx_live_store.sequence ||
        *refusal > AOTX_COG_UNAVAILABLE || count > AOTX_RECALL_BATCH || tail > AOTX_COG_IMAGE ||
        aotx_live.total != 64 + (uint64_t)count * aotx_live_auto_stride() + tail ||
        (*refusal ? count || tail : !count || count != aotx_live.count || aotx_live.status) ||
        (aotx_live.status && aotx_live.status != *refusal)) return AOTX_COG_REFERENCE;
    return AOTX_COG_OK;
}
__device__ __forceinline__ uint32_t aotx_live_auto_recorded(uint32_t i) {
    const unsigned char *p = aotx_live.input + 64 + i * aotx_live_auto_stride();
    unsigned char *q = aotx_live.requests + 64 + i * AOTX_RECALL_QUERY;
    uint32_t status = aotx_live.text_mode ? aotx_live_text_recorded(q, p + 64) :
        (aotx_cog_equal(q, p + 64, AOTX_RECALL_QUERY) ? 0 : AOTX_COG_REFERENCE);
    if (status || !aotx_cog_equal(p, aotx_live.prefixes[i], 64)) return status ? status : AOTX_COG_REFERENCE;
    if (aotx_live.text_mode) for (uint32_t j = 0; j < AOTX_RECALL_QUERY; ++j) q[j] = p[64 + j];
    aotx_recall_result *out = aotx_live.results + i;
    for (uint32_t j = 0; j < sizeof(*out); ++j) ((unsigned char *)out)[j] = 0;
    out->cut = aotx_live_store.sequence;
    for (uint32_t j = 0; j < 16; ++j) { out->request_id[j] = q[j]; out->selection_id[j] = q[48 + j]; }
    for (uint32_t j = 0; j < AOTX_RECALL_SELECTION; ++j) out->selection[j] = p[64 + AOTX_RECALL_QUERY + j];
    return aotx_live_selected(q, out);
}
__device__ inline void aotx_live_auto_fatal(uint32_t error) {
    aotx_live.fatal = 1; ++aotx_live.refused; aotx_live.received = 0;
    aotx_live_note(aotx_live_result_op(), error, 0); aotx_live.phase = AOTX_LIVE_IDLE;
}
#endif
