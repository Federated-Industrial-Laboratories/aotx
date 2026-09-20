/* Purpose: Add scoped appraisal work to each admitted automatic source batch.
 * Owns: Pending work IDs and payload encoding; the caller publishes atomically.
 * Launch shape: Serial capacity admission and one encoding thread per source.
 * Lifetime: Retained source through completion, cancellation or explicit removal. */
#include "appraisal/appraisal.cuh"
#include "appraisal/schema.cuh"
#include "cognitive/live_auto_rows.cuh"

__device__ uint32_t aotx_appraisal_queue_prepare(uint32_t *objects, uint32_t *payload) {
    aotx_appraisal_refresh();
    aotx_appraisal.enqueue_count = 0;
    if (aotx_appraisal.config >= aotx_live_store.count) return 0;
    const unsigned char *c = aotx_live_store.objects[aotx_appraisal.config];
    const unsigned char *p = aotx_live_store.payload + aotx_cog_u64(c + AOTX_CO_OFFSET);
    if (!(aotx_cog_u32(p + 12) & AOTX_APPRAISAL_WRITE)) return 0;
    if (!aotx_cog_equal(p + 40, aotx_appraisal_processor, 32)) return AOTX_COG_LAYOUT;
    uint32_t retained = aotx_live.text_mode == 2 ? aotx_live.count : aotx_live.auto_count, count = 0;
    for (uint32_t i = 0; i < retained; ++i)
        if (!aotx_cog_zero(aotx_live.retain_rows[i] + 112, 16)) aotx_appraisal.enqueue_rows[count++] = i;
    if (count > AOTX_COG_OBJECTS - aotx_live_store.count - *objects ||
        count * AOTX_APPRAISAL_QUEUE_BYTES > AOTX_COG_PAYLOAD - aotx_live_store.bytes - *payload ||
        *objects + count > UINT64_MAX - aotx_live_store.sequence) return AOTX_COG_CAPACITY;
    uint64_t next = aotx_live_store.sequence + *objects + 1;
    uint32_t attempts = 0, limit = aotx_live_store.count + count + 3 * retained + aotx_live.count;
    for (uint32_t i = 0; i < count; ++i) {
        unsigned char *id = aotx_appraisal.enqueue_ids[i];
        for (uint32_t j = 0; j < 8; ++j) id[j] = "AOTXAPQ1"[j];
        bool used;
        do {
            if (!next || attempts++ >= limit) return AOTX_COG_CAPACITY;
            aotx_cog_put(id + 8, next++, 8);
            used = aotx_cog_latest(&aotx_live_store, id) >= 0;
            for (uint32_t j = 0; !used && j < retained; ++j)
                for (uint32_t k = 32; !used && k <= 64; k += 16) used = aotx_cog_equal(id, aotx_live.retain_rows[j] + k);
            for (uint32_t j = 0; !used && aotx_live.text_mode != 2 && j < aotx_live.count; ++j)
                used = aotx_cog_equal(id, aotx_live.requests + 64 + j * AOTX_RECALL_QUERY);
        } while (used);
    }
    aotx_appraisal.enqueue_count = count; aotx_appraisal.enqueue_first = *objects;
    aotx_appraisal.enqueue_offset = *payload;
    *objects += count; *payload += count * AOTX_APPRAISAL_QUEUE_BYTES;
    if (aotx_live.intake_mode) aotx_intake.objects = *objects;
    return 0;
}
__device__ void aotx_appraisal_queue_encode(unsigned char *tail, uint32_t i, uint32_t objects) {
    uint32_t queued = 0;
    while (queued < aotx_appraisal.enqueue_count && aotx_appraisal.enqueue_rows[queued] != i) ++queued;
    if (queued == aotx_appraisal.enqueue_count) return;
    const unsigned char *in = aotx_live.retain_rows[i];
    const unsigned char *q = aotx_retain_source(in);
    const unsigned char *c = aotx_live_store.objects[aotx_appraisal.config];
    const unsigned char *cp = aotx_live_store.payload + aotx_cog_u64(c + AOTX_CO_OFFSET);
    uint32_t index = aotx_appraisal.enqueue_first + queued;
    uint32_t offset = aotx_appraisal.enqueue_offset + queued * AOTX_APPRAISAL_QUEUE_BYTES;
    unsigned char *r = tail + AOTX_COG_HEADER + index * AOTX_COG_OBJECT;
    unsigned char *p = tail + AOTX_COG_HEADER + objects * AOTX_COG_OBJECT + offset;
    uint64_t sequence = aotx_live_store.sequence + index + 1;
    aotx_cog_put(r, 1, 2); aotx_cog_put(r + AOTX_CO_KIND, AOTX_COG_POLICY, 2);
    for (uint32_t j = 0; j < 16; ++j) {
        r[AOTX_CO_ID + j] = aotx_appraisal.enqueue_ids[queued][j]; r[AOTX_CO_LINEAGE + j] = aotx_live_store.lineage[j];
        r[AOTX_CO_OWNER + j] = q[16 + j]; r[AOTX_CO_ROOM + j] = q[32 + j];
        r[AOTX_CO_SOURCE + j] = in[32 + j]; r[AOTX_CO_SUBJECT + j] = in[112 + j];
        p[16 + j] = c[AOTX_CO_ID + j];
        if (aotx_context_flags(q) & AOTX_RECALL_TASKS) p[40 + j] = q[AOTX_RECALL_EXTENSION + 16 + j];
    }
    aotx_cog_put(r + AOTX_CO_VERSION, aotx_live_store.pressure_percent ? sequence : 1, 8);
    aotx_cog_put(r + AOTX_CO_CREATED, sequence, 8); aotx_cog_put(r + AOTX_CO_UPDATED, sequence, 8);
    aotx_cog_put(r + AOTX_CO_SOURCE_VERSION, aotx_live_store.pressure_percent ? aotx_live_store.sequence + i * 3 + 1 : 1, 8);
    aotx_cog_put(r + AOTX_CO_SCOPE, aotx_cog_u32(q + 152), 4);
    aotx_cog_put(r + AOTX_CO_SOURCE_KIND, AOTX_COG_INFERRED, 4);
    aotx_cog_put(r + AOTX_CO_RETENTION, 2, 4); aotx_cog_put(r + AOTX_CO_IMPORTANCE, AOTX_COG_UNKNOWN, 4);
    aotx_cog_put(r + AOTX_CO_POLICY, 1, 8); aotx_cog_put(r + AOTX_CO_OFFSET, offset, 8);
    aotx_cog_put(r + AOTX_CO_BYTES, AOTX_APPRAISAL_QUEUE_BYTES, 8);
    for (uint32_t j = 0; j < 8; ++j) p[j] = "AOTXAPQ1"[j];
    aotx_cog_put(p + 8, 1, 4); aotx_cog_put(p + 32, aotx_cog_u64(c + AOTX_CO_VERSION), 8);
    for (uint32_t j = 0; j < 32; ++j) p[64 + j] = cp[40 + j];
    int task = aotx_cog_latest(&aotx_live_store, p + 40);
    if (task >= 0) {
        const unsigned char *t = aotx_live_store.objects[task], *tp = aotx_live_store.payload + aotx_cog_u64(t + AOTX_CO_OFFSET);
        uint64_t bytes = aotx_cog_u64(t + AOTX_CO_BYTES);
        if (!aotx_cog_cold(t) && aotx_cog_u16(t + AOTX_CO_KIND) == AOTX_COG_CUE &&
            aotx_cog_u32(t + AOTX_CO_SOURCE_KIND) == AOTX_COG_AUTHORED &&
            !(aotx_cog_u32(t + AOTX_CO_FLAGS) & AOTX_COG_TOMBSTONE) && aotx_cog_u32(t + AOTX_CO_EVIDENCE) != 3 &&
            (!aotx_cog_u64(t + AOTX_CO_EXPIRY) || aotx_cog_u64(t + AOTX_CO_EXPIRY) > aotx_live_store.sequence + objects) &&
            !aotx_cog_superseded(&aotx_live_store, t) && aotx_cog_scope(r, t) &&
            aotx_appraisal_magic(tp, bytes, "AOTXMEM1") && bytes >= 32 &&
            aotx_cog_u32(tp + 8) == 1 && aotx_cog_zero(tp + 16, 16) &&
            aotx_cog_u32(tp + 12) && bytes == 32ull + aotx_cog_u32(tp + 12)) {
            for (uint32_t j = 0; j < 16; ++j) p[128 + j] = t[AOTX_CO_ID + j];
            aotx_cog_put(p + 144, aotx_cog_u64(t + AOTX_CO_VERSION), 8);
        }
    }
}
