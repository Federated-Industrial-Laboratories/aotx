/* Purpose: Encode completed queue versions and their source-backed evidence.
 * Owns: Canonical object tails; the caller validates and publishes the whole batch.
 * Launch shape: One thread encodes each independent source row.
 * Lifetime: A complete recorded decision and exact replay. */
#ifndef AOTX_APPRAISAL_ENCODE_CUH
#define AOTX_APPRAISAL_ENCODE_CUH
#include "appraisal/appraisal.cuh"
#include "appraisal/correction.cuh"
#include "cognitive/lookup.cuh"

__device__ inline void aotx_appraisal_tail_header(unsigned char *tail, uint32_t objects, uint32_t bytes) {
    for (uint32_t j = 0; j < 8; ++j) tail[j] = "AOTXLOG1"[j];
    aotx_cog_put(tail + 8, 1, 4); aotx_cog_put(tail + 12, AOTX_COG_HEADER, 4);
    aotx_cog_put(tail + 16, AOTX_COG_OBJECT, 4); aotx_cog_put(tail + 20, objects, 4);
    aotx_cog_put(tail + 24, bytes, 8); aotx_cog_put(tail + 32, aotx_live_store.sequence + 1, 8);
    aotx_cog_put(tail + 40, aotx_live_store.tick + 1, 8);
    for (uint32_t j = 0; j < 16; ++j) tail[48 + j] = aotx_live_store.lineage[j];
    aotx_cog_put(tail + 64, AOTX_COG_HEADER, 8);
    aotx_cog_put(tail + 72, AOTX_COG_HEADER + objects * AOTX_COG_OBJECT, 8);
    aotx_cog_put(tail + 80, AOTX_COG_HEADER + objects * AOTX_COG_OBJECT + bytes, 8);
    aotx_cog_put(tail + 88, 1, 4); aotx_cog_policy_write(tail, &aotx_live_store);
}
__device__ inline uint32_t aotx_appraisal_ids(void) {
    uint32_t count = aotx_appraisal.count;
    for (uint32_t i = 0; i < count; ++i) for (uint32_t k = 0; k < 2; ++k) {
        unsigned char *id = aotx_appraisal.rows[i].ids[k];
        for (uint32_t j = 0; j < 8; ++j) id[j] = (k ? "AOTXREL1" : "AOTXAPA1")[j];
        uint64_t suffix = aotx_live_store.sequence + i + 1;
        for (;;) {
            aotx_cog_put(id + 8, suffix, 8);
            if (aotx_cog_latest(&aotx_live_store, id) < 0) break;
            if (suffix > UINT64_MAX - count) return AOTX_COG_CAPACITY;
            suffix += count;
        }
    }
    return 0;
}
__device__ inline void aotx_appraisal_encode(unsigned char *tail, uint32_t row, uint32_t status) {
    const aotx_appraisal_row *item = aotx_appraisal.rows + row;
    const unsigned char *old = aotx_live_store.objects[item->queue];
    const unsigned char *before = aotx_live_store.payload + aotx_cog_u64(old + AOTX_CO_OFFSET);
    uint32_t per = status ? 1 : 3, count = aotx_appraisal.count * per;
    uint32_t each = AOTX_APPRAISAL_QUEUE_BYTES + (status ? 0 : AOTX_APPRAISAL_ASSESS_BYTES + AOTX_APPRAISAL_RELATION_BYTES);
    uint64_t queue_version = aotx_live_store.pressure_percent ? aotx_live_store.sequence + row * per + 1 :
        aotx_cog_u64(old + AOTX_CO_VERSION) + 1;
    for (uint32_t k = 0; k < per; ++k) {
        uint32_t index = row * per + k, offset = row * each + (k ? AOTX_APPRAISAL_QUEUE_BYTES : 0) +
            (k == 2 ? AOTX_APPRAISAL_ASSESS_BYTES : 0);
        uint32_t bytes = !k ? AOTX_APPRAISAL_QUEUE_BYTES : k == 1 ? AOTX_APPRAISAL_ASSESS_BYTES : AOTX_APPRAISAL_RELATION_BYTES;
        uint64_t sequence = aotx_live_store.sequence + index + 1;
        unsigned char *r = tail + AOTX_COG_HEADER + index * AOTX_COG_OBJECT;
        unsigned char *p = tail + AOTX_COG_HEADER + count * AOTX_COG_OBJECT + offset;
        for (uint32_t j = 0; j < AOTX_COG_OBJECT; ++j) r[j] = old[j];
        aotx_cog_put(r + AOTX_CO_KIND, !k ? AOTX_COG_POLICY : k == 1 ? AOTX_COG_APPRAISAL : AOTX_COG_RELATIONSHIP, 2);
        aotx_cog_put(r + AOTX_CO_UPDATED, sequence, 8);
        aotx_cog_put(r + AOTX_CO_OFFSET, offset, 8); aotx_cog_put(r + AOTX_CO_BYTES, bytes, 8);
        aotx_cog_put(r + AOTX_CO_RETENTION, status == AOTX_COG_DENIED ? 2 : 0, 4);
        if (!k) {
            aotx_cog_put(r + AOTX_CO_VERSION, queue_version, 8);
            for (uint32_t j = 0; j < AOTX_APPRAISAL_QUEUE_BYTES; ++j) p[j] = before[j];
            aotx_cog_put(p + 12, !status ? AOTX_APPRAISAL_COMPLETE : status == AOTX_COG_DENIED ? AOTX_APPRAISAL_INTERRUPTED : AOTX_APPRAISAL_REFUSED, 4);
            aotx_cog_put(p + 56, status, 4);
            for (uint32_t j = 0; j < 32; ++j) p[96 + j] = aotx_intake.rows[row].model[j];
            continue;
        }
        for (uint32_t j = 0; j < 16; ++j) r[AOTX_CO_ID + j] = item->ids[k - 1][j];
        aotx_cog_put(r + AOTX_CO_CREATED, sequence, 8);
        aotx_cog_put(r + AOTX_CO_VERSION, aotx_live_store.pressure_percent ? sequence : 1, 8);
        uint32_t ref = k == 1 ? 96 : 136;
        for (uint32_t j = 0; j < 16; ++j) p[ref + j] = old[AOTX_CO_ID + j];
        aotx_cog_put(p + ref + 16, queue_version, 8);
        for (uint32_t j = 0; j < 32; ++j) {
            p[(k == 1 ? 32 : 72) + j] = aotx_appraisal_processor[j];
            p[(k == 1 ? 64 : 104) + j] = aotx_intake.rows[row].model[j];
        }
        if (item->correction) {
            uint32_t prior = item->prior[item->correction - 1];
            if (k == 2) prior = aotx_appraisal_old_relation(prior);
            if (prior != UINT32_MAX) {
                const unsigned char *pr = aotx_live_store.objects[prior];
                for (uint32_t j = 0; j < 16; ++j) r[AOTX_CO_SUPERSEDES + j] = pr[AOTX_CO_ID + j];
                aotx_cog_put(r + AOTX_CO_SUPER_VERSION, aotx_cog_u64(pr + AOTX_CO_VERSION), 8);
            }
        }
        if (k == 1) {
            aotx_cog_put(p, 2, 4);
            for (uint32_t j = 0; j < 5; ++j) aotx_cog_put(p + 4 + j * 4, item->values[j], 4);
            aotx_cog_put(p + 24, 1, 4);
            aotx_cog_put(p + 120, item->quote_start, 4); aotx_cog_put(p + 124, item->quote_length, 4);
        } else {
            for (uint32_t j = 0; j < 8; ++j) p[j] = "AOTXREL1"[j];
            aotx_cog_put(p + 8, 1, 4); aotx_cog_put(p + 12, 1, 4);
            for (uint32_t j = 0; j < 4; ++j) aotx_cog_put(p + 16 + j * 4, item->values[5 + j], 4);
            for (uint32_t j = 0; j < 16; ++j) p[32 + j] = item->task[j];
            aotx_cog_put(p + 48, item->quote_start, 4); aotx_cog_put(p + 52, item->quote_length, 4);
            aotx_cog_put(p + 56, item->task_start, 4); aotx_cog_put(p + 60, item->task_length, 4);
            aotx_cog_put(p + 64, item->commitment_start, 4); aotx_cog_put(p + 68, item->commitment_length, 4);
        }
    }
}
#endif
