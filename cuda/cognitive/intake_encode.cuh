/* Purpose: Encode admitted interpretations in the combined memory decision.
 * Owns: Generated IDs, canonical payloads and deterministic correction effects.
 * Launch shape: One thread per source, with serial batch capacity and ID checks.
 * Lifetime: One atomic decision and its exact replay. */
#ifndef AOTX_COGNITIVE_INTAKE_ENCODE_CUH
#define AOTX_COGNITIVE_INTAKE_ENCODE_CUH
#include "cognitive/intake_parse.cuh"
#include "model/load.cuh"
#include "model/decode_state.cuh"

__device__ inline uint32_t aotx_intake_recorded(uint32_t row) {
    const unsigned char *p = aotx_live.input + 64 + row * AOTX_LIVE_INTAKE_ROW + AOTX_LIVE_AUTO_ROW;
    aotx_intake_row *r = aotx_intake.rows + row;
    r->count = r->bytes = r->status = 0;
    uint32_t slot = aotx_cog_u32(aotx_live.prefixes[row]);
    if (aotx_live_bindings[slot].auto_retain != 2) return aotx_cog_zero(p, AOTX_INTAKE_META + AOTX_INTAKE_REPLY) ? 0 : AOTX_COG_REFERENCE;
    uint32_t bytes = aotx_cog_u32(p + 4), role = aotx_decode.role;
    if (aotx_cog_u32(p) != 1 || !bytes || bytes > AOTX_INTAKE_REPLY ||
        aotx_cog_u32(p + 72) > AOTX_INTAKE_ITEMS || !aotx_cog_zero(p + 76, 52) ||
        !aotx_cog_zero(p + AOTX_INTAKE_META + bytes, AOTX_INTAKE_REPLY - bytes) ||
        !aotx_cog_equal(p + 40, aotx_intake_processor, 32) || role >= AOTX_MODEL_ROLES ||
        !aotx_model_load.resident[role].active ||
        !aotx_cog_equal(p + 8, aotx_model_load.resident[role].body.digest, 32)) return AOTX_COG_LAYOUT;
    r->bytes = bytes;
    for (uint32_t j = 0; j < bytes; ++j) r->reply[j] = p[AOTX_INTAKE_META + j];
    for (uint32_t j = 0; j < 32; ++j) r->model[j] = p[8 + j];
    uint32_t status = aotx_intake_parse(row);
    return status ? status : r->count == aotx_cog_u32(p + 72) ? AOTX_COG_OK : AOTX_COG_REFERENCE;
}
__device__ inline void aotx_intake_metadata(uint32_t row) {
    uint32_t slot = aotx_cog_u32(aotx_live.prefixes[row]);
    if (aotx_live_bindings[slot].auto_retain != 2) return;
    const aotx_intake_row *r = aotx_intake.rows + row;
    unsigned char *p = aotx_live.choices + 64 + row * AOTX_LIVE_INTAKE_ROW + AOTX_LIVE_AUTO_ROW;
    aotx_cog_put(p, 1, 4); aotx_cog_put(p + 4, r->bytes, 4); aotx_cog_put(p + 72, r->count, 4);
    for (uint32_t j = 0; j < 32; ++j) { p[8 + j] = r->model[j]; p[40 + j] = aotx_intake_processor[j]; }
    for (uint32_t j = 0; j < r->bytes; ++j) p[AOTX_INTAKE_META + j] = r->reply[j];
}
__device__ inline uint32_t aotx_intake_prepare_rows(void) {
    uint64_t objects = 3 * aotx_live.auto_count, payload = 0;
    for (uint32_t i = 0; i < aotx_live.auto_count; ++i) payload += aotx_retain_payload_bytes(aotx_live.retain_rows[i]);
    for (uint32_t i = 0; i < aotx_live.count; ++i) {
        uint32_t slot = aotx_cog_u32(aotx_live.prefixes[i]);
        if (aotx_live_bindings[slot].auto_retain != 2) { aotx_intake.rows[i].count = 0; continue; }
        uint32_t status = aotx_intake_parse(i);
        if (status) return status;
        const aotx_intake_row *r = aotx_intake.rows + i;
        objects += r->count;
        for (uint32_t j = 0; j < r->count; ++j) payload += AOTX_INTAKE_PAYLOAD + r->items[j].length;
    }
    if (objects > AOTX_COG_OBJECTS - aotx_live_store.count || payload > AOTX_COG_PAYLOAD - aotx_live_store.bytes ||
        objects > UINT64_MAX - aotx_live_store.sequence) return AOTX_COG_CAPACITY;
    aotx_intake.objects = (uint32_t)objects; aotx_intake.payload = (uint32_t)payload;
    uint64_t next = aotx_live_store.sequence + 1;
    uint32_t attempts = 0, limit = aotx_live_store.count + aotx_live.count + aotx_intake.objects;
    for (uint32_t i = 0; i < aotx_live.count; ++i) for (uint32_t j = 0; j < aotx_intake.rows[i].count; ++j) {
        aotx_intake_item *item = aotx_intake.rows[i].items + j;
        if (item->target) {
            const unsigned char *target = aotx_live.results[i].selection + 16 + (item->target - 1) * 32;
            for (uint32_t a = 0; a < i; ++a) for (uint32_t b = 0; b < aotx_intake.rows[a].count; ++b) {
                uint32_t prior = aotx_intake.rows[a].items[b].target;
                if (prior && aotx_cog_equal(target, aotx_live.results[a].selection + 16 + (prior - 1) * 32, 24))
                    return AOTX_COG_REFERENCE;
            }
        }
        for (uint32_t k = 0; k < 8; ++k) item->id[k] = "AOTXGEN1"[k];
        bool used;
        do {
            if (!next || attempts++ >= limit) return AOTX_COG_CAPACITY;
            aotx_cog_put(item->id + 8, next++, 8);
            used = aotx_cog_latest(&aotx_live_store, item->id) >= 0;
            for (uint32_t k = 0; !used && k < aotx_live.count; ++k)
                used = aotx_cog_equal(item->id, aotx_live.requests + 64 + k * AOTX_RECALL_QUERY);
            for (uint32_t k = 0; !used && k < aotx_live.auto_count; ++k)
                used = aotx_cog_equal(item->id, aotx_live.retain_rows[k] + 48) ||
                    aotx_cog_equal(item->id, aotx_live.retain_rows[k] + 64);
        } while (used);
    }
    return AOTX_COG_OK;
}
__device__ inline void aotx_intake_encode(unsigned char *tail, uint32_t row) {
    if (row >= aotx_live.count || !aotx_intake.rows[row].count) return;
    uint32_t parent = 0, first = aotx_live.auto_count * 3, offset = 0;
    for (uint32_t j = 0; j < aotx_live.auto_count; ++j) {
        offset += aotx_retain_payload_bytes(aotx_live.retain_rows[j]);
        if (aotx_live.auto_rows[j] == row) parent = j;
    }
    for (uint32_t i = 0; i < row; ++i) {
        first += aotx_intake.rows[i].count;
        for (uint32_t j = 0; j < aotx_intake.rows[i].count; ++j)
            offset += AOTX_INTAKE_PAYLOAD + aotx_intake.rows[i].items[j].length;
    }
    const unsigned char *q = aotx_live.requests + 64 + row * AOTX_RECALL_QUERY, *in = aotx_live.retain_rows[parent];
    uint64_t source_version = aotx_live_store.pressure_percent ? aotx_live_store.sequence + parent * 3 + 1 : 1;
    for (uint32_t j = 0; j < aotx_intake.rows[row].count; ++j) {
        const aotx_intake_item *item = aotx_intake.rows[row].items + j;
        unsigned char *r = tail + AOTX_COG_HEADER + (first + j) * AOTX_COG_OBJECT;
        unsigned char *p = tail + AOTX_COG_HEADER + aotx_intake.objects * AOTX_COG_OBJECT + offset;
        uint64_t seq = aotx_live_store.sequence + first + j + 1;
        uint32_t bytes = AOTX_INTAKE_PAYLOAD + item->length;
        aotx_cog_put(r, 1, 2); aotx_cog_put(r + AOTX_CO_KIND, aotx_intake_kind(item->kind), 2);
        for (uint32_t k = 0; k < 16; ++k) {
            r[AOTX_CO_ID + k] = item->id[k]; r[AOTX_CO_LINEAGE + k] = aotx_live_store.lineage[k];
            r[AOTX_CO_OWNER + k] = q[16 + k]; r[AOTX_CO_ROOM + k] = q[32 + k];
            r[AOTX_CO_SOURCE + k] = in[32 + k]; r[AOTX_CO_EMBEDDING + k] = in[64 + k];
        }
        aotx_cog_put(r + AOTX_CO_VERSION, aotx_live_store.pressure_percent ? seq : 1, 8);
        aotx_cog_put(r + AOTX_CO_CREATED, seq, 8); aotx_cog_put(r + AOTX_CO_UPDATED, seq, 8);
        aotx_cog_put(r + AOTX_CO_SOURCE_VERSION, source_version, 8);
        aotx_cog_put(r + AOTX_CO_EMBED_VERSION, aotx_live_store.pressure_percent ? source_version + 1 : 1, 8);
        aotx_cog_put(r + AOTX_CO_OFFSET, offset, 8); aotx_cog_put(r + AOTX_CO_BYTES, bytes, 8);
        aotx_cog_put(r + AOTX_CO_SCOPE, aotx_cog_u32(q + 152), 4);
        aotx_cog_put(r + AOTX_CO_SOURCE_KIND, AOTX_COG_INFERRED, 4);
        aotx_cog_put(r + AOTX_CO_IMPORTANCE, AOTX_COG_UNKNOWN, 4); aotx_cog_put(r + AOTX_CO_POLICY, 1, 8);
        if (item->target) {
            const unsigned char *target = aotx_live.results[row].selection + 16 + (item->target - 1) * 32;
            for (uint32_t k = 0; k < 16; ++k) r[AOTX_CO_SUPERSEDES + k] = target[k];
            aotx_cog_put(r + AOTX_CO_SUPER_VERSION, aotx_cog_u64(target + 16), 8);
        }
        for (uint32_t k = 0; k < 8; ++k) p[k] = "AOTXMEM3"[k];
        aotx_cog_put(p + 8, 3, 4); aotx_cog_put(p + 12, item->length, 4);
        aotx_cog_put(p + 16, item->kind, 4); aotx_cog_put(p + 20, item->start, 4);
        for (uint32_t k = 0; k < 32; ++k) { p[24 + k] = aotx_intake.rows[row].model[k]; p[56 + k] = aotx_intake_processor[k]; }
        for (uint32_t k = 0; k < item->length; ++k) p[AOTX_INTAKE_PAYLOAD + k] = q[4640 + item->start + k];
        offset += bytes;
    }
}
__device__ inline uint32_t aotx_intake_context(uint32_t row) {
    aotx_recall_result *out = aotx_live.results + row;
    const unsigned char *q = aotx_live.requests + 64 + row * AOTX_RECALL_QUERY;
    uint32_t count = 0;
    for (uint32_t j = 0; j < out->count; ++j) {
        unsigned char *entry = out->selection + 16 + j * 32;
        if (aotx_cog_superseded(&aotx_live_candidate, aotx_live_candidate.objects[out->index[j]])) {
            if (aotx_recall_reason(q, entry) != AOTX_RECALL_SEMANTIC) return AOTX_COG_STALE;
            continue;
        }
        for (uint32_t k = 0; k < 32; ++k) out->selection[16 + count * 32 + k] = entry[k];
        ++count;
    }
    for (uint32_t j = 16 + count * 32; j < AOTX_RECALL_SELECTION; ++j) out->selection[j] = 0;
    aotx_cog_put(out->selection + 4, count, 4); out->count = count;
    return AOTX_COG_OK;
}
#endif
