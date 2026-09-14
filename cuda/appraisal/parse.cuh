/* Purpose: Validate complete appraisal responses against exact admitted sources.
 * Owns: Independent JSON, numeric, quote and correction target admission.
 * Launch shape: One source row per parsing thread in the live batch.
 * Lifetime: Candidate output before recorded memory admission. */
#ifndef AOTX_APPRAISAL_PARSE_CUH
#define AOTX_APPRAISAL_PARSE_CUH
#include "appraisal/correction.cuh"
#include "cognitive/intake_parse.cuh"

__device__ inline const unsigned char *aotx_appraisal_source(uint32_t index, uint32_t *length) {
    if (index >= aotx_live_store.count) return 0;
    const unsigned char *r = aotx_live_store.objects[index];
    uint64_t at = aotx_cog_u64(r + AOTX_CO_OFFSET), bytes = aotx_cog_u64(r + AOTX_CO_BYTES);
    if (aotx_cog_u16(r + AOTX_CO_KIND) != AOTX_COG_EVENT ||
        aotx_cog_u32(r + AOTX_CO_SOURCE_KIND) == AOTX_COG_INFERRED || at > aotx_live_store.bytes ||
        bytes > aotx_live_store.bytes - at || bytes < 32) return 0;
    const unsigned char *p = aotx_live_store.payload + at;
    *length = aotx_cog_u32(p + 12);
    if (!aotx_cog_equal(p, (const unsigned char *)"AOTXMEM1", 8) || aotx_cog_u32(p + 8) != 1 ||
        !aotx_cog_zero(p + 16, 16) || !*length || *length > AOTX_RECALL_TEXT || bytes != 32ull + *length ||
        !aotx_recall_utf8(p + 32, *length)) return 0;
    return p + 32;
}
__device__ inline const unsigned char *aotx_appraisal_task_source(uint32_t row, uint32_t *length) {
    *length = 0;
    const aotx_appraisal_row *a = aotx_appraisal.rows + row;
    if (a->source >= aotx_live_store.count || a->task_source >= aotx_live_store.count || aotx_cog_zero(a->task, 16)) return 0;
    const unsigned char *r = aotx_live_store.objects[a->task_source], *event = aotx_live_store.objects[a->source];
    uint64_t at = aotx_cog_u64(r + AOTX_CO_OFFSET), bytes = aotx_cog_u64(r + AOTX_CO_BYTES);
    uint64_t expiry = aotx_cog_u64(r + AOTX_CO_EXPIRY);
    if (aotx_cog_u16(r + AOTX_CO_KIND) != AOTX_COG_CUE || aotx_cog_u32(r + AOTX_CO_SOURCE_KIND) != AOTX_COG_AUTHORED ||
        !aotx_cog_equal(r + AOTX_CO_ID, a->task) || !aotx_cog_equal(r + AOTX_CO_OWNER, event + AOTX_CO_OWNER, 32) ||
        aotx_cog_u32(r + AOTX_CO_SCOPE) != aotx_cog_u32(event + AOTX_CO_SCOPE) ||
        aotx_cog_u32(r + AOTX_CO_FLAGS) & AOTX_COG_TOMBSTONE || aotx_cog_u32(r + AOTX_CO_EVIDENCE) == 3 ||
        (expiry && expiry <= aotx_live_store.sequence) ||
        aotx_cog_latest(&aotx_live_store, r + AOTX_CO_ID) != (int)a->task_source || aotx_cog_superseded(&aotx_live_store, r) ||
        at > aotx_live_store.bytes || bytes > aotx_live_store.bytes - at || bytes < 32) return 0;
    const unsigned char *p = aotx_live_store.payload + at;
    uint32_t n = aotx_cog_u32(p + 12);
    if (!aotx_cog_equal(p, (const unsigned char *)"AOTXMEM1", 8) || aotx_cog_u32(p + 8) != 1 ||
        !aotx_cog_zero(p + 16, 16) || !n || n > AOTX_RECALL_TEXT || bytes != 32ull + n || !aotx_recall_utf8(p + 32, n)) return 0;
    *length = n; return p + 32;
}
__device__ inline uint32_t aotx_appraisal_target(uint32_t row, uint32_t target) {
    const aotx_appraisal_row *r = aotx_appraisal.rows + row;
    if (!target || target > r->prior_count || target > AOTX_RECALL_LIMIT ||
        r->source >= aotx_live_store.count) return AOTX_COG_REFERENCE;
    uint32_t index = r->prior[target - 1];
    if (index >= aotx_live_store.count) return AOTX_COG_REFERENCE;
    const unsigned char *old = aotx_live_store.objects[index], *source = aotx_live_store.objects[r->source];
    uint64_t at = aotx_cog_u64(old + AOTX_CO_OFFSET), bytes = aotx_cog_u64(old + AOTX_CO_BYTES);
    if (at > aotx_live_store.bytes || bytes > aotx_live_store.bytes - at || bytes != AOTX_APPRAISAL_ASSESS_BYTES ||
        aotx_cog_u16(old + AOTX_CO_KIND) != AOTX_COG_APPRAISAL || aotx_cog_u32(aotx_live_store.payload + at) != 2)
        return AOTX_COG_FORMAT;
    if (aotx_cog_u32(old + AOTX_CO_SOURCE_KIND) != AOTX_COG_INFERRED ||
        aotx_cog_u32(old + AOTX_CO_FLAGS) & (AOTX_COG_PROTECTED | AOTX_COG_TOMBSTONE) ||
        aotx_cog_u32(old + AOTX_CO_EVIDENCE) == 3 || aotx_cog_zero(source + AOTX_CO_SUBJECT, 16) ||
        !aotx_cog_equal(old + AOTX_CO_SUBJECT, source + AOTX_CO_SUBJECT) ||
        !aotx_cog_equal(old + AOTX_CO_OWNER, source + AOTX_CO_OWNER, 32) ||
        aotx_cog_u32(old + AOTX_CO_SCOPE) != aotx_cog_u32(source + AOTX_CO_SCOPE)) return AOTX_COG_DENIED;
    uint64_t expiry = aotx_cog_u64(old + AOTX_CO_EXPIRY);
    if ((expiry && expiry <= aotx_live_store.sequence) ||
        aotx_cog_latest(&aotx_live_store, old + AOTX_CO_ID) != (int)index ||
        aotx_cog_superseded(&aotx_live_store, old)) return AOTX_COG_STALE;
    int prior_source = aotx_cog_find(&aotx_live_store, old + AOTX_CO_SOURCE, aotx_cog_u64(old + AOTX_CO_SOURCE_VERSION));
    uint32_t length = 0;
    if (prior_source < 0 || !aotx_appraisal_source((uint32_t)prior_source, &length)) return AOTX_COG_SOURCE;
    const unsigned char *event = aotx_live_store.objects[prior_source];
    if (!aotx_cog_equal(event + AOTX_CO_SUBJECT, source + AOTX_CO_SUBJECT) ||
        !aotx_cog_equal(event + AOTX_CO_OWNER, source + AOTX_CO_OWNER, 32) ||
        aotx_cog_u32(event + AOTX_CO_SCOPE) != aotx_cog_u32(source + AOTX_CO_SCOPE)) return AOTX_COG_DENIED;
    int latest = aotx_cog_latest(&aotx_live_store, event + AOTX_CO_ID);
    if (latest < 0) return AOTX_COG_SOURCE;
    const unsigned char *current = aotx_live_store.objects[latest];
    expiry = aotx_cog_u64(current + AOTX_CO_EXPIRY);
    if (aotx_cog_u32(current + AOTX_CO_FLAGS) & AOTX_COG_TOMBSTONE ||
        aotx_cog_u32(current + AOTX_CO_EVIDENCE) == 3 || (expiry && expiry <= aotx_live_store.sequence) ||
        !aotx_cog_equal(current + AOTX_CO_OWNER, source + AOTX_CO_OWNER, 32) ||
        aotx_cog_u32(current + AOTX_CO_SCOPE) != aotx_cog_u32(source + AOTX_CO_SCOPE)) return AOTX_COG_DENIED;
    const unsigned char *p = aotx_live_store.payload + at;
    uint32_t start = aotx_cog_u32(p + 120), quote = aotx_cog_u32(p + 124);
    if (!quote || start > length || quote > length - start) return AOTX_COG_SOURCE;
    const unsigned char *text = aotx_live_store.payload + aotx_cog_u64(event + AOTX_CO_OFFSET) + 32;
    if (!aotx_recall_utf8(text + start, quote)) return AOTX_COG_SOURCE;
    uint32_t relation = aotx_appraisal_old_relation(index);
    if (relation != UINT32_MAX &&
        (aotx_cog_u32(aotx_live_store.objects[relation] + AOTX_CO_FLAGS) & AOTX_COG_PROTECTED)) return AOTX_COG_DENIED;
    return AOTX_COG_OK;
}
__device__ inline uint32_t aotx_appraisal_quote_parse(aotx_intake_reader *r, unsigned char *scratch,
    const unsigned char *source, uint32_t bytes, uint32_t *start, uint32_t *length) {
    aotx_intake_space(r); *start = *length = 0;
    if (r->bytes - r->at >= 2 && r->p[r->at] == '"' && r->p[r->at + 1] == '"') { r->at += 2; return AOTX_COG_OK; }
    if (!aotx_intake_string(r, scratch, length) || *length > bytes) return AOTX_COG_FORMAT;
    uint32_t found = 0;
    for (uint32_t at = 0; at <= bytes - *length; ++at)
        if (aotx_cog_equal(source + at, scratch, *length)) { *start = at; ++found; }
    return found == 1 ? AOTX_COG_OK : AOTX_COG_SOURCE;
}
__device__ inline bool aotx_appraisal_key_take(aotx_intake_reader *r, const char *key) {
    if (!aotx_intake_take(r, '"')) return false;
    for (uint32_t j = 0; key[j]; ++j)
        if (r->at == r->bytes || r->p[r->at++] != (unsigned char)key[j]) return false;
    if (r->at == r->bytes || r->p[r->at++] != '"') return false;
    return aotx_intake_take(r, ':');
}
__device__ inline uint32_t aotx_appraisal_parse_body(uint32_t row) {
    aotx_appraisal_row *out = aotx_appraisal.rows + row;
    aotx_intake_row *text = aotx_intake.rows + row;
    if (!text->bytes || text->bytes > AOTX_INTAKE_REPLY) return AOTX_COG_FORMAT;
    uint32_t bytes = 0;
    const unsigned char *source = aotx_appraisal_source(out->source, &bytes);
    const unsigned char *q = aotx_live.requests + 64 + row * AOTX_RECALL_QUERY;
    if (!source || bytes != aotx_cog_u32(q + 148) || !aotx_cog_equal(source, q + 4640, bytes) ||
        aotx_cog_zero(aotx_live_store.objects[out->source] + AOTX_CO_SUBJECT, 16)) return AOTX_COG_SOURCE;
    aotx_intake_reader r = {text->reply, 0, text->bytes};
    if (!aotx_intake_take(&r, '{')) return AOTX_COG_FORMAT;
    bool supported = false;
    const char *fields[AOTX_APPRAISAL_VALUES] = {"benefit", "harm", "arousal", "consequence", "confidence",
        "regard_gain", "regard_loss", "trust_gain", "trust_loss"};
    for (uint32_t j = 0; j < AOTX_APPRAISAL_VALUES; ++j) {
        if ((j && !aotx_intake_take(&r, ',')) || !aotx_appraisal_key_take(&r, fields[j]) ||
            !aotx_intake_number(&r, out->values + j)) return AOTX_COG_FORMAT;
        uint32_t value = out->values[j];
        if (j == 3 ? value > 4 : value > AOTX_COG_SCALE && value != AOTX_COG_UNKNOWN) return AOTX_COG_FORMAT;
        supported |= value != (j == 3 ? 0 : AOTX_COG_UNKNOWN);
    }
    if (!aotx_intake_take(&r, ',') || !aotx_appraisal_key_take(&r, "evidence")) return AOTX_COG_FORMAT;
    uint32_t status = aotx_appraisal_quote_parse(&r, text->quote, source, bytes, &out->quote_start, &out->quote_length);
    if (status) return status;
    if (supported != (out->quote_length != 0)) return AOTX_COG_SOURCE;
    if (!aotx_intake_take(&r, ',') || !aotx_appraisal_key_take(&r, "task")) return AOTX_COG_FORMAT;
    status = aotx_appraisal_quote_parse(&r, text->quote, source, bytes, &out->task_start, &out->task_length);
    if (status) return status;
    if (!aotx_intake_take(&r, ',') || !aotx_appraisal_key_take(&r, "commitment")) return AOTX_COG_FORMAT;
    status = aotx_appraisal_quote_parse(&r, text->quote, source, bytes, &out->commitment_start, &out->commitment_length);
    if (status) return status;
    if (!aotx_intake_take(&r, ',') || !aotx_appraisal_key_take(&r, "correction") ||
        !aotx_intake_number(&r, &out->correction) || !aotx_intake_take(&r, '}')) return AOTX_COG_FORMAT;
    aotx_intake_space(&r);
    if (r.at != r.bytes) return AOTX_COG_FORMAT;
    if (!out->quote_length && (out->task_length || out->commitment_length || out->correction)) return AOTX_COG_SOURCE;
    if (out->values[7] != AOTX_COG_UNKNOWN || out->values[8] != AOTX_COG_UNKNOWN) {
        uint32_t task_bytes = 0;
        const unsigned char *task = aotx_appraisal_task_source(row, &task_bytes);
        if (!task || out->task_length != task_bytes || !aotx_cog_equal(source + out->task_start, task, task_bytes)) return AOTX_COG_REFERENCE;
    }
    return out->correction ? aotx_appraisal_target(row, out->correction) : AOTX_COG_OK;
}
#endif
