/* Purpose: Parse exact source quotations from a bounded model response.
 * Owns: JSON checks, unique source spans and current correction targets.
 * Launch shape: One parsing thread for each source in a batch.
 * Lifetime: Staged interpretation and its recorded replay. */
#ifndef AOTX_COGNITIVE_INTAKE_PARSE_CUH
#define AOTX_COGNITIVE_INTAKE_PARSE_CUH
#include "cognitive/intake.cuh"
#include "cognitive/intake_schema.cuh"
#include "cognitive/recall_format.cuh"

typedef struct aotx_intake_reader {
    const unsigned char *p;
    uint32_t at, bytes;
} aotx_intake_reader;
__device__ inline void aotx_intake_space(aotx_intake_reader *r) {
    while (r->at < r->bytes && (r->p[r->at] == ' ' || r->p[r->at] == '\t' ||
        r->p[r->at] == '\n' || r->p[r->at] == '\r')) ++r->at;
}
__device__ inline bool aotx_intake_take(aotx_intake_reader *r, unsigned char c) {
    aotx_intake_space(r);
    if (r->at == r->bytes || r->p[r->at] != c) return false;
    ++r->at; return true;
}
__device__ inline bool aotx_intake_number(aotx_intake_reader *r, uint32_t *out) {
    aotx_intake_space(r);
    uint32_t start = r->at, value = 0;
    while (r->at < r->bytes && r->p[r->at] >= '0' && r->p[r->at] <= '9') {
        uint32_t digit = r->p[r->at++] - '0';
        if (value > (UINT32_MAX - digit) / 10) return false;
        value = value * 10 + digit;
    }
    if (r->at == start || (r->at > start + 1 && r->p[start] == '0')) return false;
    *out = value; return true;
}
__device__ inline bool aotx_intake_hex4(aotx_intake_reader *r, uint32_t *value) {
    if (r->bytes - r->at < 4) return false;
    *value = 0;
    for (uint32_t j = 0; j < 4; ++j) {
        uint32_t c = r->p[r->at++], digit;
        if (c >= '0' && c <= '9') digit = c - '0';
        else if (c >= 'a' && c <= 'f') digit = c - 'a' + 10;
        else if (c >= 'A' && c <= 'F') digit = c - 'A' + 10;
        else return false;
        *value = *value * 16 + digit;
    }
    return true;
}
__device__ inline bool aotx_intake_string(aotx_intake_reader *r, unsigned char *out, uint32_t *bytes) {
    if (!aotx_intake_take(r, '"')) return false;
    uint32_t n = 0;
    while (r->at < r->bytes) {
        uint32_t c = r->p[r->at++];
        if (c == '"') { *bytes = n; return n && aotx_recall_utf8(out, n); }
        if (c < 32) return false;
        if (c == '\\') {
            if (r->at == r->bytes) return false;
            c = r->p[r->at++];
            if (c == 'n') c = '\n';
            else if (c == 't') c = '\t';
            else if (c == 'u') {
                if (!aotx_intake_hex4(r, &c)) return false;
                if (c >= 0xd800 && c <= 0xdbff) {
                    if (r->bytes - r->at < 6 || r->p[r->at++] != '\\' || r->p[r->at++] != 'u') return false;
                    uint32_t low;
                    if (!aotx_intake_hex4(r, &low) || low < 0xdc00 || low > 0xdfff) return false;
                    c = 0x10000 + ((c - 0xd800) << 10) + low - 0xdc00;
                } else if (c >= 0xdc00 && c <= 0xdfff) return false;
                uint32_t count = c < 0x80 ? 1 : c < 0x800 ? 2 : c < 0x10000 ? 3 : 4;
                if (count > AOTX_RECALL_TEXT - n) return false;
                if (count == 1) out[n++] = (unsigned char)c;
                else {
                    out[n++] = (unsigned char)((count == 2 ? 0xc0 : count == 3 ? 0xe0 : 0xf0) | (c >> (6 * (count - 1))));
                    for (uint32_t j = count - 1; j; --j) out[n++] = (unsigned char)(0x80 | ((c >> (6 * (j - 1))) & 63));
                }
                continue;
            } else if (c != '"' && c != '\\' && c != '/') return false;
        }
        if (n == AOTX_RECALL_TEXT) return false;
        out[n++] = (unsigned char)c;
    }
    return false;
}
__device__ inline uint32_t aotx_intake_target(uint32_t row, uint32_t target) {
    const aotx_recall_result *selected = aotx_live.results + row;
    if (!target || target > selected->count) return AOTX_COG_REFERENCE;
    const unsigned char *entry = selected->selection + 16 + (target - 1) * 32;
    const unsigned char *q = aotx_live.requests + 64 + row * AOTX_RECALL_QUERY;
    /* The recorded pre-write selection already validates transitive access at this cut. */
    uint32_t index = selected->index[target - 1];
    if (index >= aotx_live_store.count) return AOTX_COG_REFERENCE;
    const unsigned char *old = aotx_live_store.objects[index];
    if (!aotx_cog_equal(entry, old + AOTX_CO_ID) ||
        aotx_cog_u64(entry + 16) != aotx_cog_u64(old + AOTX_CO_VERSION) ||
        aotx_cog_latest(&aotx_live_store, entry) != (int)index ||
        aotx_cog_superseded(&aotx_live_store, old)) return AOTX_COG_STALE;
    const unsigned char *p = aotx_live_store.payload + aotx_cog_u64(old + AOTX_CO_OFFSET);
    if (aotx_cog_cold(old)) return AOTX_COG_UNAVAILABLE;
    if (!aotx_intake_payload(p, aotx_cog_u64(old + AOTX_CO_BYTES)) ||
        aotx_cog_u32(p + 16) < AOTX_INTAKE_ASSERTION ||
        aotx_cog_u16(old + AOTX_CO_KIND) != AOTX_COG_ASSERTION ||
        aotx_cog_u32(old + AOTX_CO_SOURCE_KIND) != AOTX_COG_INFERRED ||
        aotx_cog_u32(old + AOTX_CO_FLAGS) & AOTX_COG_PROTECTED ||
        !aotx_cog_equal(old + AOTX_CO_OWNER, q + 16, 32) ||
        aotx_cog_u32(old + AOTX_CO_SCOPE) != aotx_cog_u32(q + 152)) return AOTX_COG_DENIED;
    return AOTX_COG_OK;
}
__device__ inline uint32_t aotx_intake_parse(uint32_t row) {
    aotx_intake_row *out = aotx_intake.rows + row;
    out->count = 0;
    if (!out->bytes || out->bytes > AOTX_INTAKE_REPLY) return AOTX_COG_FORMAT;
    aotx_intake_reader r = {out->reply, 0, out->bytes};
    const unsigned char *q = aotx_live.requests + 64 + row * AOTX_RECALL_QUERY, *source = q + 4640;
    uint32_t length = aotx_cog_u32(q + 148);
    if (!length || length > AOTX_RECALL_TEXT || !aotx_recall_utf8(source, length) || !aotx_intake_take(&r, '[')) return AOTX_COG_FORMAT;
    aotx_intake_space(&r);
    if (r.at < r.bytes && r.p[r.at] != ']') for (;;) {
        if (out->count == AOTX_INTAKE_ITEMS) return AOTX_COG_CAPACITY;
        aotx_intake_item *item = out->items + out->count;
        *item = {};
        if (!aotx_intake_take(&r, '[') || !aotx_intake_number(&r, &item->kind) ||
            !aotx_intake_take(&r, ',') || !aotx_intake_string(&r, out->quote, &item->length) ||
            !aotx_intake_take(&r, ',') || !aotx_intake_number(&r, &item->target) || !aotx_intake_take(&r, ']') ||
            !item->kind || item->kind > AOTX_INTAKE_CORRECTION || item->length > length ||
            (item->kind == AOTX_INTAKE_CORRECTION ? !item->target : item->target != 0)) return AOTX_COG_FORMAT;
        uint32_t found = 0;
        for (uint32_t at = 0; at <= length - item->length; ++at)
            if (aotx_cog_equal(source + at, out->quote, item->length)) { item->start = at; ++found; }
        if (found != 1) return AOTX_COG_SOURCE;
        if (item->target) {
            uint32_t status = aotx_intake_target(row, item->target);
            if (status) return status;
        }
        for (uint32_t j = 0; j < out->count; ++j) {
            const aotx_intake_item *old = out->items + j;
            if ((old->kind == item->kind && old->start == item->start && old->length == item->length) ||
                (item->target && old->target == item->target)) return AOTX_COG_REFERENCE;
        }
        ++out->count;
        aotx_intake_space(&r);
        if (r.at < r.bytes && r.p[r.at] == ',') { ++r.at; continue; }
        break;
    }
    if (!aotx_intake_take(&r, ']')) return AOTX_COG_FORMAT;
    aotx_intake_space(&r);
    return r.at == r.bytes ? AOTX_COG_OK : AOTX_COG_FORMAT;
}
#endif
