/* Purpose: Derive typed request audit rows after the recorded choice is complete.
 * Owns: One bounded query batch and one bounded choice batch.
 * Threading: One disk reader; no recall or prompt construction.
 * Lifetime: Partial transfers are discarded when the journal reader closes. */
#include "disk/drain/transcript_live.h"
#include "cognitive/live.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define AOTX_AUDIT_QUERIES (AOTX_LIVE_HEADER + AOTX_RECALL_BATCH * AOTX_LIVE_QUERY_ROW)
struct aotx_transcript_live {
    unsigned char query[AOTX_AUDIT_QUERIES], choice[AOTX_LIVE_CHOICES];
    unsigned char query_id[16], active_id[16];
    uint32_t op, total, received, query_bytes;
    uint64_t query_tick;
};
static uint64_t audit_get(const unsigned char *p, unsigned n) {
    uint64_t value = 0;
    for (unsigned i = 0; i < n; ++i) value |= (uint64_t)p[i] << (i * 8);
    return value;
}
static int audit_zero(const unsigned char *p, uint32_t n) {
    for (uint32_t i = 0; i < n; ++i) if (p[i]) return 0;
    return 1;
}
static void audit_hex(char out[33], const unsigned char *p) {
    static const char digits[] = "0123456789abcdef";
    for (unsigned i = 0; i < 16; ++i) { out[2*i] = digits[p[i] >> 4]; out[2*i+1] = digits[p[i] & 15]; }
    out[32] = 0;
}
static int audit_header(const unsigned char *p, uint32_t bytes, const char *magic,
                        uint32_t row, int choice) {
    if (bytes < AOTX_LIVE_HEADER) return 0;
    uint32_t n = (uint32_t)audit_get(p + 8, 4);
    return n <= AOTX_RECALL_BATCH && (n || choice) && bytes == AOTX_LIVE_HEADER + n * row &&
        !memcmp(p, magic, 8) && audit_get(p + 12, 4) == 1 && audit_get(p + 40, 4) == row &&
        audit_zero(p + (choice ? 48 : 44), choice ? 16 : 20);
}
static int audit_queries(const aotx_transcript_live *s) {
    if (!audit_header(s->query, s->query_bytes, "AOTXLIV1", AOTX_LIVE_QUERY_ROW, 0)) return 0;
    uint32_t n = (uint32_t)audit_get(s->query + 8, 4);
    for (uint32_t i = 0; i < n; ++i) {
        const unsigned char *r = s->query + 64 + i * AOTX_LIVE_QUERY_ROW;
        if (audit_get(r, 4) >= 256 || !audit_zero(r + 4, 12) || !audit_zero(r + 40, 24) ||
            audit_get(r + 64 + 148, 4) > AOTX_RECALL_TEXT) return 0;
    }
    return 1;
}
static int audit_choices(const aotx_transcript_live *s) {
    if (!audit_header(s->choice, s->total, "AOTXCHO1", AOTX_LIVE_CHOICE_ROW, 1)) return 0;
    uint32_t status = (uint32_t)audit_get(s->choice + 44, 4);
    uint32_t n = (uint32_t)audit_get(s->choice + 8, 4);
    if (status) return status <= AOTX_COG_DENIED && !n;
    if (n != audit_get(s->query + 8, 4) || memcmp(s->query + 16, s->choice + 16, 24)) return 0;
    for (uint32_t i = 0; i < n; ++i) {
        const unsigned char *q = s->query + 64 + i * AOTX_LIVE_QUERY_ROW;
        const unsigned char *r = s->choice + 64 + i * AOTX_LIVE_CHOICE_ROW;
        const unsigned char *selection = r + 64;
        uint32_t count = (uint32_t)audit_get(selection + 4, 4);
        if (memcmp(q, r, 64) || audit_get(selection, 4) != 1 || count > AOTX_RECALL_LIMIT ||
            !audit_zero(selection + 8, 8) ||
            !audit_zero(selection + 16 + count * 32, (AOTX_RECALL_LIMIT - count) * 32)) return 0;
        for (uint32_t j = 0; j < count; ++j) {
            const unsigned char *entry = selection + 16 + j * 32;
            if (audit_get(entry + 24, 4) != 1 || !audit_zero(entry + 28, 4)) return 0;
        }
    }
    return 1;
}
static int audit_rows(aotx_transcript_live *s, aotx_live_audit_row emit, void *context) {
    if (!audit_queries(s) || !audit_choices(s)) return 0;
    uint32_t status = (uint32_t)audit_get(s->choice + 44, 4);
    uint32_t n = (uint32_t)audit_get(s->query + 8, 4);
    for (uint32_t i = 0; i < n; ++i) {
        const unsigned char *r = s->query + 64 + i * AOTX_LIVE_QUERY_ROW, *q = r + 64;
        char ids[6][33], text[2048];
        audit_hex(ids[0], s->query + 16); audit_hex(ids[1], r + 16);
        audit_hex(ids[2], q + 16); audit_hex(ids[3], q + 32);
        audit_hex(ids[4], q); audit_hex(ids[5], q + 48);
        int used = snprintf(text, sizeof(text),
            "lineage %s conversation %s principal %s room %s scope %u ordinal %llu "
            "request %s selection %s cut %llu status %u objects",
            ids[0], ids[1], ids[2], ids[3], (uint32_t)audit_get(q + 152, 4),
            (unsigned long long)audit_get(r + 32, 8), ids[4], ids[5],
            (unsigned long long)audit_get(s->query + 32, 8), status);
        if (used < 0 || (size_t)used >= sizeof(text)) return -1;
        if (!status) {
            const unsigned char *selection = s->choice + 64 + i * AOTX_LIVE_CHOICE_ROW + 64;
            uint32_t count = (uint32_t)audit_get(selection + 4, 4);
            for (uint32_t j = 0; j < count; ++j) {
                const unsigned char *entry = selection + 16 + j * 32;
                char id[33]; audit_hex(id, entry);
                int more = snprintf(text + used, sizeof(text) - (size_t)used, "%s%s@%llu",
                    j ? "," : " ", id, (unsigned long long)audit_get(entry + 16, 8));
                if (more < 0 || (size_t)more >= sizeof(text) - (size_t)used) return -1;
                used += more;
            }
        }
        if (emit(context, (uint32_t)audit_get(r, 4), s->query_tick, q + 4640,
            (uint32_t)audit_get(q + 148, 4), text, (uint32_t)used, status)) return -1;
    }
    return 0;
}
int aotx_transcript_live_take(aotx_transcript_live **state, const aotx_record_header *h,
                            aotx_live_audit_row emit, void *context) {
    if (!*state) { *state = calloc(1, sizeof(**state)); if (!*state) return -1; }
    aotx_transcript_live *s = *state;
    const unsigned char *p = aotx_record_body(h);
    if (h->cls != AOTX_CLASS_A || h->body_len < AOTX_LIVE_PART || h->body_len > AOTX_BODY_BYTES ||
        audit_get(p, 4) != AOTX_LIVE_SCHEMA || audit_zero(p + 8, 16)) goto discard;
    uint32_t op = (uint32_t)audit_get(p + 4, 4), total = (uint32_t)audit_get(p + 24, 4);
    uint32_t offset = (uint32_t)audit_get(p + 28, 4), n = h->body_len - AOTX_LIVE_PART;
    if (op != AOTX_LIVE_QUERY && op != AOTX_LIVE_CHOICE) goto discard;
    uint32_t cap = op == AOTX_LIVE_QUERY ? AOTX_AUDIT_QUERIES : AOTX_LIVE_CHOICES;
    if (!total || total > cap || offset >= total || offset % AOTX_LIVE_DATA ||
        n != (total - offset < AOTX_LIVE_DATA ? total - offset : AOTX_LIVE_DATA)) goto discard;
    if (!offset) {
        if (op == AOTX_LIVE_QUERY) {
            s->query_bytes = 0; memcpy(s->query_id, p + 8, 16); s->query_tick = h->tick;
        } else if (!s->query_bytes || memcmp(s->query_id, p + 8, 16)) goto discard;
        s->op = op; s->total = total; s->received = 0; memcpy(s->active_id, p + 8, 16);
    }
    if (s->op != op || s->total != total || s->received != offset ||
        memcmp(s->active_id, p + 8, 16)) goto discard;
    memcpy((op == AOTX_LIVE_QUERY ? s->query : s->choice) + offset, p + AOTX_LIVE_PART, n);
    s->received += n;
    if (s->received == total) {
        if (op == AOTX_LIVE_QUERY) s->query_bytes = total;
        else {
            int result = audit_rows(s, emit, context);
            s->query_bytes = 0; s->op = 0;
            return result;
        }
        s->op = 0;
    }
    return 0;
discard:
    s->op = 0; s->received = 0; s->query_bytes = 0;
    return 0;
}
void aotx_transcript_live_close(aotx_transcript_live *state) { free(state); }
