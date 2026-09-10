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
    unsigned char query[AOTX_AUDIT_QUERIES], choice[AOTX_LIVE_AUTO_BYTES];
    unsigned char query_id[16], active_id[16];
    uint32_t op, total, received, query_bytes, query_op;
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
    int automatic = choice && !strcmp(magic, "AOTXACH1");
    uint64_t tail = automatic ? audit_get(p + 48, 8) : 0;
    return n <= AOTX_RECALL_BATCH && (n || choice) && tail <= AOTX_COG_IMAGE &&
        bytes == AOTX_LIVE_HEADER + (uint64_t)n * row + tail &&
        !memcmp(p, magic, 8) && audit_get(p + 12, 4) == 1 && audit_get(p + 40, 4) == row &&
        audit_zero(p + (automatic ? 56 : choice ? 48 : 44), automatic ? 8 : choice ? 16 : 20);
}
static int audit_queries(const aotx_transcript_live *s) {
    const char *magic = s->query_op == AOTX_LIVE_TEXT ? "AOTXTXT1" : "AOTXLIV1";
    if (!audit_header(s->query, s->query_bytes, magic, AOTX_LIVE_QUERY_ROW, 0)) return 0;
    uint32_t n = (uint32_t)audit_get(s->query + 8, 4);
    for (uint32_t i = 0; i < n; ++i) {
        const unsigned char *r = s->query + 64 + i * AOTX_LIVE_QUERY_ROW;
        if (audit_get(r, 4) >= 256 || audit_get(r + 4, 4) > 1 || !audit_zero(r + 8, 8) || !audit_zero(r + 40, 24) ||
            audit_get(r + 64 + 148, 4) > AOTX_RECALL_TEXT) return 0;
    }
    return 1;
}
static uint32_t audit_choice_row(const aotx_transcript_live *s) {
    return s->op == AOTX_LIVE_AUTO_CHOICE ? AOTX_LIVE_AUTO_ROW :
        s->query_op == AOTX_LIVE_TEXT ? AOTX_LIVE_TEXT_CHOICE_ROW : AOTX_LIVE_CHOICE_ROW;
}
static uint32_t audit_selection_offset(const aotx_transcript_live *s) {
    return s->op == AOTX_LIVE_AUTO_CHOICE || s->query_op == AOTX_LIVE_TEXT ? 64 + AOTX_RECALL_QUERY : 64;
}
static int audit_focus(const unsigned char *raw, const unsigned char *prepared, unsigned flags) {
    uint32_t before = (uint32_t)audit_get(raw + 144, 4), after = (uint32_t)audit_get(prepared + 144, 4);
    if (!flags) return before == after && !memcmp(raw + 4448, prepared + 4448, 192);
    if (before > AOTX_RECALL_PINS || after < before || after > AOTX_RECALL_PINS ||
        memcmp(raw + 4448, prepared + 4448, before * 24) ||
        !audit_zero(raw + 4448 + before * 24, (AOTX_RECALL_PINS - before) * 24) ||
        !audit_zero(prepared + 4448 + after * 24, (AOTX_RECALL_PINS - after) * 24)) return 0;
    for (uint32_t i = before; i < after; ++i) {
        const unsigned char *entry = prepared + 4448 + i * 24;
        if (audit_zero(entry, 16) || !audit_get(entry + 16, 8)) return 0;
        for (uint32_t j = 0; j < i; ++j)
            if (!memcmp(entry, prepared + 4448 + j * 24, 24)) return 0;
    }
    return 1;
}
static int audit_prepared(const unsigned char *raw, const unsigned char *prepared, unsigned flags) {
    uint32_t width = (uint32_t)audit_get(prepared + 128, 4);
    return width && width <= AOTX_RECALL_WIDTH && !audit_zero(prepared + 64, 32) &&
        !audit_zero(prepared + 96, 32) && audit_get(raw + 148, 4) <= AOTX_LIVE_TEXT_BYTES &&
        audit_zero(prepared + 160 + width * 4, (AOTX_RECALL_WIDTH - width) * 4) &&
        audit_zero(raw + 64, 68) && audit_zero(raw + 160, 4096) &&
        !memcmp(raw, prepared, 64) && !memcmp(raw + 132, prepared + 132, 12) &&
        !memcmp(raw + 148, prepared + 148, 12) && !memcmp(raw + 4256, prepared + 4256, 192) &&
        audit_focus(raw, prepared, flags) && !memcmp(raw + 4640, prepared + 4640, AOTX_RECALL_QUERY - 4640);
}
static int audit_retained(const unsigned char *raw, const unsigned char *row) {
    const unsigned char *r = row + AOTX_LIVE_TEXT_CHOICE_ROW;
    if (audit_zero(r, AOTX_LIVE_RETAINED_ROW)) return 1;
    uint32_t focus = (uint32_t)audit_get(r + 160, 4);
    return audit_get(r, 4) == audit_get(raw, 4) && audit_get(r + 4, 4) == 1 &&
        !memcmp(r + 8, raw + 16, 16) && audit_get(r + 24, 8) == audit_get(raw + 32, 8) &&
        !memcmp(r + 32, raw + 64, 16) && !audit_zero(r + 48, 16) && !audit_zero(r + 64, 16) &&
        audit_zero(r + 80, 24) && audit_get(r + 104, 8) == 1 && !memcmp(r + 112, raw + 80, 16) &&
        audit_get(r + 128, 4) == UINT32_MAX && audit_zero(r + 132, 28) &&
        focus && focus <= AOTX_RECALL_PINS && audit_zero(r + 164, 28) &&
        !memcmp(r + 192 + (focus - 1) * 24, r + 48, 16) && audit_get(r + 208 + (focus - 1) * 24, 8) == 1 &&
        audit_zero(r + 192 + focus * 24, (AOTX_RECALL_PINS - focus) * 24);
}
static int audit_auto_tail(const aotx_transcript_live *s, uint32_t count) {
    const unsigned char *p = s->choice;
    uint32_t retained = 0;
    for (uint32_t i = 0; i < count; ++i)
        retained += !audit_zero(p + 64 + i * AOTX_LIVE_AUTO_ROW + AOTX_LIVE_TEXT_CHOICE_ROW, AOTX_LIVE_RETAINED_ROW);
    uint64_t bytes = audit_get(p + 48, 8), base = 128 + retained * 3 * 256;
    if (!retained || bytes < base) return 0;
    const unsigned char *tail = p + 64 + count * AOTX_LIVE_AUTO_ROW;
    uint64_t payload = audit_get(tail + 24, 8);
    return !memcmp(tail, "AOTXLOG1", 8) && audit_get(tail + 8, 4) == 1 &&
        audit_get(tail + 12, 4) == 128 && audit_get(tail + 16, 4) == 256 &&
        audit_get(tail + 20, 4) == retained * 3 && payload <= AOTX_COG_PAYLOAD && bytes == base + payload &&
        audit_get(p + 32, 8) != UINT64_MAX && audit_get(tail + 32, 8) == audit_get(p + 32, 8) + 1 &&
        !memcmp(tail + 48, p + 16, 16) && audit_get(tail + 64, 8) == 128 &&
        audit_get(tail + 72, 8) == base && audit_get(tail + 80, 8) == bytes &&
        audit_get(tail + 88, 4) == 1 && audit_zero(tail + 92, 36);
}
static int audit_choices(const aotx_transcript_live *s) {
    int automatic = s->op == AOTX_LIVE_AUTO_CHOICE;
    const char *magic = automatic ? "AOTXACH1" : s->query_op == AOTX_LIVE_TEXT ? "AOTXTCH1" : "AOTXCHO1";
    uint32_t row_bytes = audit_choice_row(s);
    if (!audit_header(s->choice, s->total, magic, row_bytes, 1)) return 0;
    uint32_t status = (uint32_t)audit_get(s->choice + 44, 4);
    uint32_t n = (uint32_t)audit_get(s->choice + 8, 4);
    if (status) return status <= AOTX_COG_DENIED && !n && (!automatic || !audit_get(s->choice + 48, 8));
    if (n != audit_get(s->query + 8, 4) || memcmp(s->query + 16, s->choice + 16, 24)) return 0;
    if (automatic && !audit_auto_tail(s, n)) return 0;
    for (uint32_t i = 0; i < n; ++i) {
        const unsigned char *q = s->query + 64 + i * AOTX_LIVE_QUERY_ROW;
        const unsigned char *r = s->choice + 64 + i * row_bytes;
        const unsigned char *selection = r + audit_selection_offset(s);
        if (s->query_op == AOTX_LIVE_TEXT && !audit_prepared(q + 64, r + 64, (unsigned)audit_get(q + 4, 4))) return 0;
        if (automatic && s->query_op == AOTX_LIVE_QUERY) {
            const unsigned char *raw = q + 64, *prepared = r + 64;
            if (memcmp(raw, prepared, 144) || memcmp(raw + 148, prepared + 148, 4300) ||
                !audit_focus(raw, prepared, (unsigned)audit_get(q + 4, 4)) ||
                memcmp(raw + 4640, prepared + 4640, AOTX_RECALL_QUERY - 4640)) return 0;
        }
        if (automatic && !audit_retained(q, r)) return 0;
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
            "request %s selection %s cut %llu status %u",
            ids[0], ids[1], ids[2], ids[3], (uint32_t)audit_get(q + 152, 4),
            (unsigned long long)audit_get(r + 32, 8), ids[4], ids[5],
            (unsigned long long)audit_get(s->query + 32, 8), status);
        if (used < 0 || (size_t)used >= sizeof(text)) return -1;
        if (!status && s->op == AOTX_LIVE_AUTO_CHOICE) {
            const unsigned char *retained = s->choice + 64 + i * AOTX_LIVE_AUTO_ROW + AOTX_LIVE_TEXT_CHOICE_ROW;
            if (!audit_zero(retained, AOTX_LIVE_RETAINED_ROW)) {
                char id[33]; audit_hex(id, retained + 48);
                int more = snprintf(text + used, sizeof(text) - (size_t)used, " retained %s@1", id);
                if (more < 0 || (size_t)more >= sizeof(text) - (size_t)used) return -1;
                used += more;
            }
        }
        int more = snprintf(text + used, sizeof(text) - (size_t)used, " objects");
        if (more < 0 || (size_t)more >= sizeof(text) - (size_t)used) return -1;
        used += more;
        if (!status) {
            const unsigned char *selection = s->choice + 64 + i * audit_choice_row(s) + audit_selection_offset(s);
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
    int request = op == AOTX_LIVE_QUERY || op == AOTX_LIVE_TEXT;
    if (!request && op != AOTX_LIVE_CHOICE && op != AOTX_LIVE_TEXT_CHOICE && op != AOTX_LIVE_AUTO_CHOICE) goto discard;
    uint32_t cap = request ? AOTX_AUDIT_QUERIES : op == AOTX_LIVE_AUTO_CHOICE ? AOTX_LIVE_AUTO_BYTES : op == AOTX_LIVE_TEXT_CHOICE ? AOTX_LIVE_TEXT_CHOICES : AOTX_LIVE_CHOICES;
    if (!total || total > cap || offset >= total || offset % AOTX_LIVE_DATA ||
        n != (total - offset < AOTX_LIVE_DATA ? total - offset : AOTX_LIVE_DATA)) goto discard;
    if (!offset) {
        if (request) {
            s->query_op = op;
            s->query_bytes = 0; memcpy(s->query_id, p + 8, 16); s->query_tick = h->tick;
        } else if (!s->query_bytes || memcmp(s->query_id, p + 8, 16) ||
                   (op != AOTX_LIVE_AUTO_CHOICE && op != (s->query_op == AOTX_LIVE_TEXT ? AOTX_LIVE_TEXT_CHOICE : AOTX_LIVE_CHOICE))) goto discard;
        s->op = op; s->total = total; s->received = 0; memcpy(s->active_id, p + 8, 16);
    }
    if (s->op != op || s->total != total || s->received != offset ||
        memcmp(s->active_id, p + 8, 16)) goto discard;
    memcpy((request ? s->query : s->choice) + offset, p + AOTX_LIVE_PART, n);
    s->received += n;
    if (s->received == total) {
        if (request) s->query_bytes = total;
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
