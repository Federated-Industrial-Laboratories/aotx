/* Purpose: Verify audits with recorded recall defaults and pending appraisal rows.
 * Owns: Distinct source batches and malformed late queue or context bytes.
 * Threading: One disk reader at N=1 and N=64.
 * Lifetime: One transcript regression process. */
#ifndef AOTX_TEST_APPRAISAL_TRANSCRIPT_H
#define AOTX_TEST_APPRAISAL_TRANSCRIPT_H
#include "appraisal/format.h"

static unsigned appraisal_audit_batch(unsigned char *q, unsigned char *c, unsigned n,
                                     unsigned text, unsigned existing, unsigned mixed) {
    auto_batch(q, c, n, text, mixed);
    unsigned retained = mixed ? (n + 1) / 2 : n;
    unsigned objects = retained * 4, base = 128 + objects * 256;
    unsigned payload = retained * (3 + AOTX_APPRAISAL_QUEUE_BYTES), bytes = base + payload;
    unsigned char *tail = c + 64 + n * AOTX_LIVE_AUTO_ROW;
    put(c + 48, bytes, 8); put(tail + 20, objects, 4); put(tail + 24, payload, 8);
    put(tail + 72, base, 8); put(tail + 80, bytes, 8);
    unsigned queued = 0;
    for (unsigned i = 0; i < n; ++i) {
        unsigned char *raw = q + 128 + i * AOTX_LIVE_QUERY_ROW;
        unsigned char *prepared = c + 128 + i * AOTX_LIVE_AUTO_ROW;
        unsigned char *a = raw + AOTX_RECALL_EXTENSION, *b = prepared + AOTX_RECALL_EXTENSION;
        if (existing) {
            memcpy(a, "AOTXCTX1", 8); put(a + 8, 1, 4); put(a + 12, 1, 4);
            put(a + 16, 300 + i, 8); put(a + 44, 1, 4);
            memcpy(b, a, AOTX_RECALL_QUERY - AOTX_RECALL_EXTENSION);
        } else { memcpy(b, "AOTXCTX1", 8); put(b + 8, 1, 4); put(b + 44, 1, 4); }
        put(b + 12, existing ? 3 : 2, 4); put(b + 36, 100 + i, 4); put(b + 40, 900000 - i, 4);
        if (mixed && i % 2) continue;
        unsigned char *held = c + 64 + i * AOTX_LIVE_AUTO_ROW + AOTX_LIVE_TEXT_CHOICE_ROW;
        unsigned char *r = tail + 128 + (retained * 3 + queued) * 256;
        unsigned at = retained * 3 + queued * AOTX_APPRAISAL_QUEUE_BYTES;
        unsigned char *p = tail + base + at;
        put(r, 1, 2); put(r + 2, 11, 2); put(r + 8, 5000 + i, 8); memcpy(r + 24, c + 16, 16);
        put(r + 40, 1, 8); put(r + 48, 18 + retained * 3 + queued, 8); memcpy(r + 56, r + 48, 8);
        memcpy(r + 64, raw + 16, 32); memcpy(r + 96, held + 32, 16); put(r + 112, 1, 8);
        memcpy(r + 120, held + 112, 16); put(r + 160, at, 8); put(r + 168, 160, 8);
        memcpy(r + 176, raw + 152, 4); put(r + 180, 4, 4); put(r + 188, 2, 4);
        put(r + 192, UINT32_MAX, 4); put(r + 232, 1, 8);
        memcpy(p, "AOTXAPQ1", 8); put(p + 8, 1, 4); put(p + 16, 7000, 8); put(p + 32, 1, 8);
        memset(p + 64, 81, 32); ++queued;
    }
    return 64 + n * AOTX_LIVE_AUTO_ROW + bytes;
}
static void appraisal_audit_case(const char *root, unsigned n, unsigned text, unsigned existing, unsigned mode) {
    unsigned qb = 64 + n * AOTX_LIVE_QUERY_ROW;
    unsigned char *q = calloc(1, qb), *c = calloc(1, AOTX_LIVE_AUTO_BYTES);
    if (!q || !c) exit(2);
    unsigned cb = appraisal_audit_batch(q, c, n, text, existing, mode == 1);
    unsigned char *tail = c + 64 + n * AOTX_LIVE_AUTO_ROW;
    unsigned char *last = tail + 128 + (n * 4 - 1) * 256;
    unsigned char *p = c + cb - 160;
    unsigned char *context = c + 128 + (n - 1) * AOTX_LIVE_AUTO_ROW + AOTX_RECALL_EXTENSION;
    if (mode == 2) last[96] ^= 1;
    if (mode == 3) last[120] ^= 1;
    if (mode == 4) last[64] ^= 1;
    if (mode == 5) last[160] ^= 1;
    if (mode == 6) put(last + 168, 159, 8);
    if (mode == 7) p[12] = 1;
    if (mode == 8) p[96] = 1;
    if (mode == 9) put(tail + 20, n * 4 + 1, 4);
    if (mode == 10) context[16] ^= 1;
    if (mode == 11) put(context + 40, 1000001, 4);
    if (mode == 12) put(context + 12, existing ? 2 : 3, 4);
    if (mode == 13) context[48] ^= 1;
    if (mode == 14) {
        unsigned char *raw = q + 128 + (n - 1) * AOTX_LIVE_QUERY_ROW + AOTX_RECALL_EXTENSION;
        memcpy(raw, context, AOTX_RECALL_QUERY - AOTX_RECALL_EXTENSION); context[36] ^= 1;
    }
    if (mode == 16) {
        unsigned char *raw = q + 128 + (n - 1) * AOTX_LIVE_QUERY_ROW + AOTX_RECALL_EXTENSION;
        memcpy(raw, context, AOTX_RECALL_QUERY - AOTX_RECALL_EXTENSION);
        put(raw + 12, 8, 4); put(context + 12, 10, 4);
    }
    if (mode == 17) last[112] ^= 1;
    if (mode >= 18) {
        put(tail + 8, 2, 4); put(tail + 96, 17, 8); put(tail + 104, 17, 8); put(tail + 124, 80, 4);
        for (unsigned i = 0; i < n; ++i) {
            unsigned char *r = tail + 128 + (3 * n + i) * 256;
            put(r + 112, 18 + 3 * i, 8);
            put(c + 64 + i * AOTX_LIVE_AUTO_ROW + AOTX_LIVE_TEXT_CHOICE_ROW + 208, 20 + 3 * i, 8);
        }
        if (mode == 19) last[112] ^= 1;
    }
    char dir[768]; snprintf(dir, sizeof(dir), "%s/appraisal-%u-%u-%u-%u", root, n, text, existing, mode);
    CHECK(!mkdir(dir, 0700), "appraisal audit directory"); aotx_transcript *reader = NULL;
    CHECK(!aotx_transcript_open(&reader, dir), "appraisal audit reader"); if (!reader) exit(2);
    aotx_fake_device d = {0}; d.tick = 10; d.record_seq = 1; d.writer = AOTX_WRITER_FEEDER;
    transfer(&d, reader, NULL, text ? 6 : 4, 1, q, qb, 0);
    CHECK(!aotx_transcript_lines(reader), "appraisal input waits for the complete choice");
    d.writer = AOTX_WRITER_SYSTEM; transfer(&d, reader, NULL, 10, 1, c, cb, mode == 15);
    unsigned expected = mode < 2 || mode == 18 ? 2 * n : 0;
    CHECK(aotx_transcript_lines(reader) == expected,
        "appraisal audit remains atomic N=%u text=%u context=%u mode=%u", n, text, existing, mode);
    CHECK(!aotx_transcript_sync(reader), "appraisal audit sync"); aotx_transcript_close(reader);
    for (unsigned i = 0; i < n; ++i) {
        char out[8192]; int present = read_text(dir, i, out, sizeof(out));
        CHECK(present == !!expected, "invalid appraisal framing publishes no conversation input");
        if (present) CHECK(occurrences(out, "\"kind\":\"line\"") == 1 && occurrences(out, "\"kind\":\"selection\"") == 1,
            "each admitted source has exactly one input and one selection audit");
    }
    free(q); free(c);
}
#endif
