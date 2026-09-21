/* Purpose: Verify semantic audit framing and effective selections from journal bytes.
 * Owns: Distinct mixed modes, interpretation counts and late corruption cases.
 * Threading: One disk reader over N=1 and N=64 recorded batches.
 * Lifetime: One transcript test process. */
#ifndef AOTX_TEST_INTAKE_TRANSCRIPT_H
#define AOTX_TEST_INTAKE_TRANSCRIPT_H
#include "cognitive/source_profile.h"
static void intake_case(const char *root, unsigned n, unsigned text, unsigned mode, unsigned sources) {
    unsigned stride = sources ? AOTX_LIVE_INTAKE_SOURCE_ROW : AOTX_LIVE_INTAKE_ROW;
    unsigned qb = 64 + n * AOTX_LIVE_QUERY_ROW;
    unsigned char *q = calloc(1, qb), *old = calloc(1, AOTX_LIVE_AUTO_BYTES), *c = calloc(1, AOTX_LIVE_RESULTS);
    if (!q || !old || !c) exit(2);
    auto_batch(q, old, n, text, mode == 1);
    unsigned retained = mode == 1 ? (n + 1) / 2 : n, items = retained;
    unsigned base = 128 + (3 * retained + items) * 256, tb = base + 3 * retained;
    header(c, sources ? "AOTXICH2" : "AOTXICH1", n, stride); put(c + 32, 17, 8); put(c + 48, tb, 8);
    for (unsigned i = 0; i < n; ++i) {
        unsigned char *r = c + 64 + i * stride, *meta = r + AOTX_LIVE_AUTO_ROW;
        memcpy(r, old + 64 + i * AOTX_LIVE_AUTO_ROW, AOTX_LIVE_AUTO_ROW);
        unsigned char *selected = meta + AOTX_INTAKE_META + AOTX_INTAKE_REPLY;
        memcpy(selected, r + 64 + AOTX_RECALL_QUERY, AOTX_RECALL_SELECTION);
        put(selected + 4, 1, 4); memset(selected + 48, 0, AOTX_RECALL_SELECTION - 48);
        if (sources) {
            unsigned char *raw = q + 128 + i * AOTX_LIVE_QUERY_ROW;
            for (unsigned copy = 0; copy < 2; ++copy) {
                unsigned char *p = copy ? r + 64 : raw, *context = p + AOTX_RECALL_EXTENSION;
                memcpy(context, "AOTXCTX2", 8); put(context + 8, 2, 4); put(context + 44, 2, 4);
                p[AOTX_RECALL_ACTOR] = i + 5;
            }
            if (!(mode == 1 && i % 2)) memcpy(r + AOTX_LIVE_TEXT_CHOICE_ROW + 112, raw + AOTX_RECALL_ACTOR, 16);
        }
        if (mode == 1 && i % 2) continue;
        if (sources) {
            unsigned char *table = r + AOTX_LIVE_INTAKE_ROW;
            put(table, 1, 4); put(table + 4, 1, 4); table[16] = i + 1; table[31] = 7;
            put(table + 32, 8 + i, 8); put(table + 40, 1, 4);
        }
        char response[96]; int bytes = snprintf(response, sizeof(response), "[[3,\"input %u\",0]]", i);
        put(meta, sources ? 2 : 1, 4); put(meta + 4, bytes, 4); memset(meta + 8, i + 1, 32); memset(meta + 40, i + 81, 32);
        put(meta + 72, 1, 4); memcpy(meta + AOTX_INTAKE_META, response, bytes);
        if (sources) {
            unsigned char *first = r + AOTX_LIVE_INTAKE_FIRST;
            memcpy(first, meta, AOTX_INTAKE_META + AOTX_INTAKE_REPLY); put(first, 2, 4);
            static const unsigned char profile[32] = AOTX_SOURCE_PROFILE_DIGEST;
            put(first + 80, AOTX_SOURCE_PROFILE, 4); put(first + 84, 1, 4); memcpy(first + 88, profile, 32);
            int first_bytes = snprintf(response, sizeof(response), "[[\"input %u\",\"statement\"]]", i);
            put(first + 4, first_bytes, 4); memset(first + AOTX_INTAKE_META, 0, AOTX_INTAKE_REPLY);
            memcpy(first + AOTX_INTAKE_META, response, first_bytes);
            put(meta + 80, 1, 4);
        }
    }
    unsigned char *tail = c + 64 + n * stride;
    memcpy(tail, old + 64 + n * AOTX_LIVE_AUTO_ROW, 128);
    put(tail + 20, 3 * retained + items, 4); put(tail + 72, base, 8); put(tail + 80, tb, 8);
    unsigned cb = 64 + n * stride + tb;
    unsigned char *last = c + 64 + (n - 1) * stride, *meta = last + AOTX_LIVE_AUTO_ROW;
    if (mode == 2) { put(c + 8, 0, 4); put(c + 44, 7, 4); put(c + 48, 0, 8); cb = 64; }
    if (mode == 4) put(meta, sources ? 1 : 2, 4);
    if (mode == 5) put(meta + 4, AOTX_INTAKE_REPLY + 1, 4);
    if (mode == 6) memset(meta + 8, 0, 32);
    if (mode == 7) memset(meta + 40, 0, 32);
    if (mode == 8) meta[sources ? 84 : 76] = 1;
    if (mode == 9) meta[AOTX_INTAKE_META + AOTX_INTAKE_REPLY - 1] = 1;
    if (mode == 10) put(meta + 72, AOTX_INTAKE_ITEMS + 1, 4);
    if (mode == 11) put(tail + 20, 3 * retained, 4);
    if (mode == 12) put(meta + AOTX_INTAKE_META + AOTX_INTAKE_REPLY + 4, 17, 4);
    if (mode == 13) meta[AOTX_INTAKE_META + AOTX_INTAKE_REPLY + 48] = 1;
    if (mode == 14) last[64 + AOTX_RECALL_QUERY + 8] = 1;
    if (mode == 15) meta[AOTX_INTAKE_META + AOTX_INTAKE_REPLY + 16] ^= 1;
    if (mode == 16) last[AOTX_LIVE_INTAKE_ROW + 4] = 17;
    if (mode == 17) last[AOTX_LIVE_INTAKE_ROW + 48] = 1;
    if (mode == 18) last[64 + AOTX_RECALL_ACTOR] ^= 1;
    unsigned char *first = last + AOTX_LIVE_INTAKE_FIRST;
    if (mode == 19) put(first, 1, 4);
    if (mode == 20) put(first + 4, AOTX_INTAKE_REPLY + 1, 4);
    if (mode == 21) first[8] ^= 1;
    if (mode == 22) memset(first + 40, 0, 32);
    if (mode == 23) first[76] ^= 1;
    if (mode == 24) first[80] = 2;
    if (mode == 25) first[AOTX_INTAKE_FIRST - 1] = 1;
    if (mode == 26) put(meta + 80, 0, 4);
    if (mode == 27) put(first + 72, 2, 4);
    if (mode == 28) put(first + 84, 0, 4);
    if (mode == 29) first[88] ^= 1;
    if (mode == 30) first[120] = 1;
    char dir[768]; snprintf(dir, sizeof(dir), "%s/intake-%u-%u-%u-%u", root, n, text, mode, sources);
    CHECK(!mkdir(dir, 0700), "semantic audit directory"); aotx_transcript *reader = NULL;
    CHECK(!aotx_transcript_open(&reader, dir), "semantic audit reader"); if (!reader) exit(2);
    aotx_fake_device d = {0}; d.tick = 10; d.record_seq = 1; d.writer = AOTX_WRITER_FEEDER;
    transfer(&d, reader, NULL, text ? 6 : 4, 1, q, qb, 0);
    CHECK(!aotx_transcript_lines(reader), "semantic input waits for its complete decision");
    d.writer = AOTX_WRITER_SYSTEM; transfer(&d, reader, NULL, 14, 1, c, cb, mode == 3);
    unsigned expected = mode < 2 ? 2 * n : mode == 2 ? n : 0;
    CHECK(aotx_transcript_lines(reader) == expected, "semantic audit remains atomic N=%u mode=%u text=%u", n, mode, text);
    CHECK(!aotx_transcript_sync(reader), "semantic audit sync"); aotx_transcript_close(reader);
    for (unsigned i = 0; i < n; ++i) {
        char out[8192], marker[128]; int present = read_text(dir, i, out, sizeof(out));
        CHECK(present == !!expected, "bad semantic framing emits no accepted input");
        if (!present || mode == 2) continue;
        if (sources && !(mode == 1 && i % 2)) {
            snprintf(marker, sizeof(marker), "targets %02x000000000000000000000000000007@%u", i + 1, i + 8);
            CHECK(strstr(out, marker), "audit reports every explicit correction target");
        }
        snprintf(marker, sizeof(marker), "interpreted %u", mode == 1 && i % 2 ? 0 : 1);
        CHECK(strstr(out, marker), "audit states the admitted interpretation count");
        snprintf(marker, sizeof(marker), "%02x000000000000000000000000000001@", i + 151);
        CHECK(strstr(out, marker), "audit includes the effective selected object");
        snprintf(marker, sizeof(marker), "%02x000000000000000000000000000002@", i + 151);
        CHECK(!strstr(out, marker), "audit excludes a removed pre-write object");
    }
    free(q); free(old); free(c);
}
#endif
