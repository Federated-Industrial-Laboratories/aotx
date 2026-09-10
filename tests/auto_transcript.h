/* Purpose: Verify atomic audit output from combined automatic memory decisions.
 * Owns: Distinct mixed-mode wire rows and damaged late result parts.
 * Threading: One disk reader; batches of 1 and 64 input rows.
 * Lifetime: One transcript test process. */
#ifndef AOTX_TEST_AUTO_TRANSCRIPT_H
#define AOTX_TEST_AUTO_TRANSCRIPT_H

static uint32_t auto_batch(unsigned char *q, unsigned char *c, uint32_t n, unsigned text, unsigned mixed) {
    unsigned char *old = calloc(1, AOTX_LIVE_TEXT_CHOICES);
    if (!old) exit(2);
    batch(q, old, n, text); put(q + 32, 17, 8);
    unsigned retained = mixed ? (n + 1) / 2 : n;
    uint32_t base = 128 + retained * 3 * 256, tail_bytes = base + retained * 3;
    header(c, "AOTXACH1", n, AOTX_LIVE_AUTO_ROW); put(c + 32, 17, 8); put(c + 48, tail_bytes, 8);
    for (unsigned i = 0; i < n; ++i) {
        unsigned char *raw = q + 64 + i * AOTX_LIVE_QUERY_ROW;
        const unsigned char *prior = old + 64 + i * (text ? AOTX_LIVE_TEXT_CHOICE_ROW : AOTX_LIVE_CHOICE_ROW);
        unsigned char *r = c + 64 + i * AOTX_LIVE_AUTO_ROW, *a = r + AOTX_LIVE_TEXT_CHOICE_ROW;
        memcpy(r, raw, 64); memcpy(r + 64, text ? prior + 64 : raw + 64, AOTX_RECALL_QUERY);
        memcpy(r + 64 + AOTX_RECALL_QUERY, prior + 64 + (text ? AOTX_RECALL_QUERY : 0), AOTX_RECALL_SELECTION);
        if (mixed && i % 2) continue;
        put(a, i, 4); put(a + 4, 1, 4); memcpy(a + 8, raw + 16, 16); memcpy(a + 24, raw + 32, 8);
        memcpy(a + 32, raw + 64, 16); put(a + 48, 1000 + i, 8); put(a + 64, 2000 + i, 8);
        put(a + 104, 1, 8); memcpy(a + 112, raw + 80, 16); put(a + 128, UINT32_MAX, 4);
        put(a + 160, 1, 4); memcpy(a + 192, a + 48, 16); put(a + 208, 1, 8);
    }
    unsigned char *tail = c + 64 + n * AOTX_LIVE_AUTO_ROW;
    memcpy(tail, "AOTXLOG1", 8); put(tail + 8, 1, 4); put(tail + 12, 128, 4);
    put(tail + 16, 256, 4); put(tail + 20, retained * 3, 4); put(tail + 24, retained * 3, 8);
    put(tail + 32, 18, 8); put(tail + 40, 6, 8); memcpy(tail + 48, c + 16, 16);
    put(tail + 64, 128, 8); put(tail + 72, base, 8); put(tail + 80, tail_bytes, 8); put(tail + 88, 1, 4);
    free(old); return 64 + n * AOTX_LIVE_AUTO_ROW + tail_bytes;
}
static void auto_case(const char *root, unsigned n, unsigned text, unsigned mode) {
    uint32_t qb = 64 + n * AOTX_LIVE_QUERY_ROW;
    unsigned char *q = calloc(1, qb), *c = calloc(1, AOTX_LIVE_AUTO_BYTES);
    if (!q || !c) exit(2);
    uint32_t cb = auto_batch(q, c, n, text, mode == 1 || mode == 13);
    unsigned char *last = c + 64 + (n - 1) * AOTX_LIVE_AUTO_ROW;
    if (mode == 2) { put(c + 8, 0, 4); put(c + 44, 2, 4); put(c + 48, 0, 8); cb = 64; }
    if (mode == 4) last[16] ^= 1;
    if (mode == 5) last[64 + 4640] ^= 1;
    if (mode == 6) last[AOTX_LIVE_TEXT_CHOICE_ROW + 48] ^= 1;
    if (mode == 7) last[AOTX_LIVE_TEXT_CHOICE_ROW + 192] ^= 1;
    if (mode == 8) c[64 + n * AOTX_LIVE_AUTO_ROW + 72] ^= 1;
    if (mode == 9) put(c + 48, 0, 8);
    if (mode == 10) memset(last + AOTX_LIVE_TEXT_CHOICE_ROW, 0, AOTX_LIVE_RETAINED_ROW);
    if (mode >= 12) {
        unsigned char *tail = c + 64 + n * AOTX_LIVE_AUTO_ROW;
        put(tail + 8, 2, 4); put(tail + 96, 17, 8); put(tail + 104, 17, 8); put(tail + 124, 80, 4);
        unsigned retained = 0;
        for (unsigned i = 0; i < n; ++i) {
            if (mode == 13 && i % 2) continue;
            put(c + 64 + i * AOTX_LIVE_AUTO_ROW + AOTX_LIVE_TEXT_CHOICE_ROW + 208, 20 + 3 * retained++, 8);
        }
        if (mode == 14) put(tail + 104, 18, 8);
        if (mode == 15) put(last + AOTX_LIVE_TEXT_CHOICE_ROW + 208, 1, 8);
    }
    char dir[768]; snprintf(dir, sizeof(dir), "%s/auto-%u-%u-%u", root, n, text, mode);
    CHECK(mkdir(dir, 0700) == 0, "automatic audit directory"); aotx_transcript *reader = NULL;
    CHECK(aotx_transcript_open(&reader, dir) == 0, "automatic audit reader"); if (!reader) exit(2);
    aotx_fake_device d; memset(&d, 0, sizeof(d)); d.tick = 10; d.record_seq = 1; d.writer = AOTX_WRITER_FEEDER;
    transfer(&d, reader, NULL, text ? 6 : 4, 1, q, qb, 0);
    CHECK(!aotx_transcript_lines(reader), "automatic input waits for the combined decision");
    d.writer = AOTX_WRITER_SYSTEM;
    transfer(&d, reader, NULL, 10, mode == 11 ? 2 : 1, c, cb, mode == 3);
    uint64_t expected = (mode < 2 || mode == 12 || mode == 13) ? 2 * n : mode == 2 ? n : 0;
    CHECK(aotx_transcript_lines(reader) == expected, "automatic audit is atomic for all rows N=%u mode=%u text=%u", n, mode, text);
    CHECK(!aotx_transcript_sync(reader), "automatic audit sync"); aotx_transcript_close(reader);
    for (unsigned i = 0; i < n; ++i) {
        char out[8192], expected_text[160]; int present = read_text(dir, i, out, sizeof(out));
        CHECK(present == !!expected, "invalid automatic result emits no accepted input");
        if (!present) continue;
        snprintf(expected_text, sizeof(expected_text), "request %02x000000000000000000000000000000", i + 1);
        CHECK(strstr(out, expected_text), "automatic audit retains the original request ID");
        snprintf(expected_text, sizeof(expected_text), "principal %02x000000000000000000000000000000", i + 80);
        CHECK(strstr(out, expected_text), "automatic audit retains the original principal");
        if (mode == 2) {
            CHECK(strstr(out, "status 2 objects") && !strstr(out, "\"kind\":\"line\""), "automatic refusal has no accepted input");
        } else {
            snprintf(expected_text, sizeof(expected_text), "retained %02x%02x0000000000000000000000000000@%u", (1000 + i) & 255, (1000 + i) >> 8,
                mode >= 12 ? 20 + 3 * (mode == 13 ? i / 2 : i) : 1);
            CHECK((mode == 1 || mode == 13) && i % 2 ? !strstr(out, "retained ") : strstr(out, expected_text) != NULL,
                "only automatic rows report their own retained ID");
            snprintf(expected_text, sizeof(expected_text), "input %u\\n\\\"byte\\\" \\\\ end", i);
            CHECK(strstr(out, expected_text), "automatic audit retains exact escaped input");
            CHECK(occurrences(out, "\"kind\":\"line\"") == 1 && occurrences(out, "\"kind\":\"selection\"") == 1,
                "automatic retention adds no extra conversation input");
        }
    }
    free(q); free(c);
}
#endif
