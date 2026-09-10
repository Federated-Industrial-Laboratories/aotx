/* Purpose: Check typed audit rows from journal bytes without source files.
 * Owns: Bounded query and choice fixtures and private transcript directories.
 * Threading: One disk reader; batches of 1 and 64 distinct agent slots.
 * Lifetime: One test process; all temporary files are removed. */
#include "disk/drain/transcript.h"
#include "cognitive/live.h"
#include "tests/disk_fake.h"
#include <fcntl.h>
#include <inttypes.h>

static void put(unsigned char *p, uint64_t v, unsigned n) {
    for (unsigned i = 0; i < n; ++i) p[i] = (unsigned char)(v >> (8 * i));
}
static void header(unsigned char *p, const char *magic, uint32_t count, uint32_t row) {
    memcpy(p, magic, 8); put(p + 8, count, 4); put(p + 12, 1, 4);
    p[16] = 71; put(p + 32, UINT64_MAX, 8); put(p + 40, row, 4);
}
static void batch(unsigned char *query, unsigned char *choice, uint32_t count, unsigned text) {
    uint32_t row_bytes = text ? AOTX_LIVE_TEXT_CHOICE_ROW : AOTX_LIVE_CHOICE_ROW;
    header(query, text ? "AOTXTXT1" : "AOTXLIV1", count, AOTX_LIVE_QUERY_ROW);
    header(choice, text ? "AOTXTCH1" : "AOTXCHO1", count, row_bytes);
    for (uint32_t i = 0; i < count; ++i) {
        unsigned char *r = query + 64 + i * AOTX_LIVE_QUERY_ROW, *q = r + 64;
        unsigned char *c = choice + 64 + i * row_bytes, *s = c + 64 + (text ? AOTX_RECALL_QUERY : 0);
        put(r, i, 4); r[16] = (unsigned char)(20 + i); put(r + 32, UINT64_MAX - i, 8);
        q[0] = (unsigned char)(i + 1); q[16] = (unsigned char)(i + 80);
        q[32] = (unsigned char)(i + 90); q[48] = (unsigned char)(i + 101);
        put(q + 152, i % 3, 4);
        int n = snprintf((char *)q + 4640, AOTX_RECALL_TEXT, "input %u\n\"byte\" \\ end", i);
        put(q + 148, (uint32_t)n, 4);
        memcpy(c, r, 64); put(s, 1, 4); put(s + 4, 2, 4);
        if (text) {
            unsigned char *prepared = c + 64;
            memcpy(prepared, q, AOTX_RECALL_QUERY);
            memset(prepared + 64, (int)(i + 1), 32);
            memset(prepared + 96, (int)(i + 81), 32); put(prepared + 128, 3, 4);
            for (unsigned j = 0; j < 3; ++j) put(prepared + 160 + j * 4, 0x3f800000u + i * 256 + j, 4);
        }
        for (uint32_t j = 0; j < 2; ++j) {
            unsigned char *entry = s + 16 + j * 32;
            entry[0] = (unsigned char)(i + 151); entry[15] = (unsigned char)(j + 1);
            put(entry + 16, UINT64_MAX - i - j, 8); put(entry + 24, 1, 4);
        }
    }
}
static void flush(aotx_fake_device *d, aotx_transcript *a, aotx_transcript *b) {
    aotx_fake_commit(d, 0);
    CHECK(aotx_transcript_block(a, d->stage) == 0, "first journal reader");
    if (b) CHECK(aotx_transcript_block(b, d->stage) == 0, "second journal reader");
}
static void transfer(aotx_fake_device *d, aotx_transcript *a, aotx_transcript *b,
                     unsigned op, unsigned id, const unsigned char *p, uint32_t bytes,
                     uint32_t omit) {
    for (uint32_t at = 0; at < bytes; at += AOTX_LIVE_DATA) {
        unsigned char part[AOTX_BODY_BYTES] = {0};
        uint32_t n = bytes - at < AOTX_LIVE_DATA ? bytes - at : AOTX_LIVE_DATA;
        if (omit && at + n == bytes) break;
        put(part, 1, 4); put(part + 4, op, 4); part[8] = (unsigned char)id;
        put(part + 24, bytes, 4); put(part + 28, at, 4); memcpy(part + 32, p + at, n);
        aotx_fake_record(d, AOTX_CLASS_A, AOTX_LIVE_RECORD, part, 32 + n);
        if (d->count == 7) flush(d, a, b);
    }
    if (d->count) flush(d, a, b);
}
static int read_text(const char *dir, uint32_t agent, char *out, size_t cap) {
    char path[1024]; snprintf(path, sizeof(path), "%s/transcript/%u.jsonl", dir, agent);
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) { out[0] = 0; return 0; }
    ssize_t n = read(fd, out, cap - 1); close(fd);
    CHECK(n >= 0 && (size_t)n < cap - 1, "bounded transcript read");
    if (n < 0) n = 0;
    out[n] = 0; return 1;
}
static unsigned occurrences(const char *text, const char *word) {
    unsigned count = 0;
    while ((text = strstr(text, word)) != NULL) { ++count; text += strlen(word); }
    return count;
}
static void focus_case(unsigned char *query, unsigned char *choice, uint32_t count,
                       uint32_t row_bytes, unsigned mode) {
    for (uint32_t i = 0; i < count; ++i) {
        unsigned char *r = query + 64 + i * AOTX_LIVE_QUERY_ROW, *q = r + 64;
        unsigned char *c = choice + 64 + i * row_bytes, *p = c + 64;
        put(r + 4, mode == 18 ? 2 : mode == 21 ? 0 : 1, 4); memcpy(c, r, 64);
        if (mode < 20) continue;
        put(q + 144, 1, 4); put(p + 144, 2, 4);
        put(q + 4448, 301 + i, 8); put(q + 4464, UINT64_MAX - i, 8); memcpy(p + 4448, q + 4448, 24);
        put(p + 4472, 501 + i, 8); put(p + 4488, 7, 8);
        if (mode == 22) p[4448] ^= 1;
        if (mode == 23) put(p + 144, 0, 4);
        if (mode == 24) put(p + 144, 9, 4);
        if (mode == 25) memcpy(p + 4472, p + 4448, 24);
        if (mode == 26) memset(p + 4472, 0, 16);
        if (mode == 27) put(p + 4488, 0, 8);
        if (mode == 28) {
            put(p + 144, 8, 4);
            for (unsigned j = 1; j < 8; ++j) { put(p + 4448 + j * 24, 501 + i + j, 8); put(p + 4464 + j * 24, j, 8); }
        }
        if (mode == 29) p[4496] = 1;
        if (mode == 30) p[4256] ^= 1;
        if (mode == 31) p[4640] ^= 1;
        if (mode == 32) put(c + 4, 0, 4);
    }
}
static void run_case(const char *root, uint32_t count, unsigned mode, unsigned text) {
    uint32_t row_bytes = text ? AOTX_LIVE_TEXT_CHOICE_ROW : AOTX_LIVE_CHOICE_ROW;
    uint32_t query_op = text ? AOTX_LIVE_TEXT : AOTX_LIVE_QUERY;
    uint32_t choice_op = text ? AOTX_LIVE_TEXT_CHOICE : AOTX_LIVE_CHOICE;
    uint32_t qb = 64 + count * AOTX_LIVE_QUERY_ROW, cb = 64 + count * row_bytes;
    unsigned char *q = calloc(1, qb), *c = calloc(1, AOTX_LIVE_TEXT_CHOICES);
    CHECK(q && c, "batch buffers"); if (!q || !c) exit(2);
    batch(q, c, count, text);
    char first[768], second[768];
    snprintf(first, sizeof(first), "%s/a-%u-%u-%u", root, count, mode, text);
    snprintf(second, sizeof(second), "%s/b-%u-%u-%u", root, count, mode, text);
    CHECK(mkdir(first, 0700) == 0 && mkdir(second, 0700) == 0, "reader directories");
    aotx_transcript *a = NULL, *b = NULL;
    CHECK(aotx_transcript_open(&a, first) == 0 && aotx_transcript_open(&b, second) == 0, "open readers");
    if (!a || !b) exit(2);
    aotx_fake_device d; memset(&d, 0, sizeof(d)); d.tick = 10; d.record_seq = 1;
    d.writer = AOTX_WRITER_FEEDER;
    if (mode == 4) { put(c + 8, 0, 4); put(c + 44, 7, 4); cb = 64; }
    if (mode == 5) c[64 + (count - 1) * row_bytes + 16] ^= 1;
    if (mode == 6) put(q + 64 + (count - 1) * AOTX_LIVE_QUERY_ROW + 64 + 148, 2049, 4);
    if (mode == 7) { put(q + 8, 65, 4); put(c + 8, 0, 4); put(c + 44, 1, 4); cb = 64; }
    if (mode == 10) {
        unsigned char *other = calloc(1, qb); if (!other) exit(2);
        memset(c, 0, AOTX_LIVE_TEXT_CHOICES); batch(other, c, count, !text); free(other);
        choice_op = text ? AOTX_LIVE_CHOICE : AOTX_LIVE_TEXT_CHOICE;
        cb = 64 + count * (text ? AOTX_LIVE_CHOICE_ROW : AOTX_LIVE_TEXT_CHOICE_ROW);
    }
    if (mode == 11) put(c + 40, row_bytes + 1, 4);
    if (mode == 12) c[64 + (count - 1) * row_bytes + 64 + 4640] ^= 1;
    if (mode == 13) q[64 + (count - 1) * AOTX_LIVE_QUERY_ROW + 64 + 64] = 1;
    if (mode == 14) c[64 + (count - 1) * row_bytes + 64 + 4272] ^= 1;
    if (mode == 15) put(c + 64 + (count - 1) * row_bytes + 64 + 128, 1025, 4);
    if (mode == 16) q[64 + (count - 1) * AOTX_LIVE_QUERY_ROW + 64 + 160] = 1;
    if (mode >= 17) focus_case(q, c, count, row_bytes, mode);
    int ordinary = mode == 8 || mode == 9;
    if (ordinary) {
        for (uint32_t i = 0; i < count; ++i) {
            if (mode == 9) {
                aotx_manifest_body m = {0}; m.agent = i; m.turn = 10 + i;
                d.writer = AOTX_WRITER_AGENT_BASE + i;
                aotx_fake_record(&d, AOTX_CLASS_B, AOTX_REC_MANIFEST, &m, sizeof(m));
                flush(&d, a, b);
            }
            char line[80]; int n = snprintf(line, sizeof(line), "task %u refused ordinary input", i);
            d.writer = AOTX_WRITER_FEEDER;
            aotx_fake_record(&d, AOTX_CLASS_A, AOTX_REC_INPUT_LINE, line, (uint32_t)n);
            flush(&d, a, b);
        }
    }
    transfer(&d, a, b, query_op, 1, q, qb, mode == 1);
    uint64_t prior = ordinary ? count : 0;
    CHECK(aotx_transcript_lines(a) == prior && aotx_transcript_lines(b) == prior, "query waits for choice");
    d.writer = AOTX_WRITER_SYSTEM;
    transfer(&d, a, b, choice_op, mode == 3 ? 2 : 1, c, cb, mode == 2);
    if (mode == 19) {
        uint32_t retain_bytes = 64 + count * 160, result_bytes = 64 + count * 384 + 128;
        unsigned char *retain = calloc(1, retain_bytes), *result = calloc(1, result_bytes);
        if (!retain || !result) exit(2);
        header(retain, "AOTXRTN1", count, 160); header(result, "AOTXRCH1", count, 384);
        put(result + 48, 128, 8);
        transfer(&d, a, b, 8, 2, retain, retain_bytes, 0);
        transfer(&d, a, b, 9, 2, result, result_bytes, 0);
        free(retain); free(result);
    }
    int accepted = mode == 0 || mode == 17 || mode == 19 || mode == 20 || mode == 28;
    uint64_t expected = ordinary ? 3 * count : accepted ? 2 * count : mode == 4 ? count : 0;
    CHECK(aotx_transcript_lines(a) == expected && aotx_transcript_lines(b) == expected,
          "N=%u mode=%u complete verdict row count", count, mode);
    CHECK(aotx_transcript_sync(a) == 0 && aotx_transcript_sync(b) == 0, "sync audit files");
    aotx_transcript_close(a); aotx_transcript_close(b);
    for (uint32_t i = 0; i < count; ++i) {
        char x[8192], y[8192], expected_text[160];
        int present = read_text(first, i, x, sizeof(x));
        CHECK(read_text(second, i, y, sizeof(y)) == present && !strcmp(x, y), "exact journal derivation");
        CHECK(present == (expected != 0), "partial or invalid batch has no accepted file");
        if (!expected) continue;
        snprintf(expected_text, sizeof(expected_text), "ordinal %" PRIu64, UINT64_MAX - i);
        CHECK(strstr(x, expected_text) != NULL, "full request ordinal");
        snprintf(expected_text, sizeof(expected_text), "request %02x000000000000000000000000000000", i + 1);
        CHECK(strstr(x, expected_text) != NULL, "exact request ID");
        snprintf(expected_text, sizeof(expected_text), "scope %u", i % 3);
        CHECK(strstr(x, expected_text) != NULL, "scope is present");
        if (mode == 4) {
            CHECK(strstr(x, "\"status\":\"refused\"") && !strstr(x, "\"kind\":\"line\""), "refusal is not accepted input");
            CHECK(strstr(x, "status 7 objects") != NULL, "refusal code");
        } else {
            snprintf(expected_text, sizeof(expected_text), "input %u\\n\\\"byte\\\" \\\\ end", i);
            CHECK(strstr(x, expected_text) != NULL, "exact escaped input");
            for (unsigned j = 0; j < 2; ++j) {
                snprintf(expected_text, sizeof(expected_text), "%02x0000000000000000000000000000%02x@%" PRIu64,
                         i + 151, j + 1, UINT64_MAX - i - j);
                CHECK(strstr(x, expected_text) != NULL, "selected ID and full version");
            }
            CHECK(occurrences(x, "\"kind\":\"line\"") == (ordinary ? 2u : 1u) &&
                  occurrences(x, "\"kind\":\"selection\"") == 1, "one input and one choice");
            uint32_t turn = mode == 9 ? 11 + i : 1;
            snprintf(expected_text, sizeof(expected_text), "\"status\":\"accepted\",\"turn\":%u", turn);
            CHECK(strstr(x, expected_text) != NULL, "accepted input uses the device turn after refused ordinary input");
            snprintf(expected_text, sizeof(expected_text), "\"status\":\"selected\",\"turn\":%u", turn);
            CHECK(strstr(x, expected_text) != NULL, "selection uses the same device turn, not its ordinal");
        }
    }
    free(q); free(c);
}
#include "auto_transcript.h"
#include "intake_transcript.h"

int main(void) {
    char root[512]; CHECK(aotx_temp_dir(root, sizeof(root)) == 0, "temporary directory");
    for (uint32_t n = 1; n <= 64; n *= 64) for (unsigned text = 0; text < 2; ++text)
        for (unsigned mode = 0; mode < (text ? 33u : 20u); ++mode)
            if (text || mode < 12 || mode >= 17) run_case(root, n, mode, text);
    for (unsigned n = 1; n <= 64; n *= 64) for (unsigned text = 0; text < 2; ++text)
        for (unsigned mode = 0; mode < 16; ++mode) auto_case(root, n, text, mode);
    for (unsigned n = 1; n <= 64; n *= 64) for (unsigned text = 0; text < 2; ++text)
        for (unsigned mode = 0; mode < 16; ++mode) intake_case(root, n, text, mode);
    aotx_remove_tree(root);
    return aotx_report("live transcript", 30000);
}
