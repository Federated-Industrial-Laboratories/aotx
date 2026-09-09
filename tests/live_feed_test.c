/* Purpose: Check exact live memory transport, framing refusal and bounded input reads.
 * Owns: Distinct N=1/N=64 files and one concurrent ring consumer per call.
 * Threading: One publisher and one reader exercise ring wrap and backpressure.
 * Lifetime: One test process; no GPU or language model is used. */
#include "tests/live_feed_fixture.h"
#include <errno.h>

static int aotx_read_change;
static char aotx_change_path[1024];
ssize_t __real_read(int fd, void *data, size_t bytes);
ssize_t __wrap_read(int fd, void *data, size_t bytes) {
    if (aotx_read_change) {
        int mode = aotx_read_change;
        aotx_read_change = 0;
        int writer = open(aotx_change_path, O_WRONLY | (mode == 1 ? O_TRUNC : O_APPEND));
        if (writer < 0) _exit(2);
        if (mode == 2 && write(writer, "x", 1) != 1) _exit(2);
        close(writer);
    }
    return __real_read(fd, data, bytes);
}
static int aotx_run_file(unsigned op, const char *path, aotx_live_capture *capture) {
    static const char *const names[] = {"", "load", "apply", "bind", "query", "", "text"};
    char line[2048];
    int n = snprintf(line, sizeof(line), "memory %s %s", names[op], path);
    CHECK(n > 0 && (size_t)n < sizeof(line), "command path fits");
    return aotx_test_command((const unsigned char *)line, (uint32_t)n, capture);
}
static void aotx_refused(unsigned op, const char *path) {
    aotx_live_capture capture;
    alarm(3);
    CHECK(aotx_run_file(op, path, &capture) == 1, "invalid file is handled");
    alarm(0);
    CHECK(capture.count == 1, "refusal emits one visible note");
    if (capture.count == 1) {
        const aotx_record_header *h = (const aotx_record_header *)capture.records;
        const char *body = (const char *)aotx_record_body(h);
        CHECK(h->type == AOTX_REC_INPUT_LINE && h->cls == AOTX_CLASS_A &&
              h->body_len > 5 && !memcmp(body, "note memory ", 12), "refusal has no typed state fragments");
    }
    aotx_test_drop(&capture);
}
static unsigned char *aotx_batch(unsigned n, unsigned op, uint32_t *bytes) {
    uint32_t row = op == AOTX_LIVE_BIND ? AOTX_LIVE_BIND_ROW : AOTX_LIVE_QUERY_ROW;
    *bytes = AOTX_LIVE_HEADER + n * row;
    unsigned char *out = calloc(1, *bytes);
    if (!out) exit(2);
    memcpy(out, op == AOTX_LIVE_BIND ? "AOTXBND1" : op == AOTX_LIVE_TEXT ? "AOTXTXT1" : "AOTXLIV1", 8);
    aotx_test_put(out + 8, n, 4); aotx_test_put(out + 12, 1, 4);
    out[16] = 71; aotx_test_put(out + 32, n, 8); aotx_test_put(out + 40, row, 4);
    /* The disk transports row bytes without interpreting their device-owned fields. */
    for (unsigned i = 0; i < n; ++i) for (uint32_t j = 0; j < row; ++j)
        out[AOTX_LIVE_HEADER + i * row + j] = (unsigned char)(i * 29 + j * 17 + n);
    return out;
}
static void aotx_batch_case(const char *dir, unsigned n, unsigned op) {
    char path[1024], fifo[1024], link[1024];
    snprintf(path, sizeof(path), "%s/batch %u %u", dir, n, op);
    snprintf(fifo, sizeof(fifo), "%s/fifo-%u-%u", dir, n, op);
    snprintf(link, sizeof(link), "%s/link-%u-%u", dir, n, op);
    uint32_t bytes;
    unsigned char *data = aotx_batch(n, op, &bytes), ids[2][16];
    aotx_test_write(path, data, bytes);
    for (unsigned repeat = 0; repeat < 2; ++repeat) {
        aotx_live_capture capture;
        CHECK(aotx_run_file(op, path, &capture) == 1, "valid batch is handled");
        aotx_test_transfer(&capture, op, data, bytes, ids[repeat]);
        aotx_test_drop(&capture);
    }
    CHECK(memcmp(ids[0], ids[1], 16), "separate transfers receive distinct random identities");
    if (op == AOTX_LIVE_TEXT || op == AOTX_LIVE_QUERY) {
        memcpy(data, op == AOTX_LIVE_TEXT ? "AOTXLIV1" : "AOTXTXT1", 8);
        aotx_test_write(path, data, bytes); aotx_refused(op, path);
        memcpy(data, op == AOTX_LIVE_TEXT ? "AOTXTXT1" : "AOTXLIV1", 8);
    }
    aotx_test_write(path, data, bytes - 1); aotx_refused(op, path);
    aotx_test_write(path, data, bytes);
    data[8] = 65; aotx_test_write(path, data, bytes); aotx_refused(op, path); data[8] = (unsigned char)n;
    data[40] ^= 1; aotx_test_write(path, data, bytes); aotx_refused(op, path); data[40] ^= 1;
    data[0] ^= 1; aotx_test_write(path, data, bytes); aotx_refused(op, path); data[0] ^= 1;
    for (int change = 1; change <= 2; ++change) {
        aotx_test_write(path, data, bytes);
        snprintf(aotx_change_path, sizeof(aotx_change_path), "%s", path);
        aotx_read_change = change; aotx_refused(op, path);
        CHECK(aotx_read_change == 0, "file size changes after the size check");
    }
    CHECK(mkfifo(fifo, 0600) == 0, "FIFO fixture opens"); aotx_refused(op, fifo);
    CHECK(symlink(path, link) == 0, "symlink fixture opens"); aotx_refused(op, link);
    aotx_refused(op, dir);
    unlink(path); unlink(fifo); unlink(link); free(data);
    printf("live feed operation=%u N=%u complete\n", op, n);
}
static void aotx_section(aotx_ccir_input *in, unsigned type, const void *data, uint32_t bytes) {
    memset(in, 0, sizeof(*in));
    in->section.type = type; in->section.schema = 1; in->section.flags = AOTX_CCIR_REQUIRED;
    in->section.id[0] = (unsigned char)type; in->section.bytes = bytes;
    in->section.alignment = 64; in->data = data;
}
static void aotx_state_case(const char *dir, unsigned n, uint32_t payload) {
    char source[1024], raw[1024];
    snprintf(source, sizeof(source), "%s/state-%u-%u", dir, n, payload);
    snprintf(raw, sizeof(raw), "%s/tail-%u-%u", dir, n, payload);
    uint32_t cb, tb;
    unsigned char *checkpoint = aotx_test_image(n, payload, 0, &cb);
    unsigned char *tail = aotx_test_image(n, payload, 1, &tb);
    unsigned char manifest[AOTX_CCIR_MANIFEST_BYTES], lineage[16] = {71};
    aotx_ccir_input input[3];
    aotx_section(&input[0], 1, manifest, sizeof(manifest));
    aotx_section(&input[1], 2, checkpoint, cb); aotx_section(&input[2], 3, tail, tb);
    aotx_ccir_manifest(manifest, input[1].section.id, input[2].section.id);
    aotx_ccir_meta meta = {n, n, 2};
    CHECK(aotx_ccir_create(source, lineage, input, 3, &meta, NULL) == 0, "source CCIR created");
    uint32_t length = 16 + cb + tb;
    unsigned char *expected = malloc(length), id[16];
    if (!expected) exit(2);
    aotx_test_put(expected, cb, 8); aotx_test_put(expected + 8, tb, 8);
    memcpy(expected + 16, checkpoint, cb); memcpy(expected + 16 + cb, tail, tb);
    aotx_live_capture capture;
    CHECK(aotx_run_file(AOTX_LIVE_LOAD, source, &capture) == 1, "load command is handled");
    aotx_test_transfer(&capture, AOTX_LIVE_LOAD, expected, length, id);
    if (payload == AOTX_COG_PAYLOAD) CHECK(capture.count > 4096, "maximum load crosses the reference inbound ring bound");
    aotx_test_drop(&capture);
    aotx_test_write(raw, tail, tb);
    CHECK(aotx_run_file(AOTX_LIVE_UPDATE, raw, &capture) == 1, "update command is handled");
    aotx_test_transfer(&capture, AOTX_LIVE_UPDATE, tail, tb, id); aotx_test_drop(&capture);
    tail[80] ^= 1; aotx_test_write(raw, tail, tb); aotx_refused(AOTX_LIVE_UPDATE, raw);
    unlink(source); unlink(raw); free(expected); free(checkpoint); free(tail);
    printf("live feed state N=%u payload=%u complete\n", n, payload);
}
static void aotx_parser_case(void) {
    static const char *const lines[] = {"memory", "memory stats", "memory queryish x", "xmemory load x",
        "note memory bind x", "memory choice x", "memory applyx x", "memory loader x", "say 0 memory query x",
        "memory textual x", "memory text_choice x"};
    for (unsigned i = 0; i < sizeof(lines) / sizeof(lines[0]); ++i) {
        aotx_live_capture capture;
        CHECK(aotx_test_command((const unsigned char *)lines[i], (uint32_t)strlen(lines[i]), &capture) == 0,
              "ordinary command is not intercepted");
        CHECK(!capture.count, "ordinary command creates no memory records"); aotx_test_drop(&capture);
    }
    aotx_live_capture capture;
    const unsigned char bad[] = "memory load bad\0path";
    CHECK(aotx_test_command(bad, sizeof(bad) - 1, &capture) == 1 && capture.count == 1,
          "embedded NUL path is refused"); aotx_test_drop(&capture);
    CHECK(aotx_test_command((const unsigned char *)"memory query", 12, &capture) == 1 && capture.count == 1,
          "missing path is refused"); aotx_test_drop(&capture);
}
int main(void) {
    alarm(90);
    char dir[1024];
    if (aotx_temp_dir(dir, sizeof(dir))) return 2;
    aotx_parser_case();
    for (unsigned n = 1; n <= 64; n *= 64) {
        aotx_batch_case(dir, n, AOTX_LIVE_BIND); aotx_batch_case(dir, n, AOTX_LIVE_QUERY);
        aotx_batch_case(dir, n, AOTX_LIVE_TEXT);
        aotx_state_case(dir, n, n * 37 + 13);
    }
    aotx_state_case(dir, AOTX_COG_OBJECTS, AOTX_COG_PAYLOAD);
    aotx_refused(AOTX_LIVE_LOAD, dir);
    aotx_remove_tree(dir);
    return aotx_report("live feed", 1000);
}
