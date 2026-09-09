/* Purpose: Supply independent live memory files and a draining inbound ring.
 * Owns: Test buffers, temporary files and one consumer thread.
 * Threading: The producer and consumer use the normal ring publication fields.
 * Lifetime: One bounded transport test. */
#ifndef AOTX_TEST_LIVE_FEED_FIXTURE_H
#define AOTX_TEST_LIVE_FEED_FIXTURE_H
#include "disk/feed/cognitive_io.h"
#include "cognitive/live.h"
#include "ccir/ccir.h"
#include "tests/disk_fake.h"
#include <fcntl.h>
#include <pthread.h>

#define AOTX_TEST_LIVE_SLOTS 64u
#define AOTX_TEST_LIVE_RECORDS ((AOTX_LIVE_BYTES + AOTX_LIVE_DATA - 1u) / AOTX_LIVE_DATA + 8u)
typedef struct aotx_live_capture {
    aotx_inbound_ring ring;
    unsigned char *records;
    uint64_t done;
    uint32_t count, bad;
} aotx_live_capture;
static uint64_t aotx_test_get(const unsigned char *p, unsigned n) {
    uint64_t value = 0;
    for (unsigned i = 0; i < n; ++i) value |= (uint64_t)p[i] << (i * 8);
    return value;
}
static void aotx_test_put(unsigned char *p, uint64_t value, unsigned n) {
    for (unsigned i = 0; i < n; ++i) p[i] = (unsigned char)(value >> (i * 8));
}
static void aotx_test_write(const char *path, const void *data, size_t bytes) {
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
    CHECK(fd >= 0, "test file opens");
    if (fd < 0) exit(2);
    CHECK(write(fd, data, bytes) == (ssize_t)bytes, "test file bytes write");
    CHECK(close(fd) == 0, "test file closes");
}
static void *aotx_test_consume(void *pointer) {
    aotx_live_capture *c = pointer;
    uint64_t at = 0, backoff = 0;
    for (;;) {
        uint64_t head = aotx_inbound_head(&c->ring);
        while (at < head) {
            const unsigned char *slot = c->ring.slots + (at & c->ring.mask) * AOTX_SLOT_BYTES;
            const aotx_record_header *h = (const aotx_record_header *)slot;
            if (aotx_load_acquire(&h->seq) != at + 1 || c->count >= AOTX_TEST_LIVE_RECORDS) c->bad = 1;
            else memcpy(c->records + (size_t)c->count++ * AOTX_SLOT_BYTES, slot, AOTX_SLOT_BYTES);
            ++at;
        }
        aotx_store_release(&c->ring.pre->consumed, at);
        if (aotx_load_acquire(&c->done) && at == aotx_inbound_head(&c->ring)) break;
        aotx_pause(&backoff);
    }
    return NULL;
}
static int aotx_test_command(const unsigned char *line, uint32_t bytes, aotx_live_capture *c) {
    memset(c, 0, sizeof(*c));
    c->ring.pre = calloc(1, sizeof(*c->ring.pre));
    c->ring.slots = calloc(AOTX_TEST_LIVE_SLOTS, AOTX_SLOT_BYTES);
    c->records = calloc(AOTX_TEST_LIVE_RECORDS, AOTX_SLOT_BYTES);
    if (!c->ring.pre || !c->ring.slots || !c->records) exit(2);
    c->ring.slot_count = AOTX_TEST_LIVE_SLOTS; c->ring.mask = AOTX_TEST_LIVE_SLOTS - 1u;
    pthread_t thread;
    if (pthread_create(&thread, NULL, aotx_test_consume, c)) exit(2);
    volatile sig_atomic_t stop = 0;
    int result = aotx_live_feed_line(line, bytes, &c->ring, &stop);
    aotx_store_release(&c->done, 1);
    CHECK(pthread_join(thread, NULL) == 0 && !c->bad, "consumer completes with valid slot publication");
    return result;
}
static void aotx_test_drop(aotx_live_capture *c) {
    free(c->ring.pre); free(c->ring.slots); free(c->records);
}
static void aotx_test_transfer(const aotx_live_capture *c, unsigned op,
                                const unsigned char *expected, uint32_t bytes,
                                unsigned char identity[16]) {
    uint32_t parts = (bytes + AOTX_LIVE_DATA - 1) / AOTX_LIVE_DATA, at = 0;
    CHECK(c->count == parts, "all transfer fragments arrive");
    CHECK(c->count > 0, "fragment checks are populated");
    if (!c->count) return;
    const unsigned char *first = c->records + AOTX_HEADER_BYTES;
    memcpy(identity, first + 8, 16);
    unsigned any = 0;
    for (unsigned i = 0; i < 16; ++i) any |= identity[i];
    CHECK(any != 0, "transfer identity is nonzero");
    for (unsigned i = 0; i < c->count; ++i) {
        const aotx_record_header *h = (const aotx_record_header *)(c->records + (size_t)i * AOTX_SLOT_BYTES);
        const unsigned char *p = aotx_record_body(h);
        uint32_t take = bytes - at;
        if (take > AOTX_LIVE_DATA) take = AOTX_LIVE_DATA;
        CHECK(aotx_record_valid(h) && h->cls == AOTX_CLASS_A && h->type == AOTX_LIVE_RECORD &&
              h->writer == AOTX_WRITER_FEEDER && h->flags == 0, "typed class A record header");
        CHECK(h->body_len == AOTX_LIVE_PART + take, "last fragment has its exact length");
        CHECK(aotx_test_get(p, 4) == 1 && aotx_test_get(p + 4, 4) == op &&
              !memcmp(p + 8, identity, 16), "schema, operation and transfer identity");
        CHECK(aotx_test_get(p + 24, 4) == bytes && aotx_test_get(p + 28, 4) == at, "total and ordered offset");
        CHECK(!memcmp(p + AOTX_LIVE_PART, expected + at, take), "exact source bytes cross the ring");
        unsigned padding = 0;
        for (uint32_t j = h->body_len; j < AOTX_BODY_BYTES; ++j) padding |= p[j];
        CHECK(!padding, "unused slot bytes are zero");
        at += take;
    }
    CHECK(at == bytes, "reassembly reaches the exact source length");
}
static unsigned char *aotx_test_image(unsigned count, uint32_t payload, int tail, uint32_t *bytes) {
    *bytes = AOTX_COG_HEADER + count * AOTX_COG_OBJECT + payload;
    unsigned char *out = calloc(1, *bytes);
    if (!out) exit(2);
    memcpy(out, tail ? "AOTXLOG1" : "AOTXOBJ1", 8);
    aotx_test_put(out + 8, 1, 4); aotx_test_put(out + 12, AOTX_COG_HEADER, 4);
    aotx_test_put(out + 16, AOTX_COG_OBJECT, 4); aotx_test_put(out + 20, count, 4);
    aotx_test_put(out + 24, payload, 8); aotx_test_put(out + 32, tail ? 1 : count, 8);
    aotx_test_put(out + 40, tail ? 2 : 1, 8); out[48] = 71;
    aotx_test_put(out + 64, AOTX_COG_HEADER, 8);
    aotx_test_put(out + 72, AOTX_COG_HEADER + count * AOTX_COG_OBJECT, 8);
    aotx_test_put(out + 80, *bytes, 8); aotx_test_put(out + 88, 1, 4);
    uint32_t at = 0;
    for (unsigned i = 0; i < count; ++i) {
        unsigned char *r = out + AOTX_COG_HEADER + i * AOTX_COG_OBJECT;
        uint32_t n = i + 1 == count ? payload - at : payload / count;
        aotx_test_put(r, 1, 2); aotx_test_put(r + 2, AOTX_COG_EVENT, 2);
        aotx_test_put(r + 8, i + 1, 8); r[24] = 71;
        aotx_test_put(r + 40, 1, 8); aotx_test_put(r + 48, i + 1, 8); aotx_test_put(r + 56, i + 1, 8);
        aotx_test_put(r + 64, 100 + i, 8); aotx_test_put(r + 160, at, 8); aotx_test_put(r + 168, n, 8);
        aotx_test_put(r + 176, AOTX_COG_INSTANCE, 4); aotx_test_put(r + 180, AOTX_COG_AUTHORED, 4);
        aotx_test_put(r + 192, AOTX_COG_UNKNOWN, 4); aotx_test_put(r + 232, 1, 8);
        for (uint32_t j = 0; j < n; ++j) out[AOTX_COG_HEADER + count * AOTX_COG_OBJECT + at + j] =
            (unsigned char)(i * 31u + j * 13u + 7u);
        at += n;
    }
    return out;
}
#endif
