/* Purpose: Run the feeder against two pipes and check the records that reach the ring.
 * Owns: One inbound ring, one line pipe and one key pipe for each case.
 * Threading: Two processes; the test reads the ring while the feeder writes it.
 * Lifetime: The run of the program. */
#include "tests/disk_fake.h"

#include <fcntl.h>

#define AOTX_SLOTS      64u
#define AOTX_WAIT_NS    15000000000ull
#define AOTX_PART_NS    200000000ull
#define AOTX_LINES_MAX  200

static char **arguments;

typedef struct taken {
    int lines;
    int clocks;
    int keys;
    char text[AOTX_LINES_MAX][AOTX_BODY_BYTES + 1];
    uint32_t length[AOTX_LINES_MAX];
    aotx_key_body key[AOTX_LINES_MAX];
} taken;

/* Consumes the slots that the feeder published, the way a device consumer would. */
static void consume(aotx_inbound_ring *ring, taken *t)
{
    uint64_t consumed = aotx_inbound_consumed(ring);
    uint64_t head = aotx_inbound_head(ring);
    while (consumed < head) {
        const unsigned char *slot = ring->slots + (consumed & ring->mask) * AOTX_SLOT_BYTES;
        const aotx_record_header *h = (const aotx_record_header *)slot;
        CHECK(aotx_record_valid(h) == 1, "a slot does not validate");
        CHECK(h->seq == consumed + 1, "a slot holds the wrong sequence");
        CHECK(h->cls == AOTX_CLASS_A, "a slot holds the wrong class");
        CHECK(h->writer == AOTX_WRITER_FEEDER, "a slot holds the wrong writer");
        if (h->type == AOTX_REC_INPUT_LINE) {
            if (t->lines < AOTX_LINES_MAX) {
                memcpy(t->text[t->lines], aotx_record_body(h), h->body_len);
                t->text[t->lines][h->body_len] = '\0';
                t->length[t->lines] = h->body_len;
            }
            t->lines++;
        } else if (h->type == AOTX_REC_KEY) {
            CHECK(h->body_len == sizeof(aotx_key_body), "a key record has the wrong body length");
            if (t->keys < AOTX_LINES_MAX && h->body_len == sizeof(aotx_key_body)) {
                memcpy(&t->key[t->keys], aotx_record_body(h), sizeof(aotx_key_body));
            }
            t->keys++;
        } else if (h->type == AOTX_REC_TICK_START) {
            aotx_clock_body clock;
            memcpy(&clock, aotx_record_body(h), sizeof(clock));
            CHECK(h->body_len == sizeof(clock), "a tick start has the wrong body length");
            CHECK(clock.wall_ns > 1000000000000000000ull, "a tick start holds no wall clock");
            t->clocks++;
        }
        consumed++;
        aotx_store_release(&ring->pre->consumed, consumed);
    }
}

static void batch(int n)
{
    aotx_map map;
    aotx_inbound_ring ring;
    taken got;
    char fd_text[16];
    char line[AOTX_BODY_BYTES * 2];
    char key_text[16];
    char *args[6];
    int pipe_fds[2];
    int key_fds[2];
    int child;
    int want;
    int want_keys;
    int i;
    uint64_t deadline;

    memset(&got, 0, sizeof(got));
    CHECK(aotx_inbound_create(AOTX_SLOTS, &map, &ring) == 0, "the ring does not open");
    CHECK(pipe(pipe_fds) == 0, "the line pipe does not open");
    CHECK(pipe(key_fds) == 0, "the key pipe does not open");
    snprintf(fd_text, sizeof(fd_text), "%d", map.fd);
    snprintf(key_text, sizeof(key_text), "%d", key_fds[0]);
    args[0] = arguments[1];
    args[1] = (char *)"--inbound-fd";
    args[2] = fd_text;
    args[3] = (char *)"--keys-fd";
    args[4] = key_text;
    args[5] = NULL;
    child = aotx_spawn(args, pipe_fds[0], -1);
    CHECK(child > 0, "the feeder does not start");
    close(pipe_fds[0]);
    close(key_fds[0]);

    /* Every line carries different content, so a wrong slot index cannot hide. */
    for (i = 0; i < n; i++) {
        int bytes = snprintf(line, sizeof(line), "input line %d of %d\n", i, n);
        CHECK(write(pipe_fds[1], line, (size_t)bytes) == bytes, "the line does not write");
        consume(&ring, &got);
    }
    /* One line longer than a body must arrive as two records. */
    memset(line, 'x', AOTX_BODY_BYTES + 10);
    line[AOTX_BODY_BYTES + 10] = '\n';
    CHECK(write(pipe_fds[1], line, AOTX_BODY_BYTES + 11) == (ssize_t)(AOTX_BODY_BYTES + 11),
          "the long line does not write");
    close(pipe_fds[1]);

    /* Every key frame carries different content, so a wrong frame cannot hide. */
    for (i = 0; i < n; i++) {
        aotx_key_body frame;
        frame.key = (uint32_t)(0x100 + i);
        frame.codepoint = (uint32_t)(0x41 + i);
        frame.action = 1u;
        frame.mods = (uint32_t)(i & 3);
        CHECK(write(key_fds[1], &frame, sizeof(frame)) == (ssize_t)sizeof(frame),
              "the key frame does not write");
        consume(&ring, &got);
    }
    /* One frame that arrives in two reads must give one record and no other. */
    {
        aotx_key_body frame;
        const unsigned char *bytes = (const unsigned char *)&frame;
        frame.key = 0x200u;
        frame.codepoint = 0x7a7au;
        frame.action = 2u;
        frame.mods = 5u;
        CHECK(write(key_fds[1], bytes, 7) == 7, "the first part of the frame does not write");
        deadline = aotx_wall_ns() + AOTX_PART_NS;
        while (aotx_wall_ns() < deadline) {
            uint64_t backoff = 0;
            consume(&ring, &got);
            aotx_pause(&backoff);
        }
        CHECK(got.keys == n, "a part of a frame gave %d records and %d were asked for",
              got.keys, n);
        CHECK(write(key_fds[1], bytes + 7, 9) == 9, "the second part of the frame does not write");
    }
    close(key_fds[1]);

    want_keys = n + 1;
    deadline = aotx_wall_ns() + AOTX_WAIT_NS;
    while (got.keys < want_keys && aotx_wall_ns() < deadline) {
        uint64_t backoff = 0;
        consume(&ring, &got);
        if (got.keys < want_keys) {
            aotx_pause(&backoff);
        }
    }
    CHECK(got.keys == want_keys, "the feeder sent %d key records and %d were asked for",
          got.keys, want_keys);
    for (i = 0; i < n && i < got.keys; i++) {
        CHECK(got.key[i].key == (uint32_t)(0x100 + i), "key %d holds the wrong code", i);
        CHECK(got.key[i].codepoint == (uint32_t)(0x41 + i), "key %d holds the wrong point", i);
        CHECK(got.key[i].action == 1u, "key %d holds the wrong action", i);
    }
    if (got.keys == want_keys) {
        CHECK(got.key[n].key == 0x200u, "the frame of two reads holds the wrong code");
        CHECK(got.key[n].codepoint == 0x7a7au, "the frame of two reads holds the wrong point");
        CHECK(got.key[n].mods == 5u, "the frame of two reads holds the wrong modifier bits");
    }

    want = n + 2;
    deadline = aotx_wall_ns() + AOTX_WAIT_NS;
    while (got.lines < want && aotx_wall_ns() < deadline) {
        uint64_t backoff = 0;
        consume(&ring, &got);
        if (got.lines < want) {
            aotx_pause(&backoff);
        }
    }
    CHECK(got.lines == want, "the feeder sent %d lines and %d were asked for", got.lines, want);
    for (i = 0; i < n && i < got.lines; i++) {
        char expect[64];
        snprintf(expect, sizeof(expect), "input line %d of %d", i, n);
        CHECK(strcmp(got.text[i], expect) == 0, "line %d holds %s", i, got.text[i]);
    }
    if (got.lines >= want) {
        CHECK(got.length[n] == AOTX_BODY_BYTES, "the first part of the long line is not full");
        CHECK(got.length[n + 1] == 10, "the second part of the long line is the wrong length");
    }

    /* The clock records go on after the end of the input, until the ring closes. */
    deadline = aotx_wall_ns() + AOTX_WAIT_NS;
    while (got.clocks < 2 && aotx_wall_ns() < deadline) {
        uint64_t backoff = 0;
        consume(&ring, &got);
        aotx_pause(&backoff);
    }
    CHECK(got.clocks >= 2, "the feeder sent %d tick starts", got.clocks);
    aotx_store_release16(&ring.pre->closed, 1);
    CHECK(aotx_wait(child) == 0, "the feeder does not end with a clean status");
    printf("batch %d: lines %d, keys %d, clocks %d\n", n, got.lines, got.keys, got.clocks);
    aotx_map_release(&map);
}

/* A preamble that names another layout version must stop the feeder with the layout status. */
static void refuse_layout(void)
{
    aotx_map map;
    aotx_inbound_ring ring;
    char fd_text[16];
    char *args[4];
    int child;
    CHECK(aotx_inbound_create(AOTX_SLOTS, &map, &ring) == 0, "the ring does not open");
    ring.pre->magic = 0;
    snprintf(fd_text, sizeof(fd_text), "%d", map.fd);
    args[0] = arguments[1];
    args[1] = (char *)"--inbound-fd";
    args[2] = fd_text;
    args[3] = NULL;
    child = aotx_spawn(args, -1, -1);
    CHECK(child > 0, "the feeder does not start");
    CHECK(aotx_wait(child) == AOTX_EXIT_LAYOUT, "a wrong magic must give status 2");
    aotx_map_release(&map);
}

int main(int argc, char **argv)
{
    arguments = argv;
    if (argc < 2) {
        printf("usage: feed_test <feed program>\n");
        return 1;
    }
    batch(1);
    batch(64);
    refuse_layout();
    return aotx_report("feed_test", 30);
}
