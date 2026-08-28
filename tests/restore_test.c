/* Purpose: Build a journal, run the restore, and check the summary and the replayed records.
 * Owns: One temporary journal and one inbound ring for each case.
 * Threading: Two processes; the test reads the ring while the restore writes it.
 * Lifetime: The run of the program. */
#include "tests/disk_fake.h"

#include <fcntl.h>

#define AOTX_SLOTS   32u
#define AOTX_WAIT_NS 15000000000ull
#define AOTX_HOLD_NS 200000000ull
#define AOTX_TEST_BOOT 0x00000000cafe0001ull

/* A class A type that this build does not name. The replay filter tests the class and not
 * a list of types, so the replay must send this record with no change to the filter. */
#define AOTX_TEST_LATER_TYPE 200u

/* Records of one tick that the replay must send: a tick start, an input line, a key and a
 * token. The first tick adds the record of the later type. */
#define AOTX_TEST_PER_TICK 4

static char **arguments;

typedef struct summary {
    uint64_t boot_id;
    uint64_t last_tick;
    uint64_t replayed;
    uint64_t state_hash;
    uint64_t restore_hash;
    int has_restore;
    int found;
} summary;

static uint64_t field(const char *text, const char *name, int base, int *found)
{
    const char *at = strstr(text, name);
    if (at == NULL) {
        return 0;
    }
    *found = 1;
    return strtoull(at + strlen(name), NULL, base);
}

/* Reads the one line that the restore prints and pulls out every field. */
static void parse(const char *path, summary *s)
{
    char text[1024];
    int fd = open(path, O_RDONLY);
    ssize_t got;
    int seen = 0;
    memset(s, 0, sizeof(*s));
    if (fd < 0) {
        return;
    }
    got = read(fd, text, sizeof(text) - 1);
    close(fd);
    if (got <= 0) {
        return;
    }
    text[got] = '\0';
    if (strncmp(text, "restore boot=", 13) != 0) {
        return;
    }
    s->found = 1;
    s->boot_id = field(text, "boot=", 16, &seen);
    s->last_tick = field(text, "last_tick=", 10, &seen);
    s->replayed = field(text, "replayed=", 10, &seen);
    s->state_hash = field(text, "state_hash=", 16, &seen);
    if (strstr(text, "restore_hash=none") == NULL) {
        s->restore_hash = field(text, "restore_hash=", 16, &s->has_restore);
    }
}

/* Writes one boot directory with n complete ticks and one tick that has no commit. */
static void build_journal(const char *dir, uint64_t boot_id, int n, int with_restore)
{
    aotx_fake_device device;
    aotx_segment_writer writer;
    char boot_dir[320];
    char body[64];
    int i;
    snprintf(boot_dir, sizeof(boot_dir), "%s/%016llx", dir, (unsigned long long)boot_id);
    CHECK(aotx_make_dir(boot_dir) == 0, "the boot directory does not open");
    CHECK(aotx_segment_open(&writer, boot_dir, 4096) == 0, "the segment does not open");
    aotx_fake_start(&device, NULL, boot_id);
    for (i = 0; i < n; i++) {
        aotx_boot_body boot;
        aotx_clock_body clock;
        aotx_commit_body commit;
        const aotx_block_header *h;
        if (i == 0) {
            memset(&boot, 0, sizeof(boot));
            boot.boot_id = boot_id;
            boot.wall_ns = 1700000000000000000ull + boot_id;
            aotx_fake_record(&device, AOTX_CLASS_A, AOTX_REC_BOOT, &boot, sizeof(boot));
        }
        if (with_restore && i == 0) {
            aotx_restore_body r;
            memset(&r, 0, sizeof(r));
            r.restored_boot_id = boot_id - 1;
            r.last_tick = 41;
            r.replayed_count = 7;
            r.state_hash = 0x00feed0000000042ull;
            aotx_fake_record(&device, AOTX_CLASS_B, AOTX_REC_RESTORE, &r, sizeof(r));
        }
        clock.wall_ns = 1700000000000000000ull + (uint64_t)i;
        aotx_fake_record(&device, AOTX_CLASS_A, AOTX_REC_TICK_START, &clock, sizeof(clock));
        snprintf(body, sizeof(body), "replay line %d of %d", i, n);
        aotx_fake_record(&device, AOTX_CLASS_A, AOTX_REC_INPUT_LINE, body, (uint32_t)strlen(body));
        {
            /* A key is class A, so the replay must send it and the device applies it again. */
            aotx_key_body frame;
            frame.key = (uint32_t)(0x300 + i);
            frame.codepoint = (uint32_t)(0x61 + i);
            frame.action = 1u;
            frame.mods = 0u;
            aotx_fake_record(&device, AOTX_CLASS_A, AOTX_REC_KEY, &frame, sizeof(frame));
        }
        {
            /* A token is class A, so the replay must send it and the device applies it
             * again instead of a new sample. */
            aotx_token_body token;
            aotx_fake_token(i, &token);
            aotx_fake_record(&device, AOTX_CLASS_A, AOTX_REC_TOKEN, &token, sizeof(token));
        }
        if (i == 0) {
            uint32_t later = 0xabcdef01u;
            aotx_fake_record(&device, AOTX_CLASS_A, (uint8_t)AOTX_TEST_LATER_TYPE, &later,
                             sizeof(later));
        }
        {
            /* A sequence event is class B, so the replay must leave it out. */
            aotx_sequence_body event;
            aotx_fake_sequence(i, AOTX_SEQ_DONE, &event);
            aotx_fake_record(&device, AOTX_CLASS_B, AOTX_REC_SEQUENCE, &event, sizeof(event));
        }
        snprintf(body, sizeof(body), "console %d", i);
        aotx_fake_record(&device, AOTX_CLASS_B, AOTX_REC_CONSOLE, body, (uint32_t)strlen(body));
        memset(&commit, 0, sizeof(commit));
        commit.state_hash = 0x00aa000000000000ull + (uint64_t)i + 1;
        commit.applied_count = (uint64_t)i + 1;
        aotx_fake_record(&device, AOTX_CLASS_A, AOTX_REC_TICK_COMMIT, &commit, sizeof(commit));
        aotx_fake_commit(&device, 0);
        h = (const aotx_block_header *)device.stage;
        CHECK(aotx_segment_put(&writer, device.stage, h->byte_len) == 0, "a frame does not write");
    }
    /* The last tick has no commit record, so the restore must leave it out. */
    aotx_fake_record(&device, AOTX_CLASS_A, AOTX_REC_TICK_START, body, 8);
    aotx_fake_record(&device, AOTX_CLASS_A, AOTX_REC_INPUT_LINE, "not committed", 13);
    aotx_fake_commit(&device, 0);
    CHECK(aotx_segment_put(&writer, device.stage,
                           ((const aotx_block_header *)device.stage)->byte_len) == 0,
          "the last frame does not write");
    CHECK(aotx_segment_close(&writer) == 0, "the segment does not close");
}

static int run_restore(const char *dir, const char *out_path, int inbound_fd)
{
    char fd_text[16];
    char *args[7];
    int out_fd = open(out_path, O_RDWR | O_CREAT | O_TRUNC, 0644);
    int child;
    int status;
    CHECK(out_fd >= 0, "the output file does not open");
    args[0] = arguments[1];
    args[1] = (char *)"--journal";
    args[2] = (char *)dir;
    if (inbound_fd < 0) {
        args[3] = (char *)"--summary";
        args[4] = NULL;
    } else {
        snprintf(fd_text, sizeof(fd_text), "%d", inbound_fd);
        args[3] = (char *)"--inbound-fd";
        args[4] = fd_text;
        args[5] = NULL;
    }
    child = aotx_spawn(args, -1, out_fd);
    CHECK(child > 0, "the restore does not start");
    if (inbound_fd < 0) {
        status = aotx_wait(child);
        close(out_fd);
        return status;
    }
    close(out_fd);
    return child;
}

/* Reads the replayed records and checks the flag, the writer, and the order. The last
 * record states the result, and the restore must not leave before the device consumes it. */
static void collect(aotx_inbound_ring *ring, int child, int n, int *lines, int *clocks,
                    int *keys, int *tokens, int *laters)
{
    uint64_t deadline = aotx_wall_ns() + AOTX_WAIT_NS;
    uint64_t consumed = aotx_inbound_consumed(ring);
    uint64_t hold;
    uint64_t backoff = 0;
    int want = AOTX_TEST_PER_TICK * n + 2;
    int seen = 0;
    int results = 0;
    *lines = 0;
    *clocks = 0;
    *keys = 0;
    *tokens = 0;
    *laters = 0;
    while (seen < want && aotx_wall_ns() < deadline) {
        uint64_t head = aotx_inbound_head(ring);
        while (consumed < head && seen < want) {
            const unsigned char *slot = ring->slots + (consumed & ring->mask) * AOTX_SLOT_BYTES;
            const aotx_record_header *h = (const aotx_record_header *)slot;
            CHECK(aotx_record_valid(h) == 1, "a slot does not validate");
            CHECK(h->writer == AOTX_WRITER_RESTORE, "a slot holds the wrong writer");
            if (h->type == AOTX_REC_RESTORE) {
                aotx_restore_body body;
                CHECK(seen == want - 1, "the result record is at place %d and not last", seen);
                CHECK(h->cls == AOTX_CLASS_B, "the result record holds the wrong class");
                CHECK(h->flags == 0, "the result record must carry no flag");
                CHECK(h->body_len == sizeof(body), "the result record has the wrong body length");
                memcpy(&body, aotx_record_body(h), sizeof(body));
                CHECK(body.restored_boot_id == AOTX_TEST_BOOT, "the result names another boot");
                CHECK(body.last_tick == (uint64_t)n, "the result gives tick %llu and %d were committed",
                      (unsigned long long)body.last_tick, n);
                CHECK(body.replayed_count == (uint64_t)(AOTX_TEST_PER_TICK * n + 1),
                      "the result counts %llu records and %d were sent",
                      (unsigned long long)body.replayed_count, AOTX_TEST_PER_TICK * n + 1);
                CHECK(body.state_hash == 0x00aa000000000000ull + (uint64_t)n,
                      "the result holds the wrong state hash");
                results++;
            } else {
                CHECK((h->flags & AOTX_FLAG_REPLAYED) != 0, "a replayed record has no flag");
                CHECK(h->cls == AOTX_CLASS_A, "a replayed record has the wrong class");
                if (h->type == AOTX_REC_INPUT_LINE) {
                    char want_text[64];
                    snprintf(want_text, sizeof(want_text), "replay line %d", *lines);
                    CHECK(memcmp(aotx_record_body(h), want_text, strlen(want_text)) == 0,
                          "replayed line %d is out of order", *lines);
                    (*lines)++;
                } else if (h->type == AOTX_REC_KEY) {
                    aotx_key_body frame;
                    CHECK(h->body_len == sizeof(frame), "a key record has the wrong body length");
                    memcpy(&frame, aotx_record_body(h), sizeof(frame));
                    CHECK(frame.key == (uint32_t)(0x300 + *keys),
                          "replayed key %d is out of order", *keys);
                    (*keys)++;
                } else if (h->type == AOTX_REC_TICK_START) {
                    (*clocks)++;
                } else if (h->type == AOTX_REC_TOKEN) {
                    aotx_token_body want_token;
                    aotx_token_body got_token;
                    aotx_fake_token(*tokens, &want_token);
                    CHECK(h->body_len == sizeof(got_token),
                          "a token record has the wrong body length");
                    memcpy(&got_token, aotx_record_body(h), sizeof(got_token));
                    CHECK(memcmp(&got_token, &want_token, sizeof(got_token)) == 0,
                          "replayed token %d holds slot %u position %u token %u flags %u",
                          *tokens, got_token.slot, got_token.position, got_token.token,
                          got_token.flags);
                    (*tokens)++;
                } else if (h->type == AOTX_TEST_LATER_TYPE) {
                    /* The filter is a class test, so a type this build does not name is
                     * still replayed. */
                    (*laters)++;
                } else {
                    CHECK(0, "a record of type %u must not be replayed", h->type);
                }
            }
            seen++;
            /* The last slot stays unconsumed, so the wait of the restore can be seen. */
            if (seen < want) {
                consumed++;
                aotx_store_release(&ring->pre->consumed, consumed);
            }
        }
        if (seen < want) {
            aotx_pause(&backoff);
        }
    }
    CHECK(seen == want, "the replay sent %d records and %d were asked for", seen, want);
    CHECK(results == 1, "the replay sent %d result records and one was asked for", results);
    hold = aotx_wall_ns() + AOTX_HOLD_NS;
    backoff = 0;
    while (aotx_wall_ns() < hold) {
        aotx_pause(&backoff);
    }
    CHECK(aotx_alive(child) == 1, "the restore left before the device consumed every record");
    aotx_store_release(&ring->pre->consumed, consumed + 1);
    CHECK(aotx_wait(child) == 0, "the restore does not end with a clean status");
}

static void batch(int n)
{
    aotx_map map;
    aotx_inbound_ring ring;
    summary s;
    char dir[256];
    char out_path[1024];
    uint64_t boot_id = AOTX_TEST_BOOT;
    int lines = 0;
    int clocks = 0;
    int keys = 0;
    int tokens = 0;
    int laters = 0;
    int child;

    CHECK(aotx_temp_dir(dir, sizeof(dir)) == 0, "the temporary directory does not open");
    build_journal(dir, boot_id, n, 0);
    snprintf(out_path, sizeof(out_path), "%s/summary.txt", dir);
    CHECK(run_restore(dir, out_path, -1) == 0, "the summary does not end with a clean status");
    parse(out_path, &s);
    CHECK(s.found == 1, "the summary line is not printed");
    CHECK(s.boot_id == boot_id, "the summary names another boot");
    CHECK(s.last_tick == (uint64_t)n, "the summary gives tick %llu and %d were committed",
          (unsigned long long)s.last_tick, n);
    CHECK(s.replayed == (uint64_t)(AOTX_TEST_PER_TICK * n + 1),
          "the summary counts %llu records and %d were asked for",
          (unsigned long long)s.replayed, AOTX_TEST_PER_TICK * n + 1);
    CHECK(s.state_hash == 0x00aa000000000000ull + (uint64_t)n, "the summary hash is wrong");
    CHECK(s.has_restore == 0, "a journal with no restore record must say none");

    CHECK(aotx_inbound_create(AOTX_SLOTS, &map, &ring) == 0, "the ring does not open");
    child = run_restore(dir, out_path, map.fd);
    collect(&ring, child, n, &lines, &clocks, &keys, &tokens, &laters);
    CHECK(lines == n, "the replay sent %d lines and %d were asked for", lines, n);
    CHECK(clocks == n, "the replay sent %d tick starts and %d were asked for", clocks, n);
    CHECK(keys == n, "the replay sent %d keys and %d were asked for", keys, n);
    CHECK(tokens == n, "the replay sent %d tokens and %d were asked for", tokens, n);
    CHECK(laters == 1, "the replay sent %d records of the later type and one was asked for",
          laters);
    printf("batch %d: replayed lines %d, keys %d, tokens %d, tick starts %d, later type %d\n",
           n, lines, keys, tokens, clocks, laters);
    aotx_map_release(&map);
    aotx_remove_tree(dir);
}

/* Checks the choice of the newest boot, the restore record, and a journal that is damaged. */
static void journals(void)
{
    summary s;
    char dir[256];
    char out_path[1024];
    char seg_path[1024];
    int fd;

    CHECK(aotx_temp_dir(dir, sizeof(dir)) == 0, "the temporary directory does not open");
    snprintf(out_path, sizeof(out_path), "%s/summary.txt", dir);
    CHECK(run_restore(dir, out_path, -1) == AOTX_EXIT_NOJOURNAL,
          "an empty journal must give the no journal status");

    build_journal(dir, 0x00000000cafe0001ull, 3, 0);
    build_journal(dir, 0x00000000cafe0002ull, 2, 1);
    CHECK(run_restore(dir, out_path, -1) == 0, "the summary does not end with a clean status");
    parse(out_path, &s);
    /* The boot record wall clock of the second journal is the later one. */
    CHECK(s.boot_id == 0x00000000cafe0002ull, "the newest boot is not chosen");
    CHECK(s.last_tick == 2, "the newest boot gives the wrong tick");
    CHECK(s.has_restore == 1, "the restore record is not read");
    CHECK(s.restore_hash == 0x00feed0000000042ull, "the restore hash is wrong");

    /* A frame that is cut short ends the readable journal at the tick before it. */
    {
        char names[16][AOTX_NAME_BYTES];
        char boot_dir[320];
        int count;
        snprintf(boot_dir, sizeof(boot_dir), "%s/%016llx", dir, 0x00000000cafe0002ull);
        count = aotx_segment_list(boot_dir, names, 16);
        CHECK(count > 0, "the boot directory holds no segment");
        snprintf(seg_path, sizeof(seg_path), "%s/%s", boot_dir, names[count - 1]);
    }
    fd = open(seg_path, O_WRONLY | O_APPEND);
    CHECK(fd >= 0, "the last segment does not open");
    CHECK(write(fd, "\x40\x00\x00\x00\x00\x00\x00\x00short", 13) == 13,
          "the damage does not write");
    close(fd);
    CHECK(run_restore(dir, out_path, -1) == 0, "a torn tail must still restore");
    parse(out_path, &s);
    CHECK(s.last_tick == 2, "a torn tail must keep the tick before it");
    aotx_remove_tree(dir);
}

int main(int argc, char **argv)
{
    arguments = argv;
    if (argc < 2) {
        printf("usage: restore_test <restore program>\n");
        return 1;
    }
    batch(1);
    batch(64);
    journals();
    return aotx_report("restore_test", 40);
}
