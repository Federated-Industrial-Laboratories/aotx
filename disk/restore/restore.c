/* Purpose: Replay the class A records of the newest complete journal into the inbound ring.
 * Owns: The block buffer, and the head field of the inbound ring while a replay runs.
 * Threading: One thread; the program is the only producer of the ring.
 * Lifetime: From the start of the scan to the exit of the program. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "disk/restore/scan.h"
#include "disk/feed/line.h"

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define AOTX_BLOCK_MAX (16u * 1024u * 1024u)

static volatile sig_atomic_t stop_flag;

static void on_signal(int number)
{
    (void)number;
    stop_flag = 1;
}

typedef struct replay {
    aotx_inbound_ring ring;
    uint64_t stop_block;
    uint64_t replayed;
    int publish_on;
} replay;

static int replay_group(replay *r, const unsigned char *block, uint32_t first,
                        uint32_t count)
{
    aotx_record_header headers[AOTX_LINE_PARTS_MAX];
    const void *bodies[AOTX_LINE_PARTS_MAX];
    uint32_t used = 0u;
    for (uint32_t i = 0u; i < count; ++i) {
        const aotx_record_header *h = aotx_block_record(block, first + i);
        if (h->cls != AOTX_CLASS_A
            || h->type == AOTX_REC_BOOT || h->type == AOTX_REC_TICK_COMMIT) {
            continue;
        }
        memcpy(&headers[used], h, sizeof(headers[used]));
        headers[used].flags = (uint16_t)(h->flags | AOTX_FLAG_REPLAYED);
        headers[used].writer = AOTX_WRITER_RESTORE;
        bodies[used] = aotx_record_body(h);
        used++;
    }
    if (used == 0u) {
        return 0;
    }
    if (r->publish_on
        && aotx_line_publish_records(&r->ring, &stop_flag, headers, bodies, used) != 0) {
        return -1;
    }
    r->replayed += used;
    return 0;
}

/* Sends every class A input record of one block, up to and including the last complete
 * tick. The flag marks the record as one that the system applied before. */
static int replay_block(void *ctx, const unsigned char *block, uint64_t index)
{
    replay *r = (replay *)ctx;
    const aotx_block_header *bh = (const aotx_block_header *)block;
    uint32_t i;
    if (index > r->stop_block) {
        return 1;
    }
    for (i = 0; i < bh->record_count; i++) {
        const aotx_record_header *h = aotx_block_record(block, i);
        if (h->cls != AOTX_CLASS_A) {
            continue;
        }
        if (h->type == AOTX_REC_BOOT || h->type == AOTX_REC_TICK_COMMIT) {
            /* The device is the only writer of a boot record and of a tick commit record,
             * and it makes both again on its own. The replay leaves out those two types
             * and sends every other class A record. The test is the class and not a list
             * of types, so a class A type that comes later replays with no change here. */
            continue;
        }
        uint32_t count = 1u;
        if (h->type == AOTX_REC_INPUT_LINE
            && (h->flags & AOTX_FLAG_FRAGMENT) == 0u) {
            while (i + count < bh->record_count && count < AOTX_LINE_PARTS_MAX) {
                const aotx_record_header *next = aotx_block_record(block, i + count);
                if (next->type != AOTX_REC_INPUT_LINE
                    || (next->flags & AOTX_FLAG_FRAGMENT) == 0u) {
                    break;
                }
                count++;
            }
        }
        if (replay_group(r, block, i, count) != 0) {
            return -1;
        }
        i += count - 1u;
    }
    return (index == r->stop_block) ? 1 : 0;
}

/* Publishes the one record that states the result of the replay. The device applies it and
 * writes it to the new journal, so no program on the host side reads the summary line. */
static int publish_result(replay *r, const aotx_journal_scan *scan)
{
    aotx_record_header h;
    aotx_restore_body body;
    memset(&h, 0, sizeof(h));
    memset(&body, 0, sizeof(body));
    body.restored_boot_id = scan->boot_id;
    body.last_tick = scan->last_tick;
    body.replayed_count = r->replayed;
    body.state_hash = scan->state_hash;
    /* The header boot identity names the journal that the records came from. The device
     * stamps its own boot identity when it writes the record to the new journal. */
    h.boot_id = scan->boot_id;
    h.writer = AOTX_WRITER_RESTORE;
    h.cls = AOTX_CLASS_B;
    h.type = AOTX_REC_RESTORE;
    h.flags = 0;
    h.body_len = sizeof(body);
    if (aotx_inbound_wait(&r->ring, &stop_flag) != 0) {
        return -1;
    }
    aotx_inbound_put(&r->ring, &h, &body);
    return 0;
}

/* Waits until the device consumed every published slot. The replay is not complete while a
 * record is still in the ring, because the device has not applied it. */
static void wait_for_device(replay *r)
{
    uint64_t backoff = 0;
    while (aotx_inbound_consumed(&r->ring) < aotx_inbound_head(&r->ring)) {
        if (aotx_inbound_closed(&r->ring) || stop_flag != 0) {
            fprintf(stderr, "restore: the device ended before it consumed every record\n");
            return;
        }
        aotx_pause(&backoff);
    }
}

static void usage(void)
{
    fprintf(stderr, "usage: aotx_restore --journal <dir> [--inbound-fd <fd>] [--summary]\n");
}

static void report(const aotx_journal_scan *scan, uint64_t replayed)
{
    printf("restore boot=%016llx last_tick=%llu replayed=%llu state_hash=%016llx",
           (unsigned long long)scan->boot_id, (unsigned long long)scan->last_tick,
           (unsigned long long)replayed, (unsigned long long)scan->state_hash);
    if (scan->has_restore) {
        printf(" restore_hash=%016llx restore_of=%016llx\n",
               (unsigned long long)scan->restore_hash, (unsigned long long)scan->restore_of);
    } else {
        printf(" restore_hash=none\n");
    }
}

int main(int argc, char **argv)
{
    aotx_journal_scan scan;
    aotx_map map;
    replay r;
    struct sigaction act;
    unsigned char *buffer;
    const char *journal = NULL;
    uint64_t blocks = 0;
    int inbound_fd = -1;
    int summary = 0;
    int torn = 0;
    int i;

    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--journal") == 0 && i + 1 < argc) {
            journal = argv[++i];
        } else if (strcmp(argv[i], "--inbound-fd") == 0 && i + 1 < argc) {
            inbound_fd = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--summary") == 0) {
            summary = 1;
        } else {
            usage();
            return AOTX_EXIT_FAULT;
        }
    }
    if (journal == NULL || (inbound_fd < 0 && !summary)) {
        usage();
        return AOTX_EXIT_FAULT;
    }

    memset(&r, 0, sizeof(r));
    memset(&act, 0, sizeof(act));
    act.sa_handler = on_signal;
    sigaction(SIGTERM, &act, NULL);
    sigaction(SIGINT, &act, NULL);

    buffer = (unsigned char *)malloc(AOTX_BLOCK_MAX);
    if (buffer == NULL) {
        fprintf(stderr, "restore: the block buffer does not fit in memory\n");
        return AOTX_EXIT_FAULT;
    }
    if (aotx_journal_latest(journal, buffer, AOTX_BLOCK_MAX, &scan) != 0) {
        fprintf(stderr, "restore: no journal in %s holds a complete tick\n", journal);
        free(buffer);
        return AOTX_EXIT_NOJOURNAL;
    }
    if (scan.torn) {
        fprintf(stderr, "restore: the journal of boot %016llx has a torn tail after %llu blocks\n",
                (unsigned long long)scan.boot_id, (unsigned long long)scan.blocks);
    }

    if (!summary) {
        if (aotx_die_with_parent() != 0) {
            fprintf(stderr, "restore: the parent death signal is not set\n");
        }
        if (aotx_map_fd(inbound_fd, &map) != 0) {
            fprintf(stderr, "restore: the ring descriptor does not map\n");
            free(buffer);
            return AOTX_EXIT_FAULT;
        }
        if (aotx_inbound_attach(&map, &r.ring) != 0) {
            fprintf(stderr, "restore: the ring preamble does not match this layout version\n");
            free(buffer);
            return AOTX_EXIT_LAYOUT;
        }
        r.publish_on = 1;
    }
    r.stop_block = scan.last_block;
    if (aotx_journal_walk(scan.dir, buffer, AOTX_BLOCK_MAX, replay_block, &r, &blocks, &torn) != 0) {
        fprintf(stderr, "restore: the replay of %s did not finish\n", scan.dir);
        free(buffer);
        return AOTX_EXIT_FAULT;
    }
    if (!summary) {
        if (publish_result(&r, &scan) != 0) {
            fprintf(stderr, "restore: the result record does not reach the ring\n");
            free(buffer);
            return AOTX_EXIT_FAULT;
        }
        wait_for_device(&r);
    }
    report(&scan, r.replayed);
    if (!summary) {
        aotx_map_release(&map);
    }
    free(buffer);
    return AOTX_EXIT_OK;
}
