/* Purpose: Copy blocks from the host ring into journal segments and derived files.
 * Owns: The open segment writer, the derived files, and the consumer cursor of the ring.
 * Threading: One thread; the program is the only writer of the cursor.
 * Lifetime: From the map of the ring to the exit of the program. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "disk/drain/derive.h"

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define AOTX_TORN_TRIES 64

static volatile sig_atomic_t stop_flag;

static void on_signal(int number)
{
    (void)number;
    stop_flag = 1;
}

/* Makes a directory and every parent of it. Returns 0 or -1. */
static int make_path(const char *path)
{
    char work[AOTX_PATH_BYTES];
    size_t i;
    size_t n = strlen(path);
    if (n + 1 > sizeof(work)) {
        return -1;
    }
    memcpy(work, path, n + 1);
    for (i = 1; i < n; i++) {
        if (work[i] == '/') {
            work[i] = '\0';
            if (aotx_make_dir(work) != 0) {
                return -1;
            }
            work[i] = '/';
        }
    }
    return aotx_make_dir(work);
}

static void usage(void)
{
    fprintf(stderr, "usage: aotx_drain --ring-fd <fd> --journal <dir>\n");
}

typedef struct drain_state {
    aotx_host_ring ring;
    aotx_segment_writer seg;
    aotx_derive derive;
    unsigned char *block;
    uint32_t block_bytes;
    uint64_t cursor;
    uint64_t expect;    /* the block sequence that comes next */
    uint64_t written;   /* the last block sequence that reached the disk */
    uint64_t gaps;
} drain_state;

/* Takes every block that the ring holds now. Returns the count taken, or -1 on a fault. */
static int drain_pass(drain_state *s)
{
    uint64_t head = aotx_host_ring_head(&s->ring);
    int taken = 0;
    for (;;) {
        aotx_take t;
        uint64_t backoff = 0;
        int status;
        int tries = 0;
        do {
            status = aotx_host_ring_take(&s->ring, s->cursor, head, s->block, s->block_bytes, &t);
            if (status != AOTX_TAKE_TORN) {
                break;
            }
            aotx_pause(&backoff);
            tries++;
        } while (tries < AOTX_TORN_TRIES);
        if (status == AOTX_TAKE_EMPTY || status == AOTX_TAKE_TORN) {
            return taken;
        }
        if (status == AOTX_TAKE_BAD) {
            fprintf(stderr, "drain: block at cursor %llu refused: %s\n",
                    (unsigned long long)s->cursor, t.reason);
            return -1;
        }
        if (t.block_seq != s->expect) {
            /* A gap means the device lost blocks. The count is reported and the drain goes
             * on, because the blocks that follow are still whole. */
            fprintf(stderr, "drain: block sequence gap, expected %llu and found %llu\n",
                    (unsigned long long)s->expect, (unsigned long long)t.block_seq);
            s->gaps++;
        }
        s->expect = t.block_seq + 1;
        if (t.kind != AOTX_BLOCK_PAD) {
            /* A pad block fills the tail of the ring and holds no record, so the journal
             * does not carry it. */
            if (aotx_segment_put(&s->seg, s->block, t.byte_len) != 0) {
                fprintf(stderr, "drain: the segment write did not finish\n");
                return -1;
            }
            if (aotx_derive_block(&s->derive, s->block) != 0) {
                fprintf(stderr, "drain: the derived write did not finish\n");
                return -1;
            }
            s->written = t.block_seq;
        }
        s->cursor += t.byte_len;
        taken++;
    }
}

static int run(drain_state *s)
{
    uint64_t backoff = 0;
    int ending = 0;
    for (;;) {
        int taken = drain_pass(s);
        if (taken < 0) {
            return AOTX_EXIT_FAULT;
        }
        if (taken > 0) {
            /* The cursor moves only after the bytes reach the disk. */
            if (aotx_segment_sync(&s->seg) != 0) {
                fprintf(stderr, "drain: the segment synchronize did not finish\n");
                return AOTX_EXIT_FAULT;
            }
            aotx_derive_sync(&s->derive, 0);
            aotx_host_ring_advance(&s->ring, s->cursor);
            backoff = 0;
            continue;
        }
        if (ending) {
            return AOTX_EXIT_OK;
        }
        if (aotx_host_ring_closed(&s->ring) || stop_flag != 0) {
            /* One more pass takes every block that the producer published before it ended. */
            ending = 1;
            continue;
        }
        aotx_pause(&backoff);
    }
}

int main(int argc, char **argv)
{
    aotx_map map;
    drain_state s;
    struct sigaction act;
    const char *journal = NULL;
    char boot_dir[AOTX_PATH_BYTES];
    int ring_fd = -1;
    int i;
    int rc;

    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--ring-fd") == 0 && i + 1 < argc) {
            ring_fd = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--journal") == 0 && i + 1 < argc) {
            journal = argv[++i];
        } else {
            usage();
            return AOTX_EXIT_FAULT;
        }
    }
    if (ring_fd < 0 || journal == NULL || strlen(journal) + 20 > AOTX_PATH_BYTES) {
        usage();
        return AOTX_EXIT_FAULT;
    }

    memset(&s, 0, sizeof(s));
    memset(&act, 0, sizeof(act));
    act.sa_handler = on_signal;
    sigaction(SIGTERM, &act, NULL);
    sigaction(SIGINT, &act, NULL);
    if (aotx_die_with_parent() != 0) {
        fprintf(stderr, "drain: the parent death signal is not set\n");
    }
    if (aotx_map_fd(ring_fd, &map) != 0) {
        fprintf(stderr, "drain: the ring descriptor does not map\n");
        return AOTX_EXIT_FAULT;
    }
    if (aotx_host_ring_attach(&map, &s.ring) != 0) {
        fprintf(stderr, "drain: the ring preamble does not match this layout version\n");
        return AOTX_EXIT_LAYOUT;
    }

    snprintf(boot_dir, sizeof(boot_dir), "%s/%016llx", journal,
             (unsigned long long)s.ring.pre->boot_id);
    if (make_path(journal) != 0 || aotx_make_dir(boot_dir) != 0) {
        fprintf(stderr, "drain: the journal directory does not open\n");
        return AOTX_EXIT_FAULT;
    }
    if (aotx_segment_open(&s.seg, boot_dir, AOTX_SEGMENT_LIMIT) != 0) {
        fprintf(stderr, "drain: the first segment does not open\n");
        return AOTX_EXIT_FAULT;
    }
    if (aotx_derive_open(&s.derive, journal, boot_dir) != 0) {
        fprintf(stderr, "drain: the derived files do not open\n");
        return AOTX_EXIT_FAULT;
    }
    s.block_bytes = (uint32_t)s.ring.data_bytes;
    s.block = (unsigned char *)malloc(s.block_bytes);
    if (s.block == NULL) {
        fprintf(stderr, "drain: the block buffer does not fit in memory\n");
        return AOTX_EXIT_FAULT;
    }
    s.cursor = aotx_host_ring_cursor(&s.ring);
    s.expect = 1;

    rc = run(&s);

    aotx_derive_close(&s.derive);
    if (aotx_segment_close(&s.seg) != 0) {
        rc = AOTX_EXIT_FAULT;
    }
    fprintf(stderr, "drain: blocks to %llu, gaps %llu, console %llu, notes %llu\n",
            (unsigned long long)s.written, (unsigned long long)s.gaps,
            (unsigned long long)s.derive.lines, (unsigned long long)s.derive.notes);
    free(s.block);
    aotx_map_release(&map);
    return rc;
}
