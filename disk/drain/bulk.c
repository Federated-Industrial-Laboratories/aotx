/* Purpose: Write the payloads of the bulk ring to files, with a checksum for each one.
 * Owns: The payload directory, the index file, and the consumer cursor of the bulk ring.
 * Threading: One thread; the program is the only writer of that cursor.
 * Lifetime: From the map of the ring to the exit of the program. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "disk/drain/bulk.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* A payload is too large for a record body, so it crosses the seam in a block of its own.
 * The block header gives the handle in first_seq and holds no record. The record in the
 * journal gives the same handle and the exact length. */

#define AOTX_TORN_TRIES 64

/* The columns of the index, so a person can read the file without another document. */
static const char index_header[] = "handle\ttick\tlength\tcrc\n";

static int put_all(int fd, const char *data, size_t bytes)
{
    size_t done = 0;
    while (done < bytes) {
        ssize_t n = write(fd, data + done, bytes - done);
        if (n <= 0) {
            return -1;
        }
        done += (size_t)n;
    }
    return 0;
}

int aotx_bulk_open(aotx_bulk *b, const char *journal, int fd, int on)
{
    char path[AOTX_PATH_BYTES + 32];
    off_t end;
    memset(b, 0, sizeof(*b));
    b->index_fd = -1;
    b->on = on;
    b->expect = 1;
    if (aotx_map_fd(fd, &b->map) != 0) {
        return -1;
    }
    if (aotx_host_ring_attach(&b->map, &b->ring) != 0) {
        return -2;
    }
    b->open = 1;
    b->cursor = aotx_host_ring_cursor(&b->ring);
    b->block_bytes = (uint32_t)b->ring.data_bytes;
    b->block = (unsigned char *)malloc(b->block_bytes);
    if (b->block == NULL) {
        return -1;
    }
    if (!on) {
        return 0;
    }
    snprintf(b->dir, sizeof(b->dir), "%s/bulk", journal);
    if (aotx_make_dir(b->dir) != 0) {
        return -1;
    }
    snprintf(path, sizeof(path), "%s/index.tsv", b->dir);
    b->index_fd = open(path, O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (b->index_fd < 0) {
        return -1;
    }
    end = lseek(b->index_fd, 0, SEEK_END);
    if (end == 0 && put_all(b->index_fd, index_header, sizeof(index_header) - 1) != 0) {
        return -1;
    }
    return 0;
}

/* Writes one payload to its own file and adds one row to the index. The name of the file is
 * the handle, so the record in the journal gives the file that holds the payload. */
static int write_payload(aotx_bulk *b, const aotx_take *t)
{
    char path[AOTX_PATH_BYTES + 32];
    char row[128];
    const unsigned char *payload = b->block + AOTX_BLOCK_HEADER_BYTES;
    uint32_t length = t->byte_len - AOTX_BLOCK_HEADER_BYTES;
    uint32_t crc = aotx_crc32c(payload, length, 0);
    int fd;
    int used;
    snprintf(path, sizeof(path), "%s/%016llx", b->dir, (unsigned long long)t->first_seq);
    fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        return -1;
    }
    if (put_all(fd, (const char *)payload, length) != 0 || fsync(fd) != 0) {
        close(fd);
        return -1;
    }
    close(fd);
    used = snprintf(row, sizeof(row), "%016llx\t%llu\t%u\t%08x\n",
                    (unsigned long long)t->first_seq, (unsigned long long)t->tick, length, crc);
    if (used < 0 || (size_t)used >= sizeof(row)) {
        return -1;
    }
    if (put_all(b->index_fd, row, (size_t)used) != 0 || fsync(b->index_fd) != 0) {
        return -1;
    }
    b->files++;
    b->bytes += length;
    return 0;
}

int aotx_bulk_pass(aotx_bulk *b)
{
    uint64_t head;
    int taken = 0;
    if (!b->open) {
        return 0;
    }
    head = aotx_host_ring_head(&b->ring);
    for (;;) {
        aotx_take t;
        uint64_t backoff = 0;
        int status;
        int tries = 0;
        do {
            status = aotx_host_ring_take(&b->ring, b->cursor, head, b->block, b->block_bytes, &t);
            if (status != AOTX_TAKE_TORN) {
                break;
            }
            aotx_pause(&backoff);
            tries++;
        } while (tries < AOTX_TORN_TRIES);
        if (status == AOTX_TAKE_EMPTY || status == AOTX_TAKE_TORN) {
            break;
        }
        if (status == AOTX_TAKE_BAD) {
            fprintf(stderr, "drain: bulk block at cursor %llu refused: %s\n",
                    (unsigned long long)b->cursor, t.reason);
            return -1;
        }
        if (t.kind != AOTX_BLOCK_BULK && t.kind != AOTX_BLOCK_PAD) {
            /* Only the journal ring carries records. A block of records on this ring is a
             * fault of the producer, and the drain does not write it to a file. */
            fprintf(stderr, "drain: a block of kind %u is not a payload\n", (unsigned)t.kind);
            return -1;
        }
        if (t.block_seq != b->expect) {
            fprintf(stderr, "drain: bulk sequence gap, expected %llu and found %llu\n",
                    (unsigned long long)b->expect, (unsigned long long)t.block_seq);
            b->gaps++;
        }
        b->expect = t.block_seq + 1;
        if (t.kind == AOTX_BLOCK_BULK && b->on && write_payload(b, &t) != 0) {
            fprintf(stderr, "drain: the payload write did not finish\n");
            return -1;
        }
        b->cursor += t.byte_len;
        taken++;
    }
    if (taken > 0) {
        /* The cursor moves only after the bytes reach the disk. */
        aotx_host_ring_advance(&b->ring, b->cursor);
    }
    return taken;
}

int aotx_bulk_closed(const aotx_bulk *b)
{
    return (!b->open) ? 1 : aotx_host_ring_closed(&b->ring);
}

void aotx_bulk_close(aotx_bulk *b)
{
    if (b->index_fd >= 0) {
        close(b->index_fd);
        b->index_fd = -1;
    }
    free(b->block);
    b->block = NULL;
    if (b->map.base != NULL) {
        aotx_map_release(&b->map);
    }
    b->open = 0;
}
