/* Purpose: Declare the reader of the bulk ring, which holds payloads that no record carries.
 * Owns: Nothing; the caller holds the state structure.
 * Threading: One thread; the drain reads the bulk ring between journal passes.
 * Lifetime: From open to close, which is the run of the drain. */
#ifndef AOTX_DISK_BULK_H
#define AOTX_DISK_BULK_H

#include "disk/wire/diskwire.h"

typedef struct aotx_bulk {
    aotx_map map;
    aotx_host_ring ring;
    unsigned char *block;
    uint32_t block_bytes;
    uint64_t cursor;
    uint64_t expect;    /* the block sequence that comes next */
    uint64_t files;     /* payloads written */
    uint64_t bytes;     /* payload bytes written */
    uint64_t gaps;      /* block sequences that the ring did not give */
    int index_fd;
    int on;             /* one when the drain writes the files */
    int open;           /* one when the ring is attached */
    char dir[AOTX_PATH_BYTES];
} aotx_bulk;

/* Maps the ring of the descriptor and opens the payload directory and its index. The drain
 * reads the ring even when it writes no file, or the ring fills and the device holds.
 * Returns 0, -1 on a fault, or -2 when the preamble names another layout version. */
int aotx_bulk_open(aotx_bulk *b, const char *journal, int fd, int on);

/* Takes every payload that the ring holds now. Returns the count taken, or -1. */
int aotx_bulk_pass(aotx_bulk *b);

int aotx_bulk_closed(const aotx_bulk *b);

void aotx_bulk_close(aotx_bulk *b);

#endif
