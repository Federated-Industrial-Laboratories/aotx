/* Purpose: Declare the read of a journal boot directory and the choice of the newest one.
 * Owns: Nothing; the caller holds the result structure and the block buffer.
 * Threading: One thread; the scan reads files and holds no state between calls.
 * Lifetime: The call. */
#ifndef AOTX_DISK_SCAN_H
#define AOTX_DISK_SCAN_H

#include "disk/wire/diskwire.h"

typedef struct aotx_journal_scan {
    uint64_t boot_id;
    uint64_t boot_wall_ns;     /* the wall clock of the boot record, or zero */
    uint64_t rank;             /* the value that orders one boot against another */
    uint64_t blocks;           /* blocks that read without a fault */
    uint64_t commits;          /* tick commit records found */
    uint64_t last_tick;        /* the tick of the last complete tick */
    uint64_t last_block;       /* the index of the block that holds that tick commit */
    uint64_t state_hash;       /* the state hash of that tick commit */
    uint64_t restore_hash;     /* the state hash of the newest restore record */
    uint64_t restore_of;       /* the boot that the newest restore record replayed */
    int has_restore;
    int torn;                  /* one when a frame did not verify */
    char dir[AOTX_PATH_BYTES];
} aotx_journal_scan;

/* Calls fn for each block of a boot directory, in read order. A frame that does not verify
 * ends the readable journal, and torn goes to one. Returns 0 or -1.
 *
 * The function returns 0 to go on, 1 to stop the walk without a fault, or -1 on a fault. */
typedef int (*aotx_block_fn)(void *ctx, const unsigned char *block, uint64_t index);
int aotx_journal_walk(const char *dir, unsigned char *buffer, uint32_t bytes,
                      aotx_block_fn fn, void *ctx, uint64_t *blocks, int *torn);

/* Reads one boot directory and fills the result. Returns 0 or -1. */
int aotx_journal_read(const char *dir, uint64_t boot_id, unsigned char *buffer, uint32_t bytes,
                      aotx_journal_scan *out);

/* Finds the newest boot directory that holds at least one complete tick. Returns 0, or -1
 * when the journal holds no such directory. */
int aotx_journal_latest(const char *journal, unsigned char *buffer, uint32_t bytes,
                        aotx_journal_scan *out);

#endif
