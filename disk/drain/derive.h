/* Purpose: Declare the derived files that the drain writes beside the journal segments.
 * Owns: Nothing; the caller holds the state structure.
 * Threading: One thread; the drain calls these functions in block order.
 * Lifetime: From open to close, which is the run of the drain. */
#ifndef AOTX_DISK_DERIVE_H
#define AOTX_DISK_DERIVE_H

#include "disk/wire/diskwire.h"

typedef struct aotx_derive {
    int console_fd;
    int bus_fd;
    int echo_fd;              /* the operator terminal, which sees every console record */
    uint64_t bus_seq;         /* the sequence of the next bus line, from one */
    uint64_t tick_start_ns;   /* the wall clock of the newest tick start record */
    uint64_t sync_ns;         /* the wall clock of the last synchronize call */
    uint64_t lines;           /* console records written */
    uint64_t notes;           /* bus lines written */
    char bus_dir[AOTX_PATH_BYTES];
    char bus_date[16];
} aotx_derive;

/* Opens the console log in the boot directory and the bus file in the journal. The derived
 * files are outputs only, and no program reads them back as inputs. */
int aotx_derive_open(aotx_derive *d, const char *journal, const char *boot_dir);

/* Writes the derived lines of one block. Returns 0 or -1. */
int aotx_derive_block(aotx_derive *d, const unsigned char *block);

/* Synchronizes the derived files when more than one second passed, or when force is one. */
int aotx_derive_sync(aotx_derive *d, int force);

void aotx_derive_close(aotx_derive *d);

#endif
