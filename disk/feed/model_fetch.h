/* Purpose: Run one model fetch child and turn its progress into input line notes.
 * Owns: The child process, its output pipe, and one part line.
 * Threading: One feeder loop; no function blocks for the child.
 * Lifetime: From initialization to close. */
#ifndef AOTX_FEED_MODEL_FETCH_H
#define AOTX_FEED_MODEL_FETCH_H

#include "disk/wire/diskwire.h"

typedef struct aotx_fetch_child {
    int pid;
    int fd;
    uint32_t fill;
    char line[512];
    char name[96];
    char directory[AOTX_PATH_BYTES];
    char program[AOTX_PATH_BYTES];
    uint64_t started;
    uint64_t ended;
    uint64_t refused;
} aotx_fetch_child;

void aotx_fetch_child_init(aotx_fetch_child *child, const char *directory);

/* Takes an input line. Returns 1 when it was a model fetch line, 0 when it was not, and
 * -1 when the ring closed while the refusal was published. */
int aotx_fetch_child_line(aotx_fetch_child *child, const unsigned char *line, uint32_t bytes,
                          const aotx_inbound_ring *ring,
                          const volatile sig_atomic_t *stop);

/* Takes available output and a completed child without waiting. */
int aotx_fetch_child_poll(aotx_fetch_child *child, const aotx_inbound_ring *ring,
                          const volatile sig_atomic_t *stop);

void aotx_fetch_child_close(aotx_fetch_child *child);

#endif
