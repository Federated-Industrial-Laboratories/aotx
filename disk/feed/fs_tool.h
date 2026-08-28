/* Purpose: Declare the host tool that reads a file under the allowed root for an agent.
 * Owns: Nothing; the caller holds the structure that this header declares.
 * Threading: One thread; the feeder is the only caller and the only producer of the ring.
 * Lifetime: From the open of the root to the close of the feeder. */
#ifndef AOTX_DISK_FS_TOOL_H
#define AOTX_DISK_FS_TOOL_H

#include "disk/wire/diskwire.h"

/* The bytes one read may take. A larger file gives the first bytes and a stated cut. The
 * figure is the device result buffer, which must hold the reply beside the prompt of the
 * turn that asked for it. */
#define AOTX_FS_CAP   4096u

/* The requests the table remembers, so a request is not executed twice. A restart starts
 * with an empty table, and the device refuses a reply for a request it already applied. */
#define AOTX_FS_SEEN  1024u

#define AOTX_FS_LINE  2048u
#define AOTX_FS_DEPTH 64      /* path components one path may hold */

typedef struct aotx_fs_tool {
    int root_fd;              /* the allowed root, or -1 when no root was given */
    int requests_fd;          /* the requests file, or -1 while the file is not there */
    int over;                 /* one while a line longer than the buffer is dropped */
    uint32_t fill;            /* bytes of a line that a read did not complete */
    uint64_t seen[AOTX_FS_SEEN]; /* the identity and one, or zero for a free place */
    uint64_t seen_at;         /* the next place in the table */
    uint64_t taken;           /* requests executed */
    uint64_t replies;         /* reply records published */
    uint64_t again;           /* requests the table already held */
    uint64_t refusals;        /* requests the root rule refused */
    uint64_t errors;          /* requests that gave an error */
    char requests_path[AOTX_PATH_BYTES];
    char line[AOTX_FS_LINE];
    unsigned char bytes[AOTX_FS_CAP];
} aotx_fs_tool;

/* Opens the allowed root and takes the name of the requests file. A requests file that is
 * already there is read from its end. A feeder that starts after a restore therefore
 * executes no request of the run before it. Returns 0, or -1 when the root does not open. */
int aotx_fs_tool_open(aotx_fs_tool *t, const char *root, const char *requests);

/* Takes the lines the requests file gained, executes them, and publishes the replies.
 * Returns the count of requests taken, or -1 when the ring closed or the stop flag went
 * to one. */
int aotx_fs_tool_poll(aotx_fs_tool *t, const aotx_inbound_ring *ring,
                      const volatile sig_atomic_t *stop);

void aotx_fs_tool_close(aotx_fs_tool *t);

#endif
