/* Purpose: Declare the host tool that reads a file under the allowed root for an agent.
 * Owns: Nothing; the caller holds the structure that this header declares.
 * Threading: One thread; the feeder is the only caller and the only producer of the ring.
 * Lifetime: From the open of the root to the close of the feeder. */
#ifndef AOTX_DISK_FS_TOOL_H
#define AOTX_DISK_FS_TOOL_H

#include "disk/wire/diskwire.h"

/* The import state that the feeder holds. The requests file can name the import tool. The
 * module directory then goes out as it goes out for a line of the standard input. */
struct aotx_import;

/* The bytes one read may take. A larger file gives the first bytes and a stated cut. The
 * figure is the device result buffer, which must hold the reply beside the prompt of the
 * turn that asked for it. */
#define AOTX_FS_CAP   4096u

/* The requests the table remembers, so a request is not executed twice. A restart starts
 * with an empty table, and the device refuses a reply for a request it already applied. */
#define AOTX_FS_SEEN  1024u

#define AOTX_FS_LINE  2048u
#define AOTX_FS_DEPTH 64      /* path components one path may hold */

/* ---- the path walk that is the boundary of every host side file operation ---- */

/* Bytes of a path that the walk takes, with the end byte. */
#define AOTX_WALK_BYTES 1024u

/* The last component of the path must name a directory. */
#define AOTX_WALK_DIR   1u
/* A component of two dots is refused, which refuses every path that leaves the base. */
#define AOTX_WALK_NO_UP 2u

typedef struct aotx_walk {
    char work[AOTX_WALK_BYTES]; /* the copy of the path that the walk cuts at each slash */
} aotx_walk;

/* Opens one path under a base directory. Every component is opened with O_NOFOLLOW, so a
 * symbolic link at any depth is refused. The state of the component gives the reason and
 * the O_NOFOLLOW gives the boundary. A link that comes between the two calls thus still
 * fails the open. The base descriptor stays open. Returns the descriptor of the last
 * component, or -1 with the status and the reason. */
int aotx_path_walk(aotx_walk *w, int base_fd, const char *path, unsigned flags,
                   uint32_t *status, const char **reason);

/* The reason the walk gives when a component of the path is not there. A caller compares
 * the reason it got with this one, so it can name a better cause in that one case. */
extern const char *const aotx_walk_absent;

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
    uint64_t imports;         /* requests that named the import tool */
    char requests_path[AOTX_PATH_BYTES];
    char line[AOTX_FS_LINE];
    unsigned char bytes[AOTX_FS_CAP];
} aotx_fs_tool;

/* Opens the allowed root and takes the name of the requests file. A requests file that is
 * already there is read from its end. A feeder that starts after a restore therefore
 * executes no request of the run before it. Returns 0, or -1 when the root does not open. */
int aotx_fs_tool_open(aotx_fs_tool *t, const char *root, const char *requests);

/* Takes the lines the requests file gained, executes them, and publishes the replies. A
 * line that names the import tool reads a module directory and publishes no reply. Returns
 * the count of requests taken, or -1 when the ring closed or the stop flag went to one. */
int aotx_fs_tool_poll(aotx_fs_tool *t, struct aotx_import *imports,
                      const aotx_inbound_ring *ring, const volatile sig_atomic_t *stop);

void aotx_fs_tool_close(aotx_fs_tool *t);

#endif
