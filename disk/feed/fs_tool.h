/* Purpose: Declare the host tools that the feeder runs for an agent under the allowed root.
 * Owns: Nothing; the caller holds the structure that this header declares.
 * Threading: One thread; the feeder is the only caller and the only producer of the ring.
 * Lifetime: From the open of the root to the close of the feeder. */
#ifndef AOTX_DISK_FS_TOOL_H
#define AOTX_DISK_FS_TOOL_H

#include "disk/wire/diskwire.h"

/* The import state that the feeder holds. The requests file can name the import tool. The
 * module directory then goes out as it goes out for a line of the standard input. */
struct aotx_import;

/* The bytes one reply may take. A larger result gives the first bytes and a stated cut. The
 * figure is the device result buffer, which must hold the reply beside the prompt of the
 * turn that asked for it. */
#define AOTX_FS_CAP   4096u

/* The requests the table remembers, so a request is not executed twice. A restart starts
 * with an empty table, and the device refuses a reply for a request it already applied. */
#define AOTX_FS_SEEN  1024u

#define AOTX_FS_LINE  2048u
#define AOTX_FS_DEPTH 64      /* path components one path may hold */

/* The bytes a file may hold for a text replacement. A larger file is refused, because the
 * tool must read the whole file to count the runs of the old text. */
#define AOTX_FS_FILE_CAP (1024u * 1024u)

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

/* Opens the directory that holds the last component of a path, and writes that component.
 * The walk rule of the whole path applies to the components before it. A last component of
 * one dot or two dots, and an empty one, are refused. Returns the descriptor of the
 * directory, or -1 with the status and the reason. */
int aotx_path_parent(int root_fd, const char *path, char *last, size_t last_bytes,
                     uint32_t *status, const char **reason);

/* ---- the state of the feeder's tools ---- */

struct aotx_children;
struct aotx_modules;

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
    /* The children and the module table are large, so they live beside the program and not
     * in this structure. A feeder with no program to run holds a null pointer here. */
    struct aotx_children *kids;
    struct aotx_modules *table;
    /* The seconds a built-in tool may run a command. A tool of the catalog takes the
     * figure of its manifest. Zero takes the default of the module table. */
    uint32_t timeout;
    char requests_path[AOTX_PATH_BYTES];
    char line[AOTX_FS_LINE];
    unsigned char bytes[AOTX_FS_CAP];
} aotx_fs_tool;

/* Opens the allowed root and takes the name of the requests file. A requests file that is
 * already there is read from its end. A feeder that starts after a restore therefore
 * executes no request of the run before it. Returns 0, or -1 when the root does not open. */
int aotx_fs_tool_open(aotx_fs_tool *t, const char *root, const char *requests);

/* Takes the lines the requests file gained, executes them, and publishes the replies. A
 * line that names the import tool reads a module directory and publishes no reply. The
 * call also takes the output of every program that still runs and answers the programs
 * that ended. Returns the count of requests taken, or -1 when the ring closed or the stop
 * flag went to one. */
int aotx_fs_tool_poll(aotx_fs_tool *t, struct aotx_import *imports,
                      const aotx_inbound_ring *ring, const volatile sig_atomic_t *stop);

void aotx_fs_tool_close(aotx_fs_tool *t);

/* ---- the reply, which every tool of the feeder publishes through these ---- */

/* Publishes one reply record. Returns 0, or -1 when the ring closed or a signal arrived. */
int aotx_fs_put_part(aotx_fs_tool *t, const aotx_inbound_ring *ring,
                     const volatile sig_atomic_t *stop, uint32_t agent, uint32_t request,
                     uint32_t status, uint32_t part, uint32_t parts, const void *data,
                     uint32_t len);

/* Publishes the reply of a request that gives no bytes. The reply is one part, and its
 * bytes are the reason. */
int aotx_fs_put_reason(aotx_fs_tool *t, const aotx_inbound_ring *ring,
                       const volatile sig_atomic_t *stop, uint32_t agent, uint32_t request,
                       uint32_t status, const char *reason);

/* Publishes bytes in parts. A part with the status ok carries content. A part with any
 * other status is the last part of the reply and its bytes are a reason. A reason that is
 * not null therefore adds one part after the content. */
int aotx_fs_put_bytes(aotx_fs_tool *t, const aotx_inbound_ring *ring,
                      const volatile sig_atomic_t *stop, uint32_t agent, uint32_t request,
                      const unsigned char *data, uint32_t len, const char *reason);

/* ---- the tools that write, which live in a file of their own ---- */

/* Writes the text to a temporary name in the directory of the path and renames it over the
 * path. The tool creates or replaces a file and refuses a directory. Returns 0, or -1 when
 * the ring closed. */
int aotx_fs_write_file(aotx_fs_tool *t, const aotx_inbound_ring *ring,
                       const volatile sig_atomic_t *stop, uint32_t agent, uint32_t request,
                       const char *path, const char *text);

/* Replaces one run of the old text with the new text in a file under the root. The tool
 * refuses a file in which the old text occurs zero times or more than one time, and states
 * the count. Returns 0, or -1 when the ring closed. */
int aotx_fs_update_file(aotx_fs_tool *t, const aotx_inbound_ring *ring,
                        const volatile sig_atomic_t *stop, uint32_t agent, uint32_t request,
                        const char *path, const char *old_text, const char *new_text);

#endif
