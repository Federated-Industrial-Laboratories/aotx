/* Purpose: Read a file under the allowed root for an agent and publish the reply in parts.
 * Owns: The descriptor of the root, the descriptor of the requests file, and the table of
 *       requests that were executed.
 * Threading: One thread; the feeder is the only caller and the only producer of the ring.
 * Lifetime: From the open of the root to the close of the feeder. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "disk/feed/fs_tool.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

/* The allowed root is the security boundary of the system. A path that leaves the root is
 * refused. A path that goes through a symbolic link is refused. A path that names anything
 * other than a regular file is refused. The reason goes back to the agent. */

/* Reports whether the table already holds the identity. */
static int already(const aotx_fs_tool *t, uint32_t request)
{
    uint32_t i;
    for (i = 0; i < AOTX_FS_SEEN; i++) {
        if (t->seen[i] == (uint64_t)request + 1u) {
            return 1;
        }
    }
    return 0;
}

static void remember(aotx_fs_tool *t, uint32_t request)
{
    t->seen[t->seen_at % AOTX_FS_SEEN] = (uint64_t)request + 1u;
    t->seen_at++;
}

const char *const aotx_walk_absent = "the file is not there";

int aotx_path_walk(aotx_walk *w, int base_fd, const char *path, unsigned flags,
                   uint32_t *status, const char **reason)
{
    struct stat state;
    char *at;
    int dir_fd;
    int depth = 0;
    *status = AOTX_TOOL_REFUSED;
    if (strlen(path) >= sizeof(w->work)) {
        *reason = "the path is too long";
        return -1;
    }
    snprintf(w->work, sizeof(w->work), "%s", path);
    dir_fd = dup(base_fd);
    if (dir_fd < 0) {
        *status = AOTX_TOOL_ERROR;
        *reason = "the base directory does not open again";
        return -1;
    }
    at = w->work;
    while (at != NULL) {
        char *end = strchr(at, '/');
        int last;
        int fd;
        if (end != NULL) {
            *end = '\0';
        }
        last = (end == NULL);
        if ((flags & AOTX_WALK_NO_UP) != 0 && strcmp(at, "..") == 0) {
            close(dir_fd);
            *reason = "the path holds a component of two dots";
            return -1;
        }
        if (at[0] == '\0' || strcmp(at, ".") == 0) {
            /* A part of the path that names the same directory moves nothing. */
            at = (end != NULL) ? end + 1 : NULL;
            continue;
        }
        if (++depth > AOTX_FS_DEPTH) {
            close(dir_fd);
            *reason = "the path holds too many components";
            return -1;
        }
        /* The state of the component names the refusal. The open that follows carries
         * O_NOFOLLOW, which is the guard: a link that comes between these two calls still
         * fails the open. This call gives the reason and not the boundary. */
        if (fstatat(dir_fd, at, &state, AT_SYMLINK_NOFOLLOW) == 0 && S_ISLNK(state.st_mode)) {
            close(dir_fd);
            *reason = "a component of the path is a symbolic link";
            return -1;
        }
        /* O_NONBLOCK keeps the open of a named pipe from holding the caller. A regular
         * file reads the same with it. */
        fd = openat(dir_fd, at, O_RDONLY | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK |
                                ((!last || (flags & AOTX_WALK_DIR) != 0) ? O_DIRECTORY : 0));
        if (fd < 0) {
            close(dir_fd);
            if (errno == ELOOP) {
                *reason = "a component of the path is a symbolic link";
                return -1;
            }
            *status = AOTX_TOOL_ERROR;
            *reason = (errno == ENOENT) ? aotx_walk_absent : "the path does not open";
            return -1;
        }
        close(dir_fd);
        dir_fd = fd;
        at = (end != NULL) ? end + 1 : NULL;
    }
    if (depth == 0) {
        close(dir_fd);
        *reason = "the path names no file";
        return -1;
    }
    *status = AOTX_TOOL_OK;
    return dir_fd;
}

/* Opens one file under the allowed root. The root is the boundary. A path that starts at
 * the root of the file system is refused before the walk. A path that holds a component of
 * two dots is refused there too. Returns the descriptor, or -1 with the status and the
 * reason. */
static int open_under(int root_fd, const char *path, uint32_t *status, const char **reason)
{
    aotx_walk walk;
    *status = AOTX_TOOL_REFUSED;
    if (path[0] == '\0') {
        *reason = "the path is empty";
        return -1;
    }
    if (path[0] == '/') {
        *reason = "the path starts at the root of the file system";
        return -1;
    }
    if (strlen(path) > AOTX_TOOL_ARG_BYTES) {
        *reason = "the path is too long";
        return -1;
    }
    return aotx_path_walk(&walk, root_fd, path, AOTX_WALK_NO_UP, status, reason);
}

/* Publishes one reply record. Returns 0, or -1 when the ring closed or a signal arrived. */
static int put_part(aotx_fs_tool *t, const aotx_inbound_ring *ring,
                    const volatile sig_atomic_t *stop, uint32_t agent, uint32_t request,
                    uint32_t status, uint32_t part, uint32_t parts, const void *data,
                    uint32_t len)
{
    aotx_record_header h;
    aotx_tool_reply_body body;
    memset(&h, 0, sizeof(h));
    memset(&body, 0, sizeof(body));
    h.writer = AOTX_WRITER_FEEDER;
    h.cls = AOTX_CLASS_A;
    h.type = AOTX_REC_TOOL_REPLY;
    h.body_len = (uint32_t)sizeof(body);
    body.agent = agent;
    body.request = request;
    body.status = status;
    body.part = part;
    body.parts = parts;
    body.len = (len > AOTX_TOOL_REPLY_BYTES) ? AOTX_TOOL_REPLY_BYTES : len;
    if (body.len > 0) {
        memcpy(body.bytes, data, body.len);
    }
    if (aotx_inbound_wait(ring, stop) != 0) {
        return -1;
    }
    aotx_inbound_put(ring, &h, &body);
    t->replies++;
    return 0;
}

/* Publishes the reply of a request that gave no bytes. The reply is one part, and its
 * bytes are the reason. */
static int put_reason(aotx_fs_tool *t, const aotx_inbound_ring *ring,
                      const volatile sig_atomic_t *stop, uint32_t agent, uint32_t request,
                      uint32_t status, const char *reason)
{
    if (status == AOTX_TOOL_REFUSED) {
        t->refusals++;
    } else {
        t->errors++;
    }
    return put_part(t, ring, stop, agent, request, status, 0u, 1u, reason,
                    (uint32_t)strlen(reason));
}

/* Publishes the bytes of a file in parts. A part with the status ok carries content. A
 * part with any other status is the last part of the reply and its bytes are the reason.
 * A file that the cap cut therefore states the cut in a part of its own. */
static int put_bytes(aotx_fs_tool *t, const aotx_inbound_ring *ring,
                     const volatile sig_atomic_t *stop, uint32_t agent, uint32_t request,
                     uint32_t len, int cut)
{
    uint32_t parts = (len + AOTX_TOOL_REPLY_BYTES - 1u) / AOTX_TOOL_REPLY_BYTES;
    uint32_t i;
    if (parts == 0) {
        parts = 1;
    }
    if (cut) {
        parts++;
        t->errors++;
    }
    for (i = 0; i + (uint32_t)(cut ? 1 : 0) < parts; i++) {
        uint32_t at = i * AOTX_TOOL_REPLY_BYTES;
        uint32_t take = len - at;
        if (take > AOTX_TOOL_REPLY_BYTES) {
            take = AOTX_TOOL_REPLY_BYTES;
        }
        if (put_part(t, ring, stop, agent, request, AOTX_TOOL_OK, i, parts, t->bytes + at,
                     take) != 0) {
            return -1;
        }
    }
    if (cut) {
        /* The count comes from the cap itself, so the reason cannot state a figure that
         * the cap does not hold. */
        char reason[96];
        int len = snprintf(reason, sizeof(reason),
                           "the file is longer than the cap and the reply holds the first"
                           " %u bytes", (unsigned)AOTX_FS_CAP);
        return put_part(t, ring, stop, agent, request, AOTX_TOOL_ERROR, parts - 1u, parts,
                        reason, (uint32_t)len);
    }
    return 0;
}

/* Reads one file under the root and publishes its reply. Returns 0 or -1. */
static int read_file(aotx_fs_tool *t, const aotx_inbound_ring *ring,
                     const volatile sig_atomic_t *stop, uint32_t agent, uint32_t request,
                     const char *path)
{
    struct stat info;
    const char *reason = "";
    uint32_t status = AOTX_TOOL_OK;
    uint32_t got = 0;
    int cut = 0;
    int fd = open_under(t->root_fd, path, &status, &reason);
    if (fd < 0) {
        return put_reason(t, ring, stop, agent, request, status, reason);
    }
    if (fstat(fd, &info) != 0 || !S_ISREG(info.st_mode)) {
        close(fd);
        return put_reason(t, ring, stop, agent, request, AOTX_TOOL_REFUSED,
                          "the path does not name a regular file");
    }
    while (got < AOTX_FS_CAP) {
        ssize_t n = read(fd, t->bytes + got, (size_t)(AOTX_FS_CAP - got));
        if (n < 0) {
            close(fd);
            return put_reason(t, ring, stop, agent, request, AOTX_TOOL_ERROR,
                              "the file does not read");
        }
        if (n == 0) {
            break;
        }
        got += (uint32_t)n;
    }
    if (got == AOTX_FS_CAP) {
        unsigned char more;
        ssize_t n = read(fd, &more, 1);
        cut = (n > 0);
    }
    close(fd);
    return put_bytes(t, ring, stop, agent, request, got, cut);
}

/* Takes one line of the requests file. Returns 0 or -1. */
static int take_line(aotx_fs_tool *t, const aotx_inbound_ring *ring,
                     const volatile sig_atomic_t *stop)
{
    char tool[32];
    char path[AOTX_TOOL_ARG_BYTES + 1];
    uint64_t request = 0;
    uint64_t agent = 0;
    if (!aotx_json_number(t->line, "\"request\":", &request) || request == 0 ||
        !aotx_json_number(t->line, "\"agent\":", &agent) ||
        !aotx_json_text(t->line, "\"tool\":\"", tool, sizeof(tool)) ||
        !aotx_json_text(t->line, "\"arg\":\"", path, sizeof(path))) {
        /* A line the reader cannot read names no request, so no reply can name it. */
        t->errors++;
        return 0;
    }
    if (already(t, (uint32_t)request)) {
        /* One request is executed one time. A file read has an effect on the host, and a
         * second read of it is a second effect. */
        t->again++;
        return 0;
    }
    remember(t, (uint32_t)request);
    t->taken++;
    if (strcmp(tool, "fs_read") != 0) {
        return put_reason(t, ring, stop, (uint32_t)agent, (uint32_t)request, AOTX_TOOL_ERROR,
                          "the tool is not fs_read");
    }
    return read_file(t, ring, stop, (uint32_t)agent, (uint32_t)request, path);
}

int aotx_fs_tool_open(aotx_fs_tool *t, const char *root, const char *requests)
{
    memset(t, 0, sizeof(*t));
    t->root_fd = -1;
    t->requests_fd = -1;
    if (root == NULL || requests == NULL) {
        return 0;
    }
    t->root_fd = open(root, O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (t->root_fd < 0) {
        return -1;
    }
    snprintf(t->requests_path, sizeof(t->requests_path), "%s", requests);
    t->requests_fd = open(t->requests_path, O_RDONLY | O_CLOEXEC);
    if (t->requests_fd >= 0 && lseek(t->requests_fd, 0, SEEK_END) < 0) {
        return -1;
    }
    return 0;
}

int aotx_fs_tool_poll(aotx_fs_tool *t, const aotx_inbound_ring *ring,
                      const volatile sig_atomic_t *stop)
{
    unsigned char buffer[4096];
    int taken = 0;
    if (t->root_fd < 0) {
        return 0;
    }
    if (t->requests_fd < 0) {
        t->requests_fd = open(t->requests_path, O_RDONLY | O_CLOEXEC);
        if (t->requests_fd < 0) {
            return 0;
        }
    }
    for (;;) {
        ssize_t n = read(t->requests_fd, buffer, sizeof(buffer));
        ssize_t i;
        if (n <= 0) {
            return taken;
        }
        for (i = 0; i < n; i++) {
            if (buffer[i] != '\n') {
                if (t->fill + 1u < AOTX_FS_LINE) {
                    t->line[t->fill++] = (char)buffer[i];
                } else {
                    /* A line longer than the buffer is not a line this reader wrote. */
                    t->over = 1;
                }
                continue;
            }
            t->line[t->fill] = '\0';
            if (!t->over) {
                if (take_line(t, ring, stop) != 0) {
                    return -1;
                }
                taken++;
            } else {
                t->errors++;
            }
            t->fill = 0;
            t->over = 0;
        }
    }
}

void aotx_fs_tool_close(aotx_fs_tool *t)
{
    if (t->root_fd >= 0) {
        close(t->root_fd);
    }
    if (t->requests_fd >= 0) {
        close(t->requests_fd);
    }
    t->root_fd = -1;
    t->requests_fd = -1;
}
