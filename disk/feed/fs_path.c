/* Purpose: Open a path under a base directory, which is the boundary of every file tool.
 * Owns: Nothing; the caller holds the walk state.
 * Threading: One thread; the feeder is the only caller.
 * Lifetime: The call. */
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

int aotx_path_parent(int root_fd, const char *path, char *last, size_t last_bytes,
                     uint32_t *status, const char **reason)
{
    aotx_walk walk;
    char head[AOTX_WALK_BYTES];
    const char *cut = strrchr(path, '/');
    const char *name = (cut != NULL) ? cut + 1 : path;
    size_t head_len;
    *status = AOTX_TOOL_REFUSED;
    if (path[0] == '\0') {
        *reason = "the path is empty";
        return -1;
    }
    if (path[0] == '/') {
        *reason = "the path starts at the root of the file system";
        return -1;
    }
    if (name[0] == '\0' || strcmp(name, ".") == 0 || strcmp(name, "..") == 0) {
        *reason = "the path names no file";
        return -1;
    }
    if (strlen(name) >= last_bytes) {
        *reason = "the path is too long";
        return -1;
    }
    snprintf(last, last_bytes, "%s", name);
    if (cut == NULL) {
        /* The file stands in the root itself, which is already open. */
        int fd = dup(root_fd);
        if (fd < 0) {
            *status = AOTX_TOOL_ERROR;
            *reason = "the root does not open again";
            return -1;
        }
        *status = AOTX_TOOL_OK;
        return fd;
    }
    head_len = (size_t)(cut - path);
    if (head_len >= sizeof(head)) {
        *reason = "the path is too long";
        return -1;
    }
    memcpy(head, path, head_len);
    head[head_len] = '\0';
    return aotx_path_walk(&walk, root_fd, head, AOTX_WALK_NO_UP | AOTX_WALK_DIR, status,
                          reason);
}
