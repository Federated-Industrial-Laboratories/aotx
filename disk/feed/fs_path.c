/* Purpose: Open a path under a base directory, and split the arguments of one tool call.
 * Owns: Nothing; the caller holds the walk state and the argument table.
 * Threading: One thread; the feeder is the only caller.
 * Lifetime: The call.
 *
 * The walk is the security boundary of every file operation of the host tools. The
 * arguments are the other side of one call. The device writes the keys of the tool call
 * into one text field, and this file gives them back as pairs. */
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

/* The reason of a refusal that names a key of the call. The text lives beside the program,
 * because a caller reads the reason after the function returns. */
static char key_reason[128];

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

/* ---- the arguments of one call ---- */

/* Reports whether the tool names the key. */
static int named(const char *key, const char *const *keys, uint32_t count)
{
    uint32_t i;
    for (i = 0; i < count; i++) {
        if (strcmp(key, keys[i]) == 0) {
            return 1;
        }
    }
    return 0;
}

const char *aotx_args_value(const aotx_args *a, const char *key)
{
    uint32_t i;
    for (i = 0; i < a->count; i++) {
        if (strcmp(a->key[i], key) == 0) {
            return a->value[i];
        }
    }
    return NULL;
}

int aotx_args_split(aotx_args *a, const char *arg, const char *const *keys, uint32_t count,
                    const char **reason)
{
    size_t len = strlen(arg);
    char *at;
    memset(a, 0, sizeof(*a));
    if (len >= sizeof(a->work)) {
        *reason = "the arguments are longer than a request body holds";
        return 0;
    }
    memcpy(a->work, arg, len + 1u);
    if (count == 0) {
        *reason = "the tool takes no argument";
        return 0;
    }
    /* A text that does not start with the separator byte is one value, which is the shape
     * of a call with one argument. The separator marks the key and value shape, so a value
     * that holds an equal sign cannot be read as a key. */
    if (a->work[0] != AOTX_ARG_SEPARATOR) {
        snprintf(a->key[0], sizeof(a->key[0]), "%s", keys[0]);
        a->value[0] = a->work;
        a->count = 1;
        return 1;
    }
    at = a->work + 1;
    while (at != NULL) {
        char *end = strchr(at, AOTX_ARG_SEPARATOR);
        char *equal;
        if (end != NULL) {
            *end = '\0';
        }
        if (at[0] != '\0') {
            if (a->count >= AOTX_ARG_MAX) {
                *reason = "the call holds more arguments than a tool takes";
                return 0;
            }
            equal = strchr(at, '=');
            if (equal == NULL) {
                *reason = "an argument holds no key and no equal sign";
                return 0;
            }
            *equal = '\0';
            if (strlen(at) >= AOTX_ARG_KEY_BYTES) {
                *reason = "an argument key is too long";
                return 0;
            }
            if (!named(at, keys, count)) {
                snprintf(key_reason, sizeof(key_reason),
                         "the argument key %.64s is not a key of this tool", at);
                *reason = key_reason;
                return 0;
            }
            snprintf(a->key[a->count], sizeof(a->key[0]), "%s", at);
            a->value[a->count] = equal + 1;
            a->count++;
        }
        at = (end != NULL) ? end + 1 : NULL;
    }
    return 1;
}
