/* Purpose: Write and update a file under the allowed root for an agent.
 * Owns: The bytes of the file under update, which live beside the program.
 * Threading: One thread; the feeder is the only caller and the only producer of the ring.
 * Lifetime: The call; no descriptor stays open after a reply. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "disk/feed/fs_tool.h"

#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

/* The bytes of the file under update. The buffer lives beside the program, because a file
 * of one megabyte must not stand on the stack of the poll loop. */
static unsigned char file_bytes[AOTX_FS_FILE_CAP];

/* The reason of a refusal that states a figure. */
static char write_reason[192];

/* Writes the temporary name that the new content takes before the rename. The name holds
 * the process and the request, so two requests cannot take one name. */
static void temp_name(char *out, size_t bytes, uint32_t request)
{
    snprintf(out, bytes, ".aotx-%u-%u", (unsigned)getpid(), request);
}

/* Opens the directory of the path and gives the last component. A directory that is not
 * there gives a reason that names the directory. Returns the descriptor, or -1. */
static int open_parent(int root_fd, const char *path, char *last, size_t last_bytes,
                       uint32_t *status, const char **reason)
{
    int fd;
    if (strlen(path) > AOTX_TOOL_ARG_BYTES) {
        *status = AOTX_TOOL_REFUSED;
        *reason = "the path is too long";
        return -1;
    }
    fd = aotx_path_parent(root_fd, path, last, last_bytes, status, reason);
    if (fd < 0 && *reason == aotx_walk_absent) {
        *reason = "the directory of the path is not there";
    }
    return fd;
}

/* Refuses a path that names a directory or a symbolic link. The tool replaces a file and
 * must not put a file over either of them. Returns 0, or 1 with the reason. */
static int stands_in_the_way(int dir_fd, const char *last, const char **reason)
{
    struct stat info;
    if (fstatat(dir_fd, last, &info, AT_SYMLINK_NOFOLLOW) != 0) {
        return 0;
    }
    if (S_ISDIR(info.st_mode)) {
        *reason = "the path names a directory";
        return 1;
    }
    if (S_ISLNK(info.st_mode)) {
        *reason = "the path names a symbolic link";
        return 1;
    }
    if (!S_ISREG(info.st_mode)) {
        *reason = "the path does not name a regular file";
        return 1;
    }
    return 0;
}

/* Writes the content to a temporary name in the directory and renames it over the last
 * component. The content comes in three runs, so a replacement needs no second buffer.
 * Returns 0, or 1 with the reason. */
static int put_file(int dir_fd, const char *last, uint32_t request,
                    const void *head, size_t head_len, const void *middle, size_t mid_len,
                    const void *tail, size_t tail_len, const char **reason)
{
    char temp[64];
    const void *runs[3];
    size_t lengths[3];
    int i;
    int fd;
    temp_name(temp, sizeof(temp), request);
    /* A name that an earlier call left behind must not stop this one. */
    unlinkat(dir_fd, temp, 0);
    fd = openat(dir_fd, temp, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0644);
    if (fd < 0) {
        *reason = "the temporary file does not open";
        return 1;
    }
    runs[0] = head;
    runs[1] = middle;
    runs[2] = tail;
    lengths[0] = head_len;
    lengths[1] = mid_len;
    lengths[2] = tail_len;
    for (i = 0; i < 3; i++) {
        const unsigned char *at = (const unsigned char *)runs[i];
        size_t left = lengths[i];
        while (left > 0) {
            ssize_t n = write(fd, at, left);
            if (n <= 0) {
                close(fd);
                unlinkat(dir_fd, temp, 0);
                *reason = "the temporary file does not write";
                return 1;
            }
            at += (size_t)n;
            left -= (size_t)n;
        }
    }
    close(fd);
    /* The rename is one step, so a reader sees the file it had or the file it gets. */
    if (renameat(dir_fd, temp, dir_fd, last) != 0) {
        unlinkat(dir_fd, temp, 0);
        *reason = "the temporary file does not take the name of the path";
        return 1;
    }
    return 0;
}

/* Publishes the reply of a tool that wrote a file. */
static int wrote(aotx_fs_tool *t, const aotx_inbound_ring *ring,
                 const volatile sig_atomic_t *stop, uint32_t agent, uint32_t request,
                 const char *path, size_t bytes)
{
    char text[AOTX_TOOL_REPLY_BYTES + 1];
    int used = snprintf(text, sizeof(text), "the file %.80s holds %llu bytes\n", path,
                        (unsigned long long)bytes);
    if (used < 0) {
        used = 0;
    }
    if ((size_t)used >= sizeof(text)) {
        used = (int)sizeof(text) - 1;
    }
    return aotx_fs_put_bytes(t, ring, stop, agent, request, (const unsigned char *)text,
                             (uint32_t)used, NULL);
}

int aotx_fs_write_file(aotx_fs_tool *t, const aotx_inbound_ring *ring,
                       const volatile sig_atomic_t *stop, uint32_t agent, uint32_t request,
                       const char *path, const char *text)
{
    char last[AOTX_WALK_BYTES];
    const char *reason = "";
    uint32_t status = AOTX_TOOL_REFUSED;
    size_t len = strlen(text);
    int dir_fd = open_parent(t->root_fd, path, last, sizeof(last), &status, &reason);
    if (dir_fd < 0) {
        return aotx_fs_put_reason(t, ring, stop, agent, request, status, reason);
    }
    if (stands_in_the_way(dir_fd, last, &reason)) {
        close(dir_fd);
        return aotx_fs_put_reason(t, ring, stop, agent, request, AOTX_TOOL_REFUSED, reason);
    }
    if (put_file(dir_fd, last, request, text, len, NULL, 0, NULL, 0, &reason) != 0) {
        close(dir_fd);
        return aotx_fs_put_reason(t, ring, stop, agent, request, AOTX_TOOL_ERROR, reason);
    }
    close(dir_fd);
    return wrote(t, ring, stop, agent, request, path, len);
}

/* Counts the places at which the old text starts in the bytes of the file. A run that
 * starts inside another run counts, so a text that is there two times is refused. */
static uint32_t count_runs(const unsigned char *bytes, size_t len, const char *old_text,
                           size_t old_len, size_t *at)
{
    uint32_t count = 0;
    size_t i;
    if (old_len == 0 || old_len > len) {
        return 0;
    }
    for (i = 0; i + old_len <= len; i++) {
        if (memcmp(bytes + i, old_text, old_len) == 0) {
            if (count == 0) {
                *at = i;
            }
            count++;
        }
    }
    return count;
}

/* Reads the whole file of an update. Returns the byte count, or -1 with the reason. */
static long take_file(int dir_fd, const char *last, const char **reason)
{
    struct stat info;
    uint32_t got = 0;
    int fd = openat(dir_fd, last, O_RDONLY | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK);
    if (fd < 0) {
        *reason = "the file does not open";
        return -1;
    }
    if (fstat(fd, &info) != 0 || !S_ISREG(info.st_mode)) {
        close(fd);
        *reason = "the path does not name a regular file";
        return -1;
    }
    if ((unsigned long long)info.st_size > (unsigned long long)AOTX_FS_FILE_CAP) {
        close(fd);
        snprintf(write_reason, sizeof(write_reason),
                 "the file is longer than the bound of %u bytes", (unsigned)AOTX_FS_FILE_CAP);
        *reason = write_reason;
        return -1;
    }
    while (got < AOTX_FS_FILE_CAP) {
        ssize_t n = read(fd, file_bytes + got, (size_t)(AOTX_FS_FILE_CAP - got));
        if (n < 0) {
            close(fd);
            *reason = "the file does not read";
            return -1;
        }
        if (n == 0) {
            break;
        }
        got += (uint32_t)n;
    }
    close(fd);
    return (long)got;
}

int aotx_fs_update_file(aotx_fs_tool *t, const aotx_inbound_ring *ring,
                        const volatile sig_atomic_t *stop, uint32_t agent, uint32_t request,
                        const char *path, const char *old_text, const char *new_text)
{
    char last[AOTX_WALK_BYTES];
    const char *reason = "";
    uint32_t status = AOTX_TOOL_REFUSED;
    size_t old_len = strlen(old_text);
    size_t new_len = strlen(new_text);
    size_t at = 0;
    uint32_t runs;
    long got;
    int dir_fd = open_parent(t->root_fd, path, last, sizeof(last), &status, &reason);
    if (dir_fd < 0) {
        return aotx_fs_put_reason(t, ring, stop, agent, request, status, reason);
    }
    if (old_len == 0) {
        close(dir_fd);
        return aotx_fs_put_reason(t, ring, stop, agent, request, AOTX_TOOL_REFUSED,
                                  "the old text holds no byte");
    }
    got = take_file(dir_fd, last, &reason);
    if (got < 0) {
        close(dir_fd);
        return aotx_fs_put_reason(t, ring, stop, agent, request, AOTX_TOOL_ERROR, reason);
    }
    runs = count_runs(file_bytes, (size_t)got, old_text, old_len, &at);
    if (runs != 1u) {
        close(dir_fd);
        /* The count states what the file holds, so an agent knows to make the old text
         * longer or to read the file again. */
        snprintf(write_reason, sizeof(write_reason),
                 "the old text occurs %u times in the file and one occurrence is needed",
                 runs);
        return aotx_fs_put_reason(t, ring, stop, agent, request, AOTX_TOOL_REFUSED,
                                  write_reason);
    }
    if (put_file(dir_fd, last, request, file_bytes, at, new_text, new_len,
                 file_bytes + at + old_len, (size_t)got - at - old_len, &reason) != 0) {
        close(dir_fd);
        return aotx_fs_put_reason(t, ring, stop, agent, request, AOTX_TOOL_ERROR, reason);
    }
    close(dir_fd);
    return wrote(t, ring, stop, agent, request, path, (size_t)got - old_len + new_len);
}
