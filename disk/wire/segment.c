/* Purpose: Write and read the journal segment files that hold one frame for each block.
 * Owns: The open segment file of a writer, and the open file of a reader.
 * Threading: One thread; a writer and a reader are not shared between threads.
 * Lifetime: From open to close. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "disk/wire/diskwire.h"

#include <dirent.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define AOTX_FRAME_HEAD_BYTES 8

int aotx_make_dir(const char *path)
{
    if (mkdir(path, 0755) == 0) {
        return 0;
    }
    /* A directory that is already there is the normal case at a restart. */
    struct stat st;
    if (stat(path, &st) == 0 && S_ISDIR(st.st_mode)) {
        return 0;
    }
    return -1;
}

static int write_all(int fd, const void *data, size_t bytes)
{
    const unsigned char *p = (const unsigned char *)data;
    size_t done = 0;
    while (done < bytes) {
        ssize_t n = write(fd, p + done, bytes - done);
        if (n <= 0) {
            return -1;
        }
        done += (size_t)n;
    }
    return 0;
}

static int read_all(int fd, void *data, size_t bytes, size_t *got)
{
    unsigned char *p = (unsigned char *)data;
    size_t done = 0;
    while (done < bytes) {
        ssize_t n = read(fd, p + done, bytes - done);
        if (n < 0) {
            return -1;
        }
        if (n == 0) {
            break;
        }
        done += (size_t)n;
    }
    *got = done;
    return 0;
}

static void segment_name(char *out, size_t bytes, const char *dir, uint64_t index)
{
    snprintf(out, bytes, "%s/seg-%06llu.seg", dir, (unsigned long long)index);
}

static int sync_dir(const char *dir)
{
    int fd = open(dir, O_RDONLY | O_DIRECTORY);
    if (fd < 0) {
        return -1;
    }
    /* The file name reaches the disk only after the directory is synchronized. */
    if (fsync(fd) != 0) {
        close(fd);
        return -1;
    }
    close(fd);
    return 0;
}

static int open_index(aotx_segment_writer *w, uint64_t index)
{
    char path[AOTX_PATH_BYTES + 32];
    segment_name(path, sizeof(path), w->dir, index);
    w->fd = open(path, O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (w->fd < 0) {
        return -1;
    }
    w->index = index;
    w->bytes = 0;
    return sync_dir(w->dir);
}

int aotx_segment_open(aotx_segment_writer *w, const char *dir, uint64_t limit)
{
    char (*names)[AOTX_NAME_BYTES] = NULL;
    int count;
    memset(w, 0, sizeof(*w));
    w->fd = -1;
    if (strlen(dir) + 1 > AOTX_PATH_BYTES) {
        return -1;
    }
    memcpy(w->dir, dir, strlen(dir) + 1);
    w->limit = (limit == 0) ? AOTX_SEGMENT_LIMIT : limit;
    count = aotx_segment_list(dir, names, 0);
    if (count < 0) {
        return -1;
    }
    /* A restart into a directory that already holds segments opens the next number, so no
     * earlier segment is written over. */
    return open_index(w, (uint64_t)count);
}

int aotx_segment_put(aotx_segment_writer *w, const unsigned char *block, uint32_t byte_len)
{
    unsigned char head[AOTX_FRAME_HEAD_BYTES];
    uint32_t crc = aotx_crc32c(block, byte_len, 0);
    uint64_t need = AOTX_FRAME_HEAD_BYTES + byte_len;
    if (w->bytes > 0 && w->bytes + need > w->limit) {
        if (aotx_segment_sync(w) != 0 || close(w->fd) != 0) {
            return -1;
        }
        w->fd = -1;
        if (open_index(w, w->index + 1) != 0) {
            return -1;
        }
    }
    memcpy(head, &byte_len, 4);
    memcpy(head + 4, &crc, 4);
    if (write_all(w->fd, head, sizeof(head)) != 0) {
        return -1;
    }
    if (write_all(w->fd, block, byte_len) != 0) {
        return -1;
    }
    w->bytes += need;
    return 0;
}

int aotx_segment_sync(aotx_segment_writer *w)
{
    return (w->fd >= 0 && fsync(w->fd) == 0) ? 0 : -1;
}

int aotx_segment_close(aotx_segment_writer *w)
{
    int rc = 0;
    if (w->fd >= 0) {
        rc = aotx_segment_sync(w);
        if (close(w->fd) != 0) {
            rc = -1;
        }
        w->fd = -1;
    }
    return rc;
}

int aotx_segment_reader_open(aotx_segment_reader *r, const char *path)
{
    r->offset = 0;
    r->fd = open(path, O_RDONLY);
    return (r->fd < 0) ? -1 : 0;
}

void aotx_segment_reader_close(aotx_segment_reader *r)
{
    if (r->fd >= 0) {
        close(r->fd);
    }
    r->fd = -1;
}

int aotx_segment_get(aotx_segment_reader *r, unsigned char *out, uint32_t out_bytes, uint32_t *got)
{
    unsigned char head[AOTX_FRAME_HEAD_BYTES];
    uint32_t byte_len;
    uint32_t crc;
    size_t read_bytes = 0;
    *got = 0;
    if (read_all(r->fd, head, sizeof(head), &read_bytes) != 0) {
        return AOTX_FRAME_TORN;
    }
    if (read_bytes == 0) {
        return AOTX_FRAME_END;
    }
    if (read_bytes != sizeof(head)) {
        return AOTX_FRAME_TORN;
    }
    memcpy(&byte_len, head, 4);
    memcpy(&crc, head + 4, 4);
    if (byte_len < AOTX_BLOCK_HEADER_BYTES || byte_len > out_bytes) {
        return AOTX_FRAME_TORN;
    }
    if (read_all(r->fd, out, byte_len, &read_bytes) != 0 || read_bytes != byte_len) {
        return AOTX_FRAME_TORN;
    }
    if (aotx_crc32c(out, byte_len, 0) != crc) {
        return AOTX_FRAME_TORN;
    }
    r->offset += AOTX_FRAME_HEAD_BYTES + byte_len;
    *got = byte_len;
    return AOTX_FRAME_OK;
}

static int is_segment(const struct dirent *e)
{
    size_t n = strlen(e->d_name);
    return n > 8 && memcmp(e->d_name, "seg-", 4) == 0 &&
           memcmp(e->d_name + n - 4, ".seg", 4) == 0;
}

int aotx_segment_list(const char *dir, char (*names)[AOTX_NAME_BYTES], int max)
{
    struct dirent **found = NULL;
    int total;
    int kept = 0;
    int i;
    total = scandir(dir, &found, NULL, alphasort);
    if (total < 0) {
        return -1;
    }
    /* The names hold a fixed number of digits, so the order of the names is the order that
     * the segments were written in. */
    for (i = 0; i < total; i++) {
        if (is_segment(found[i]) && strlen(found[i]->d_name) + 1 <= AOTX_NAME_BYTES) {
            if (names != NULL && kept < max) {
                memcpy(names[kept], found[i]->d_name, strlen(found[i]->d_name) + 1);
            }
            kept++;
        }
        free(found[i]);
    }
    free(found);
    return kept;
}
