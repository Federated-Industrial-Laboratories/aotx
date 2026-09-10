/* Purpose: Replace a live CCIR file with a complete compacted incarnation.
 * Owns: Both writer leases through replacement and directory synchronization.
 * Threading: One writer copies a bounded section batch; readers keep complete views.
 * Lifetime: Failure before rename keeps the old file; later failure retains the new view. */
#include "disk/ccir/internal.h"
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#include <string.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

static int same_path(int fd, const char *path) {
    struct stat a, b;
    return fstat(fd, &a) || lstat(path, &b) || !S_ISREG(b.st_mode) ||
        a.st_dev != b.st_dev || a.st_ino != b.st_ino ? AOTX_CCIR_CHANGED : AOTX_CCIR_OK;
}
static int pending_path(char out[PATH_MAX], const char *path, const aotx_ccir_view *view) {
    size_t length = strlen(path);
    if (!length || length > PATH_MAX - 42u) return AOTX_CCIR_LIMIT;
    memcpy(out, path, length); memcpy(out + length, ".pending-", 9);
    for (uint32_t i = 0; i < 16; ++i) snprintf(out + length + 9 + i * 2, 3, "%02x", view->incarnation[i]);
    return AOTX_CCIR_OK;
}
int aotx_ccir_pending_clear(const char *path, const aotx_ccir_view *view, const aotx_ccir_limits *limits) {
    char pending[PATH_MAX]; struct stat st;
    int status = pending_path(pending, path, view), fd = -1;
    if (status) return status;
    if (lstat(pending, &st)) return errno == ENOENT ? AOTX_CCIR_OK : AOTX_CCIR_IO;
    aotx_ccir_view abandoned;
    unsigned char commit[AOTX_CCIR_COMMIT];
    status = aotx_ccir_lock(pending, 1, 0, &fd);
    if (!status) status = aotx_ccir_load(fd, limits, &abandoned);
    if (!status) status = aotx_ccir_pread(fd, commit, sizeof(commit), abandoned.commit_offset);
    if (!status && (abandoned.generation != 1 || memcmp(abandoned.lineage, view->lineage, 16) ||
        memcmp(commit + 16, view->commit_digest, 32))) status = AOTX_CCIR_CHANGED;
    if (!status) status = same_path(fd, pending);
    if (!status && unlink(pending)) status = AOTX_CCIR_IO;
    if (fd >= 0) close(fd);
    return status;
}
static int temporary_open(const char *path, int *fd) {
    char parent[PATH_MAX]; size_t length = strlen(path);
    if (!length || length >= sizeof(parent)) return AOTX_CCIR_LIMIT;
    memcpy(parent, path, length + 1);
    char *slash = strrchr(parent, '/');
    if (!slash) strcpy(parent, ".");
    else if (slash == parent) slash[1] = 0;
    else *slash = 0;
    *fd = open(parent, O_TMPFILE | O_RDWR | O_CLOEXEC, 0600);
    if (*fd < 0) return errno == EOPNOTSUPP || errno == EINVAL || errno == EISDIR ?
        AOTX_CCIR_UNSUPPORTED : AOTX_CCIR_IO;
    struct flock owner = {0};
    owner.l_type = F_WRLCK; owner.l_whence = SEEK_SET; owner.l_len = 1;
    if (fcntl(*fd, F_OFD_SETLK, &owner) || flock(*fd, LOCK_EX | LOCK_NB)) {
        close(*fd); *fd = -1; return AOTX_CCIR_IO;
    }
    return AOTX_CCIR_OK;
}
int aotx_ccir_writer_replace(aotx_ccir_view *view, const char *path,
    const aotx_ccir_input *inputs, uint32_t count, const aotx_ccir_meta *meta,
    const aotx_ccir_limits *limits) {
    if (!view || view->fd < 0 || !path || !inputs || !count || count > AOTX_CCIR_SECTIONS || !meta)
        return AOTX_CCIR_INVALID;
    aotx_ccir_limits bounds;
    int status = aotx_ccir_limits_get(limits, &bounds);
    if (status) return status;
    int old = view->fd;
    if (flock(old, LOCK_EX | LOCK_NB)) return errno == EWOULDBLOCK ? AOTX_CCIR_BUSY : AOTX_CCIR_IO;
    status = same_path(old, path);
    if (!status) status = aotx_ccir_pending_clear(path, view, &bounds);
    aotx_ccir_input batch[AOTX_CCIR_SECTIONS];
    memcpy(batch, inputs, count * sizeof(*inputs));
    for (uint32_t i = 0; !status && i < count; ++i) if (batch[i].source == AOTX_CCIR_REUSE) {
        uint32_t j = 0;
        while (j < view->count && memcmp(batch[i].section.id, view->sections[j].id, 16)) ++j;
        if (j == view->count || batch[i].section.type != view->sections[j].type ||
            batch[i].section.schema != view->sections[j].schema || batch[i].section.flags != view->sections[j].flags ||
            batch[i].section.bytes != view->sections[j].bytes || batch[i].section.alignment != view->sections[j].alignment)
            status = AOTX_CCIR_INVALID;
        else {
            batch[i].source = AOTX_CCIR_FILE; batch[i].fd = old;
            batch[i].source_offset = view->sections[j].offset;
        }
    }
    char pending[PATH_MAX], descriptor[64];
    if (!status) status = pending_path(pending, path, view);
    aotx_ccir_view next;
    memset(&next, 0, sizeof(next)); next.fd = -1;
    int fd = -1, linked = 0, replaced = 0;
    if (!status) status = temporary_open(path, &fd);
    if (!status) status = aotx_ccir_initialize(fd, view->lineage, view->commit_digest, batch, count, meta, &bounds, &next);
    if (!status) status = same_path(old, path);
    if (!status) {
        snprintf(descriptor, sizeof(descriptor), "/proc/self/fd/%d", fd);
        if (linkat(AT_FDCWD, descriptor, AT_FDCWD, pending, AT_SYMLINK_FOLLOW)) status = AOTX_CCIR_IO;
        else linked = 1;
    }
    if (!status) status = same_path(fd, pending);
    if (!status) {
        if (rename(pending, path)) status = AOTX_CCIR_IO;
        else {
            *view = next; fd = -1; replaced = 1;
            status = aotx_ccir_parent_sync(path);
        }
    }
    if (linked && !replaced && !same_path(fd, pending)) unlink(pending);
    if (fd >= 0) close(fd);
    if (replaced && flock(view->fd, LOCK_UN) && !status) status = AOTX_CCIR_IO;
    if (flock(old, LOCK_UN) && !status) status = AOTX_CCIR_IO;
    if (replaced) close(old);
    return status;
}
