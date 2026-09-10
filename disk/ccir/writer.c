/* Purpose: Append generations under one continuous exclusive writer lease.
 * Owns: The writer view and its selected durable generation.
 * Threading: One disk writer; each call takes a complete section batch.
 * Lifetime: Writer open through view close. */
#include "disk/ccir/internal.h"
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <sys/file.h>

int aotx_ccir_writer_open(const char *path, const aotx_ccir_limits *limits, aotx_ccir_view *view)
{
    if (!path || !view) return AOTX_CCIR_INVALID;
    aotx_ccir_limits bounds;
    memset(view, 0, sizeof(*view)); view->fd = -1;
    int status = aotx_ccir_limits_get(limits, &bounds), fd = -1;
    if (!status) status = aotx_ccir_lock(path, 1, 0, &fd);
    if (!status) status = aotx_ccir_load(fd, &bounds, view);
    if (!status) status = aotx_ccir_pending_clear(path, view, &bounds);
    if (!status && flock(fd, LOCK_UN)) status = AOTX_CCIR_IO;
    if (status && fd >= 0) close(fd);
    if (status) view->fd = -1;
    return status;
}
int aotx_ccir_writer_append(aotx_ccir_view *view, const aotx_ccir_input *inputs,
    uint32_t count, const aotx_ccir_meta *meta, const aotx_ccir_limits *limits)
{
    aotx_ccir_limits bounds;
    if (!view || view->fd < 0) return AOTX_CCIR_INVALID;
    int status = aotx_ccir_limits_get(limits, &bounds);
    int fd = view->fd;
    if (!status && flock(fd, LOCK_EX | LOCK_NB)) return errno == EWOULDBLOCK ? AOTX_CCIR_BUSY : AOTX_CCIR_IO;
    if (!status) {
        status = aotx_ccir_write_generation(fd, view, inputs, count, meta, &bounds);
        aotx_ccir_view next;
        int loaded = aotx_ccir_load(fd, &bounds, &next);
        if (!loaded) *view = next;
        else if (!status) status = loaded;
        if (flock(fd, LOCK_UN) && !status) status = AOTX_CCIR_IO;
    }
    return status;
}
int aotx_ccir_writer_sync(aotx_ccir_view *view, const char *path) {
    if (!view || view->fd < 0 || !path) return AOTX_CCIR_INVALID;
    return fsync(view->fd) ? AOTX_CCIR_IO : aotx_ccir_parent_sync(path);
}
