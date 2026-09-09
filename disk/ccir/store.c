/* Purpose: Create, append and compact CCIR files under exclusive leases.
 * Owns: Creation of file incarnations and parent-directory sync.
 * Threading: One writer for each file; source compaction is quiescent.
 * Lifetime: A transaction retains each lease until its result is known. */
#include "disk/ccir/internal.h"
#include <errno.h>
#include <string.h>
#include <sys/random.h>
#include <unistd.h>

static int incarnation(unsigned char out[16])
{
    size_t at = 0u;
    while (at < 16u) {
        ssize_t got = getrandom(out + at, 16u - at, 0);
        if (got < 0 && errno == EINTR) continue;
        if (got <= 0) return AOTX_CCIR_IO;
        at += (size_t)got;
    }
    return aotx_ccir_zero(out, 16u) ? AOTX_CCIR_IO : AOTX_CCIR_OK;
}

static int create_file(const char *path, const unsigned char lineage[16],
                       const unsigned char *previous,
                       const aotx_ccir_input *inputs, uint32_t count,
                       const aotx_ccir_meta *meta, const aotx_ccir_limits *limits)
{
    unsigned char page[AOTX_CCIR_DATA];
    aotx_ccir_view old, verified;
    int fd, rc;
    if (!path || !lineage || aotx_ccir_zero(lineage, 16u)) return AOTX_CCIR_INVALID;
    memset(&old, 0, sizeof(old));
    rc = incarnation(old.incarnation);
    if (rc) return rc;
    memset(page, 0, sizeof(page));
    memcpy(page, "AOTXCCIR", 8u);
    aotx_ccir_put(page + 8, 1u, 2u);
    aotx_ccir_put(page + 12, AOTX_CCIR_PAGE, 4u);
    memcpy(page + 24, lineage, 16u);
    memcpy(page + 40, old.incarnation, 16u);
    aotx_ccir_put(page + 56, AOTX_CCIR_PAGE, 8u);
    aotx_ccir_put(page + 64, 2u * AOTX_CCIR_PAGE, 8u);
    aotx_ccir_put(page + 72, AOTX_CCIR_PAGE, 4u);
    aotx_ccir_put(page + 76, AOTX_CCIR_ROW, 4u);
    aotx_ccir_put(page + 80, AOTX_CCIR_DATA, 8u);
    aotx_ccir_hash(page, AOTX_CCIR_HASH_OFFSET, old.prologue_digest);
    memcpy(page + AOTX_CCIR_HASH_OFFSET, old.prologue_digest, 32u);
    if (previous) memcpy(old.commit_digest, previous, 32u);
    rc = aotx_ccir_lock(path, 1, 1, &fd);
    if (rc) return rc;
    rc = aotx_ccir_pwrite(fd, page, sizeof(page), 0u);
    if (!rc) rc = aotx_ccir_write_generation(fd, &old, inputs, count, meta, limits);
    if (!rc) rc = aotx_ccir_load(fd, limits, &verified);
    if (!rc) rc = aotx_ccir_parent_sync(path);
    if (rc) unlink(path);
    close(fd);
    return rc;
}

int aotx_ccir_create(const char *path, const unsigned char lineage[16],
                     const aotx_ccir_input *inputs, uint32_t count,
                     const aotx_ccir_meta *meta, const aotx_ccir_limits *limits)
{
    aotx_ccir_limits bounds;
    int rc = aotx_ccir_limits_get(limits, &bounds);
    return rc ? rc : create_file(path, lineage, NULL, inputs, count, meta, &bounds);
}

static int append_file(const char *path, const aotx_ccir_revision *expected,
                       const aotx_ccir_input *inputs, uint32_t count,
                       const aotx_ccir_meta *meta, const aotx_ccir_limits *limits)
{
    aotx_ccir_view old;
    aotx_ccir_limits bounds;
    int fd, rc;
    if (!path) return AOTX_CCIR_INVALID;
    rc = aotx_ccir_limits_get(limits, &bounds);
    if (rc) return rc;
    rc = aotx_ccir_lock(path, 1, 0, &fd);
    if (rc) return rc;
    rc = aotx_ccir_load(fd, &bounds, &old);
    if (!rc && expected &&
        (memcmp(expected->prologue_digest, old.prologue_digest, 32u) ||
         memcmp(expected->commit_digest, old.commit_digest, 32u))) rc = AOTX_CCIR_CHANGED;
    if (!rc) rc = aotx_ccir_write_generation(fd, &old, inputs, count, meta, &bounds);
    close(fd);
    return rc;
}

int aotx_ccir_append(const char *path, const aotx_ccir_input *inputs,
                     uint32_t count, const aotx_ccir_meta *meta,
                     const aotx_ccir_limits *limits)
{
    return append_file(path, NULL, inputs, count, meta, limits);
}

int aotx_ccir_append_if(const char *path, const aotx_ccir_revision *expected,
                        const aotx_ccir_input *inputs, uint32_t count,
                        const aotx_ccir_meta *meta, const aotx_ccir_limits *limits)
{
    if (!expected) return AOTX_CCIR_INVALID;
    return append_file(path, expected, inputs, count, meta, limits);
}

int aotx_ccir_compact(const char *source, const char *destination,
                      const aotx_ccir_limits *limits)
{
    aotx_ccir_view old;
    aotx_ccir_limits bounds;
    aotx_ccir_input inputs[AOTX_CCIR_SECTIONS];
    uint32_t i;
    int fd, rc;
    if (!source || !destination) return AOTX_CCIR_INVALID;
    rc = aotx_ccir_limits_get(limits, &bounds);
    if (rc) return rc;
    rc = aotx_ccir_lock(source, 1, 0, &fd);
    if (rc) return rc;
    rc = aotx_ccir_load(fd, &bounds, &old);
    if (!rc) {
        memset(inputs, 0, sizeof(inputs));
        for (i = 0; i < old.count; i++) {
            inputs[i].section = old.sections[i];
            inputs[i].source = AOTX_CCIR_FILE;
            inputs[i].fd = fd;
            inputs[i].source_offset = old.sections[i].offset;
        }
        rc = create_file(destination, old.lineage, old.commit_digest,
                         inputs, old.count, &old.meta, &bounds);
    }
    close(fd);
    return rc;
}
