/* Purpose: Select a complete CCIR generation and read bounded section batches.
 * Owns: Validation of the prologue, both roots and all selected extents.
 * Threading: The caller holds a file lease.
 * Lifetime: The selected view lasts until close. */
#include "disk/ccir/internal.h"
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static int prologue(int fd, aotx_ccir_view *view)
{
    unsigned char page[AOTX_CCIR_PAGE], digest[32];
    int rc = aotx_ccir_pread(fd, page, sizeof(page), 0u);
    if (rc) return rc;
    aotx_ccir_hash(page, AOTX_CCIR_HASH_OFFSET, digest);
    if (memcmp(page, "AOTXCCIR", 8u) ||
        memcmp(digest, page + AOTX_CCIR_HASH_OFFSET, 32u)) return AOTX_CCIR_INVALID;
    if (aotx_ccir_u16(page + 8) != 1u || aotx_ccir_u16(page + 10) ||
        aotx_ccir_u64(page + 16)) return AOTX_CCIR_UNSUPPORTED;
    if (aotx_ccir_u32(page + 12) != AOTX_CCIR_PAGE ||
        aotx_ccir_u64(page + 56) != AOTX_CCIR_PAGE ||
        aotx_ccir_u64(page + 64) != 2u * AOTX_CCIR_PAGE ||
        aotx_ccir_u32(page + 72) != AOTX_CCIR_PAGE ||
        aotx_ccir_u32(page + 76) != AOTX_CCIR_ROW ||
        aotx_ccir_u64(page + 80) != AOTX_CCIR_DATA ||
        aotx_ccir_zero(page + 24, 16u) || aotx_ccir_zero(page + 40, 16u) ||
        !aotx_ccir_zero(page + 88, AOTX_CCIR_HASH_OFFSET - 88u))
        return AOTX_CCIR_INVALID;
    memcpy(view->lineage, page + 24, 16u);
    memcpy(view->incarnation, page + 40, 16u);
    memcpy(view->prologue_digest, digest, 32u);
    return AOTX_CCIR_OK;
}

static int sections(int fd, const unsigned char *commit,
                    const aotx_ccir_limits *limits, aotx_ccir_view *view)
{
    unsigned char rows[AOTX_CCIR_SECTIONS * AOTX_CCIR_ROW], digest[32];
    uint64_t length = aotx_ccir_u64(commit + 88);
    uint32_t i, j;
    int rc;
    view->count = aotx_ccir_u32(commit + 80);
    view->directory_offset = aotx_ccir_u64(commit + 72);
    if (view->count < 2u || view->count > AOTX_CCIR_SECTIONS)
        return AOTX_CCIR_INVALID;
    if (view->count > limits->sections) return AOTX_CCIR_LIMIT;
    if (aotx_ccir_u32(commit + 84) != AOTX_CCIR_ROW ||
        length != (uint64_t)view->count * AOTX_CCIR_ROW ||
        view->directory_offset < AOTX_CCIR_DATA || view->directory_offset % 128u ||
        view->directory_offset > view->commit_offset ||
        length != view->commit_offset - view->directory_offset)
        return AOTX_CCIR_INVALID;
    rc = aotx_ccir_pread(fd, rows, (size_t)length, view->directory_offset);
    if (rc) return rc;
    aotx_ccir_hash(rows, (size_t)length, digest);
    if (memcmp(digest, commit + 96, 32u)) return AOTX_CCIR_INVALID;
    for (i = 0; i < view->count; i++) {
        aotx_ccir_section *s = &view->sections[i];
        rc = aotx_ccir_decode_row(rows + i * AOTX_CCIR_ROW, s, limits,
                                   view->directory_offset);
        if (rc) return rc;
        for (j = 0; j < i; j++) {
            const aotx_ccir_section *p = &view->sections[j];
            if (!memcmp(p->id, s->id, 16u) ||
                (s->offset < p->offset + p->bytes && p->offset < s->offset + s->bytes))
                return AOTX_CCIR_INVALID;
        }
        rc = aotx_ccir_hash_fd(fd, s->offset, s->bytes, digest);
        if (rc) return rc;
        if (memcmp(digest, s->digest, 32u)) return AOTX_CCIR_INVALID;
    }
    return aotx_ccir_profile(fd, view, commit);
}

static int root(int fd, uint32_t slot, uint64_t file_bytes,
                const aotx_ccir_limits *limits, aotx_ccir_view *view,
                int *present, uint64_t *hint)
{
    unsigned char page[AOTX_CCIR_PAGE], commit[AOTX_CCIR_COMMIT], digest[32];
    int rc = aotx_ccir_pread(fd, page, sizeof(page), (slot + 1u) * AOTX_CCIR_PAGE);
    *present = 1; *hint = 0;
    if (rc) return rc;
    if (aotx_ccir_zero(page, sizeof(page))) {
        *present = 0; return AOTX_CCIR_INVALID;
    }
    aotx_ccir_hash(page, AOTX_CCIR_HASH_OFFSET, digest);
    if (memcmp(page, "AOTXROOT", 8u) ||
        memcmp(digest, page + AOTX_CCIR_HASH_OFFSET, 32u)) return AOTX_CCIR_INVALID;
    *hint = aotx_ccir_u64(page + 16);
    if (aotx_ccir_u16(page + 8) != 1u) return AOTX_CCIR_UNSUPPORTED;
    if (!aotx_ccir_zero(page + 10, 6u) ||
        !aotx_ccir_zero(page + 112, AOTX_CCIR_HASH_OFFSET - 112u) ||
        memcmp(page + 80, view->prologue_digest, 32u)) return AOTX_CCIR_INVALID;
    view->root_slot = slot;
    view->generation = *hint;
    view->commit_offset = aotx_ccir_u64(page + 24);
    view->end = aotx_ccir_u64(page + 40);
    if (!view->generation || aotx_ccir_u64(page + 32) != AOTX_CCIR_COMMIT ||
        view->end > file_bytes || view->end < AOTX_CCIR_DATA + AOTX_CCIR_COMMIT ||
        view->commit_offset % 128u ||
        view->commit_offset != view->end - AOTX_CCIR_COMMIT)
        return AOTX_CCIR_INVALID;
    rc = aotx_ccir_pread(fd, commit, sizeof(commit), view->commit_offset);
    if (rc) return rc;
    aotx_ccir_hash(commit, sizeof(commit), digest);
    if (memcmp(digest, page + 48, 32u) || memcmp(commit, "AOTXCMT1", 8u) ||
        aotx_ccir_u64(commit + 8) != view->generation ||
        aotx_ccir_u64(commit + 128) != view->end ||
        !aotx_ccir_zero(commit + 184, AOTX_CCIR_COMMIT - 184u))
        return AOTX_CCIR_INVALID;
    memcpy(view->commit_digest, digest, 32u);
    view->meta.checkpoint_sequence = aotx_ccir_u64(commit + 48);
    view->meta.durable_sequence = aotx_ccir_u64(commit + 56);
    view->meta.source_tick = aotx_ccir_u64(commit + 64);
    if (view->meta.checkpoint_sequence > view->meta.durable_sequence)
        return AOTX_CCIR_INVALID;
    return sections(fd, commit, limits, view);
}

int aotx_ccir_load(int fd, const aotx_ccir_limits *limits, aotx_ccir_view *view)
{
    struct stat st;
    aotx_ccir_view candidates[2];
    uint64_t hint[2];
    int rc[2], present[2], chosen, i;
    memset(view, 0, sizeof(*view));
    view->fd = -1;
    if (fstat(fd, &st) || st.st_size < 0) return AOTX_CCIR_IO;
    if ((uint64_t)st.st_size > limits->file_bytes) return AOTX_CCIR_LIMIT;
    if (st.st_size < AOTX_CCIR_DATA) return AOTX_CCIR_INVALID;
    rc[0] = prologue(fd, view);
    if (rc[0]) return rc[0];
    candidates[0] = candidates[1] = *view;
    for (i = 0; i < 2; i++)
        rc[i] = root(fd, (uint32_t)i, (uint64_t)st.st_size, limits,
                     &candidates[i], &present[i], &hint[i]);
    chosen = rc[0] == AOTX_CCIR_OK ? 0 : rc[1] == AOTX_CCIR_OK ? 1 : -1;
    if (!rc[0] && !rc[1]) {
        if (hint[0] == hint[1] && memcmp(candidates[0].commit_digest,
                                       candidates[1].commit_digest, 32u))
            return AOTX_CCIR_INVALID;
        chosen = hint[1] > hint[0] ? 1 : 0;
    }
    for (i = 0; i < 2; i++) {
        if (rc[i] == AOTX_CCIR_IO) return rc[i];
        if ((rc[i] == AOTX_CCIR_UNSUPPORTED || rc[i] == AOTX_CCIR_LIMIT) &&
            (chosen < 0 || hint[i] >= hint[chosen])) return rc[i];
    }
    if (chosen < 0) return AOTX_CCIR_INVALID;
    *view = candidates[chosen];
    view->fd = fd;
    view->fallback = present[1 - chosen] && rc[1 - chosen] != AOTX_CCIR_OK;
    view->trailing_bytes = (uint64_t)st.st_size - view->end;
    return AOTX_CCIR_OK;
}

int aotx_ccir_open(const char *path, const aotx_ccir_limits *limits,
                   aotx_ccir_view *view)
{
    aotx_ccir_limits bounds;
    int fd, rc;
    if (!path || !view) return AOTX_CCIR_INVALID;
    memset(view, 0, sizeof(*view)); view->fd = -1;
    rc = aotx_ccir_limits_get(limits, &bounds);
    if (rc) return rc;
    rc = aotx_ccir_lock(path, 0, 0, &fd);
    if (rc) return rc;
    rc = aotx_ccir_load(fd, &bounds, view);
    if (rc) { close(fd); view->fd = -1; }
    return rc;
}
void aotx_ccir_close(aotx_ccir_view *view)
{
    if (view && view->fd >= 0) { close(view->fd); view->fd = -1; }
}
int aotx_ccir_read_batch(const aotx_ccir_view *view,
                         const aotx_ccir_read *reads, uint32_t count)
{
    uint32_t i;
    if (!view || view->fd < 0 || !reads || !count || count > AOTX_CCIR_SECTIONS)
        return AOTX_CCIR_INVALID;
    for (i = 0; i < count; i++) {
        const aotx_ccir_read *r = &reads[i];
        if (r->section >= view->count || (!r->data && r->bytes) ||
            r->offset > view->sections[r->section].bytes ||
            r->bytes > view->sections[r->section].bytes - r->offset)
            return AOTX_CCIR_INVALID;
    }
    for (i = 0; i < count; i++) {
        const aotx_ccir_read *r = &reads[i];
        int rc = aotx_ccir_pread(view->fd, r->data, r->bytes,
                                  view->sections[r->section].offset + r->offset);
        if (rc) return rc;
    }
    return AOTX_CCIR_OK;
}
