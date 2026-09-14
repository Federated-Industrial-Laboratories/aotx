/* Purpose: Create an exact policy bundle from image and metadata byte extents.
 * Owns: Header encoding, source validation, and exclusive output publication.
 * Threading: One caller writes the complete extent batch.
 * Lifetime: The output is closed and synced before a successful return. */
#include "disk/policy/file.h"
#include "disk/ccir/internal.h"
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int aotx_policy_file_write(const char *path, const aotx_policy_source *s) {
    if (!path || !*path || !s || !s->provenance || !s->license ||
        !s->provenance_bytes || !s->license_bytes || (s->image_bytes && !s->image))
        return AOTX_CCIR_INVALID;
    if (s->image_bytes > AOTX_POLICY_IMAGE_BYTES || s->provenance_bytes > AOTX_POLICY_METADATA_BYTES ||
        s->license_bytes > AOTX_POLICY_METADATA_BYTES || s->provenance_bytes > UINT32_MAX ||
        s->license_bytes > UINT32_MAX || (s->entry && strlen(s->entry) >= 64)) return AOTX_CCIR_LIMIT;
    uint64_t total = AOTX_POLICY_FILE_HEADER + (uint64_t)s->image_bytes +
        s->provenance_bytes + s->license_bytes;
    if (total > SIZE_MAX) return AOTX_CCIR_LIMIT;
    unsigned char *h = calloc(1, (size_t)total);
    if (!h) return AOTX_CCIR_IO;
    const aotx_policy_config *c = &s->config;
    memcpy(h, "AOTXPL01", 8); aotx_ccir_put(h + 8, 1, 4);
    aotx_ccir_put(h + 12, c->mode, 4); aotx_ccir_put(h + 16, c->abi ? c->abi : AOTX_POLICY_ABI, 4);
    aotx_ccir_put(h + 20, c->state_schema, 4); aotx_ccir_put(h + 24, c->state_bytes, 4);
    aotx_ccir_put(h + 28, c->architecture, 4); aotx_ccir_put(h + 32, c->threads, 4);
    aotx_ccir_put(h + 36, c->registers, 4); aotx_ccir_put(h + 40, c->shared_bytes, 4);
    aotx_ccir_put(h + 44, c->local_bytes, 4); aotx_ccir_put(h + 48, c->pressure, 4);
    aotx_ccir_put(h + 52, c->minimum_move, 4); aotx_ccir_put(h + 56, c->backoff, 4);
    aotx_ccir_put(h + 60, c->format, 4); aotx_ccir_put(h + 64, s->image_bytes, 8);
    aotx_ccir_put(h + 72, s->provenance_bytes, 4); aotx_ccir_put(h + 76, s->license_bytes, 4);
    if (s->entry) memcpy(h + 80, s->entry, strlen(s->entry));
    if (s->image_bytes) {
        memcpy(h + AOTX_POLICY_FILE_HEADER, s->image, s->image_bytes);
        aotx_ccir_hash(s->image, s->image_bytes, h + 144);
    }
    memcpy(h + AOTX_POLICY_FILE_HEADER + s->image_bytes, s->provenance, s->provenance_bytes);
    memcpy(h + AOTX_POLICY_FILE_HEADER + s->image_bytes + s->provenance_bytes, s->license, s->license_bytes);
    aotx_policy_file file;
    int rc = aotx_policy_file_decode(h, (size_t)total, &file);
    aotx_policy_file_close(&file);
    int fd = -1;
    if (!rc) rc = aotx_ccir_lock(path, 1, 1, &fd);
    if (!rc) rc = aotx_ccir_pwrite(fd, h, (size_t)total, 0);
    if (!rc && fsync(fd)) rc = AOTX_CCIR_IO;
    if (!rc) rc = aotx_ccir_parent_sync(path);
    if (fd >= 0) { if (rc) unlink(path); close(fd); }
    free(h);
    return rc;
}
