/* Purpose: Publish an explicit compatible policy update as a separate complete file.
 * Owns: New index and history buffers plus an unnamed output file.
 * Threading: One reader holds the original lease until atomic output publication.
 * Lifetime: All prior sections and replay bytes remain unchanged. */
#include "disk/runtime/policy.h"
#include "disk/ccir/internal.h"
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int publish(const char *path, const aotx_ccir_view *old,
    const aotx_ccir_input *inputs, uint32_t count) {
    char parent[PATH_MAX], descriptor[64]; size_t n = strlen(path);
    if (!n || n >= sizeof(parent)) return AOTX_CCIR_LIMIT;
    memcpy(parent, path, n + 1);
    char *slash = strrchr(parent, '/');
    if (!slash) strcpy(parent, ".");
    else if (slash == parent) slash[1] = 0;
    else *slash = 0;
    int fd = open(parent, O_TMPFILE | O_RDWR | O_CLOEXEC, 0600);
    if (fd < 0) return AOTX_CCIR_IO;
    aotx_ccir_limits limits; aotx_ccir_default_limits(&limits);
    aotx_ccir_view next;
    int rc = aotx_ccir_initialize(fd, old->lineage, old->commit_digest,
        inputs, count, &old->meta, &limits, &next);
    if (!rc) rc = aotx_runtime_dependencies(&next);
    if (!rc) {
        snprintf(descriptor, sizeof(descriptor), "/proc/self/fd/%d", fd);
        if (linkat(AT_FDCWD, descriptor, AT_FDCWD, path, AT_SYMLINK_FOLLOW))
            rc = errno == EEXIST ? AOTX_CCIR_EXISTS : AOTX_CCIR_IO;
        else rc = aotx_ccir_parent_sync(path);
    }
    close(fd);
    return rc;
}
static int replace_assets(const aotx_ccir_view *view, aotx_runtime_index *index,
    const aotx_policy_file *next, const unsigned char *history, size_t bytes, const char *output) {
    aotx_ccir_input inputs[AOTX_CCIR_SECTIONS] = {0};
    uint32_t count = view->count;
    for (uint32_t i = 0; i < count; ++i) {
        inputs[i].section = view->sections[i]; inputs[i].source = AOTX_CCIR_FILE;
        inputs[i].fd = view->fd; inputs[i].source_offset = view->sections[i].offset;
    }
    int prior = -1;
    for (uint32_t i = 0; i < index->count; ++i)
        if (!strcmp((const char *)index->rows[i] + 64, "policy-history.bin")) prior = (int)i;
    if (prior < 0) {
        if (count == AOTX_CCIR_SECTIONS || index->count == AOTX_CCIR_SECTIONS) return AOTX_CCIR_LIMIT;
        prior = (int)index->count++;
        unsigned char *row = index->rows[prior]; memset(row, 0, AOTX_RUNTIME_ROW);
        row[0] = AOTX_CCIR_ASSET;
        /* New IDs use the first free ordinal and never reuse an existing section ID. */
        uint64_t ordinal = 1;
        for (;;) {
            aotx_ccir_put(row + 8, ordinal, 8);
            if (aotx_runtime_section(view, row) < 0) break;
            if (++ordinal == 0) return AOTX_CCIR_LIMIT;
        }
        aotx_ccir_put(row + 16, 4, 4); strcpy((char *)row + 64, "policy-history.bin");
        aotx_ccir_input *in = inputs + count++;
        memcpy(in->section.id, row, 16); in->section.type = AOTX_CCIR_ASSET;
        in->section.schema = 1; in->section.flags = AOTX_CCIR_REQUIRED; in->section.alignment = 4096;
    }
    for (uint32_t i = 0; i < index->count; ++i) {
        unsigned char *row = index->rows[i]; const char *name = (const char *)row + 64;
        const void *data = NULL; size_t length = 0;
        if (!strcmp(name, "policy.bin")) { data = next->buffer; length = next->buffer_bytes; }
        else if ((int)i == prior) { data = history; length = bytes; }
        if (!data) continue;
        uint32_t at = 0;
        while (at < count && memcmp(inputs[at].section.id, row, 16)) ++at;
        if (at == count) return AOTX_CCIR_INVALID;
        inputs[at].source = AOTX_CCIR_MEMORY; inputs[at].data = data;
        inputs[at].section.bytes = length; aotx_ccir_hash(data, length, inputs[at].section.digest);
        aotx_ccir_put(row + 24, length, 8); memcpy(row + 32, inputs[at].section.digest, 32);
    }
    aotx_ccir_put(index->header + 16, index->count, 4);
    aotx_ccir_put(index->header + 20, aotx_ccir_u32(index->header + 20) | AOTX_RUNTIME_POLICY_HISTORY, 4);
    for (uint32_t i = 0; i < count; ++i) if (inputs[i].section.type == AOTX_CCIR_RUNTIME) {
        inputs[i].source = AOTX_CCIR_MEMORY; inputs[i].data = index->header;
        inputs[i].section.schema = aotx_runtime_schema(aotx_ccir_u32(index->header + 20));
        inputs[i].section.bytes = AOTX_RUNTIME_HEADER + index->count * AOTX_RUNTIME_ROW;
        memset(inputs[i].section.digest, 0, 32);
    }
    return publish(output, view, inputs, count);
}
int aotx_runtime_policy_update(const char *source, const char *expected, const char *bundle,
    const char *destination) {
    if (!source || !expected || strlen(expected) != 64 || !bundle || !destination) return AOTX_CCIR_INVALID;
    aotx_ccir_view view;
    int rc = aotx_ccir_open(source, NULL, &view);
    if (rc) return rc;
    aotx_runtime_index *index = malloc(sizeof(*index));
    aotx_policy_file old = {0}, next = {0}; aotx_policy_history revisions;
    unsigned char *prior = NULL, *history = NULL; size_t prior_bytes = 0;
    if (!index) rc = AOTX_CCIR_IO;
    if (!rc) rc = aotx_runtime_index_read(view.fd, &view, index);
    if (!rc) rc = aotx_runtime_dependencies(&view);
    if (!rc) rc = aotx_runtime_policy_read(&view, index, &old, &revisions, &prior, &prior_bytes);
    char digest[65]; aotx_sha256_text(old.digest, digest);
    if (!rc && (!old.config.mode || strcmp(expected, digest))) rc = AOTX_CCIR_CHANGED;
    if (!rc) rc = aotx_policy_file_read(bundle, NULL, 0, &next);
    if (!rc && (!aotx_policy_compatible(&old.config, &next.config) ||
        (next.config.mode == AOTX_POLICY_NATIVE &&
         next.config.architecture != aotx_ccir_u32(index->header + 36)))) rc = AOTX_CCIR_UNSUPPORTED;
    if (!rc && (!memcmp(old.digest, next.digest, 32) || revisions.count == AOTX_POLICY_REVISIONS))
        rc = AOTX_CCIR_LIMIT;
    uint64_t last = 0;
    if (!rc) rc = aotx_runtime_policy_replay(&view, &old, &revisions, &last);
    size_t head = prior_bytes ? prior_bytes : 64, bytes = head + 64 + old.buffer_bytes;
    if (!rc) { history = calloc(1, bytes); if (!history) rc = AOTX_CCIR_IO; }
    if (!rc) {
        if (prior_bytes) memcpy(history, prior, prior_bytes);
        memcpy(history, "AOTXPH01", 8); aotx_ccir_put(history + 8, 1, 4);
        aotx_ccir_put(history + 12, revisions.count + 1, 4); memcpy(history + 16, next.digest, 32);
        aotx_ccir_put(history + head, old.buffer_bytes, 8); aotx_ccir_put(history + head + 8, last, 8);
        memcpy(history + head + 16, old.digest, 32); memcpy(history + head + 64, old.buffer, old.buffer_bytes);
        aotx_policy_history check;
        rc = aotx_policy_history_decode(history, bytes, &next, &check);
        if (!rc) rc = replace_assets(&view, index, &next, history, bytes, destination);
    }
    free(history); free(prior); free(index);
    aotx_policy_file_close(&old); aotx_policy_file_close(&next); aotx_ccir_close(&view);
    return rc;
}
