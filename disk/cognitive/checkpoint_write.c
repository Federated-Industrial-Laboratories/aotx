/* Purpose: Publish complete memory checkpoints as CCIR generations.
 * Owns: The continuous writer lease and section batch; state bytes come from CUDA.
 * Threading: One disk writer serializes each complete snapshot.
 * Lifetime: The configured memory mirror file. */
#include "cognitive/checkpoint_io.h"
#include "disk/runtime/replay.h"
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static void aotx_cp_section(aotx_ccir_input *in, uint32_t type, const void *data, uint64_t bytes) {
    memset(in, 0, sizeof(*in));
    in->section.type = type; in->section.schema = type == AOTX_CCIR_MANIFEST ? 2 : 1;
    in->section.flags = AOTX_CCIR_REQUIRED; in->section.id[0] = (unsigned char)type;
    in->section.bytes = bytes; in->section.alignment = 8;
    in->source = AOTX_CCIR_MEMORY; in->data = data;
}
static void aotx_cp_digest(const void *data, size_t bytes, unsigned char digest[32]) {
    aotx_sha256 hash;
    aotx_sha256_init(&hash); aotx_sha256_update(&hash, data, bytes); aotx_sha256_final(&hash, digest);
}
static int aotx_cp_same(aotx_checkpoint_disk *d, const unsigned char *image, uint32_t base,
    int *same) {
    *same = 0;
    const aotx_ccir_view *v = &d->view;
    uint32_t live = UINT32_MAX, state = UINT32_MAX;
    for (uint32_t j = 0; j < v->count; ++j) {
        if (v->sections[j].type == AOTX_CCIR_LIVE && (v->sections[j].flags & AOTX_CCIR_REQUIRED)) live = j;
        if (v->sections[j].type == AOTX_CCIR_CHECKPOINT) state = j;
    }
    if (live == UINT32_MAX || state == UINT32_MAX || memcmp(v->lineage, image + 32, 16)) return AOTX_CCIR_CHANGED;
    unsigned char head[AOTX_CP_HEADER];
    aotx_ccir_read read = {live, 0, sizeof(head), head};
    int status = aotx_ccir_read_batch(v, &read, 1);
    if (status) return status;
    uint64_t prior = aotx_cp_get(head + 64, 8), revision = aotx_cp_get(image + 64, 8);
    if (revision < prior) return AOTX_CCIR_CHANGED;
    if (revision > prior) return AOTX_CCIR_OK;
    if (v->sections[live].bytes != base || v->sections[state].bytes != aotx_cp_get(image + 24, 8))
        return AOTX_CCIR_CHANGED;
    unsigned char *bytes = malloc(base);
    if (!bytes) return AOTX_CCIR_IO;
    read.bytes = base; read.data = bytes;
    status = aotx_ccir_read_batch(v, &read, 1);
    /* Capture tick and executed search counts are observations, not memory operations. */
    if (!status && memcmp(bytes, image, 72)) status = AOTX_CCIR_CHANGED;
    uint32_t at = 80, count = (uint32_t)aotx_cp_get(image + 16, 4);
    for (uint32_t j = 0; !status && j < count; ++j) {
        uint32_t skip = AOTX_CP_HEADER + j * AOTX_CP_ROW + 128 + AOTX_RECALL_QUERY + 12;
        if (memcmp(bytes + at, image + at, skip - at)) status = AOTX_CCIR_CHANGED;
        at = skip + 4;
    }
    if (!status && memcmp(bytes + at, image + at, base - at)) status = AOTX_CCIR_CHANGED;
    free(bytes);
    unsigned char digest[32];
    aotx_cp_digest(image + base, (size_t)aotx_cp_get(image + 24, 8), digest);
    if (!status && memcmp(digest, v->sections[state].digest, 32)) status = AOTX_CCIR_CHANGED;
    if (!status) *same = 1;
    return status;
}
int aotx_checkpoint_file_write(aotx_checkpoint_disk *d, const unsigned char *image, uint64_t bytes) {
    uint32_t base = 0;
    int status = aotx_checkpoint_framing(image, bytes, &base);
    if (status) return status;
    aotx_ccir_input inputs[AOTX_CCIR_SECTIONS];
    unsigned char manifest[AOTX_CCIR_MANIFEST_BYTES];
    aotx_cp_section(inputs, AOTX_CCIR_MANIFEST, manifest, sizeof(manifest));
    aotx_cp_section(inputs + 1, AOTX_CCIR_CHECKPOINT, image + base, bytes - base);
    aotx_cp_section(inputs + 2, AOTX_CCIR_LIVE, image, base);
    inputs[1].section.schema = (uint16_t)aotx_cp_get(image + base + 8, 4);
    aotx_ccir_live_manifest(manifest, inputs[1].section.id, inputs[2].section.id);
    manifest[20] = (unsigned char)inputs[1].section.schema;
    aotx_ccir_meta meta = {aotx_cp_get(image + 48, 8), aotx_cp_get(image + 48, 8), aotx_cp_get(image + 56, 8)};
    if (d->view.fd < 0) {
        struct stat st;
        if (lstat(d->path, &st)) {
            if (d->runtime) return AOTX_CCIR_CHANGED;
            status = aotx_ccir_create(d->path, image + 32, inputs, 3, &meta, NULL);
            if (status) return status;
        }
        status = aotx_ccir_writer_open(d->path, NULL, &d->view);
        if (status) return status;
    }
    struct stat path, fd;
    if (lstat(d->path, &path) || fstat(d->view.fd, &fd) || !S_ISREG(path.st_mode) ||
        path.st_dev != fd.st_dev || path.st_ino != fd.st_ino) return AOTX_CCIR_CHANGED;
    if (d->runtime && !d->runtime_verified) {
        unsigned char revision[32]; aotx_runtime_revision(&d->view, revision);
        if (memcmp(revision, d->runtime_revision, 32)) return AOTX_CCIR_CHANGED;
        d->runtime_verified = 1;
    }
    int same = 0;
    status = aotx_cp_same(d, image, base, &same);
    if (status) return status;
    int runtime = 0;
    for (uint32_t i = 0; i < d->view.count; ++i)
        runtime |= d->view.sections[i].type == AOTX_CCIR_MANIFEST && d->view.sections[i].schema == 3;
    if (runtime) return aotx_runtime_checkpoint_write(d, image, bytes, base, same);
    if (d->runtime) return AOTX_CCIR_UNSUPPORTED;
    if (same) return aotx_ccir_writer_sync(&d->view, d->path);
    uint32_t count = 3;
    for (uint32_t j = 0; j < d->view.count; ++j) {
        const aotx_ccir_section *old = d->view.sections + j;
        if (old->type == AOTX_CCIR_MANIFEST || old->type == AOTX_CCIR_CHECKPOINT ||
            (old->type == AOTX_CCIR_LIVE && (old->flags & AOTX_CCIR_REQUIRED))) {
            uint32_t i = old->type == AOTX_CCIR_MANIFEST ? 0 : old->type == AOTX_CCIR_CHECKPOINT ? 1 : 2;
            memcpy(inputs[i].section.id, old->id, 16);
            continue;
        }
        if (count == AOTX_CCIR_SECTIONS) return AOTX_CCIR_LIMIT;
        memset(inputs + count, 0, sizeof(*inputs));
        inputs[count].section = *old; inputs[count++].source = AOTX_CCIR_REUSE;
    }
    aotx_ccir_live_manifest(manifest, inputs[1].section.id, inputs[2].section.id);
    manifest[20] = (unsigned char)inputs[1].section.schema;
    for (uint32_t i = 0; i < 3; ++i) {
        unsigned char digest[32];
        aotx_cp_digest(inputs[i].data, (size_t)inputs[i].section.bytes, digest);
        for (uint32_t j = 0; j < d->view.count; ++j)
            if (!memcmp(inputs[i].section.id, d->view.sections[j].id, 16) &&
                inputs[i].section.schema == d->view.sections[j].schema &&
                inputs[i].section.bytes == d->view.sections[j].bytes &&
                !memcmp(digest, d->view.sections[j].digest, 32)) inputs[i].source = AOTX_CCIR_REUSE;
    }
    if (inputs[1].section.schema == 2) {
        uint64_t packed = AOTX_CCIR_DATA + AOTX_CCIR_COMMIT + AOTX_CCIR_PAGE;
        for (uint32_t i = 0; i < count; ++i) {
            if (inputs[i].section.bytes > INT64_MAX - packed - AOTX_CCIR_PAGE - AOTX_CCIR_ROW)
                return AOTX_CCIR_LIMIT;
            packed += inputs[i].section.bytes + AOTX_CCIR_PAGE + AOTX_CCIR_ROW;
        }
        int replace = (uint64_t)fd.st_size > packed && (uint64_t)fd.st_size - packed > packed;
        for (uint32_t j = 0; j < d->view.count; ++j) if (d->view.sections[j].type == AOTX_CCIR_CHECKPOINT) {
            unsigned char head[AOTX_COG_HEADER];
            aotx_ccir_read read = {j, 0, sizeof(head), head};
            status = aotx_ccir_read_batch(&d->view, &read, 1);
            if (status) return status;
            if (d->view.sections[j].schema != 2 || inputs[1].section.bytes < d->view.sections[j].bytes ||
                aotx_cp_get(image + base + 96, 8) > aotx_cp_get(head + 96, 8)) replace = 1;
        }
        if (replace) return aotx_ccir_writer_replace(&d->view, d->path, inputs, count, &meta, NULL);
    }
    status = aotx_ccir_writer_append(&d->view, inputs, count, &meta, NULL);
    if (status == AOTX_CCIR_LIMIT && inputs[1].section.schema == 2)
        status = aotx_ccir_writer_replace(&d->view, d->path, inputs, count, &meta, NULL);
    return status;
}
