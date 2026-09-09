/* Purpose: Transfer validated CCIR image sections to and from device host glue.
 * Owns: Bounded disk buffers and file metadata consistency checks.
 * Threading: One reader lease until all output sections are written.
 * Lifetime: One offline checkpoint export. */
#include "cognitive/io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t aotx_cognitive_le(const unsigned char *p, unsigned bytes) {
    uint64_t value = 0;
    for (unsigned i = 0; i < bytes; ++i) value |= (uint64_t)p[i] << (8 * i);
    return value;
}
static int aotx_cognitive_binding(const aotx_cognitive_file *file) {
    const unsigned char *p = file->checkpoint;
    const aotx_ccir_meta *meta = &file->view.meta;
    if (file->checkpoint_bytes < AOTX_COG_HEADER ||
        memcmp(p + 48, file->view.lineage, 16) ||
        aotx_cognitive_le(p + 32, 8) != meta->checkpoint_sequence ||
        aotx_cognitive_le(p + 40, 8) > meta->source_tick) return AOTX_CCIR_INVALID;
    if (!file->tail_bytes)
        return meta->checkpoint_sequence == meta->durable_sequence &&
               aotx_cognitive_le(p + 40, 8) == meta->source_tick ? AOTX_CCIR_OK : AOTX_CCIR_INVALID;
    p = file->tail;
    if (file->tail_bytes < AOTX_COG_HEADER) return AOTX_CCIR_INVALID;
    uint64_t first = aotx_cognitive_le(p + 32, 8), count = aotx_cognitive_le(p + 20, 4);
    if (!first || !count || count > AOTX_COG_OBJECTS || first > UINT64_MAX - (count - 1) ||
        memcmp(p + 48, file->view.lineage, 16) || first + count - 1 != meta->durable_sequence ||
        aotx_cognitive_le(p + 40, 8) != meta->source_tick) return AOTX_CCIR_INVALID;
    return AOTX_CCIR_OK;
}

int aotx_cognitive_file_open(const char *path, aotx_cognitive_file *file) {
    memset(file, 0, sizeof(*file));
    file->view.fd = -1;
    file->tail_index = UINT32_MAX;
    int status = aotx_ccir_open(path, NULL, &file->view);
    if (status) return status;
    for (uint32_t i = 0; i < file->view.count; ++i) {
        const aotx_ccir_section *s = &file->view.sections[i];
        if (s->type == AOTX_CCIR_MANIFEST) file->manifest_index = i;
    }
    aotx_ccir_read manifest = {file->manifest_index, 0, sizeof(file->manifest), file->manifest};
    status = aotx_ccir_read_batch(&file->view, &manifest, 1);
    if (status) goto failed;
    file->checkpoint_index = UINT32_MAX;
    for (uint32_t i = 0; i < file->view.count; ++i) {
        const aotx_ccir_section *s = &file->view.sections[i];
        if (s->type == AOTX_CCIR_CHECKPOINT && !memcmp(s->id, file->manifest + 24, 16)) {
            file->checkpoint_index = i; file->checkpoint_bytes = s->bytes;
        }
        if (s->type == AOTX_CCIR_TAIL && !memcmp(s->id, file->manifest + 40, 16)) {
            file->tail_index = i; file->tail_bytes = s->bytes;
        }
    }
    if (file->checkpoint_index == UINT32_MAX || file->checkpoint_bytes < AOTX_COG_HEADER ||
        file->checkpoint_bytes > AOTX_COG_IMAGE || file->tail_bytes > AOTX_COG_IMAGE ||
        (file->tail_index != UINT32_MAX && file->tail_bytes < AOTX_COG_HEADER)) {
        status = AOTX_CCIR_LIMIT; goto failed;
    }
    file->checkpoint = malloc((size_t)file->checkpoint_bytes);
    file->tail = file->tail_bytes ? malloc((size_t)file->tail_bytes) : NULL;
    if (!file->checkpoint || (file->tail_bytes && !file->tail)) { status = AOTX_CCIR_IO; goto failed; }
    aotx_ccir_read reads[2] = {
        {file->checkpoint_index, 0, (size_t)file->checkpoint_bytes, file->checkpoint},
        {file->tail_index, 0, (size_t)file->tail_bytes, file->tail}};
    status = aotx_ccir_read_batch(&file->view, reads, file->tail_bytes ? 2 : 1);
    if (!status) status = aotx_cognitive_binding(file);
    if (!status) return AOTX_CCIR_OK;
failed:
    aotx_cognitive_file_close(file);
    return status;
}

int aotx_cognitive_file_write(aotx_cognitive_file *file, const char *path,
                              const unsigned char *checkpoint, uint64_t bytes) {
    if (bytes < AOTX_COG_HEADER || bytes > AOTX_COG_IMAGE ||
        memcmp(checkpoint + 48, file->view.lineage, 16) ||
        aotx_cognitive_le(checkpoint + 32, 8) != file->view.meta.durable_sequence ||
        aotx_cognitive_le(checkpoint + 40, 8) != file->view.meta.source_tick) return AOTX_CCIR_INVALID;
    aotx_ccir_input inputs[AOTX_CCIR_SECTIONS];
    memset(inputs, 0, sizeof(inputs));
    unsigned char manifest[AOTX_CCIR_MANIFEST_BYTES];
    aotx_ccir_manifest(manifest, file->view.sections[file->checkpoint_index].id, NULL);
    uint32_t count = 0;
    for (uint32_t i = 0; i < file->view.count; ++i) {
        if (i == file->tail_index) continue;
        aotx_ccir_input *in = &inputs[count++];
        in->section = file->view.sections[i];
        in->source = AOTX_CCIR_FILE; in->fd = file->view.fd;
        in->source_offset = in->section.offset;
        if (i == file->checkpoint_index) {
            in->source = AOTX_CCIR_MEMORY; in->data = checkpoint; in->section.bytes = bytes;
        } else if (i == file->manifest_index) {
            in->source = AOTX_CCIR_MEMORY; in->data = manifest;
        }
    }
    aotx_ccir_meta meta = file->view.meta;
    meta.checkpoint_sequence = meta.durable_sequence;
    return aotx_ccir_create(path, file->view.lineage, inputs, count, &meta, NULL);
}

void aotx_cognitive_file_close(aotx_cognitive_file *file) {
    if (file->view.fd >= 0) aotx_ccir_close(&file->view);
    free(file->checkpoint); free(file->tail);
    file->checkpoint = file->tail = NULL;
}
int aotx_cognitive_file_options(int argc, char **argv) {
    if (argc == 2 && !strcmp(argv[1], "--help")) {
        puts("Use: aotx_ccir_state INPUT OUTPUT\nRestore typed state on the GPU and write a new checkpoint file.\n"
             "The output must not exist. This command does not load a language model.");
        return 1;
    }
    if (argc != 3) { fputs("Use: aotx_ccir_state INPUT OUTPUT\n", stderr); return -1; }
    return 0;
}
void aotx_cognitive_file_report(int status, uint64_t sequence, uint64_t bytes, uint32_t fallback) {
    static const char *const state[] = {
        "valid", "invalid format", "capacity limit", "invalid reference", "scope conflict",
        "version conflict", "sequence conflict", "source conflict", "invalid media layout",
        "missing object", "stale object", "access denied"};
    if (status) {
        const char *reason = status >= 200 && status < 212 ? state[status - 200] :
            status == 100 ? "device transfer or launch failed" : aotx_ccir_status_text(status);
        fprintf(stderr, "state export failed: %s (%d)\n", reason, status);
    }
    else printf("checkpoint sequence=%llu bytes=%llu fallback=%u\n",
                (unsigned long long)sequence, (unsigned long long)bytes, fallback);
}
