/* Purpose: Pack independent byte fixtures for the recorded recall command.
 * Owns: Temporary input buffers and distinct optional file sections.
 * Threading: One CCIR create call takes the complete section batch.
 * Lifetime: The parent test owns the input and output paths. */
#include "ccir/ccir.h"
#include "cognitive/format.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t aotx_get(const unsigned char *p, unsigned bytes) {
    uint64_t value = 0;
    for (unsigned i = 0; i < bytes; ++i) value |= (uint64_t)p[i] << (8 * i);
    return value;
}
static unsigned char *aotx_read(const char *path, size_t *bytes) {
    FILE *file = fopen(path, "rb");
    if (!file) return NULL;
    unsigned char *data = malloc(AOTX_COG_IMAGE + 1u);
    if (!data) { fclose(file); return NULL; }
    *bytes = fread(data, 1, AOTX_COG_IMAGE + 1u, file);
    int failed = ferror(file) || *bytes < AOTX_COG_HEADER || *bytes > AOTX_COG_IMAGE;
    fclose(file);
    if (failed) { free(data); return NULL; }
    return data;
}
static void aotx_section(aotx_ccir_input *in, uint32_t type, uint32_t id,
                          const void *data, size_t bytes) {
    memset(in, 0, sizeof(*in));
    in->section.type = type; in->section.schema = type < 4 ? 1 : 17;
    in->section.flags = type < 4 ? AOTX_CCIR_REQUIRED : 0;
    in->section.id[0] = (unsigned char)id; in->section.id[1] = (unsigned char)(id >> 8);
    in->section.id[15] = 0x91; in->section.alignment = 64;
    in->section.bytes = bytes; in->data = data;
}
int main(int argc, char **argv) {
    if (argc != 5) return 2;
    char *end = NULL;
    unsigned long n = strtoul(argv[4], &end, 10);
    if (!end || *end || (n != 1 && n != 64)) return 2;
    size_t checkpoint_bytes, tail_bytes = 0;
    unsigned have_tail = strcmp(argv[2], "-") != 0;
    unsigned char *checkpoint = aotx_read(argv[1], &checkpoint_bytes);
    unsigned char *tail = have_tail ? aotx_read(argv[2], &tail_bytes) : NULL;
    if (!checkpoint || (have_tail && !tail)) { free(checkpoint); free(tail); return 1; }
    unsigned char manifest[AOTX_CCIR_MANIFEST_BYTES], optional[64][97];
    aotx_ccir_input inputs[67];
    aotx_section(&inputs[0], 1, 1, manifest, sizeof(manifest));
    aotx_section(&inputs[1], 2, 2, checkpoint, checkpoint_bytes);
    if (have_tail) aotx_section(&inputs[2], 3, 3, tail, tail_bytes);
    aotx_ccir_manifest(manifest, inputs[1].section.id, have_tail ? inputs[2].section.id : NULL);
    for (unsigned i = 0; i < n; ++i) {
        for (unsigned j = 0; j < sizeof(optional[i]); ++j)
            optional[i][j] = (unsigned char)(i * 29 + j * 13 + n);
        aotx_section(&inputs[2 + have_tail + i], 9000 + i, 100 + i, optional[i], sizeof(optional[i]));
    }
    unsigned count = (unsigned)n + 2 + have_tail;
    aotx_ccir_input swap = inputs[0]; inputs[0] = inputs[count - 1]; inputs[count - 1] = swap;
    uint64_t sequence = aotx_get(checkpoint + 32, 8);
    aotx_ccir_meta meta = {sequence, have_tail ?
        aotx_get(tail + 32, 8) + aotx_get(tail + 20, 4) - 1 : sequence,
        aotx_get((have_tail ? tail : checkpoint) + 40, 8)};
    int status = aotx_ccir_create(argv[3], checkpoint + 48, inputs, count, &meta, NULL);
    free(checkpoint); free(tail);
    if (status) fprintf(stderr, "fixture failed: %s\n", aotx_ccir_status_text(status));
    return status ? 1 : 0;
}
