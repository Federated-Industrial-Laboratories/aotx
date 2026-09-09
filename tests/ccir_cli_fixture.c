/* Purpose: Supply unordered directories and a second writer for CLI checks.
 * Owns: Bounded distinct file sections and command arguments.
 * Threading: Each call takes the normal CCIR file lease.
 * Lifetime: The parent test owns the output files. */
#include "disk/ccir/ccir.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void section(aotx_ccir_input *in, uint32_t type, uint32_t id,
                     const void *data, size_t bytes)
{
    memset(in, 0, sizeof(*in));
    in->section.type = type; in->section.schema = 1u;
    in->section.flags = type <= 3u ? AOTX_CCIR_REQUIRED : 0u;
    in->section.id[0] = (unsigned char)id;
    in->section.id[1] = (unsigned char)(id >> 8u);
    in->section.alignment = 128u; in->section.bytes = bytes; in->data = data;
}
static int make(const char *path, uint32_t n, uint32_t order, uint32_t tail,
                uint32_t serial)
{
    aotx_ccir_input inputs[67], temp;
    aotx_ccir_meta meta = {1u, tail ? 2u : 1u, 100u};
    unsigned char manifest[96] = {0}, payload[64][97], state[47], recorded[39];
    unsigned char lineage[16] = {0};
    uint32_t i, j, count = n + 2u + tail;
    lineage[0] = (unsigned char)n; lineage[1] = (unsigned char)serial;
    lineage[2] = (unsigned char)order; lineage[3] = (unsigned char)(tail + 1u);
    for (i = 0; i < sizeof(state); i++) state[i] = (unsigned char)(serial + i * 7u);
    for (i = 0; i < sizeof(recorded); i++) recorded[i] = (unsigned char)(serial + i * 17u);
    section(&inputs[0], 1u, 1u, manifest, sizeof(manifest));
    section(&inputs[1], 2u, 2u, state, sizeof(state));
    for (i = 0; i < n; i++) {
        for (j = 0; j < sizeof(payload[i]); j++)
            payload[i][j] = (unsigned char)(serial + i * 31u + j * 19u);
        section(&inputs[2u + i], 8000u + i, 3u + i, payload[i], sizeof(payload[i]));
        inputs[2u + i].section.schema = (uint16_t)(3u + i);
    }
    if (tail) section(&inputs[n + 2u], 3u, n + 3u, recorded, sizeof(recorded));
    aotx_ccir_manifest(manifest, inputs[1].section.id, tail ? inputs[n + 2u].section.id : NULL);
    if (order == 1u) {
        for (i = 0; i < count / 2u; i++) {
            temp = inputs[i]; inputs[i] = inputs[count - 1u - i]; inputs[count - 1u - i] = temp;
        }
    } else if (order == 2u && tail) {
        temp = inputs[2]; inputs[2] = inputs[n + 2u]; inputs[n + 2u] = temp;
    }
    return aotx_ccir_create(path, lineage, inputs, count, &meta, NULL);
}
static int add(const char *path, uint32_t serial)
{
    aotx_ccir_view view;
    aotx_ccir_input inputs[AOTX_CCIR_SECTIONS];
    aotx_ccir_meta meta;
    unsigned char extra[79];
    uint32_t count, i;
    int rc = aotx_ccir_open(path, NULL, &view);
    if (rc) return rc;
    count = view.count; meta = view.meta;
    if (count == AOTX_CCIR_SECTIONS) { aotx_ccir_close(&view); return AOTX_CCIR_LIMIT; }
    memset(inputs, 0, sizeof(inputs));
    for (i = 0; i < count; i++) {
        inputs[i].section = view.sections[i]; inputs[i].source = AOTX_CCIR_REUSE;
    }
    aotx_ccir_close(&view);
    for (i = 0; i < sizeof(extra); i++) extra[i] = (unsigned char)(serial * 29u + i);
    section(&inputs[count], 9000u, 1000u + serial, extra, sizeof(extra));
    return aotx_ccir_append(path, inputs, count + 1u, &meta, NULL);
}
int main(int argc, char **argv)
{
    uint32_t n, order, tail, serial;
    int rc;
    if (argc == 4 && !strcmp(argv[1], "add")) rc = add(argv[2], (uint32_t)strtoul(argv[3], NULL, 10));
    else if (argc == 7 && !strcmp(argv[1], "make")) {
        n = (uint32_t)strtoul(argv[3], NULL, 10); order = (uint32_t)strtoul(argv[4], NULL, 10);
        tail = (uint32_t)strtoul(argv[5], NULL, 10); serial = (uint32_t)strtoul(argv[6], NULL, 10);
        if ((n != 1u && n != 64u) || order > 2u || tail > 1u || serial > 64u) return 2;
        rc = make(argv[2], n, order, tail, serial);
    } else return 2;
    printf("ccir fixture: %s\n", aotx_ccir_status_text(rc));
    return rc;
}
