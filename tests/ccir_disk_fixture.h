/* Purpose: Supply distinct section batches and bounded file helpers for CCIR checks.
 * Owns: Fixed fixture buffers and assertion counts.
 * Threading: One test process, with child processes for fault checks.
 * Lifetime: One test executable. */
#ifndef AOTX_CCIR_DISK_FIXTURE_H
#define AOTX_CCIR_DISK_FIXTURE_H
#include "disk/ccir/internal.h"
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static unsigned int aotx_checks, aotx_failures;
#define CHECK(x) do { aotx_checks++; if (!(x)) { aotx_failures++; \
    fprintf(stderr, "line %d: %s\n", __LINE__, #x); } } while (0)

typedef struct aotx_ccir_fixture {
    aotx_ccir_input inputs[66];
    unsigned char manifest[AOTX_CCIR_MANIFEST_BYTES];
    unsigned char payload[65][512];
    unsigned char lineage[16];
    aotx_ccir_meta meta;
    uint32_t count;
} aotx_ccir_fixture;

static void fixture(aotx_ccir_fixture *f, uint32_t n)
{
    uint32_t i, j;
    memset(f, 0, sizeof(*f));
    f->count = n + 2u; f->lineage[0] = 73u; f->lineage[15] = (unsigned char)n;
    f->meta.checkpoint_sequence = f->meta.durable_sequence = n;
    f->meta.source_tick = 100u + n;
    for (i = 0; i < f->count; i++) {
        aotx_ccir_input *in = &f->inputs[i];
        in->section.type = i < 2u ? i + 1u : 100u + i;
        in->section.schema = (uint16_t)(i < 2u ? 1u : i + 3u);
        in->section.flags = i < 2u ? AOTX_CCIR_REQUIRED : 0u;
        in->section.id[0] = (unsigned char)(i + 1u);
        in->section.id[15] = (unsigned char)n;
        in->section.alignment = 128u;
        if (i) {
            for (j = 0; j < 512u; j++)
                f->payload[i - 1u][j] = (unsigned char)((i * 71u + j * 31u) % 253u);
            in->data = f->payload[i - 1u]; in->section.bytes = 257u + i;
        }
    }
    aotx_ccir_manifest(f->manifest, f->inputs[1].section.id, NULL);
    f->inputs[0].data = f->manifest; f->inputs[0].section.bytes = sizeof(f->manifest);
}
static int copy_file(const char *source, const char *destination)
{
    unsigned char buffer[65536];
    int in = open(source, O_RDONLY), out, rc = 0;
    ssize_t n;
    if (in < 0) return 1;
    out = open(destination, O_CREAT | O_TRUNC | O_WRONLY, 0600);
    if (out < 0) { close(in); return 1; }
    while ((n = read(in, buffer, sizeof(buffer))) > 0) {
        ssize_t at = 0;
        while (at < n) {
            ssize_t put = write(out, buffer + at, (size_t)(n - at));
            if (put <= 0) { rc = 1; break; }
            at += put;
        }
        if (rc) break;
    }
    if (n < 0) rc = 1;
    close(in); close(out);
    return rc;
}
static int reopen_generation(const char *path, uint64_t generation)
{
    aotx_ccir_view view;
    int rc = aotx_ccir_open(path, NULL, &view);
    if (rc) return 0;
    rc = view.generation == generation;
    aotx_ccir_close(&view);
    return rc;
}
#endif
