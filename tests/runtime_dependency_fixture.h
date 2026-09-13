/* Purpose: Supply complete runtime metadata for disk dependency and startup checks.
 * Owns: A small model, distinct data modules and encoded section batches.
 * Threading: One disk test process at batch sizes one and 64.
 * Lifetime: Fixture buffers belong to the calling test. */
#ifndef AOTX_RUNTIME_DEPENDENCY_FIXTURE_H
#define AOTX_RUNTIME_DEPENDENCY_FIXTURE_H
#include "disk/runtime/runtime.h"
#include "disk/ccir/internal.h"
#include "disk/modelfile/manifest.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static unsigned checks, failures;
#define CHECK(x) do { ++checks; if (!(x)) { ++failures; fprintf(stderr, "line %d: %s\n", __LINE__, #x); } } while (0)
typedef struct aotx_dependency_fixture {
    aotx_ccir_input input[AOTX_CCIR_SECTIONS];
    unsigned char manifest[96], memory[128], live[128], replay[128 + 8 + 576], model[96];
    aotx_runtime_index index;
    char models[8192], settings[1024], module[64][512], body[64][128];
    uint32_t count;
} aotx_dependency_fixture;

static void section(aotx_dependency_fixture *f, unsigned kind, unsigned schema,
    const void *data, size_t bytes) {
    aotx_ccir_input *in = f->input + f->count;
    in->section.id[0] = (unsigned char)kind; in->section.id[8] = (unsigned char)f->count;
    in->section.type = kind; in->section.schema = (uint16_t)schema;
    in->section.flags = AOTX_CCIR_REQUIRED; in->section.alignment = 128;
    in->section.bytes = bytes; in->data = data; ++f->count;
}
static void asset(aotx_dependency_fixture *f, const char *name, unsigned kind,
    const void *data, size_t bytes) {
    aotx_ccir_input *in = f->input + f->count;
    section(f, AOTX_CCIR_ASSET, 1, data, bytes);
    unsigned char *row = f->index.rows[f->index.count++];
    memcpy(row, in->section.id, 16); aotx_ccir_put(row + 16, kind, 4);
    aotx_ccir_put(row + 24, bytes, 8); aotx_ccir_hash(data, bytes, row + 32);
    strcpy((char *)row + 64, name);
}
static void make(aotx_dependency_fixture *f, unsigned n, unsigned defect) {
    memset(f, 0, sizeof(*f));
    section(f, 1, 3, f->manifest, 96); section(f, 2, 1, f->memory, 128);
    section(f, 4, 1, f->live, 128); section(f, 5, 1, f->index.header, 0);
    section(f, 7, 1, f->replay, 128);
    aotx_ccir_live_manifest(f->manifest, f->input[1].section.id, f->input[2].section.id);
    aotx_ccir_put(f->manifest + 8, 3, 4); memcpy(f->manifest + 72, f->input[3].section.id, 16);
    memcpy(f->replay, "AOTXRPL1", 8); aotx_ccir_put(f->replay + 8, 1, 4); aotx_ccir_put(f->replay + 12, 1, 4);
    memcpy(f->model, "GGUF", 4); aotx_ccir_put(f->model + 4, 3, 4); aotx_ccir_put(f->model + 8, 1, 8);
    aotx_ccir_put(f->model + 24, 6, 8); memcpy(f->model + 32, "weight", 6);
    aotx_ccir_put(f->model + 38, 1, 4); aotx_ccir_put(f->model + 42, 8, 8);
    for (unsigned i = 0; i < 32; ++i) f->model[64 + i] = (unsigned char)(3 * i + n);
    asset(f, "weights/model.gguf", 1, f->model, sizeof(f->model));
    aotx_manifest_entry entry = {0};
    strcpy(entry.name, "model"); strcpy(entry.role, "language"); strcpy(entry.path, "weights/model.gguf");
    strcpy(entry.source, "local:model"); strcpy(entry.revision, "1"); strcpy(entry.license, "Apache-2.0");
    aotx_sha256_text(f->index.rows[0] + 32, entry.sha256); entry.bytes = sizeof(f->model);
    if (defect == 1) strcpy(entry.path, "weights/missing.gguf");
    if (defect == 2) entry.sha256[0] = entry.sha256[0] == '0' ? '1' : '0';
    if (defect == 3) ++entry.bytes;
    if (defect == 4) entry.license[0] = 0;
    if (defect == 5) strcpy(entry.role, "embedding");
    CHECK(!aotx_manifest_write_line(f->models, sizeof(f->models), &entry));
    if (defect == 6) {
        size_t size = strlen(f->models); memmove(f->models + size, f->models, size + 1);
    }
    asset(f, defect == 7 ? "unused.jsonl" : "manifest.jsonl", 1, f->models, strlen(f->models));
    strcpy(f->settings, "sample.seed = 19\n");
    if (defect == 8) strcpy(f->settings, "unknown.key = 1\n");
    if (defect == 9) strcpy(f->settings, "window.on = 1\n");
    if (defect == 10) strcpy(f->settings, "sample.temperature = -1\n");
    asset(f, defect == 11 ? "unused-settings" : "settings", 1, f->settings, strlen(f->settings));
    for (unsigned i = 0; i < n; ++i) {
        char name[256], module[64];
        if (!i) strcpy(module, "conductor"); else snprintf(module, sizeof(module), "member-%u", i);
        if (!i && defect == 12) strcpy(module, "unbound");
        snprintf(f->module[i], sizeof(f->module[i]), "kind: %s\nname: %s\nversion: 1\nbody: %s\n",
            defect == 13 && i + 1 == n ? "tool" : "role", module,
            defect == 14 && i + 1 == n ? "missing.txt" : defect == 15 && i + 1 == n ? "../outside" : "body.txt");
        if (defect == 16 && i + 1 == n) strcat(f->module[i], "kind: skill\n");
        snprintf(name, sizeof(name), "modules/%s/module.manifest", module);
        asset(f, name, 2, f->module[i], strlen(f->module[i]));
        snprintf(name, sizeof(name), "modules/%s/body.txt", module);
        snprintf(f->body[i], sizeof(f->body[i]), "Keep the requirement for member %u with source %u.\n", i, 71 + i * 3);
        asset(f, name, 2, f->body[i], strlen(f->body[i]));
    }
    const char *reference = "{\"file\":\"weights/model.gguf\"}\n";
    if (defect == 20) reference = "{\"file\":\"missing-vector.bin\"}\n";
    asset(f, "steer.jsonl", 1, reference, strlen(reference));
    if (defect == 21) reference = "{\"file\":\"/outside-probe.bin\"}\n";
    asset(f, "probes.jsonl", 1, reference, strlen(reference));
    reference = defect == 22 ? "{\"composite\":[\"weights/model.gguf\",\"missing.bin\"]}\n" :
        "{\"composite\":[\"weights/model.gguf\",\"weights/model.gguf\"]}\n";
    asset(f, "affect/calibration.jsonl", 1, reference, strlen(reference));
    unsigned char *h = f->index.header;
    memcpy(h, "AOTXRT01", 8); aotx_ccir_put(h + 8, 1, 4); aotx_ccir_put(h + 12, AOTX_RUNTIME_ROW, 4);
    aotx_ccir_put(h + 16, f->index.count, 4); aotx_ccir_put(h + 24, AOTX_WIRE_LAYOUT, 4);
    aotx_ccir_put(h + 28, 64, 4); aotx_ccir_put(h + 32, 8192, 4);
    aotx_ccir_put(h + 36, 86, 4); aotx_ccir_put(h + 40, 16777216, 8); aotx_ccir_put(h + 48, 1, 4);
    strcpy((char *)h + 64, defect == 17 ? "language,language" : defect == 18 ? "language,,embedding" : "language");
    memcpy(h + 128, f->input[4].section.id, 16);
    if (defect == 19) aotx_ccir_put(h + 20, AOTX_RUNTIME_AFFECT, 4);
    f->input[3].section.bytes = AOTX_RUNTIME_HEADER + f->index.count * AOTX_RUNTIME_ROW;
    if (defect >= 23) {
        unsigned char *p = f->replay;
        aotx_ccir_put(p + 12, 2, 4); aotx_ccir_put(p + 16, 75, 8); aotx_ccir_put(p + 24, 17, 8);
        aotx_ccir_put(p + 32, 1234, 8); aotx_ccir_put(p + 40, defect == 27 ? 2 : 1, 8);
        aotx_ccir_put(p + 48, 1, 8); aotx_ccir_put(p + 56, 584, 8);
        aotx_ccir_put(p + 64, 7, 8); aotx_ccir_put(p + 72, 1, 8); aotx_ccir_put(p + 128, 576, 8);
        aotx_block_header *block = (aotx_block_header *)(p + 136);
        block->magic = AOTX_BLOCK_MAGIC; block->layout = AOTX_WIRE_LAYOUT; block->byte_len = 576;
        block->boot_id = 75; block->tick = 17; block->block_seq = 1; block->record_count = 2;
        for (unsigned j = 0; j < 2; ++j) {
            aotx_record_header *r = (aotx_record_header *)(p + 200 + j * 256);
            r->magic = AOTX_WIRE_MAGIC; r->layout = AOTX_WIRE_LAYOUT; r->header_bytes = 64;
            r->boot_id = 75; r->tick = 17; r->seq = j + 1; r->cls = AOTX_CLASS_A;
            r->type = j ? AOTX_REC_TICK_COMMIT : AOTX_REC_INPUT_LINE;
            r->body_len = j ? sizeof(aotx_commit_body) : 1;
            if (j) {
                ((aotx_commit_body *)((unsigned char *)r + 64))->state_hash = 1234;
                if (defect == 26) r->cls = AOTX_CLASS_B;
            } else *((unsigned char *)r + 64) = 'x';
        }
        f->input[4].section.bytes = sizeof(f->replay);
        memcpy(f->live, "AOTXLCP1", 8);
        aotx_ccir_put(f->live + 64, defect == 25 ? 8 : 7, 8);
        aotx_ccir_put(f->live + 72, defect == 24 ? 18 : 17, 8);
    }
}
#endif
