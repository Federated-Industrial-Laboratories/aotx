/* Purpose: Check complete runtime indices and bounded model reads across file generations.
 * Owns: Distinct model assets and temporary files for one and 64 components.
 * Threading: One process takes reader and writer leases in sequence.
 * Lifetime: Every test source and descriptor ends with its case. */
#include "tests/ccir_disk_fixture.h"
#include "disk/runtime/assets.h"
#include "disk/modelfile/modelfile.h"

typedef struct aotx_runtime_fixture {
    aotx_ccir_input inputs[71];
    unsigned char manifest[96], memory[128], live[128], replay[128];
    aotx_runtime_index index;
    unsigned char model[64][96];
    unsigned char lineage[16];
    aotx_ccir_meta meta;
    uint32_t count;
} aotx_runtime_fixture;
static void input(aotx_runtime_fixture *f, uint32_t at, uint32_t kind, uint16_t schema,
                    const void *data, uint64_t bytes) {
    aotx_ccir_input *in = f->inputs + at;
    in->section.type = kind; in->section.schema = schema; in->section.flags = AOTX_CCIR_REQUIRED;
    in->section.id[0] = (unsigned char)kind; in->section.id[8] = (unsigned char)at;
    in->section.alignment = 128; in->section.bytes = bytes; in->data = data;
}
static void make(aotx_runtime_fixture *f, uint32_t n) {
    memset(f, 0, sizeof(*f)); f->count = n + 5; f->lineage[0] = 72;
    f->meta = (aotx_ccir_meta){1, 1, 1};
    input(f, 0, 1, 3, f->manifest, 96); input(f, 1, 2, 1, f->memory, 128);
    input(f, 2, 4, 1, f->live, 128);
    input(f, 3, 5, 1, f->index.header, AOTX_RUNTIME_HEADER + n * AOTX_RUNTIME_ROW);
    input(f, 4, 7, 1, f->replay, 128);
    aotx_ccir_live_manifest(f->manifest, f->inputs[1].section.id, f->inputs[2].section.id);
    aotx_ccir_put(f->manifest + 8, 3, 4); memcpy(f->manifest + 72, f->inputs[3].section.id, 16);
    unsigned char *h = f->index.header;
    memcpy(h, "AOTXRT01", 8); aotx_ccir_put(h + 8, 1, 4); aotx_ccir_put(h + 12, AOTX_RUNTIME_ROW, 4);
    aotx_ccir_put(h + 16, n, 4); aotx_ccir_put(h + 24, AOTX_WIRE_LAYOUT, 4);
    aotx_ccir_put(h + 28, 64, 4); aotx_ccir_put(h + 32, 8192, 4);
    aotx_ccir_put(h + 36, 86, 4); aotx_ccir_put(h + 40, 16777216, 8); aotx_ccir_put(h + 48, 1, 4);
    strcpy((char *)h + 64, "language"); memcpy(h + 128, f->inputs[4].section.id, 16);
    memcpy(f->replay, "AOTXRPL1", 8); aotx_ccir_put(f->replay + 8, 1, 4); aotx_ccir_put(f->replay + 12, 1, 4);
    for (uint32_t i = 0; i < n; ++i) {
        unsigned char *model = f->model[i];
        memcpy(model, "GGUF", 4); aotx_ccir_put(model + 4, 3, 4); aotx_ccir_put(model + 8, 1, 8);
        aotx_ccir_put(model + 24, 6, 8); memcpy(model + 32, "weight", 6);
        aotx_ccir_put(model + 38, 1, 4); aotx_ccir_put(model + 42, 8, 8);
        for (uint32_t j = 0; j < 32; ++j) model[64 + j] = (unsigned char)(i * 13 + j * 7 + 1);
        input(f, i + 5, 6, 1, model, 96);
        unsigned char *row = f->index.rows[i];
        memcpy(row, f->inputs[i + 5].section.id, 16); aotx_ccir_put(row + 16, 1, 4);
        aotx_ccir_put(row + 24, 96, 8); aotx_ccir_hash(model, 96, row + 32);
        snprintf((char *)row + 64, 256, "weights/%u.gguf", i);
    }
}
static void cases(const char *path, const char *copy, uint32_t n) {
    aotx_runtime_fixture *f = calloc(1, sizeof(*f)); CHECK(f != NULL); if (!f) return;
    make(f, n);
    CHECK(!aotx_ccir_create(path, f->lineage, f->inputs, f->count, &f->meta, NULL));
    CHECK(copy_file(path, copy) == 0); CHECK(reopen_generation(copy, 1)); unlink(copy);
    CHECK(aotx_asset_begin(path) == 0);
    aotx_modelfile *held = NULL;
    for (uint32_t i = 0; i < n; ++i) {
        aotx_manifest_entry e = {0}; unsigned char got[32];
        snprintf(e.name, sizeof(e.name), "model-%u", i);
        strcpy(e.path, (char *)f->index.rows[i] + 64); e.bytes = 96;
        aotx_sha256_text(f->index.rows[i] + 32, e.sha256);
        CHECK(aotx_manifest_check(path, &e) == 0);
        aotx_modelfile *model = NULL; CHECK(aotx_modelfile_open_entry(path, &e, &model) == 0);
        if (!model) continue;
        CHECK(aotx_modelfile_tensor_count(model) == 1);
        CHECK(aotx_modelfile_read(model, 0, sizeof(got), got) == 0 && !memcmp(got, f->model[i] + 64, 32));
        CHECK(aotx_modelfile_read(model, 31, 2, got) != 0);
        FILE *stream = aotx_asset_stream(path, e.path); CHECK(stream != NULL);
        if (stream) {
            CHECK(fseek(stream, -32, SEEK_END) == 0 && fread(got, 1, 32, stream) == 32);
            CHECK(!memcmp(got, f->model[i] + 64, 32) && fgetc(stream) == EOF);
            CHECK(fseek(stream, 97, SEEK_SET) != 0); fclose(stream);
        }
        aotx_asset asset; CHECK(!aotx_asset_open(path, e.path, &asset));
        aotx_modelfile *bad = NULL;
        CHECK(aotx_modelfile_open_extent(e.name, asset.fd, asset.offset, 95, &bad) != 0 && !bad);
        CHECK(aotx_modelfile_open_extent(e.name, asset.fd, UINT64_MAX, 1, &bad) != 0 && !bad);
        aotx_asset_close(&asset);
        if (!i) held = model; else aotx_modelfile_close(model);
    }
    aotx_asset_end();
    CHECK(aotx_ccir_append(path, f->inputs, f->count, &f->meta, NULL) == AOTX_CCIR_BUSY);
    aotx_modelfile_close(held);
    for (uint32_t i = 0; i < f->count; ++i) f->inputs[i].source = AOTX_CCIR_REUSE;
    CHECK(aotx_ccir_append(path, f->inputs, f->count, &f->meta, NULL) == 0);
    CHECK(reopen_generation(path, 2));
    aotx_ccir_view writer; CHECK(!aotx_ccir_writer_open(path, NULL, &writer));
    CHECK(!aotx_ccir_writer_replace(&writer, path, f->inputs, f->count, &f->meta, NULL));
    aotx_ccir_close(&writer);
    aotx_asset asset; CHECK(!aotx_asset_open(path, "weights/0.gguf", &asset));
    unsigned char got[32]; CHECK(!aotx_asset_read(&asset, 64, 32, got) && !memcmp(got, f->model[0] + 64, 32));
    CHECK(aotx_asset_read(&asset, UINT64_MAX, 1, got) == AOTX_CCIR_INVALID);
    aotx_asset_close(&asset); unlink(path);
    for (unsigned defect = 0; defect < 9; ++defect) {
        make(f, n);
        if (defect == 0) f->index.rows[0][32] ^= 1;
        if (defect == 1) aotx_ccir_put(f->index.rows[0] + 24, 97, 8);
        if (defect == 2) strcpy((char *)f->index.rows[0] + 64, "../model.gguf");
        if (defect == 3) f->index.rows[0][320] = 1;
        if (defect == 4) f->index.header[128] ^= 1;
        if (defect == 5) f->index.header[48] = 2;
        if (defect == 6) f->index.rows[0][16] = 3;
        if (defect == 7) f->replay[16] = 1;
        if (defect == 8) f->manifest[88] = 1;
        CHECK(aotx_ccir_create(copy, f->lineage, f->inputs, f->count, &f->meta, NULL) != 0);
        CHECK(access(copy, F_OK) != 0);
    }
    free(f);
}
int main(void) {
    aotx_ccir_limits limits;
    aotx_ccir_default_limits(&limits);
    CHECK(limits.section_bytes == limits.file_bytes);
    char root[] = "/tmp/aotx-runtime-XXXXXX", path[256], copy[256];
    CHECK(mkdtemp(root) != NULL);
    snprintf(path, sizeof(path), "%s/state.aotxccir", root); snprintf(copy, sizeof(copy), "%s/copy.aotxccir", root);
    cases(path, copy, 1); cases(path, copy, 64); rmdir(root);
    printf("runtime disk: %u checks, %u failures\n", aotx_checks, aotx_failures);
    return aotx_failures ? 1 : 0;
}
