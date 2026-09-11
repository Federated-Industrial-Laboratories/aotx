/* Purpose: Refuse invalid runtime references, incompatible files and changed startup generations.
 * Owns: Distinct data modules, prepared metadata and changed source generations.
 * Threading: One disk process checks complete batches at one and 64 modules.
 * Lifetime: All temporary files, reader handles and writer leases end with each case. */
#include "runtime_dependency_fixture.h"
#include "disk/runtime/activate.h"
#include "cognitive/checkpoint_io.h"
#include "profile/profile.cuh"

static void prepared(aotx_dependency_fixture *f) {
    const char *phrases = "I cannot\n";
    asset(f, "quality/refusal-phrases.txt", 1, phrases, strlen(phrases));
    aotx_ccir_put(f->index.header + 16, f->index.count, 4);
    f->input[3].section.bytes = AOTX_RUNTIME_HEADER + f->index.count * AOTX_RUNTIME_ROW;
#ifdef AOTX_AFFECT
    aotx_ccir_put(f->index.header + 20, AOTX_RUNTIME_AFFECT, 4);
#endif
    memcpy(f->memory, "AOTXOBJ1", 8); aotx_ccir_put(f->memory + 8, 1, 4);
    aotx_ccir_put(f->memory + 12, 256, 4);
    aotx_ccir_put(f->memory + 32, 1, 8); aotx_ccir_put(f->memory + 40, 1, 8);
    f->memory[48] = 73; aotx_ccir_put(f->memory + 80, 128, 8);
    memcpy(f->live, "AOTXLCP1", 8); aotx_ccir_put(f->live + 8, 1, 4);
    aotx_ccir_put(f->live + 12, AOTX_CP_ROW, 4); aotx_ccir_put(f->live + 24, 128, 8);
    f->live[32] = 73; aotx_ccir_put(f->live + 48, 1, 8); aotx_ccir_put(f->live + 56, 1, 8);
    aotx_ccir_put(f->live + 64, 1, 8); aotx_ccir_put(f->live + 72, 1, 8);
}
static void test(const char *root, unsigned n, unsigned changed) {
    char path[256], journal[256];
    snprintf(path, sizeof(path), "%s/state.aotxccir", root);
    snprintf(journal, sizeof(journal), "%s/journal", root);
    aotx_dependency_fixture *f = calloc(1, sizeof(*f)); CHECK(f != NULL); if (!f) return;
    make(f, n, 0); prepared(f);
    unsigned char lineage[16] = {73}; aotx_ccir_meta meta = {1, 1, 1};
    CHECK(!aotx_ccir_create(path, lineage, f->input, f->count, &meta, NULL));
    aotx_runtime_boot boot;
    int rc = aotx_runtime_prepare(path, journal, 86, &boot); CHECK(!rc);
    if (rc) { unlink(path); free(f); return; }
    unsigned char *image = NULL; uint32_t bytes = 0;
    CHECK(!aotx_checkpoint_file_read(path, &image, &bytes));
    if (changed) {
        f->body[n - 1][0] = 'X';
        for (uint32_t i = 0; i < f->index.count; ++i) {
            aotx_ccir_input *in = f->input + i + 5;
            if (in->data == f->body[n - 1])
                aotx_ccir_hash(in->data, in->section.bytes, f->index.rows[i] + 32);
        }
        CHECK(!aotx_ccir_append(path, f->input, f->count, &meta, NULL));
    }
    aotx_checkpoint_ring ring = {0}; ring.boot = 75;
    aotx_checkpoint_disk disk = {0}; disk.view.fd = -1;
    disk.runtime = 1; disk.runtime_sequence = 100; disk.path = path; disk.journal = journal; disk.ring = &ring;
    memcpy(disk.runtime_revision, boot.revision, 32);
    if (image) {
        rc = aotx_checkpoint_file_write(&disk, image, bytes);
        CHECK(rc == (changed ? AOTX_CCIR_CHANGED : AOTX_CCIR_BUSY));
        CHECK(disk.runtime_verified == !changed);
        CHECK(disk.view.generation == (changed ? 2u : 1u));
        CHECK(!ring.consumed && !ring.generation);
        if (changed) {
            CHECK(aotx_checkpoint_file_write(&disk, image, bytes) == AOTX_CCIR_CHANGED);
            CHECK(!disk.runtime_verified && !ring.consumed);
        }
    }
    aotx_ccir_close(&disk.view); aotx_runtime_release(&boot); free(image); free(f);
    CHECK(!rmdir(journal)); CHECK(!unlink(path));
}
static void compatibility(const char *root, unsigned n) {
    char path[256], journal[256];
    snprintf(path, sizeof(path), "%s/state.aotxccir", root);
    snprintf(journal, sizeof(journal), "%s/journal", root);
    aotx_dependency_fixture *f = calloc(1, sizeof(*f)); CHECK(f != NULL); if (!f) return;
    for (unsigned defect = 0; defect < 6; ++defect) {
        make(f, n, 0); prepared(f); unsigned char *h = f->index.header;
        if (defect == 0) aotx_ccir_put(h + 20, aotx_ccir_u32(h + 20) ^ AOTX_RUNTIME_AFFECT, 4);
        if (defect == 1) aotx_ccir_put(h + 24, AOTX_WIRE_LAYOUT + 1, 4);
        if (defect == 2) aotx_ccir_put(h + 28, AOTX_SLOTS + 1, 4);
        if (defect == 3) aotx_ccir_put(h + 32, AOTX_COG_OBJECTS + 1, 4);
        if (defect == 4) aotx_ccir_put(h + 40, (uint64_t)AOTX_COG_PAYLOAD + 1, 8);
        if (defect == 5) aotx_ccir_put(h + 36, 87, 4);
        unsigned char lineage[16] = {73}; aotx_ccir_meta meta = {1, 1, 1};
        CHECK(!aotx_ccir_create(path, lineage, f->input, f->count, &meta, NULL));
        aotx_runtime_boot boot;
        CHECK(aotx_runtime_prepare(path, journal, 86, &boot) == AOTX_CCIR_UNSUPPORTED);
        CHECK(!boot.owned && !boot.root[0]);
        CHECK(access(journal, F_OK) != 0);
        aotx_runtime_release(&boot); rmdir(journal); CHECK(!unlink(path));
    }
    free(f);
}
static void index_references(const char *root, unsigned n) {
    char path[256], journal[256];
    snprintf(path, sizeof(path), "%s/reference.aotxccir", root);
    snprintf(journal, sizeof(journal), "%s/reference-journal", root);
    aotx_dependency_fixture *f = calloc(1, sizeof(*f)); CHECK(f != NULL); if (!f) return;
    for (unsigned mode = 0; mode < 4; ++mode) {
        make(f, n, 0); prepared(f);
        aotx_ccir_input inputs[141];
        memcpy(inputs, f->input, f->count * sizeof(*inputs));
        aotx_ccir_input *extra = inputs + f->count;
        *extra = f->input[3]; extra->section.id[15] = 99; extra->section.flags = 0;
        if (mode != 2) {
            extra->section.schema = 99; extra->section.bytes = 8; extra->data = "BADINDEX";
        }
        if (mode == 1 || mode == 2) memcpy(f->manifest + 72, extra->section.id, 16);
        if (mode == 3) f->manifest[87] = 77;
        unsigned char lineage[16] = {73}; aotx_ccir_meta meta = {1, 1, 1};
        int made = aotx_ccir_create(path, lineage, inputs, f->count + 1, &meta, NULL);
        CHECK(mode ? made != 0 : made == 0);
        if (mode) CHECK(access(path, F_OK) != 0);
        if (!made) {
            aotx_runtime_boot boot;
            int rc = aotx_runtime_prepare(path, journal, 86, &boot);
            CHECK(mode ? rc != 0 : rc == 0);
            aotx_runtime_release(&boot);
            if (!rc) CHECK(!rmdir(journal));
            CHECK(!unlink(path));
        }
    }
    free(f);
}
int main(void) {
    char root[] = "/tmp/aotx-runtime-publication-XXXXXX";
    CHECK(mkdtemp(root) != NULL);
    test(root, 1, 0); test(root, 1, 1); test(root, 64, 0); test(root, 64, 1);
    compatibility(root, 1); compatibility(root, 64);
    index_references(root, 1); index_references(root, 64);
    CHECK(!rmdir(root));
    printf("runtime publication: %u checks, %u failures\n", checks, failures);
    return failures ? 1 : 0;
}
