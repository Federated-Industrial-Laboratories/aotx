/* Purpose: Check required runtime metadata and refusal of incomplete component batches.
 * Owns: A small model and distinct data modules at batch sizes one and 64.
 * Threading: One disk process inspects files without CUDA or embedded execution.
 * Lifetime: All files and source buffers end with their case. */
#include "runtime_dependency_fixture.h"

static void cases(const char *path, unsigned n) {
    aotx_dependency_fixture *f = calloc(1, sizeof(*f)); CHECK(f != NULL); if (!f) return;
    unsigned char lineage[16] = {73}; aotx_ccir_meta meta = {1, 1, 1};
    for (unsigned defect = 0; defect < 28; ++defect) {
        make(f, n, defect);
        int made = aotx_ccir_create(path, lineage, f->input, f->count, &meta, NULL);
        if (defect == 24 || defect == 25) {
            CHECK(made != 0); CHECK(access(path, F_OK) != 0); continue;
        }
        CHECK(!made);
        aotx_ccir_view view;
        int rc = aotx_ccir_open(path, NULL, &view); CHECK(!rc);
        if (!rc) {
            rc = aotx_runtime_dependencies(&view);
            CHECK(defect && defect != 23 ? rc != 0 : rc == 0);
            aotx_ccir_close(&view);
        }
        CHECK(!unlink(path));
    }
    free(f);
}
static void controls(const char *path, unsigned n) {
    aotx_dependency_fixture *f = calloc(1, sizeof(*f)); CHECK(f != NULL); if (!f) return;
    unsigned char lineage[16] = {74}; aotx_ccir_meta meta = {1, 1, 1};
    for (unsigned kind = 0; kind < 3; ++kind) for (unsigned defect = 0; defect < 9; ++defect) {
        make(f, n, 0);
        unsigned char *raw = f->control[kind];
        if (defect == 1) raw[24 + n % 32] ^= 1;
        if (defect == 2) raw[56 + n % 32] ^= 1;
        if (defect == 3) raw[12] = 2;
        if (defect == 4) raw[16] = (kind + 1) % 3 + 1;
        if (defect == 5) raw[568 + n % 32] ^= 1;
        if (defect == 6) raw[600] = 1;
        if (defect == 8) { memcpy(raw, "AOTXCTL2", 8); raw[8] = 2; raw[20] = 1; }
        for (unsigned i = 0; i < f->index.count; ++i) for (unsigned j = 0; j < f->count; ++j) {
            if (f->input[j].data != raw || memcmp(f->index.rows[i], f->input[j].section.id, 16)) continue;
            if (defect == 7) {
                --f->input[j].section.bytes;
                aotx_ccir_put(f->index.rows[i] + 24, f->input[j].section.bytes, 8);
            }
            aotx_ccir_hash(raw, f->input[j].section.bytes, f->index.rows[i] + 32);
        }
        CHECK(!aotx_ccir_create(path, lineage, f->input, f->count, &meta, NULL));
        aotx_ccir_view view;
        int rc = aotx_ccir_open(path, NULL, &view); CHECK(!rc);
        if (!rc) {
            if (defect == 8 && kind == 0) {
                CHECK(!aotx_control_reference(&view, &f->index, "control/vector.aotxvec", AOTX_CONTROL_VECTOR, 1));
                CHECK(aotx_control_reference(&view, &f->index, "control/vector.aotxvec", AOTX_CONTROL_VECTOR, 0));
            }
            rc = aotx_runtime_dependencies(&view);
            CHECK(defect ? rc != 0 : rc == 0);
            aotx_ccir_close(&view);
        }
        CHECK(!unlink(path));
    }
    free(f);
}
int main(void) {
    char root[] = "/tmp/aotx-dependencies-XXXXXX", path[256];
    CHECK(mkdtemp(root) != NULL); snprintf(path, sizeof(path), "%s/state.aotxccir", root);
    cases(path, 1); cases(path, 64); controls(path, 1); controls(path, 64); CHECK(!rmdir(root));
    printf("runtime dependencies: %u checks, %u failures\n", checks, failures);
    return failures ? 1 : 0;
}
