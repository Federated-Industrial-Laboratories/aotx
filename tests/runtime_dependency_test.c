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
int main(void) {
    char root[] = "/tmp/aotx-dependencies-XXXXXX", path[256];
    CHECK(mkdtemp(root) != NULL); snprintf(path, sizeof(path), "%s/state.aotxccir", root);
    cases(path, 1); cases(path, 64); CHECK(!rmdir(root));
    printf("runtime dependencies: %u checks, %u failures\n", checks, failures);
    return failures ? 1 : 0;
}
