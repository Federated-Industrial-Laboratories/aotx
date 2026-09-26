/* Purpose: Check exact control evidence through complete-file activation and reads.
 * Owns: Distinct portable component batches and damaged reference cases.
 * Threading: One disk process at batch sizes one and 64.
 * Lifetime: Each complete file is removed after its checks. */
#include "tests/runtime_dependency_fixture.h"
#include "disk/runtime/qualification.h"

typedef struct aotx_qualification_batch {
    unsigned char vector[64][96], binding[64][AOTX_CONTROL_BYTES];
    char qualification[64][8192], evidence[5][96];
} aotx_qualification_batch;

static void cases(const char *path, unsigned n) {
    aotx_dependency_fixture *f = calloc(1, sizeof(*f));
    aotx_qualification_batch *batch = calloc(1, sizeof(*batch));
    CHECK(f != NULL && batch != NULL); if (!f || !batch) { free(f); free(batch); return; }
    unsigned char lineage[16] = {75}; aotx_ccir_meta meta = {1, 1, 1};
    const char *keys[] = {"binding", "source", "commitments", "examples", "calibration", "acceptance", "consumer"};
    for (unsigned defect = 0; defect < 8; ++defect) {
        make(f, 1, 0);
        for (unsigned j = 0; j < 5; ++j) {
            char name[256]; snprintf(name, sizeof(name), "identity/evidence-%u.txt", j);
            snprintf(batch->evidence[j], sizeof(batch->evidence[j]), "Source %u, evidence file %u.\n", n, j);
            asset(f, name, 1, batch->evidence[j], strlen(batch->evidence[j]));
        }
        for (unsigned i = 0; i < n; ++i) {
            unsigned bad = i + 1 == n ? defect : 0;
            char name[256], control[256], binding[256], names[7][256];
            if (!i) strcpy(control, "control/vector.aotxvec");
            else snprintf(control, sizeof(control), "control/vector-%u.aotxvec", i);
            snprintf(binding, sizeof(binding), "%s.binding", control);
            if (i) {
                memcpy(batch->vector[i], f->model, sizeof(f->model)); batch->vector[i][95] ^= i;
                memcpy(batch->binding[i], f->control[0], AOTX_CONTROL_BYTES);
                aotx_ccir_hash(batch->vector[i], sizeof(f->model), batch->binding[i] + 568);
                asset(f, control, 1, batch->vector[i], sizeof(f->model));
                asset(f, binding, 1, batch->binding[i], AOTX_CONTROL_BYTES);
            }
            strcpy(names[0], binding);
            for (unsigned j = 1; j < 6; ++j) snprintf(names[j], sizeof(names[j]), "identity/evidence-%u.txt", j - 1);
            strcpy(names[6], "modules/conductor/body.txt");
            char *q = batch->qualification[i];
            snprintf(q, sizeof(batch->qualification[i]), "{\"schema\":%u,\"kind\":1,\"status\":\"accepted\",\"checks\":%u,\"doses\":[%u]",
                bad == 3 ? 2 : 1, bad == 4 ? 127 : 255, 5000 + i);
            for (unsigned j = 0; j < 7; ++j) {
                unsigned char digest[32]; char hex[65], part[512]; unsigned found = 0;
                for (unsigned k = 0; k < f->index.count; ++k)
                    if (!strcmp(names[j], (char *)f->index.rows[k] + 64)) {
                        memcpy(digest, f->index.rows[k] + 32, 32); ++found;
                    }
                CHECK(found == 1);
                if ((bad == 1 && j == 1) || (bad == 5 && j == 6)) digest[(i + j) % 32] ^= 1;
                if (bad == 2 && j == 2) strcpy(names[j], "identity/missing.txt");
                aotx_sha256_text(digest, hex);
                snprintf(part, sizeof(part), ",\"%s\":{\"file\":\"%s\",\"sha256\":\"%s\"}", keys[j], names[j], hex);
                strcat(q, part);
            }
            strcat(q, "}");
            snprintf(name, sizeof(name), "%s.qualification", bad == 7 ? "missing" : control);
            asset(f, name, 1, q, strlen(q) - (bad == 6));
        }
        unsigned components = 0;
        for (unsigned i = 0; i < f->index.count; ++i)
            components += strstr((char *)f->index.rows[i] + 64, ".qualification") != NULL;
        CHECK(components == n && f->count <= AOTX_CCIR_SECTIONS);
        aotx_ccir_put(f->index.header + 16, f->index.count, 4);
        f->input[3].section.bytes = AOTX_RUNTIME_HEADER + f->index.count * AOTX_RUNTIME_ROW;
        CHECK(!aotx_ccir_create(path, lineage, f->input, f->count, &meta, NULL));
        aotx_ccir_view view; int rc = aotx_ccir_open(path, NULL, &view); CHECK(!rc);
        if (!rc) {
            rc = aotx_runtime_dependencies(&view); CHECK(defect ? rc != 0 : rc == 0);
            aotx_ccir_close(&view);
        }
        if (!defect) {
            unsigned char previous[32] = {0}; aotx_control_permit permit;
            for (unsigned i = 0; i < n; ++i) {
                char name[256];
                if (!i) strcpy(name, "control/vector.aotxvec");
                else snprintf(name, sizeof(name), "control/vector-%u.aotxvec", i);
                CHECK(!aotx_qualification_read(path, name, 1, &permit));
                CHECK(permit.status == 1 && permit.count == 1 && permit.dose[0] == (int)(5000 + i));
                CHECK(memcmp(previous, permit.digest, 32)); memcpy(previous, permit.digest, 32);
            }
            CHECK(!aotx_qualification_read(path, "control/probe.aotxprb", 2, &permit) && !permit.status);
        }
        CHECK(!unlink(path));
    }
    free(batch); free(f);
}
int main(void) {
    char root[] = "/tmp/aotx-qualification-runtime-XXXXXX", path[256];
    CHECK(mkdtemp(root) != NULL); snprintf(path, sizeof(path), "%s/state.aotxccir", root);
    cases(path, 1); cases(path, 64); CHECK(!rmdir(root));
    printf("qualification runtime: %u checks, %u failures\n", checks, failures);
    return failures || checks < 3000;
}
