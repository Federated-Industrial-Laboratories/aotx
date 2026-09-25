/* Purpose: Check appraisal feature admission, packaging and durable runtime promotion.
 * Owns: Complete file fixtures, source manifests and framed journal batches.
 * Threading: One disk process tests distinct batches at one and 64 rows.
 * Lifetime: Each case removes its complete files and source bytes. */
#include "appraisal_runtime_disk_fixture.h"
#include <dirent.h>
#include <errno.h>
#include <sys/wait.h>
#include "disk/policy/file.h"
#include "appraisal_runtime_disk_replay.h"

static void admission(const char *root, unsigned n, unsigned defect) {
    aotx_dependency_fixture *f = calloc(1, sizeof(*f)); CHECK(f != NULL); if (!f) return;
    make(f, n, 0); uint64_t bytes = 0;
    unsigned char *memory = memory_batch(f, n, &bytes), prior[96], digest[32];
    if (!memory) { free(f); return; }
    unsigned char *config = row_payload(memory, 0), *queue = row_payload(memory, 3 * n - 2);
    unsigned char *assessment = row_payload(memory, 3 * n - 1), *relation = row_payload(memory, 3 * n);
    if (defect == 1) profile(f, 0);
    if (defect == 2) f->index.header[192] ^= 1;
    if (defect == 3) f->index.header[224] ^= 1;
    if (defect == 4) config[40] ^= 1;
    if (defect == 5) queue[64] ^= 1;
    if (defect == 6) assessment[32] ^= 1;
    if (defect == 7) relation[72] ^= 1;
    if (defect == 8) queue[96] ^= 1;
    if (defect == 9) assessment[64] ^= 1;
    if (defect == 10) relation[104] ^= 1;
    if (defect == 11) memset(queue + 96, 0, 32);
    if (defect == 12) { profile(f, 0); f->index.header[188] = 1; }
    if (defect == 13) queue[7] = '2';
    if (defect == 14) aotx_ccir_put(memory + AOTX_COG_HEADER + (3 * n - 2) * AOTX_COG_OBJECT + AOTX_CO_BYTES, 128, 8);
    if (defect == 15) aotx_ccir_put(memory + AOTX_COG_HEADER + AOTX_CO_OFFSET, bytes, 8);
    if (defect == 16) f->input[3].section.schema = 3;
    if (defect == 17) f->index.header[20] |= 64;
    if (defect == 18) strcpy((char *)f->index.header + 64, "embedding");
    if (defect == 19) {
        historical(f, prior, digest);
        memcpy(queue + 96, digest, 32); memcpy(assessment + 64, digest, 32); memcpy(relation + 104, digest, 32);
    }
    if (defect == 20) { memcpy(prior, f->model, 96); prior[95] ^= 7; asset(f, "weights/unused.gguf", 1, prior, 96);
        memcpy(queue + 96, f->index.rows[f->index.count - 1] + 32, 32); profile(f, 1); }
    if (defect == 21) aotx_ccir_put(assessment, 3, 4);
    if (defect == 22) aotx_ccir_put(relation + 8, 2, 4);
    if (defect == 23) aotx_ccir_put(queue + 12, 4, 4);
    if (defect == 24) { aotx_ccir_put(queue + 12, AOTX_APPRAISAL_PENDING, 4); memset(queue + 96, 0, 32); }
    if (defect == 25) f->index.header[188] = 2;
    if (defect == 26) memset(f->index.header + 224, 0, 32);
    if (defect == 27) strcpy((char *)f->index.header + 64, "language,language-q4");
    aotx_policy_file policy = {0};
    if (defect >= 28 && defect < 30) {
        aotx_policy_source s = {0}; s.config.mode = AOTX_POLICY_NATIVE; s.config.abi = defect - 27;
        s.config.state_schema = 1; s.config.state_bytes = 16; s.config.architecture = 86;
        s.config.threads = s.config.registers = 64; s.config.format = 1;
        s.config.minimum_move = s.config.backoff = 1;
        s.entry = "aotx_policy_entry"; s.provenance = "local source"; s.provenance_bytes = 12;
        s.license = "Apache-2.0"; s.license_bytes = 10;
        s.image = ".version 8.0\n.target sm_86\n.address_size 64\n.visible .entry aotx_policy_entry() { ret; }\n";
        s.image_bytes = strlen(s.image);
        char bundle[256]; snprintf(bundle, sizeof(bundle), "%s/policy.bin", root);
        CHECK(!aotx_policy_file_write(bundle, &s)); CHECK(!aotx_policy_file_read(bundle, NULL, 0, &policy));
        CHECK(!unlink(bundle));
        if (policy.buffer) {
            asset(f, "policy.bin", 3, policy.buffer, policy.buffer_bytes);
            aotx_ccir_put(f->index.header + 20, aotx_ccir_u32(f->index.header + 20) | AOTX_RUNTIME_POLICY, 4);
            profile(f, 1);
        }
    }
    if (defect == 30) {
        const unsigned char old[32] = AOTX_APPRAISAL_LEGACY_PROCESSOR_BYTES;
        memcpy(f->index.header + 192, old, 32); memcpy(config + 40, old, 32);
        for (unsigned i = 0; i < n; ++i) {
            memcpy(row_payload(memory, 3 * i + 1) + 64, old, 32);
            memcpy(row_payload(memory, 3 * i + 2) + 32, old, 32);
            memcpy(row_payload(memory, 3 * i + 3) + 72, old, 32);
        }
    }
    int valid = defect == 0 || defect == 19 || defect == 24 || defect >= 28;
    char path[256], journal[256]; snprintf(path, sizeof(path), "%s/admit.aotxccir", root);
    snprintf(journal, sizeof(journal), "%s/admit-journal", root);
    unsigned char lineage[16] = {73}; aotx_ccir_meta meta = {1, 1, 1};
    int rc = aotx_ccir_create(path, lineage, f->input, f->count, &meta, NULL);
    aotx_ccir_view view = {0}; view.fd = -1;
    if (!rc) rc = aotx_ccir_open(path, NULL, &view);
    if (!rc) rc = aotx_runtime_dependencies(&view);
    if ((rc == 0) != valid) fprintf(stderr, "admission n=%u case=%u status=%d\n", n, defect, rc);
    CHECK((rc == 0) == valid);
    if (valid && !rc) {
        aotx_runtime_index *index = malloc(sizeof(*index)); CHECK(index != NULL);
        if (index) {
            CHECK(!aotx_runtime_index_read(view.fd, &view, index));
            CHECK(aotx_runtime_schema(aotx_ccir_u32(index->header + 20)) == 4);
            CHECK(!memcmp(index->header + 192, config + 40, 32));
            CHECK(!memcmp(index->header + 224, f->index.rows[0] + 32, 32)); free(index);
        }
        aotx_ccir_close(&view);
        unsigned char *saved = NULL; uint32_t saved_bytes = 0;
        CHECK(!aotx_checkpoint_file_read(path, &saved, &saved_bytes));
        CHECK(saved && saved_bytes == bytes + 128 && !memcmp(saved + 128, memory, bytes)); free(saved);
        aotx_runtime_boot boot; rc = aotx_runtime_prepare(path, journal, 86, &boot); CHECK(!rc);
        if (!rc) {
            CHECK(boot.features & AOTX_RUNTIME_APPRAISAL);
            if (policy.buffer) {
                aotx_policy_file saved_policy; CHECK(!aotx_policy_file_read(boot.policy, NULL, 0, &saved_policy));
                CHECK(saved_policy.buffer_bytes == policy.buffer_bytes && !memcmp(saved_policy.digest, policy.digest, 32));
                CHECK(saved_policy.config.abi == defect - 27); aotx_policy_file_close(&saved_policy);
            }
            aotx_runtime_release(&boot); CHECK(!rmdir(journal));
        }
        CHECK(!aotx_runtime_promote_shared(path)); CHECK(!aotx_ccir_open(path, NULL, &view));
        CHECK(!aotx_runtime_dependencies(&view));
        index = malloc(sizeof(*index)); CHECK(index != NULL);
        if (index) {
            CHECK(!aotx_runtime_index_read(view.fd, &view, index));
            CHECK((aotx_ccir_u32(index->header + 20) & (AOTX_RUNTIME_SHARED | AOTX_RUNTIME_APPRAISAL)) ==
                (AOTX_RUNTIME_SHARED | AOTX_RUNTIME_APPRAISAL));
            CHECK(!memcmp(index->header + 188, f->index.header + 188, 68)); free(index);
        }
    }
    aotx_ccir_close(&view); unlink(path); aotx_policy_file_close(&policy); free(memory); free(f);
}
static void pack(unsigned n) {
    aotx_dependency_fixture *f = calloc(1, sizeof(*f));
    aotx_runtime_pack *p = calloc(1, sizeof(*p)); CHECK(f && p);
    if (!f || !p) { free(f); free(p); return; }
    make(f, n, 0); uint64_t bytes = 0; unsigned char *memory = memory_batch(f, n, &bytes);
    if (!memory) { free(f); free(p); return; }
    FILE *file = tmpfile(); CHECK(file != NULL);
    if (file) {
        CHECK(fwrite(f->models, 1, strlen(f->models), file) == strlen(f->models)); CHECK(!fflush(file));
        for (unsigned defect = 0; defect < 4; ++defect) {
            profile(f, 0); p->index = f->index; p->count = f->count;
            memcpy(p->inputs, f->input, sizeof(f->input)); p->memory = memory; p->memory_bytes = bytes;
            for (unsigned i = 0; i < p->count; ++i) if (p->inputs[i].data == f->models) {
                p->inputs[i].source = AOTX_CCIR_FILE; p->inputs[i].fd = fileno(file);
            }
            unsigned char *q = row_payload(memory, 3 * n - 2);
            if (defect == 1) q[64] ^= 1;
            if (defect == 2) q[96] ^= 1;
            if (defect == 3) strcpy((char *)p->index.header + 64, "embedding");
            int rc = aotx_runtime_pack_appraisal(p); CHECK((rc == 0) == (defect == 0));
            if (!defect) { CHECK(aotx_runtime_schema(aotx_ccir_u32(p->index.header + 20)) == 4);
                CHECK(!memcmp(p->index.header + 192, processor, 32));
                CHECK(!memcmp(p->index.header + 224, f->index.rows[0] + 32, 32)); }
            if (defect == 1) q[64] ^= 1;
            if (defect == 2) q[96] ^= 1;
        }
        fclose(file);
    }
    free(memory); free(f); free(p);
}
static void mirror(const char *root, unsigned n, unsigned memory_only) {
    aotx_dependency_fixture *f = calloc(1, sizeof(*f)); CHECK(f != NULL); if (!f) return;
    make(f, n, 0); uint64_t bytes = 0; unsigned char *memory = memory_batch(f, n, &bytes);
    if (!memory) { free(f); return; }
    memcpy(f->memory, memory, 128); aotx_ccir_put(f->memory + 20, 0, 4); aotx_ccir_put(f->memory + 24, 0, 8);
    aotx_ccir_put(f->memory + 72, 128, 8); aotx_ccir_put(f->memory + 80, 128, 8);
    f->input[1].data = f->memory; f->input[1].section.bytes = 128; aotx_ccir_put(f->live + 24, 128, 8); profile(f, 0);
    char path[256], journal[256], segment[288]; snprintf(path, sizeof(path), "%s/mirror.aotxccir", root);
    snprintf(journal, sizeof(journal), "%s/mirror-journal", root); snprintf(segment, sizeof(segment), "%s/seg-000000.seg", journal);
    CHECK(!mkdir(journal, 0700)); unsigned char lineage[16] = {73}; aotx_ccir_meta meta = {1, 1, 1};
    CHECK(!aotx_ccir_create(path, lineage, f->input, f->count, &meta, NULL));
    aotx_checkpoint_ring ring = {0}; ring.boot = 75;
    aotx_checkpoint_disk disk = {0}; disk.view.fd = -1; disk.runtime = 1; disk.runtime_sequence = 1;
    disk.path = path; disk.journal = journal; disk.ring = &ring;
    CHECK(!aotx_ccir_writer_open(path, NULL, &disk.view));
    unsigned char *blocks = malloc((n + 1) * APPRAISAL_BLOCK); CHECK(blocks != NULL);
    if (blocks) {
        block_batch(blocks, n + 1, memory_only ? 14 : AOTX_APPRAISAL_CONTROL + n % 3);
        aotx_ccir_put(blocks + AOTX_BLOCK_HEADER_BYTES + AOTX_HEADER_BYTES + 4, 14, 4);
        aotx_segment_writer writer; CHECK(!aotx_segment_open(&writer, journal, 0));
        CHECK(!aotx_segment_put(&writer, blocks, APPRAISAL_BLOCK)); CHECK(!aotx_segment_sync(&writer));
        unsigned char initial[256]; memcpy(initial, f->live, 128); memcpy(initial + 128, f->memory, 128);
        CHECK(!aotx_runtime_checkpoint_write(&disk, initial, sizeof(initial), 128, 1));
        for (unsigned i = 1; i <= n; ++i) CHECK(!aotx_segment_put(&writer, blocks + i * APPRAISAL_BLOCK, APPRAISAL_BLOCK));
        CHECK(!aotx_segment_close(&writer));
        FILE *replay = NULL; uint64_t replay_bytes = 0; uint32_t features = 0;
        CHECK(!aotx_runtime_replay_collect_features(journal, 75, n + 1, 2, 2 * n + 1, 1048576,
            &replay, &replay_bytes, &features));
        CHECK(features == (memory_only ? 0u : AOTX_RUNTIME_APPRAISAL));
        CHECK(replay_bytes == 128 + (n + 1) * (8 + APPRAISAL_BLOCK)); if (replay) fclose(replay);
        uint64_t stored = memory_only ? bytes : 128;
        unsigned char *image = malloc((size_t)stored + 128); CHECK(image != NULL);
        if (image) {
            memcpy(image, initial, 128); memcpy(image + 128, memory_only ? memory : f->memory, stored);
            aotx_ccir_put(image + 24, stored, 8); aotx_ccir_put(image + 48, 2, 8);
            aotx_ccir_put(image + 56, n + 1, 8); aotx_ccir_put(image + 64, 2, 8); aotx_ccir_put(image + 72, n + 1, 8);
            aotx_ccir_put(image + 128 + 32, 2, 8); aotx_ccir_put(image + 128 + 40, n + 1, 8);
            disk.runtime_sequence = 2 * n + 1;
            CHECK(!aotx_runtime_checkpoint_write(&disk, image, stored + 128, 128, 0));
            CHECK(!aotx_runtime_dependencies(&disk.view));
            aotx_runtime_index *index = malloc(sizeof(*index)); CHECK(index != NULL);
            if (index) {
                CHECK(!aotx_runtime_index_read(disk.view.fd, &disk.view, index));
                CHECK(aotx_ccir_u32(index->header + 20) & AOTX_RUNTIME_APPRAISAL);
                CHECK(!memcmp(index->header + 192, processor, 32));
                unsigned char profile_bytes[68]; memcpy(profile_bytes, index->header + 188, 68);
                CHECK(!aotx_runtime_appraisal_checkpoint(&disk.view, index, f->memory, 128, 0));
                CHECK(!memcmp(profile_bytes, index->header + 188, 68)); free(index);
            }
            aotx_ccir_close(&disk.view);
            unsigned char *saved = NULL; uint32_t saved_bytes = 0;
            CHECK(!aotx_checkpoint_file_read(path, &saved, &saved_bytes));
            CHECK(saved && saved_bytes == stored + 128 && !memcmp(saved, image, saved_bytes)); free(saved); free(image);
        }
        free(blocks);
    }
    aotx_ccir_close(&disk.view); CHECK(!unlink(segment)); CHECK(!rmdir(journal)); CHECK(!unlink(path));
    free(memory); free(f);
}
static void replay_admission(const char *root, unsigned n) {
    for (unsigned op = 14; op <= 17; ++op) for (unsigned declared = 0; declared < 2; ++declared) {
        aotx_dependency_fixture *f = calloc(1, sizeof(*f)); CHECK(f != NULL); if (!f) return;
        make(f, n, 23); prepared(f, 128); profile(f, declared);
        aotx_ccir_put(f->live + 64, 7, 8); aotx_ccir_put(f->live + 72, 17, 8);
        aotx_record_header *r = (aotx_record_header *)(f->replay + 200);
        r->type = 33; r->body_len = 32; aotx_ccir_put((unsigned char *)r + AOTX_HEADER_BYTES + 4, op, 4);
        if (op == AOTX_APPRAISAL_RESULT) {
            unsigned char *part = (unsigned char *)r + AOTX_HEADER_BYTES;
            r->body_len = 96; aotx_ccir_put(part, 1, 4); aotx_ccir_put(part + 24, 64, 4);
            memcpy(part + 32, "AOTXAPS1", 8); aotx_ccir_put(part + 40, 1, 4);
            aotx_ccir_put(part + 64, AOTX_COG_DENIED, 4);
        }
        char path[256]; snprintf(path, sizeof(path), "%s/replay.aotxccir", root);
        unsigned char lineage[16] = {73}; aotx_ccir_meta meta = {1, 1, 1};
        CHECK(!aotx_ccir_create(path, lineage, f->input, f->count, &meta, NULL));
        aotx_ccir_view view; int rc = aotx_ccir_open(path, NULL, &view); CHECK(!rc);
        if (!rc) { rc = aotx_runtime_dependencies(&view); CHECK((rc == 0) == (declared || op == 14)); aotx_ccir_close(&view); }
        CHECK(!unlink(path)); free(f);
    }
}
static void legacy(unsigned n) {
    uint64_t start = 128 + (uint64_t)n * 256, bytes = start + (uint64_t)n * 32;
    unsigned char *memory = calloc(1, (size_t)bytes); CHECK(memory != NULL); if (!memory) return;
    memcpy(memory, "AOTXOBJ1", 8); aotx_ccir_put(memory + 8, 1, 4);
    aotx_ccir_put(memory + 12, 128, 4); aotx_ccir_put(memory + 16, 256, 4);
    aotx_ccir_put(memory + 20, n, 4); aotx_ccir_put(memory + 24, n * 32, 8);
    aotx_ccir_put(memory + 64, 128, 8); aotx_ccir_put(memory + 72, start, 8); aotx_ccir_put(memory + 80, bytes, 8);
    for (unsigned i = 0; i < n; ++i) {
        unsigned char *r = memory + 128 + i * 256;
        aotx_ccir_put(r + AOTX_CO_KIND, AOTX_COG_APPRAISAL, 2);
        aotx_ccir_put(r + AOTX_CO_OFFSET, i * 32, 8); aotx_ccir_put(r + AOTX_CO_BYTES, 32, 8);
        aotx_ccir_put(memory + start + i * 32, 1, 4);
    }
    uint32_t required = UINT32_MAX;
    CHECK(!aotx_runtime_appraisal_scan(memory, -1, 0, bytes, NULL, &required)); CHECK(!required);
    free(memory);
}
static void disk_file(const char *path, const void *data, size_t bytes) {
    char parent[512]; CHECK(strlen(path) < sizeof(parent)); strcpy(parent, path);
    for (char *p = parent + 1; *p; ++p) if (*p == '/') {
        *p = 0; CHECK(!mkdir(parent, 0700) || errno == EEXIST); *p = '/';
    }
    FILE *file = fopen(path, "wb"); CHECK(file != NULL); if (!file) return;
    CHECK(fwrite(data, 1, bytes, file) == bytes); CHECK(!fclose(file));
}
static void remove_files(const char *path) {
    DIR *dir = opendir(path); if (!dir) { CHECK(!unlink(path)); return; }
    struct dirent *e;
    while ((e = readdir(dir))) {
        if (!strcmp(e->d_name, ".") || !strcmp(e->d_name, "..")) continue;
        char child[768]; snprintf(child, sizeof(child), "%s/%s", path, e->d_name); remove_files(child);
    }
    closedir(dir); CHECK(!rmdir(path));
}
static void command(const char *root, unsigned n, const char *program) {
    aotx_dependency_fixture *f = calloc(1, sizeof(*f)); CHECK(f != NULL); if (!f) return;
    make(f, n, 0); uint64_t bytes = 0; unsigned char *memory = memory_batch(f, n, &bytes);
    if (!memory) { free(f); return; }
    char source[256], model[288], modules[288], input[288], output[288], bad[288], path[512], journal[288];
    snprintf(source, sizeof(source), "%s/source", root); snprintf(model, sizeof(model), "%s/models", source);
    snprintf(modules, sizeof(modules), "%s/modules", source); snprintf(input, sizeof(input), "%s/memory.aotxccir", source);
    snprintf(output, sizeof(output), "%s/complete.aotxccir", root); snprintf(bad, sizeof(bad), "%s/bad.aotxccir", root);
    snprintf(journal, sizeof(journal), "%s/command-journal", root);
    snprintf(path, sizeof(path), "%s/weights/model.gguf", model); disk_file(path, f->model, sizeof(f->model));
    snprintf(path, sizeof(path), "%s/manifest.jsonl", model); disk_file(path, f->models, strlen(f->models));
    snprintf(path, sizeof(path), "%s/quality/refusal-phrases.txt", model); disk_file(path, "I cannot\n", 9);
    for (unsigned i = 0; i < f->index.count; ++i) if (aotx_ccir_u32(f->index.rows[i] + 16) == 2) {
        snprintf(path, sizeof(path), "%s/%s", source, f->index.rows[i] + 64);
        disk_file(path, f->input[i + 5].data, f->input[i + 5].section.bytes);
    }
    unsigned char manifest[96]; aotx_ccir_manifest(manifest, f->input[1].section.id, NULL);
    aotx_ccir_input inputs[2] = {f->input[0], f->input[1]}; inputs[0].section.schema = 1; inputs[0].data = manifest;
    unsigned char lineage[16] = {73}; aotx_ccir_meta meta = {1, 1, 1};
    for (unsigned defect = 0; defect < 3; ++defect) {
        unsigned char *q = row_payload(memory, 3 * n - 2);
        if (defect == 1) q[64] ^= 1;
        if (defect == 2) q[96] ^= 1;
        CHECK(!aotx_ccir_create(input, lineage, inputs, 2, &meta, NULL));
        pid_t pid = fork(); CHECK(pid >= 0);
        if (!pid) { execl(program, program, "--memory", input, "--models", model, "--roles", "language",
            "--modules", modules, "--output", defect ? bad : output, (char *)NULL); _exit(127); }
        if (pid > 0) { int status = 0; CHECK(waitpid(pid, &status, 0) == pid);
            CHECK(WIFEXITED(status) && WEXITSTATUS(status) == (defect ? 1 : 0)); }
        CHECK(access(bad, F_OK) != 0); CHECK(!unlink(input));
        if (defect == 1) q[64] ^= 1;
        if (defect == 2) q[96] ^= 1;
    }
    remove_files(source);
    aotx_runtime_boot boot; int rc = aotx_runtime_prepare(output, journal, 86, &boot); CHECK(!rc);
    if (!rc) { CHECK(boot.features & AOTX_RUNTIME_APPRAISAL); aotx_runtime_release(&boot); CHECK(!rmdir(journal)); }
    unsigned char *saved = NULL; uint32_t saved_bytes = 0; CHECK(!aotx_checkpoint_file_read(output, &saved, &saved_bytes));
    CHECK(saved && saved_bytes == bytes + 128 && !memcmp(saved + 128, memory, bytes));
    free(saved); CHECK(!unlink(output)); free(memory); free(f);
}
int main(int argc, char **argv) {
    if (argc > 2) return 2;
    char root[] = "/tmp/aotx-appraisal-file-XXXXXX"; CHECK(mkdtemp(root) != NULL);
    unsigned batches[] = {1, 64};
    for (unsigned i = 0; i < 2; ++i) {
        for (unsigned j = 0; j < 31; ++j) admission(root, batches[i], j);
        pack(batches[i]); mirror(root, batches[i], 0); mirror(root, batches[i], 1); replay_admission(root, batches[i]);
        legacy(batches[i]); legacy(AOTX_COG_OBJECTS + batches[i]);
        unsigned flags[] = {0, AOTX_FLAG_REPLAYED, AOTX_FLAG_REPLAY, AOTX_FLAG_REPLAYED | AOTX_FLAG_REPLAY};
        for (unsigned flag = 0; flag < sizeof(flags) / sizeof(flags[0]); ++flag)
            for (unsigned j = 0; j < 21; ++j) fragmented(root, batches[i], j, flags[flag]);
        for (unsigned j = 21; j < 40; ++j) fragmented(root, batches[i], j, 0);
        if (argc == 2) command(root, batches[i], argv[1]);
    }
    CHECK(!rmdir(root));
    printf("appraisal runtime disk: %u checks, %u failures\n", checks, failures);
    return failures ? 1 : 0;
}
