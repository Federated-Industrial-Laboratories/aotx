/* Purpose: Check policy trust, malformed bundles, and complete runtime asset recovery.
 * Owns: Distinct policy revisions and temporary files at batch sizes one and 64.
 * Threading: One disk process; native images remain inert byte extents.
 * Lifetime: All files and buffers end with their case. */
#include "tests/runtime_dependency_fixture.h"
#include "disk/policy/file.h"
#include "disk/runtime/activate.h"
#include "disk/runtime/assets.h"
#include <fcntl.h>
#include <sys/stat.h>

static void source(aotx_policy_source *s, unsigned mode, unsigned row, char provenance[128]) {
    memset(s, 0, sizeof(*s));
    s->config = (aotx_policy_config){mode, 1, 16, 0, 64, 0, 0, 0, row % 101, row + 1, row + 3, 0};
    snprintf(provenance, 128, "source %u; compiler options %u\n", row, 11 + row * 13);
    s->provenance = provenance; s->provenance_bytes = strlen(provenance);
    s->license = "Apache-2.0\n"; s->license_bytes = strlen(s->license);
    if (mode == AOTX_POLICY_NATIVE) {
        s->config.architecture = 86; s->config.registers = 64;
        s->config.state_schema = row + 7; s->config.state_bytes = 16 + row;
        s->config.format = 1; s->entry = "aotx_policy_entry";
        s->image = ".version 8.0\n.target sm_86\n.address_size 64\n"
            ".visible .entry aotx_policy_entry() { ret; }\n";
        s->image_bytes = strlen(s->image);
    }
}
static void altered(const aotx_policy_file *file, size_t at, uint64_t value,
    unsigned width, int expected) {
    unsigned char *bytes = malloc(file->buffer_bytes + 1); CHECK(bytes != NULL);
    if (!bytes) return;
    memcpy(bytes, file->buffer, file->buffer_bytes);
    aotx_ccir_put(bytes + at, value, width);
    aotx_policy_file bad;
    CHECK(aotx_policy_file_decode(bytes, file->buffer_bytes, &bad) == expected);
    CHECK(!bad.buffer && !bad.image);
    aotx_policy_file_close(&bad); free(bytes);
}
static void malformed(const aotx_policy_file *file) {
    altered(file, 0, 'B', 1, AOTX_CCIR_INVALID);
    altered(file, 8, 2, 4, AOTX_CCIR_UNSUPPORTED);
    altered(file, 12, 4, 4, AOTX_CCIR_UNSUPPORTED);
    altered(file, 16, 2, 4, AOTX_CCIR_UNSUPPORTED);
    altered(file, 20, 0, 4, AOTX_CCIR_INVALID);
    altered(file, 24, 0, 4, AOTX_CCIR_INVALID);
    altered(file, 24, (uint64_t)AOTX_POLICY_STATE_BYTES + 1, 4, AOTX_CCIR_LIMIT);
    altered(file, 32, 0, 4, AOTX_CCIR_INVALID);
    altered(file, 32, 1025, 4, AOTX_CCIR_INVALID);
    altered(file, 48, 101, 4, AOTX_CCIR_INVALID);
    altered(file, 52, 0, 4, AOTX_CCIR_INVALID);
    altered(file, 56, 0, 4, AOTX_CCIR_INVALID);
    altered(file, 64, UINT64_MAX, 8, AOTX_CCIR_LIMIT);
    altered(file, 72, 0, 4, AOTX_CCIR_INVALID);
    altered(file, 76, 0, 4, AOTX_CCIR_INVALID);
    altered(file, 72, (uint64_t)AOTX_POLICY_METADATA_BYTES + 1, 4, AOTX_CCIR_LIMIT);
    altered(file, 176, 1, 1, AOTX_CCIR_INVALID);
    altered(file, 255, 1, 1, AOTX_CCIR_INVALID);
    altered(file, (size_t)(file->provenance - file->buffer), 0, 1, AOTX_CCIR_INVALID);
    altered(file, (size_t)(file->license - file->buffer), 0, 1, AOTX_CCIR_INVALID);
    aotx_policy_file bad;
    CHECK(aotx_policy_file_decode(file->buffer, 255, &bad) == AOTX_CCIR_INVALID);
    CHECK(aotx_policy_file_decode(file->buffer, file->buffer_bytes - 1, &bad) == AOTX_CCIR_INVALID);
    unsigned char *extra = malloc(file->buffer_bytes + 1); CHECK(extra != NULL);
    if (extra) {
        memcpy(extra, file->buffer, file->buffer_bytes); extra[file->buffer_bytes] = 1;
        CHECK(aotx_policy_file_decode(extra, file->buffer_bytes + 1, &bad) == AOTX_CCIR_INVALID);
        free(extra);
    }
    if (file->config.mode == AOTX_POLICY_NATIVE) {
        altered(file, 28, 0, 4, AOTX_CCIR_INVALID);
        altered(file, 36, 0, 4, AOTX_CCIR_INVALID);
        altered(file, 60, 3, 4, AOTX_CCIR_UNSUPPORTED);
        altered(file, 80, '/', 1, AOTX_CCIR_INVALID);
        altered(file, 143, 'x', 1, AOTX_CCIR_INVALID);
        altered(file, 144, file->buffer[144] ^ 1, 1, AOTX_POLICY_FILE_DIGEST);
        altered(file, 256, file->buffer[256] ^ 1, 1, AOTX_POLICY_FILE_DIGEST);
    } else {
        altered(file, 20, 2, 4, AOTX_CCIR_UNSUPPORTED);
        altered(file, 24, 32, 4, AOTX_CCIR_UNSUPPORTED);
        altered(file, 28, 86, 4, AOTX_CCIR_INVALID);
        altered(file, 32, 32, 4, AOTX_CCIR_INVALID);
        altered(file, 60, 1, 4, AOTX_CCIR_INVALID);
        altered(file, 80, 'a', 1, AOTX_CCIR_INVALID);
        altered(file, 144, 1, 1, AOTX_CCIR_INVALID);
    }
}
static void files(const char *root, unsigned n) {
    unsigned start = checks;
    unsigned char previous[32] = {0};
    for (unsigned i = 0; i < n; ++i) for (unsigned mode = 1; mode <= 3; ++mode) {
        char path[256], link[256], provenance[128], trust[65];
        snprintf(path, sizeof(path), "%s/policy-%u.bin", root, i);
        snprintf(link, sizeof(link), "%s/link-%u.bin", root, i);
        aotx_policy_source s; source(&s, mode, i, provenance);
        CHECK(!aotx_policy_file_write(path, &s));
        CHECK(aotx_policy_file_write(path, &s) == AOTX_CCIR_EXISTS);
        aotx_policy_file file;
        CHECK(!aotx_policy_file_read(path, NULL, 0, &file));
        if (!file.buffer) continue;
        CHECK(!memcmp(&file.config, &s.config, sizeof(s.config)));
        CHECK(file.provenance_bytes == s.provenance_bytes &&
            !memcmp(file.provenance, s.provenance, s.provenance_bytes));
        CHECK(file.license_bytes == s.license_bytes && !memcmp(file.license, s.license, s.license_bytes));
        CHECK(memcmp(file.digest, previous, 32)); memcpy(previous, file.digest, 32);
        if (mode == 3) {
            CHECK(file.image_bytes == s.image_bytes && !memcmp(file.image, s.image, s.image_bytes));
            CHECK(file.image[s.image_bytes] == 0 && file.image != file.buffer + 256);
        } else CHECK(!file.image && !file.image_bytes);
        aotx_sha256_text(file.digest, trust);
        aotx_policy_file got;
        CHECK(!aotx_policy_file_read(path, trust, 1, &got));
        CHECK(got.buffer_bytes == file.buffer_bytes && !memcmp(got.buffer, file.buffer, file.buffer_bytes));
        aotx_policy_file_close(&got);
        CHECK(aotx_policy_file_read(path, NULL, 1, &got) == (mode == 3 ? AOTX_POLICY_FILE_TRUST : 0));
        aotx_policy_file_close(&got);
        trust[0] = trust[0] == '0' ? '1' : '0';
        CHECK(aotx_policy_file_read(path, trust, 1, &got) == AOTX_POLICY_FILE_TRUST);
        CHECK(!got.buffer && !got.image);
        CHECK(aotx_policy_file_read(path, "z", 1, &got) == AOTX_POLICY_FILE_TRUST);
        int fd = open(path, O_RDONLY); CHECK(fd >= 0);
        CHECK(!aotx_policy_file_extent(fd, 0, file.buffer_bytes, &got));
        CHECK(!memcmp(got.digest, file.digest, 32)); aotx_policy_file_close(&got);
        CHECK(aotx_policy_file_extent(fd, UINT64_MAX, 1, &got) == AOTX_CCIR_INVALID);
        CHECK(aotx_policy_file_extent(fd, 1, file.buffer_bytes, &got) == AOTX_CCIR_INVALID);
        CHECK(!close(fd));
        CHECK(!symlink(path, link));
        CHECK(aotx_policy_file_read(link, NULL, 0, &got) != 0); CHECK(!unlink(link));
        malformed(&file);
        if (mode == 3) {
            s.provenance = "Changed source\n"; s.provenance_bytes = strlen(s.provenance);
            CHECK(!unlink(path)); CHECK(!aotx_policy_file_write(path, &s));
            aotx_sha256_text(file.digest, trust);
            CHECK(aotx_policy_file_read(path, trust, 1, &got) == AOTX_POLICY_FILE_TRUST);
        }
        aotx_policy_file_close(&file); CHECK(!unlink(path));
        CHECK(aotx_policy_file_read(path, NULL, 0, &got) == AOTX_CCIR_IO);
    }
    printf("policy files N=%u: %u checks\n", n, checks - start);
}
static void limits(const char *root) {
    char path[256], provenance[128];
    snprintf(path, sizeof(path), "%s/limits.bin", root);
    aotx_policy_source s; source(&s, AOTX_POLICY_NATIVE, 0, provenance);
    s.config.state_bytes = AOTX_POLICY_STATE_BYTES;
    ++s.image_bytes;
    CHECK(!aotx_policy_file_write(path, &s));
    aotx_policy_file file; CHECK(!aotx_policy_file_read(path, NULL, 0, &file));
    CHECK(file.config.state_bytes == AOTX_POLICY_STATE_BYTES);
    CHECK(file.image_bytes == s.image_bytes && file.image[s.image_bytes - 1] == 0);
    aotx_policy_file_close(&file); CHECK(!unlink(path));
    s.config.state_bytes = AOTX_POLICY_STATE_BYTES + 1;
    CHECK(aotx_policy_file_write(path, &s) == AOTX_CCIR_LIMIT);
    CHECK(access(path, F_OK) != 0);
    s.config.state_bytes = 16; s.provenance_bytes = (size_t)AOTX_POLICY_METADATA_BYTES + 1;
    CHECK(aotx_policy_file_write(path, &s) == AOTX_CCIR_LIMIT);
    s.provenance_bytes = 0;
    CHECK(aotx_policy_file_write(path, &s) == AOTX_CCIR_INVALID);
    s.provenance_bytes = strlen(provenance); s.image_bytes = (size_t)AOTX_POLICY_IMAGE_BYTES + 1;
    CHECK(aotx_policy_file_write(path, &s) == AOTX_CCIR_LIMIT);
    s.image_bytes = strlen(s.image); s.image = NULL;
    CHECK(aotx_policy_file_write(path, &s) == AOTX_CCIR_INVALID);
    CHECK(access(path, F_OK) != 0);
}
static void runtime_make(aotx_dependency_fixture *f, unsigned n, const aotx_policy_file *file) {
    make(f, n, 0);
#ifdef AOTX_AFFECT
    static const char phrases[] = "cannot comply\n";
    asset(f, "quality/refusal-phrases.txt", 1, phrases, sizeof(phrases) - 1);
    aotx_ccir_put(f->index.header + 20, AOTX_RUNTIME_AFFECT, 4);
#endif
    if (file) {
        asset(f, "policy.bin", 3, file->buffer, file->buffer_bytes);
        aotx_ccir_put(f->index.header + 20,
            aotx_ccir_u32(f->index.header + 20) | AOTX_RUNTIME_POLICY, 4);
    }
    aotx_ccir_put(f->index.header + 16, f->index.count, 4);
    f->input[3].section.bytes = AOTX_RUNTIME_HEADER + f->index.count * AOTX_RUNTIME_ROW;
    f->input[3].section.schema = aotx_runtime_schema(aotx_ccir_u32(f->index.header + 20));
}
static void runtimes(const char *root, unsigned n) {
    unsigned start = checks;
    char path[256], bundle[256], provenance[128];
    snprintf(path, sizeof(path), "%s/runtime.aotxccir", root);
    snprintf(bundle, sizeof(bundle), "%s/native.bin", root);
    aotx_dependency_fixture *f = calloc(1, sizeof(*f)); CHECK(f != NULL);
    if (!f) return;
    unsigned char lineage[16] = {37}; aotx_ccir_meta meta = {1, 1, 1};
    aotx_policy_source s; source(&s, AOTX_POLICY_NATIVE, n, provenance);
    CHECK(!aotx_policy_file_write(bundle, &s));
    aotx_policy_file file; CHECK(!aotx_policy_file_read(bundle, NULL, 0, &file));
    CHECK(!unlink(bundle));
    if (!file.buffer) { free(f); return; }
    for (unsigned profile = 0; profile < 4; ++profile) {
        runtime_make(f, n, profile >= 2 ? &file : NULL);
        if (profile & 1) {
            aotx_runtime_shared_profile shared; aotx_runtime_shared_current(&shared);
            aotx_runtime_shared_write(f->index.header, &shared);
            f->input[3].section.schema = aotx_runtime_schema(aotx_ccir_u32(f->index.header + 20));
        }
        CHECK(!aotx_ccir_create(path, lineage, f->input, f->count, &meta, NULL));
        aotx_ccir_view view; CHECK(!aotx_ccir_open(path, NULL, &view));
        CHECK(!aotx_runtime_dependencies(&view)); aotx_ccir_close(&view);
        aotx_runtime_boot boot; CHECK(!aotx_runtime_prepare(path, root, 86, &boot));
        if (profile >= 2) {
            CHECK(boot.policy[0] && (boot.features & AOTX_RUNTIME_POLICY));
            aotx_policy_file got; CHECK(!aotx_policy_file_read(boot.policy, NULL, 0, &got));
            CHECK(got.buffer_bytes == file.buffer_bytes && !memcmp(got.digest, file.digest, 32));
            aotx_policy_file_close(&got);
            char temporary[1024]; strcpy(temporary, boot.policy); aotx_runtime_release(&boot);
            CHECK(access(temporary, F_OK) != 0);
        } else { CHECK(!boot.policy[0]); aotx_runtime_release(&boot); }
        CHECK(!aotx_runtime_promote_shared(path));
        CHECK(!aotx_ccir_open(path, NULL, &view));
        CHECK(!aotx_runtime_dependencies(&view));
        aotx_runtime_index *index = malloc(sizeof(*index)); CHECK(index != NULL);
        if (index) {
            CHECK(!aotx_runtime_index_read(view.fd, &view, index));
            int at = aotx_runtime_section(&view, f->input[3].section.id);
            CHECK(at >= 0 && view.sections[at].schema == (profile >= 2 ? 3 : 2));
            CHECK((aotx_ccir_u32(index->header + 20) & AOTX_RUNTIME_SHARED) != 0);
            if (profile >= 2) {
                at = aotx_runtime_section(&view, f->input[f->count - 1].section.id);
                aotx_policy_file got;
                CHECK(at >= 0 && !aotx_policy_file_extent(view.fd, view.sections[at].offset,
                    view.sections[at].bytes, &got));
                CHECK(!memcmp(got.digest, file.digest, 32)); aotx_policy_file_close(&got);
            }
            free(index);
        }
        aotx_ccir_close(&view); CHECK(!unlink(path));
    }
    for (unsigned defect = 0; defect < 8; ++defect) {
        runtime_make(f, n, &file);
        unsigned char *h = f->index.header, *row = f->index.rows[f->index.count - 1];
        if (defect == 0) aotx_ccir_put(h + 20, aotx_ccir_u32(h + 20) & ~AOTX_RUNTIME_POLICY, 4);
        if (defect == 1) { --f->count; --f->index.count; aotx_ccir_put(h + 16, f->index.count, 4);
            f->input[3].section.bytes -= AOTX_RUNTIME_ROW; }
        if (defect == 2) aotx_ccir_put(row + 16, 1, 4);
        if (defect == 3) strcpy((char *)row + 64, "../policy.bin");
        if (defect == 4) f->input[3].section.schema = 2;
        if (defect == 5) aotx_ccir_put(h + 36, 87, 4);
        if (defect == 6) {
            asset(f, "policy.bin", 3, file.buffer, file.buffer_bytes);
            aotx_ccir_put(h + 16, f->index.count, 4); f->input[3].section.bytes += AOTX_RUNTIME_ROW;
        }
        if (defect == 7) strcpy((char *)row + 64, "other.bin");
        int rc = aotx_ccir_create(path, lineage, f->input, f->count, &meta, NULL);
        if (defect == 5) {
            CHECK(!rc); aotx_ccir_view view; CHECK(!aotx_ccir_open(path, NULL, &view));
            CHECK(aotx_runtime_dependencies(&view) == AOTX_CCIR_UNSUPPORTED);
            aotx_ccir_close(&view); aotx_runtime_boot boot;
            CHECK(aotx_runtime_prepare(path, root, 87, &boot) == AOTX_CCIR_UNSUPPORTED);
            CHECK(!boot.owned && !boot.policy[0]); CHECK(!unlink(path));
        } else { CHECK(rc != 0); CHECK(access(path, F_OK) != 0); }
    }
    aotx_policy_file_close(&file); free(f);
    printf("policy runtime N=%u: %u checks\n", n, checks - start);
}
int main(void) {
    char root[] = "/tmp/aotx-policy-files-XXXXXX";
    CHECK(mkdtemp(root) != NULL);
    files(root, 1); files(root, 64); limits(root); runtimes(root, 1); runtimes(root, 64);
    CHECK(!rmdir(root));
    printf("policy disk: %u checks, %u failures\n", checks, failures);
    return failures || checks < 16000 ? 1 : 0;
}
