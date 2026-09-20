/* Purpose: Verify compatible updates preserve complete files and refuse invalid maps.
 * Owns: Distinct policy bundles and runtime asset batches at N=1 and N=64.
 * Threading: One disk test; native code remains inert.
 * Lifetime: All temporary files are removed after their case. */
#include "tests/runtime_dependency_fixture.h"
#include "disk/runtime/policy.h"
#include "disk/runtime/activate.h"
#include <errno.h>
#include <fcntl.h>
#include <sys/stat.h>

static int fail_link;
int __real_linkat(int, const char *, int, const char *, int);
int __wrap_linkat(int a, const char *b, int c, const char *d, int e) {
    if (fail_link) { errno = EIO; return -1; }
    return __real_linkat(a, b, c, d, e);
}
static void bundle(const char *path, unsigned row, unsigned mode, unsigned abi, aotx_policy_file *out) {
    aotx_policy_source s = {0}; char provenance[64];
    snprintf(provenance, sizeof(provenance), "policy source %u", row);
    s.config = (aotx_policy_config){mode, 1, 16, 0, 64, 0, 0, 0, row % 101, 1, 8, 0, abi};
    s.provenance = provenance; s.provenance_bytes = strlen(provenance);
    s.license = "Apache-2.0"; s.license_bytes = 10;
    if (mode == AOTX_POLICY_NATIVE) {
        s.config.architecture = 86; s.config.registers = 32; s.config.format = 1;
        s.entry = "aotx_policy_entry";
        s.image = ".version 8.0\n.target sm_86\n.address_size 64\n.visible .entry aotx_policy_entry() { ret; }\n";
        s.image_bytes = strlen(s.image);
    }
    CHECK(!aotx_policy_file_write(path, &s));
    CHECK(!aotx_policy_file_read(path, NULL, 0, out));
}
#include "policy_update_replay.h"
static void test(unsigned n, unsigned mode) {
    char root[] = "/tmp/aotx-policy-update-XXXXXX", source[256], output[256], second[256], oldpath[256], newpath[256], badpath[256];
    CHECK(mkdtemp(root) != NULL);
    snprintf(source, sizeof(source), "%s/source", root); snprintf(output, sizeof(output), "%s/output", root);
    snprintf(second, sizeof(second), "%s/second", root); snprintf(oldpath, sizeof(oldpath), "%s/old", root);
    snprintf(newpath, sizeof(newpath), "%s/new", root); snprintf(badpath, sizeof(badpath), "%s/bad", root);
    aotx_policy_file old = {0}, next = {0}, bad = {0};
    bundle(oldpath, n, mode, 1, &old); bundle(newpath, n + 1, mode, 1, &next);
    bundle(badpath, n + 2, mode, 2, &bad);
    aotx_dependency_fixture *f = calloc(1, sizeof(*f)); CHECK(f != NULL);
    if (!f) exit(1);
    make(f, n, 0); unsigned char *replay = saved_replay(f, n, &old, 1, 0);
    asset(f, "policy.bin", 3, old.buffer, old.buffer_bytes);
    aotx_ccir_put(f->index.header + 16, f->index.count, 4);
    aotx_ccir_put(f->index.header + 20, AOTX_RUNTIME_POLICY, 4);
    f->input[3].section.schema = 3; f->input[3].section.bytes = 256 + f->index.count * AOTX_RUNTIME_ROW;
    unsigned char lineage[16] = {71}; aotx_ccir_meta meta = {1, 1, 1};
    CHECK(!aotx_ccir_create(source, lineage, f->input, f->count, &meta, NULL));
    char from[65], wrong[65]; aotx_sha256_text(old.digest, from); aotx_sha256_text(next.digest, wrong);
    aotx_ccir_view before; CHECK(!aotx_ccir_open(source, NULL, &before));
    unsigned char original[32], after[32]; CHECK(!aotx_ccir_hash_fd(before.fd, 0, before.end, original));
    CHECK(aotx_runtime_policy_update(source, wrong, newpath, output) == AOTX_CCIR_CHANGED);
    CHECK(aotx_runtime_policy_update(source, from, badpath, output) == AOTX_CCIR_UNSUPPORTED);
    CHECK(aotx_runtime_policy_update(source, from, oldpath, output) != 0);
    CHECK(access(output, F_OK) != 0);
    fail_link = 1;
    CHECK(aotx_runtime_policy_update(source, from, newpath, output) == AOTX_CCIR_IO);
    fail_link = 0; CHECK(access(output, F_OK) != 0);
    CHECK(!aotx_runtime_policy_update(source, from, newpath, output));
    CHECK(aotx_runtime_policy_update(source, from, newpath, output) == AOTX_CCIR_EXISTS);
    CHECK(!aotx_ccir_hash_fd(before.fd, 0, before.end, after) && !memcmp(original, after, 32));
    aotx_ccir_view changed; int opened = aotx_ccir_open(output, NULL, &changed); CHECK(!opened);
    if (opened) exit(1);
    CHECK(!aotx_runtime_dependencies(&changed));
    CHECK(!memcmp(before.lineage, changed.lineage, 16) && memcmp(before.incarnation, changed.incarnation, 16));
    CHECK(!memcmp(&before.meta, &changed.meta, sizeof(before.meta)));
    for (uint32_t i = 0; i < before.count; ++i) {
        const aotx_ccir_section *s = before.sections + i;
        if (s->type == AOTX_CCIR_RUNTIME || !memcmp(s->digest, old.digest, 32)) continue;
        int at = aotx_runtime_section(&changed, s->id); CHECK(at >= 0);
        if (at >= 0) CHECK(changed.sections[at].bytes == s->bytes && !memcmp(changed.sections[at].digest, s->digest, 32));
    }
    aotx_runtime_index index; CHECK(!aotx_runtime_index_read(changed.fd, &changed, &index));
    aotx_policy_file got; aotx_policy_history history; unsigned char *raw = NULL; size_t bytes = 0;
    CHECK(!aotx_runtime_policy_read(&changed, &index, &got, &history, &raw, &bytes));
    CHECK(history.count == 1 && history.rows[0].last_decision == n &&
        !memcmp(history.rows[0].digest, old.digest, 32) && !memcmp(got.digest, next.digest, 32));
    CHECK(bytes == 128 + old.buffer_bytes && !memcmp(raw + 128, old.buffer, old.buffer_bytes));
    aotx_policy_history malformed;
    raw[16] ^= 1; CHECK(aotx_policy_history_decode(raw, bytes, &got, &malformed) != 0); raw[16] ^= 1;
    raw[80] ^= 1; CHECK(aotx_policy_history_decode(raw, bytes, &got, &malformed) != 0); raw[80] ^= 1;
    CHECK(aotx_policy_history_decode(raw, bytes - 1, &got, &malformed) != 0);
    aotx_ccir_put(raw + 72, n + 1, 8);
    CHECK(!aotx_policy_history_decode(raw, bytes, &got, &malformed));
    uint64_t last = 0; CHECK(aotx_runtime_policy_replay(&changed, &got, &malformed, &last) == AOTX_CCIR_INVALID);
    free(raw); aotx_policy_file_close(&got); aotx_ccir_close(&changed);
    CHECK(aotx_runtime_policy_update(output, wrong, oldpath, second) == AOTX_CCIR_INVALID);
    CHECK(access(second, F_OK) != 0);
    if (mode == AOTX_POLICY_NATIVE) {
        CHECK(aotx_policy_file_read(newpath, NULL, 1, &got) == AOTX_POLICY_FILE_TRUST);
        CHECK(!aotx_policy_file_read(newpath, wrong, 1, &got)); aotx_policy_file_close(&got);
    }
    aotx_ccir_close(&before); free(replay); free(f);
    aotx_policy_file_close(&old); aotx_policy_file_close(&next); aotx_policy_file_close(&bad);
    CHECK(!unlink(source)); CHECK(!unlink(output)); CHECK(!unlink(oldpath)); CHECK(!unlink(newpath)); CHECK(!unlink(badpath));
    CHECK(!rmdir(root));
}
static void history_limit(unsigned n) {
    char root[] = "/tmp/aotx-policy-chain-XXXXXX", paths[10][256], bundles[10][256];
    CHECK(mkdtemp(root) != NULL);
    aotx_policy_file policies[10] = {0};
    for (unsigned i = 0; i < 10; ++i) {
        snprintf(paths[i], sizeof(paths[i]), "%s/runtime-%u", root, i);
        snprintf(bundles[i], sizeof(bundles[i]), "%s/policy-%u", root, i);
        bundle(bundles[i], 17 + n + i, AOTX_POLICY_RULES, 1, policies + i);
    }
    aotx_dependency_fixture *f = calloc(1, sizeof(*f)); CHECK(f != NULL); if (!f) exit(1);
    make(f, n, 0); asset(f, "policy.bin", 3, policies[0].buffer, policies[0].buffer_bytes);
    aotx_ccir_put(f->index.header + 16, f->index.count, 4);
    aotx_ccir_put(f->index.header + 20, AOTX_RUNTIME_POLICY, 4);
    f->input[3].section.schema = 3; f->input[3].section.bytes = 256 + f->index.count * AOTX_RUNTIME_ROW;
    unsigned char lineage[16] = {87}; aotx_ccir_meta meta = {1, 1, 1};
    CHECK(!aotx_ccir_create(paths[0], lineage, f->input, f->count, &meta, NULL)); free(f);
    for (unsigned i = 1; i < 10; ++i) {
        char digest[65]; aotx_sha256_text(policies[i - 1].digest, digest);
        CHECK(aotx_runtime_policy_update(paths[i - 1], digest, bundles[i], paths[i]) ==
            (i <= AOTX_POLICY_REVISIONS ? 0 : AOTX_CCIR_LIMIT));
    }
    CHECK(access(paths[9], F_OK) != 0);
    size_t first = 64 + policies[0].buffer_bytes, second = 64 + policies[1].buffer_bytes;
    size_t bytes = 64 + first + second;
    unsigned char *raw = calloc(1, bytes); CHECK(raw != NULL); if (!raw) exit(1);
    memcpy(raw, "AOTXPH01", 8); aotx_ccir_put(raw + 8, 1, 4); aotx_ccir_put(raw + 12, 2, 4);
    memcpy(raw + 16, policies[2].digest, 32);
    for (unsigned i = 0; i < 2; ++i) {
        size_t at = i ? 64 + first : 64;
        aotx_ccir_put(raw + at, policies[i].buffer_bytes, 8);
        aotx_ccir_put(raw + at + 8, i ? 9 : 4, 8);
        memcpy(raw + at + 16, policies[i].digest, 32);
        memcpy(raw + at + 64, policies[i].buffer, policies[i].buffer_bytes);
    }
    aotx_policy_history history;
    CHECK(!aotx_policy_history_decode(raw, bytes, policies + 2, &history));
    CHECK(history.count == 2 && history.rows[0].last_decision == 4 && history.rows[1].last_decision == 9);
    aotx_ccir_put(raw + 64 + first + 8, 3, 8);
    CHECK(aotx_policy_history_decode(raw, bytes, policies + 2, &history) == AOTX_CCIR_INVALID);
    aotx_ccir_put(raw + 64 + first + 8, 9, 8);
    CHECK(first == second);
    memcpy(raw + 64 + first + 16, policies[0].digest, 32);
    memcpy(raw + 64 + first + 64, policies[0].buffer, policies[0].buffer_bytes);
    CHECK(aotx_policy_history_decode(raw, bytes, policies + 2, &history) == AOTX_CCIR_INVALID);
    free(raw);
    for (unsigned i = 0; i < 10; ++i) {
        aotx_policy_file_close(policies + i); CHECK(!unlink(bundles[i]));
        if (i < 9) CHECK(!unlink(paths[i]));
    }
    CHECK(!rmdir(root));
}
int main(void) {
    for (unsigned n = 1; n <= 64; n *= 64) {
        test(n, AOTX_POLICY_RULES); test(n, AOTX_POLICY_NATIVE);
        replay_cases(n, AOTX_POLICY_RULES); replay_cases(n, AOTX_POLICY_NATIVE); history_limit(n);
    }
    printf("policy update: %u checks, %u failures\n", checks, failures);
    return failures ? 1 : 0;
}
