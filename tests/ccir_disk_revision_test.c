/* Purpose: Check conditional append identity and exact rejection without writes.
 * Owns: Distinct section batches, revision tokens and temporary files.
 * Threading: One process interleaves two directory updates under normal leases.
 * Lifetime: Every case removes its files. */
#include "tests/ccir_disk_fixture.h"

static void revision(const aotx_ccir_view *view, aotx_ccir_revision *out)
{
    memcpy(out->prologue_digest, view->prologue_digest, 32u);
    memcpy(out->commit_digest, view->commit_digest, 32u);
}
static void file_digest(const char *path, unsigned char digest[32], uint64_t *bytes)
{
    struct stat st;
    int fd = open(path, O_RDONLY);
    CHECK(fd >= 0);
    CHECK(fstat(fd, &st) == 0);
    *bytes = (uint64_t)st.st_size;
    CHECK(aotx_ccir_hash_fd(fd, 0u, *bytes, digest) == 0);
    close(fd);
}
static void unchanged(const char *path, const unsigned char before[32], uint64_t bytes)
{
    unsigned char after[32];
    uint64_t after_bytes;
    file_digest(path, after, &after_bytes);
    CHECK(bytes == after_bytes && !memcmp(before, after, 32u));
}
static void cases(const char *path, const char *copy, uint32_t n)
{
    aotx_ccir_fixture f;
    aotx_ccir_view view;
    aotx_ccir_revision first, current, wrong;
    aotx_ccir_input all[67];
    unsigned char digest[32], extra[64], got[64];
    uint64_t bytes;
    uint32_t i;
    fixture(&f, n);
    CHECK(aotx_ccir_create(path, f.lineage, f.inputs, f.count, &f.meta, NULL) == 0);
    CHECK(aotx_ccir_open(path, NULL, &view) == 0);
    revision(&view, &first); aotx_ccir_close(&view);
    f.meta.checkpoint_sequence++; f.meta.durable_sequence++;
    f.payload[0][0] ^= 117u;
    CHECK(aotx_ccir_append_if(path, &first, f.inputs, f.count, &f.meta, NULL) == 0);
    CHECK(reopen_generation(path, 2u));
    file_digest(path, digest, &bytes);
    CHECK(aotx_ccir_append_if(path, &first, f.inputs, f.count, &f.meta, NULL) == AOTX_CCIR_CHANGED);
    unchanged(path, digest, bytes);
    CHECK(aotx_ccir_open(path, NULL, &view) == 0);
    revision(&view, &current); aotx_ccir_close(&view);
    wrong = current; wrong.prologue_digest[n % 32u] ^= 1u;
    CHECK(aotx_ccir_append_if(path, &wrong, f.inputs, f.count, &f.meta, NULL) == AOTX_CCIR_CHANGED);
    unchanged(path, digest, bytes);
    wrong = current; wrong.commit_digest[(n + 11u) % 32u] ^= 1u;
    CHECK(aotx_ccir_append_if(path, &wrong, f.inputs, f.count, &f.meta, NULL) == AOTX_CCIR_CHANGED);
    unchanged(path, digest, bytes);
    CHECK(aotx_ccir_append_if(path, NULL, f.inputs, f.count, &f.meta, NULL) == AOTX_CCIR_INVALID);
    unchanged(path, digest, bytes);
    memcpy(all, f.inputs, f.count * sizeof(all[0]));
    memset(&all[f.count], 0, sizeof(all[0]));
    all[f.count].section.type = 9000u; all[f.count].section.schema = 71u;
    all[f.count].section.alignment = 128u; all[f.count].section.bytes = sizeof(extra);
    all[f.count].section.id[0] = 251u; all[f.count].section.id[15] = (unsigned char)n;
    all[f.count].data = extra;
    for (i = 0; i < sizeof(extra); i++) extra[i] = (unsigned char)(i * 23u + n);
    CHECK(aotx_ccir_append(path, all, f.count + 1u, &f.meta, NULL) == 0);
    file_digest(path, digest, &bytes);
    CHECK(aotx_ccir_append_if(path, &current, f.inputs, f.count, &f.meta, NULL) == AOTX_CCIR_CHANGED);
    unchanged(path, digest, bytes);
    CHECK(aotx_ccir_open(path, NULL, &view) == 0);
    CHECK(view.count == f.count + 1u && view.generation == 3u);
    {
        aotx_ccir_read read = {f.count, 0u, sizeof(got), got};
        CHECK(aotx_ccir_read_batch(&view, &read, 1u) == 0);
        CHECK(!memcmp(got, extra, sizeof(got)));
    }
    revision(&view, &current); aotx_ccir_close(&view);
    CHECK(aotx_ccir_compact(path, copy, NULL) == 0);
    CHECK(aotx_ccir_append_if(copy, &current, f.inputs, f.count, &f.meta, NULL) == AOTX_CCIR_CHANGED);
    CHECK(reopen_generation(copy, 1u));
    CHECK(aotx_ccir_open(path, NULL, &view) == 0);
    revision(&view, &current); aotx_ccir_close(&view);
    CHECK(aotx_ccir_append_if(path, &current, all, f.count + 1u, &f.meta, NULL) == 0);
    CHECK(reopen_generation(path, 4u));
    CHECK(copy_file(path, copy) == 0);
    CHECK(reopen_generation(copy, 4u));
    unlink(path); unlink(copy);
}
int main(void)
{
    char directory[] = "/tmp/aotx-ccir-revision-XXXXXX", path[256], copy[256];
    if (!mkdtemp(directory)) return 1;
    snprintf(path, sizeof(path), "%s/state.aotxccir", directory);
    snprintf(copy, sizeof(copy), "%s/copy.aotxccir", directory);
    cases(path, copy, 1u); cases(path, copy, 64u);
    rmdir(directory);
    printf("ccir disk revision: %u checks, %u failed\n", aotx_checks, aotx_failures);
    return aotx_failures ? 1 : 0;
}
