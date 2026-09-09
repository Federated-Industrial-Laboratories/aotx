/* Purpose: Check CCIR section batches, strict parsing and compaction.
 * Owns: Temporary files and distinct payloads for one and 64 sections.
 * Threading: One process takes file leases in sequence.
 * Lifetime: Each case removes its temporary files. */
#include "tests/ccir_disk_fixture.h"
#include <sys/file.h>

static void exact(const char *path, const aotx_ccir_fixture *f, uint64_t generation)
{
    aotx_ccir_view view;
    aotx_ccir_read reads[65];
    unsigned char got[65][512];
    uint32_t i;
    int rc = aotx_ccir_open(path, NULL, &view);
    CHECK(rc == AOTX_CCIR_OK);
    if (rc) return;
    CHECK(view.generation == generation && view.count == f->count);
    CHECK(!memcmp(view.lineage, f->lineage, 16u));
    CHECK(view.meta.durable_sequence == f->meta.durable_sequence);
    memset(got, 0, sizeof(got));
    for (i = 1; i < f->count; i++) {
        reads[i - 1u] = (aotx_ccir_read){i, 0u, (size_t)f->inputs[i].section.bytes,
                                         got[i - 1u]};
        CHECK(!memcmp(view.sections[i].id, f->inputs[i].section.id, 16u));
    }
    CHECK(aotx_ccir_read_batch(&view, reads, f->count - 1u) == AOTX_CCIR_OK);
    for (i = 1; i < f->count; i++)
        CHECK(!memcmp(got[i - 1u], f->inputs[i].data, reads[i - 1u].bytes));
    memset(got, 219, sizeof(got));
    reads[f->count - 2u].offset = UINT64_MAX;
    CHECK(aotx_ccir_read_batch(&view, reads, f->count - 1u) == AOTX_CCIR_INVALID);
    CHECK(got[0][0] == 219u);
    CHECK(aotx_ccir_append(path, f->inputs, f->count, &f->meta, NULL) == AOTX_CCIR_BUSY);
    aotx_ccir_close(&view);
}

/* Recompute enclosing digests so structural changes reach the real parser. */
static void reseal(int fd, uint32_t root_slot, unsigned char *rows,
                   unsigned char commit[256], unsigned char root[4096])
{
    uint32_t count = aotx_ccir_u32(commit + 80);
    uint64_t offset = aotx_ccir_u64(commit + 72);
    if (rows) {
        CHECK(aotx_ccir_pwrite(fd, rows, count * 128u, offset) == 0);
        aotx_ccir_hash(rows, count * 128u, commit + 96);
    }
    CHECK(aotx_ccir_pwrite(fd, commit, 256u, aotx_ccir_u64(root + 24)) == 0);
    aotx_ccir_hash(commit, 256u, root + 48);
    aotx_ccir_hash(root, 4064u, root + 4064);
    CHECK(aotx_ccir_pwrite(fd, root, 4096u, (root_slot + 1u) * 4096u) == 0);
}

static void corruptions(const char *path, const char *bad, uint32_t n)
{
    unsigned int kind;
    for (kind = 0; kind < 19u; kind++) {
        aotx_ccir_view view;
        unsigned char rows[66u * 128u], commit[256], root[4096];
        uint32_t slot;
        uint64_t commit_offset, directory_offset;
        int fd, expected = AOTX_CCIR_OK, rc;
        CHECK(copy_file(path, bad) == 0);
        CHECK(aotx_ccir_open(bad, NULL, &view) == 0);
        slot = view.root_slot; commit_offset = view.commit_offset;
        directory_offset = view.directory_offset;
        aotx_ccir_close(&view);
        fd = open(bad, O_RDWR);
        CHECK(fd >= 0);
        CHECK(aotx_ccir_pread(fd, root, sizeof(root), (slot + 1u) * 4096u) == 0);
        CHECK(aotx_ccir_pread(fd, commit, sizeof(commit), commit_offset) == 0);
        CHECK(aotx_ccir_pread(fd, rows, (n + 2u) * 128u, directory_offset) == 0);
        if (kind == 0u) rows[2u * 128u + 6u] = AOTX_CCIR_REQUIRED;
        if (kind == 1u) rows[4u] = 2u;
        if (kind == 2u) rows[2u * 128u + 88u] = 1u;
        if (kind == 3u) memcpy(rows + 2u * 128u + 8u, rows + 8u, 16u);
        if (kind == 4u) memcpy(rows + 2u * 128u + 24u, rows + 24u, 8u);
        if (kind == 5u) memset(rows + 2u * 128u + 24u, 255, 8u);
        if (kind == 6u) aotx_ccir_put(rows + 2u * 128u + 48u, 3u, 4u);
        if (kind == 7u) rows[2u * 128u + 54u] = 1u;
        if (kind == 8u) rows[2u * 128u + 52u] = 1u;
        if (kind == 9u) aotx_ccir_put(rows + 2u * 128u + 40u, UINT64_MAX, 8u);
        if (kind == 10u) commit[184u] = 1u;
        if (kind == 11u) aotx_ccir_put(commit + 72u, 0u, 8u);
        if (kind == 12u) aotx_ccir_put(root + 32u, 257u, 8u);
        if (kind == 13u) root[112u] = 1u;
        if (kind == 14u) aotx_ccir_put(commit + 80u, 257u, 4u);
        if (kind == 15u) aotx_ccir_put(commit + 80u, 0u, 4u);
        if (kind == 16u) aotx_ccir_put(rows + 2u * 128u + 32u, 0u, 8u);
        if (kind == 17u) rows[2u * 128u + 6u] = 2u;
        if (kind == 18u) aotx_ccir_put(commit + 56u, 0u, 8u);
        if (kind == 0u || kind == 1u || kind == 8u) expected = AOTX_CCIR_UNSUPPORTED;
        reseal(fd, slot, kind == 11u || kind == 14u || kind == 15u ? NULL : rows,
               commit, root);
        close(fd);
        rc = aotx_ccir_open(bad, NULL, &view);
        CHECK(rc == expected);
        if (!rc) {
            CHECK(view.generation == 1u && view.fallback == 1u);
            aotx_ccir_close(&view);
        }
        unlink(bad);
    }
}

static void cases(const char *path, const char *copy, uint32_t n)
{
    aotx_ccir_fixture f;
    aotx_ccir_view view, packed;
    aotx_ccir_limits limits;
    aotx_ccir_section original[66];
    uint64_t old_bytes, last_end;
    uint32_t i;
    int fd;
    fixture(&f, n);
    CHECK(aotx_ccir_create(path, f.lineage, f.inputs, f.count, &f.meta, NULL) == 0);
    CHECK(aotx_ccir_create(path, f.lineage, f.inputs, f.count, &f.meta, NULL) == AOTX_CCIR_EXISTS);
    exact(path, &f, 1u);
    CHECK(aotx_ccir_open(path, NULL, &view) == 0);
    memcpy(original, view.sections, f.count * sizeof(original[0]));
    aotx_ccir_close(&view);
    f.payload[0][0] ^= 93u; f.meta.durable_sequence++; f.meta.checkpoint_sequence++;
    for (i = 2u; i < f.count; i++) f.inputs[i].source = AOTX_CCIR_REUSE;
    CHECK(aotx_ccir_append(path, f.inputs, f.count, &f.meta, NULL) == 0);
    exact(path, &f, 2u);
    CHECK(aotx_ccir_open(path, NULL, &view) == 0);
    for (i = 2u; i < f.count; i++) CHECK(original[i].offset == view.sections[i].offset);
    last_end = view.end;
    aotx_ccir_close(&view);
    corruptions(path, copy, n);
    CHECK(copy_file(path, copy) == 0);
    fd = open(copy, O_RDWR); CHECK(fd >= 0);
    CHECK(ftruncate(fd, (off_t)last_end - 1) == 0); close(fd);
    CHECK(aotx_ccir_open(copy, NULL, &view) == 0);
    CHECK(view.generation == 1u && view.fallback && view.trailing_bytes);
    aotx_ccir_close(&view); unlink(copy);
    CHECK(copy_file(path, copy) == 0);
    fd = open(copy, O_RDWR); CHECK(fd >= 0);
    CHECK(aotx_ccir_pwrite(fd, "x", 1u, original[2].offset) == 0); close(fd);
    CHECK(aotx_ccir_open(copy, NULL, &view) == AOTX_CCIR_INVALID);
    unlink(copy);
    aotx_ccir_default_limits(&limits); limits.sections = 2u;
    CHECK(aotx_ccir_open(path, &limits, &view) == AOTX_CCIR_LIMIT);
    aotx_ccir_default_limits(&limits); limits.section_bytes = 100u;
    CHECK(aotx_ccir_open(path, &limits, &view) == AOTX_CCIR_LIMIT);
    aotx_ccir_default_limits(&limits); limits.file_bytes = AOTX_CCIR_DATA;
    limits.section_bytes = 100u;
    CHECK(aotx_ccir_open(path, &limits, &view) == AOTX_CCIR_LIMIT);
    f.meta.checkpoint_sequence--; f.meta.durable_sequence--;
    CHECK(aotx_ccir_append(path, f.inputs, f.count, &f.meta, NULL) == AOTX_CCIR_INVALID);
    f.meta.checkpoint_sequence++; f.meta.durable_sequence++;
    CHECK(reopen_generation(path, 2u));
    for (i = 2u; i < f.count; i++) f.inputs[i].source = AOTX_CCIR_MEMORY;
    CHECK(aotx_ccir_append(path, f.inputs, f.count, &f.meta, NULL) == 0);
    CHECK(aotx_ccir_open(path, NULL, &view) == 0);
    old_bytes = view.end; aotx_ccir_close(&view);
    CHECK(aotx_ccir_compact(path, copy, NULL) == 0);
    exact(copy, &f, 1u); exact(path, &f, 3u);
    CHECK(aotx_ccir_open(copy, NULL, &packed) == 0);
    CHECK(aotx_ccir_open(path, NULL, &view) == 0);
    CHECK(packed.end < old_bytes && !memcmp(packed.lineage, view.lineage, 16u));
    CHECK(memcmp(packed.incarnation, view.incarnation, 16u) != 0);
    for (i = 0u; i < f.count; i++)
        CHECK(!memcmp(packed.sections[i].digest, view.sections[i].digest, 32u));
    aotx_ccir_close(&packed); aotx_ccir_close(&view);
    CHECK(aotx_ccir_compact(path, copy, NULL) == AOTX_CCIR_EXISTS);
    fd = open(path, O_RDWR); CHECK(fd >= 0 && flock(fd, LOCK_EX | LOCK_NB) == 0);
    CHECK(aotx_ccir_open(path, NULL, &view) == AOTX_CCIR_BUSY);
    CHECK(aotx_ccir_compact(path, copy, NULL) == AOTX_CCIR_BUSY);
    close(fd);
    unlink(path); unlink(copy);
}

int main(void)
{
    char directory[] = "/tmp/aotx-ccir-disk-XXXXXX", path[256], copy[256];
    if (!mkdtemp(directory)) return 1;
    snprintf(path, sizeof(path), "%s/state.aotxccir", directory);
    snprintf(copy, sizeof(copy), "%s/copy.aotxccir", directory);
    cases(path, copy, 1u); cases(path, copy, 64u);
    rmdir(directory);
    printf("ccir disk: %u checks, %u failed\n", aotx_checks, aotx_failures);
    return aotx_failures ? 1 : 0;
}
