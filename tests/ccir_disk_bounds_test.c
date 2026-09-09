/* Purpose: Check framing limits and streamed immutable sections.
 * Owns: Fixed buffers, distinct input batches and temporary files.
 * Threading: One process operates on each file in sequence.
 * Lifetime: Every case removes its files. */
#include "tests/ccir_disk_fixture.h"

static void headers(const char *source, const char *bad, uint32_t n)
{
    static const unsigned int offsets[] = {8, 10, 12, 16, 24, 40, 56, 64, 72, 76, 80, 88};
    aotx_ccir_fixture f;
    aotx_ccir_view view;
    unsigned int i;
    fixture(&f, n);
    CHECK(aotx_ccir_create(source, f.lineage, f.inputs, f.count, &f.meta, NULL) == 0);
    for (i = 0; i < sizeof(offsets) / sizeof(offsets[0]); i++) {
        unsigned char page[4096];
        int fd, rc;
        CHECK(copy_file(source, bad) == 0);
        fd = open(bad, O_RDWR); CHECK(fd >= 0);
        CHECK(aotx_ccir_pread(fd, page, sizeof(page), 0u) == 0);
        if (offsets[i] == 24u || offsets[i] == 40u) memset(page + offsets[i], 0, 16u);
        else page[offsets[i]] ^= 4u;
        aotx_ccir_hash(page, 4064u, page + 4064u);
        CHECK(aotx_ccir_pwrite(fd, page, sizeof(page), 0u) == 0);
        close(fd);
        rc = aotx_ccir_open(bad, NULL, &view);
        CHECK(rc == (offsets[i] == 8u || offsets[i] == 10u || offsets[i] == 16u ?
                     AOTX_CCIR_UNSUPPORTED : AOTX_CCIR_INVALID));
        unlink(bad);
    }
    CHECK(aotx_ccir_open(source, NULL, &view) == 0);
    {
        uint64_t cuts[] = {0, 1, 4095, 4096, 8191, 8192, 12287, 12288,
                            view.directory_offset - 1u, view.commit_offset, view.end - 1u};
        aotx_ccir_close(&view);
        for (i = 0; i < sizeof(cuts) / sizeof(cuts[0]); i++) {
            int fd;
            CHECK(copy_file(source, bad) == 0);
            fd = open(bad, O_RDWR); CHECK(fd >= 0);
            CHECK(ftruncate(fd, (off_t)cuts[i]) == 0); close(fd);
            CHECK(aotx_ccir_open(bad, NULL, &view) == AOTX_CCIR_INVALID);
            unlink(bad);
        }
    }
    CHECK(copy_file(source, bad) == 0);
    {
        int fd = open(bad, O_RDWR);
        unsigned char zero[4096] = {0};
        CHECK(fd >= 0);
        CHECK(aotx_ccir_pwrite(fd, zero, sizeof(zero), 4096u) == 0);
        close(fd); CHECK(aotx_ccir_open(bad, NULL, &view) == AOTX_CCIR_INVALID);
    }
    unlink(bad);
    f.inputs[2].section.flags = AOTX_CCIR_REQUIRED;
    CHECK(aotx_ccir_append(source, f.inputs, f.count, &f.meta, NULL) == AOTX_CCIR_UNSUPPORTED);
    CHECK(reopen_generation(source, 1u));
    f.inputs[2].section.flags = 0u;
    CHECK(aotx_ccir_append(source, f.inputs, AOTX_CCIR_SECTIONS + 1u, &f.meta, NULL) == AOTX_CCIR_LIMIT);
    CHECK(reopen_generation(source, 1u));
    unlink(source);
}

static void streaming(const char *path, const char *input, const char *compact, uint32_t n)
{
    aotx_ccir_fixture f;
    aotx_ccir_view view;
    aotx_ccir_read reads[64];
    unsigned char buffer[4096], got[64][31];
    unsigned int i, j;
    int fd;
    fixture(&f, n);
    fd = open(input, O_CREAT | O_EXCL | O_RDWR, 0600); CHECK(fd >= 0);
    for (i = 0; i < sizeof(buffer); i++) buffer[i] = (unsigned char)((i * 37u + n) % 251u);
    for (i = 0; i < 513u; i++) CHECK(write(fd, buffer, sizeof(buffer)) == (ssize_t)sizeof(buffer));
    f.inputs[2].source = AOTX_CCIR_FILE; f.inputs[2].fd = fd;
    f.inputs[2].source_offset = 17u;
    f.inputs[2].section.bytes = 2u * 1024u * 1024u + 113u;
    CHECK(aotx_ccir_create(path, f.lineage, f.inputs, f.count, &f.meta, NULL) == 0);
    close(fd); unlink(input);
    CHECK(aotx_ccir_compact(path, compact, NULL) == 0);
    CHECK(aotx_ccir_open(compact, NULL, &view) == 0);
    for (i = 0; i < n; i++) {
        reads[i] = (aotx_ccir_read){2u, (uint64_t)i * 317u, sizeof(got[i]), got[i]};
    }
    CHECK(aotx_ccir_read_batch(&view, reads, n) == 0);
    for (i = 0; i < n; i++)
        for (j = 0; j < sizeof(got[i]); j++)
            CHECK(got[i][j] == buffer[(17u + i * 317u + j) % sizeof(buffer)]);
    aotx_ccir_close(&view);
    unlink(path); unlink(compact);
}

int main(void)
{
    char directory[] = "/tmp/aotx-ccir-bounds-XXXXXX", path[256], copy[256], input[256];
    if (!mkdtemp(directory)) return 1;
    snprintf(path, sizeof(path), "%s/state.aotxccir", directory);
    snprintf(copy, sizeof(copy), "%s/copy.aotxccir", directory);
    snprintf(input, sizeof(input), "%s/input.bin", directory);
    headers(path, copy, 1u); headers(path, copy, 64u);
    streaming(path, input, copy, 1u); streaming(path, input, copy, 64u);
    rmdir(directory);
    printf("ccir disk bounds: %u checks, %u failed\n", aotx_checks, aotx_failures);
    return aotx_failures ? 1 : 0;
}
