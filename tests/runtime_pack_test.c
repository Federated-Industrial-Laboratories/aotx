/* Purpose: Check streamed component batches against the configured file capacity.
 * Owns: Distinct auxiliary assets, including an asset larger than 64 MiB.
 * Threading: One disk process checks one and 64 component files.
 * Lifetime: Each case closes its descriptors and removes its temporary files. */
#include "disk/runtime/pack.h"
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static unsigned checks, failures;
#define CHECK(x) do { ++checks; if (!(x)) { ++failures; \
    fprintf(stderr, "line %d: %s\n", __LINE__, #x); } } while (0)
static void test(unsigned n) {
    char root[] = "/tmp/aotx-runtime-pack-XXXXXX", path[256];
    CHECK(mkdtemp(root) != NULL);
    uint64_t bytes[64]; unsigned char marker[64];
    for (unsigned i = 0; i < n; ++i) {
        snprintf(path, sizeof(path), "%s/asset-%02u.bin", root, i);
        bytes[i] = i + 1 == n ? 65u * 1024u * 1024u + i : 256u + i;
        marker[i] = (unsigned char)(13 * i + 7);
        int fd = open(path, O_WRONLY | O_CREAT | O_EXCL, 0600); CHECK(fd >= 0);
        if (fd < 0) continue;
        CHECK(!ftruncate(fd, (off_t)bytes[i]));
        CHECK(pwrite(fd, marker + i, 1, (off_t)bytes[i] - 1) == 1);
        CHECK(!close(fd));
    }
    aotx_runtime_pack *p = calloc(1, sizeof(*p)); CHECK(p != NULL);
    if (p) {
        p->source.fd = -1; p->count = 5;
        CHECK(!aotx_runtime_pack_tree(p, root, "", "", 1));
        CHECK(p->count == n + 5 && p->index.count == n);
        for (unsigned i = 0; i < n && i < p->index.count; ++i) {
            const aotx_ccir_input *in = p->inputs + 5 + i;
            char name[64]; snprintf(name, sizeof(name), "asset-%02u.bin", i);
            CHECK(!strcmp(name, (const char *)p->index.rows[i] + 64));
            CHECK(in->section.bytes == bytes[i] && aotx_ccir_u64(p->index.rows[i] + 24) == bytes[i]);
            unsigned char got = 0;
            CHECK(pread(in->fd, &got, 1, (off_t)bytes[i] - 1) == 1 && got == marker[i]);
            if (i) CHECK(memcmp(p->index.rows[i - 1] + 32, p->index.rows[i] + 32, 32));
        }
        aotx_runtime_pack_close(p); free(p);
    }
    for (unsigned i = 0; i < n; ++i) {
        snprintf(path, sizeof(path), "%s/asset-%02u.bin", root, i); CHECK(!unlink(path));
    }
    CHECK(!rmdir(root));
}
int main(void) {
    test(1); test(64);
    printf("runtime pack: %u checks, %u failures\n", checks, failures);
    return failures || checks < 500 ? 1 : 0;
}
