/* Purpose: Bind original model bytes and a relocated manifest into a runtime file.
 * Owns: The generated manifest until its descriptor enters the component batch.
 * Threading: One packager checks each source model and its tensor metadata.
 * Lifetime: One complete file creation. */
#include "disk/runtime/pack.h"
#include <stdio.h>
#include "disk/modelfile/manifest.h"
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int aotx_runtime_pack_models(aotx_runtime_pack *p, const char *store) {
    aotx_manifest_entry entries[8], original[8];
    int count = aotx_manifest_read(store, entries, 8);
    if (count <= 0) return AOTX_CCIR_INVALID;
    memcpy(original, entries, count * sizeof(*entries));
    FILE *manifest = tmpfile();
    if (!manifest) return AOTX_CCIR_IO;
    int rc = 0;
    for (int i = 0; i < count && !rc; ++i) {
        aotx_manifest_entry *e = entries + i;
        unsigned char expected[32];
        if (!e->source[0] || !e->revision[0] || !e->license[0] ||
            aotx_manifest_digest(e->sha256, expected)) {
            rc = AOTX_CCIR_INVALID; break;
        }
        aotx_modelfile *model = NULL;
        rc = aotx_modelfile_open_entry(store, e, &model);
        if (rc) break;
        aotx_modelfile_close(model);
        char path[AOTX_MANIFEST_PATH], name[64], line[AOTX_MANIFEST_LINE];
        if (aotx_manifest_path(path, sizeof(path), store, e->path)) { rc = AOTX_CCIR_LIMIT; break; }
        snprintf(name, sizeof(name), "weights/%u.gguf", (unsigned)i);
        rc = aotx_runtime_pack_asset(p, path, name, 1);
        if (rc) break;
        const aotx_ccir_section *asset = &p->inputs[p->count - 1].section;
        if (asset->bytes != e->bytes || memcmp(asset->digest, expected, 32)) {
            rc = AOTX_CCIR_INVALID; break;
        }
        strcpy(e->path, name);
        if (aotx_manifest_write_line(line, sizeof(line), e) || fputs(line, manifest) == EOF) rc = AOTX_CCIR_IO;
    }
    if (!rc && fflush(manifest)) rc = AOTX_CCIR_IO;
    if (!rc) {
        char path[64]; snprintf(path, sizeof(path), "/proc/self/fd/%d", fileno(manifest));
        rc = aotx_runtime_pack_asset(p, path, "manifest.jsonl", 1);
    }
    fclose(manifest);
    if (!rc) rc = aotx_runtime_pack_vision(p, store, original, entries, (unsigned)count);
    if (!rc) rc = aotx_runtime_pack_tree(p, store, "", "", 1);
    return rc;
}
