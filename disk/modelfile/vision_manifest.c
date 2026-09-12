/* Purpose: Bind one complete vision component to an exact language model entry.
 * Owns: Bounded manifest bytes until the two entries are validated.
 * Threading: One disk reader processes the component pair.
 * Lifetime: One manifest read from a store or runtime file. */
#include "disk/modelfile/vision.h"
#include "disk/modelfile/manifest.h"
#include "disk/runtime/assets.h"
#include <errno.h>
#include <string.h>

int aotx_vision_manifest_text(const char *text, size_t bytes, aotx_manifest_entry entries[2])
{
    aotx_manifest_entry pair[2];
    unsigned count = 0;
    if (!text || !entries || bytes > 2u*AOTX_MANIFEST_LINE || memchr(text, 0, bytes)) return -1;
    while (bytes) {
        const char *end = memchr(text, '\n', bytes);
        size_t n = end ? (size_t)(end-text)+1 : bytes;
        char line[AOTX_MANIFEST_LINE];
        if (n >= sizeof(line) || count == 2) return -1;
        memcpy(line, text, n); line[n] = 0;
        if (aotx_manifest_line(line, pair + count)) return -1;
        aotx_manifest_entry *e = pair + count++;
        if (!e->source[0] || !e->revision[0] || !e->license[0] || !e->bytes) return -1;
        text += n; bytes -= n;
    }
    if (count != 2 || (strcmp(pair[0].role, "language") && strcmp(pair[0].role, "language-q4")) ||
        strcmp(pair[1].role, "vision") || !strcmp(pair[0].sha256, pair[1].sha256) ||
        !strcmp(pair[0].name, pair[1].name) || !strcmp(pair[0].path, pair[1].path)) return -1;
    memcpy(entries, pair, sizeof(pair)); return 2;
}
int aotx_vision_manifest(const char *store, aotx_manifest_entry entries[2])
{
    aotx_asset asset;
    errno = 0;
    if (aotx_asset_open(store, "vision.jsonl", &asset)) return errno == ENOENT ? 0 : -1;
    char text[2u*AOTX_MANIFEST_LINE];
    int rc = -1;
    if (asset.bytes && asset.bytes <= sizeof(text) &&
        !aotx_asset_read(&asset, 0, (size_t)asset.bytes, text))
        rc = aotx_vision_manifest_text(text, (size_t)asset.bytes, entries);
    aotx_asset_close(&asset);
    return rc;
}
int aotx_vision_pair(const aotx_manifest_entry pair[2], const aotx_manifest_entry *models,
                      unsigned count)
{
    int found = -1;
    for (unsigned i = 0; i < count; ++i) {
        const aotx_manifest_entry *e = models + i, *p = pair;
        if (strcmp(e->role, p->role) || strcmp(e->name, p->name) || strcmp(e->path, p->path) ||
            strcmp(e->sha256, p->sha256) || e->bytes != p->bytes ||
            strcmp(e->source, p->source) || strcmp(e->revision, p->revision) ||
            strcmp(e->license, p->license)) continue;
        if (found >= 0) return -1;
        found = (int)i;
    }
    return found;
}
