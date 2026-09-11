/* Purpose: Read bounded assets as file streams and list asset names.
 * Owns: Stream cookies and temporary directory listings.
 * Threading: One loading thread processes the requested asset batch.
 * Lifetime: Streams end at fclose; listings belong to the caller. */
#include "disk/runtime/assets.h"
#include "disk/ccir/internal.h"
#include <dirent.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>

typedef struct aotx_asset_cookie { aotx_asset asset; uint64_t cursor; } aotx_asset_cookie;
static ssize_t asset_read(void *context, char *out, size_t bytes) {
    aotx_asset_cookie *c = context;
    if (bytes > c->asset.bytes - c->cursor) bytes = (size_t)(c->asset.bytes - c->cursor);
    if (aotx_asset_read(&c->asset, c->cursor, bytes, out)) { errno = EIO; return -1; }
    c->cursor += bytes;
    return (ssize_t)bytes;
}
static int asset_seek(void *context, off64_t *offset, int whence) {
    aotx_asset_cookie *c = context;
    uint64_t base = whence == SEEK_SET ? 0 : whence == SEEK_CUR ? c->cursor : c->asset.bytes;
    if (whence != SEEK_SET && whence != SEEK_CUR && whence != SEEK_END) return -1;
    if ((*offset < 0 && (uint64_t)(-(*offset + 1)) + 1 > base) ||
        (*offset >= 0 && (uint64_t)*offset > c->asset.bytes - base)) return -1;
    c->cursor = *offset < 0 ? base - ((uint64_t)(-(*offset + 1)) + 1) : base + (uint64_t)*offset;
    *offset = (off64_t)c->cursor;
    return 0;
}
static int asset_close(void *context) {
    aotx_asset_cookie *c = context;
    aotx_asset_close(&c->asset); free(c);
    return 0;
}
FILE *aotx_asset_stream(const char *store, const char *name) {
    aotx_asset_cookie *c = calloc(1, sizeof(*c));
    if (!c) return NULL;
    if (aotx_asset_open(store, name, &c->asset)) { free(c); return NULL; }
    cookie_io_functions_t io = {asset_read, NULL, asset_seek, asset_close};
    FILE *stream = fopencookie(c, "r", io);
    if (!stream) asset_close(c);
    return stream;
}
static int wanted(const char *name, const char *suffix) {
    size_t n = strlen(name), s = strlen(suffix);
    return n >= s && !strchr(name, '/') && !strcmp(name + n - s, suffix);
}
static int names_order(const void *a, const void *b) { return strcmp(a, b); }
int aotx_asset_names(const char *store, const char *prefix, const char *suffix,
                     char (*names)[AOTX_RUNTIME_NAME], unsigned capacity) {
    unsigned count = 0;
    if (!store || !prefix || !suffix || !names || !capacity) return -1;
    if (aotx_asset_is_runtime(store)) {
        aotx_runtime_index *index = malloc(sizeof(*index));
        if (!index) return -1;
        int rc = aotx_asset_index(store, index);
        size_t prefix_bytes = strlen(prefix);
        for (uint32_t i = 0; !rc && i < index->count; ++i) {
            const char *name = (const char *)index->rows[i] + 64;
            if (strncmp(name, prefix, prefix_bytes) || !wanted(name + prefix_bytes, suffix)) continue;
            if (count == capacity) { rc = AOTX_CCIR_LIMIT; break; }
            strcpy(names[count++], name + prefix_bytes);
        }
        free(index);
        if (rc) return -1;
    } else {
        char path[2048]; int n = snprintf(path, sizeof(path), "%s/%s", store, prefix);
        if (n < 0 || (size_t)n >= sizeof(path)) return -1;
        DIR *dir = opendir(path);
        if (!dir) return errno == ENOENT ? 0 : -1;
        struct dirent *at; int rc = 0;
        while ((at = readdir(dir))) {
            if (!wanted(at->d_name, suffix)) continue;
            if (count == capacity || strlen(at->d_name) >= AOTX_RUNTIME_NAME) { rc = -1; break; }
            strcpy(names[count++], at->d_name);
        }
        closedir(dir);
        if (rc) return -1;
    }
    qsort(names, count, AOTX_RUNTIME_NAME, names_order);
    return (int)count;
}
