/* Purpose: Supply bounded asset reads with a shared lease for one loading batch.
 * Owns: The batch directory and independent source descriptor duplicates.
 * Threading: One loading thread; no descriptor remains after its consumer closes.
 * Lifetime: One batch or one read when no batch is open. */
#include "disk/runtime/assets.h"
#include "disk/ccir/internal.h"
#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

typedef struct aotx_asset_batch {
    char path[1024];
    aotx_ccir_view view;
    aotx_runtime_index index;
} aotx_asset_batch;
static aotx_asset_batch *batch;

int aotx_asset_is_runtime(const char *store) {
    struct stat st;
    return store && !stat(store, &st) && S_ISREG(st.st_mode);
}
static int batch_open(const char *store, aotx_asset_batch **out) {
    *out = NULL;
    if (!store || strlen(store) >= sizeof((*out)->path)) return AOTX_CCIR_LIMIT;
    aotx_asset_batch *b = calloc(1, sizeof(*b));
    if (!b) return AOTX_CCIR_IO;
    b->view.fd = -1;
    int rc = aotx_ccir_open(store, NULL, &b->view);
    if (!rc) rc = aotx_runtime_index_read(b->view.fd, &b->view, &b->index);
    if (rc) { aotx_ccir_close(&b->view); free(b); return rc; }
    memcpy(b->path, store, strlen(store) + 1); *out = b;
    return 0;
}
int aotx_asset_begin(const char *store) {
    if (batch) return AOTX_CCIR_BUSY;
    return aotx_asset_is_runtime(store) ? batch_open(store, &batch) : 0;
}
void aotx_asset_end(void) {
    if (batch) { aotx_ccir_close(&batch->view); free(batch); batch = NULL; }
}
int aotx_asset_index(const char *store, aotx_runtime_index *index) {
    if (!store || !index) return AOTX_CCIR_INVALID;
    aotx_asset_batch *b = batch, *local = NULL;
    if (!b || strcmp(store, b->path)) {
        int rc = batch_open(store, &local);
        if (rc) return rc;
        b = local;
    }
    *index = b->index;
    if (local) { aotx_ccir_close(&local->view); free(local); }
    return 0;
}
static int asset_runtime(const char *store, const char *name, aotx_asset *asset) {
    aotx_asset_batch *b = batch, *local = NULL;
    if (!b || strcmp(store, b->path)) {
        int rc = batch_open(store, &local);
        if (rc) return rc;
        b = local;
    }
    int rc = AOTX_CCIR_INVALID;
    for (uint32_t i = 0; i < b->index.count; ++i) {
        const unsigned char *p = b->index.rows[i];
        if (strcmp(name, (const char *)p + 64)) continue;
        int at = aotx_runtime_section(&b->view, p);
        const aotx_ccir_section *s = &b->view.sections[at];
        asset->fd = fcntl(b->view.fd, F_DUPFD_CLOEXEC, 0);
        asset->offset = s->offset; asset->bytes = s->bytes;
        memcpy(asset->digest, s->digest, 32);
        rc = asset->fd < 0 ? AOTX_CCIR_IO : AOTX_CCIR_OK;
        break;
    }
    if (local) { aotx_ccir_close(&local->view); free(local); }
    if (rc == AOTX_CCIR_INVALID) errno = ENOENT;
    return rc;
}
int aotx_asset_open(const char *store, const char *name, aotx_asset *asset) {
    if (!asset) return AOTX_CCIR_INVALID;
    memset(asset, 0, sizeof(*asset)); asset->fd = -1;
    if (!store || !name) return AOTX_CCIR_INVALID;
    if (aotx_asset_is_runtime(store)) {
        if (!aotx_runtime_name(name)) return AOTX_CCIR_INVALID;
        return asset_runtime(store, name, asset);
    }
    char path[2048]; struct stat st;
    int n = name[0] == '/' ? snprintf(path, sizeof(path), "%s", name) :
        snprintf(path, sizeof(path), "%s/%s", store, name);
    if (n < 0 || (size_t)n >= sizeof(path)) return AOTX_CCIR_LIMIT;
    asset->fd = open(path, O_RDONLY | O_CLOEXEC | O_NONBLOCK);
    if (asset->fd < 0) return AOTX_CCIR_IO;
    if (fstat(asset->fd, &st) || !S_ISREG(st.st_mode) || st.st_size < 0) {
        aotx_asset_close(asset); return AOTX_CCIR_INVALID;
    }
    asset->bytes = (uint64_t)st.st_size;
    return 0;
}
void aotx_asset_close(aotx_asset *asset) {
    if (asset && asset->fd >= 0) { close(asset->fd); asset->fd = -1; }
}
int aotx_asset_read(const aotx_asset *asset, uint64_t offset, size_t bytes, void *out) {
    if (!asset || asset->fd < 0 || (!out && bytes) || offset > asset->bytes ||
        bytes > asset->bytes - offset) return AOTX_CCIR_INVALID;
    return aotx_ccir_pread(asset->fd, out, bytes, asset->offset + offset);
}
