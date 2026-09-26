/* Purpose: Verify control evidence through directory and complete-file readers.
 * Owns: Short file leases and a bounded qualification buffer.
 * Threading: One disk reader for each control batch.
 * Lifetime: No file or buffer remains after a call. */
#include "disk/runtime/qualification.h"
#include "disk/runtime/assets.h"
#include "disk/ccir/internal.h"
#include "disk/wire/diskwire.h"
#include <errno.h>
#include <string.h>

static int related(const aotx_qualification *q, const char *asset, unsigned kind) {
    char name[AOTX_RUNTIME_NAME];
    int n = snprintf(name, sizeof(name), "%s.binding", asset);
    return n < 0 || (size_t)n >= sizeof(name) || kind != q->kind ||
        strcmp(q->reference[0].name, name);
}
static int read_asset(const char *store, const char *name, unsigned char *raw, size_t room,
    size_t *length) {
    aotx_asset asset;
    int rc = aotx_asset_open(store, name, &asset);
    if (rc) return 1;
    int bad = !asset.bytes || asset.bytes > room;
    if (!bad) { *length = (size_t)asset.bytes; bad = aotx_asset_read(&asset, 0, *length, raw); }
    aotx_asset_close(&asset); return bad;
}
static int hash_asset(const char *store, const aotx_control_reference_file *ref) {
    aotx_asset asset;
    if (aotx_asset_open(store, ref->name, &asset)) return 1;
    int bad = !asset.bytes || asset.bytes > 16u * 1024u * 1024u;
    aotx_sha256 state; aotx_sha256_init(&state);
    unsigned char raw[16384], digest[32];
    for (uint64_t at = 0; !bad && at < asset.bytes;) {
        size_t n = asset.bytes - at < sizeof(raw) ? (size_t)(asset.bytes - at) : sizeof(raw);
        bad = aotx_asset_read(&asset, at, n, raw);
        if (!bad) aotx_sha256_update(&state, raw, n);
        at += n;
    }
    aotx_sha256_final(&state, digest);
    aotx_asset_close(&asset);
    return bad || memcmp(digest, ref->digest, 32);
}
int aotx_qualification_read(const char *store, const char *asset, unsigned kind,
    aotx_control_permit *permit) {
    char name[AOTX_RUNTIME_NAME]; aotx_asset file;
    memset(permit, 0, sizeof(*permit));
    if (!aotx_runtime_name(asset)) return 1;
    int n = snprintf(name, sizeof(name), "%s.qualification", asset);
    if (n < 0 || (size_t)n >= sizeof(name)) return 1;
    errno = 0;
    if (aotx_asset_open(store, name, &file)) return errno == ENOENT ? 0 : 1;
    unsigned char raw[AOTX_QUALIFICATION_BYTES]; size_t length = (size_t)file.bytes;
    int bad = !file.bytes || file.bytes > sizeof(raw);
    if (!bad) bad = aotx_asset_read(&file, 0, length, raw);
    aotx_asset_close(&file);
    aotx_qualification q;
    if (bad || aotx_qualification_parse(raw, length, &q) || related(&q, asset, kind)) return 1;
    for (unsigned i = 0; i < AOTX_QUALIFICATION_REFERENCES; ++i)
        if (hash_asset(store, q.reference + i)) return 1;
    aotx_sha256 state; aotx_sha256_init(&state);
    aotx_sha256_update(&state, raw, length); aotx_sha256_final(&state, q.permit.digest);
    unsigned char binding[AOTX_CONTROL_BYTES], wanted[32]; aotx_control_identity identity;
    unsigned positions; FILE *in = aotx_asset_stream(store, asset);
    bad = read_asset(store, q.reference[0].name, binding, sizeof(binding), &length) ||
        length != sizeof(binding) || aotx_control_decode(binding, kind, &identity, wanted, &positions);
    if (!bad) bad = aotx_control_read(store, asset, kind, &identity, in, &positions);
    if (in) fclose(in);
    if (bad) return 1;
    *permit = q.permit; return 0;
}

static int named(const aotx_runtime_index *index, const char *name) {
    for (unsigned i = 0; i < index->count; ++i)
        if (!strcmp((const char *)index->rows[i] + 64, name)) return (int)i;
    return -1;
}
int aotx_qualification_references(const aotx_ccir_view *view, const aotx_runtime_index *index) {
    const char suffix[] = ".qualification";
    for (unsigned i = 0; i < index->count; ++i) {
        const char *name = (const char *)index->rows[i] + 64;
        size_t length = strlen(name), cut = sizeof(suffix) - 1;
        if (length <= cut || strcmp(name + length - cut, suffix)) continue;
        if (aotx_ccir_u32(index->rows[i] + 16) != 1) return AOTX_CCIR_INVALID;
        int at = aotx_runtime_section(view, index->rows[i]);
        if (at < 0) return AOTX_CCIR_INVALID;
        const aotx_ccir_section *s = view->sections + at;
        unsigned char raw[AOTX_QUALIFICATION_BYTES]; aotx_qualification q;
        if (!s->bytes || s->bytes > sizeof(raw) ||
            aotx_ccir_pread(view->fd, raw, (size_t)s->bytes, s->offset) ||
            aotx_qualification_parse(raw, (size_t)s->bytes, &q)) return AOTX_CCIR_INVALID;
        char asset[AOTX_RUNTIME_NAME]; memcpy(asset, name, length - cut); asset[length - cut] = 0;
        if (related(&q, asset, q.kind) || named(index, asset) < 0) return AOTX_CCIR_INVALID;
        for (unsigned j = 0; j < AOTX_QUALIFICATION_REFERENCES; ++j) {
            at = named(index, q.reference[j].name);
            if (at < 0 || !aotx_ccir_u64(index->rows[at] + 24) ||
                aotx_ccir_u64(index->rows[at] + 24) > 16u * 1024u * 1024u ||
                memcmp(q.reference[j].digest, index->rows[at] + 32, 32)) return AOTX_CCIR_INVALID;
        }
        if (aotx_control_reference(view, index, asset, q.kind, q.kind == AOTX_CONTROL_VECTOR))
            return AOTX_CCIR_INVALID;
    }
    return 0;
}
