/* Purpose: Encode section rows and check the data-state profile.
 * Owns: Explicit byte offsets for rows and the profile manifest.
 * Threading: The caller holds a file lease.
 * Lifetime: One generation check. */
#include "disk/ccir/internal.h"
#include "disk/runtime/runtime.h"
#include <string.h>

void aotx_ccir_manifest(unsigned char out[AOTX_CCIR_MANIFEST_BYTES],
                        const unsigned char checkpoint[16],
                        const unsigned char tail[16])
{
    memset(out, 0, AOTX_CCIR_MANIFEST_BYTES);
    memcpy(out, "AOTXDATA", 8u);
    aotx_ccir_put(out + 8, 1u, 4u);
    aotx_ccir_put(out + 12, 1u, 4u);
    aotx_ccir_put(out + 16, 256u, 4u);
    aotx_ccir_put(out + 20, 1u, 4u);
    memcpy(out + 24, checkpoint, 16u);
    if (tail) memcpy(out + 40, tail, 16u);
}
void aotx_ccir_encode_row(const aotx_ccir_section *s, unsigned char row[128])
{
    memset(row, 0, 128u);
    aotx_ccir_put(row, s->type, 4u);
    aotx_ccir_put(row + 4, s->schema, 2u);
    aotx_ccir_put(row + 6, s->flags, 2u);
    memcpy(row + 8, s->id, 16u);
    aotx_ccir_put(row + 24, s->offset, 8u);
    aotx_ccir_put(row + 32, s->bytes, 8u);
    aotx_ccir_put(row + 40, s->bytes, 8u);
    aotx_ccir_put(row + 48, s->alignment, 4u);
    memcpy(row + 56, s->digest, 32u);
}
void aotx_ccir_live_manifest(unsigned char out[AOTX_CCIR_MANIFEST_BYTES],
    const unsigned char checkpoint[16], const unsigned char live[16])
{
    aotx_ccir_manifest(out, checkpoint, NULL);
    aotx_ccir_put(out + 8, 2, 4);
    memcpy(out + 56, live, 16);
}
int aotx_ccir_decode_row(const unsigned char row[128], aotx_ccir_section *s,
                         const aotx_ccir_limits *limits, uint64_t end)
{
    s->type = aotx_ccir_u32(row);
    s->schema = aotx_ccir_u16(row + 4);
    s->flags = aotx_ccir_u16(row + 6);
    memcpy(s->id, row + 8, 16u);
    s->offset = aotx_ccir_u64(row + 24);
    s->bytes = aotx_ccir_u64(row + 32);
    s->alignment = aotx_ccir_u32(row + 48);
    memcpy(s->digest, row + 56, 32u);
    if (!s->type || !s->schema || (s->flags & ~AOTX_CCIR_REQUIRED) ||
        aotx_ccir_zero(s->id, 16u) || !s->bytes || !s->alignment ||
        s->alignment > AOTX_CCIR_PAGE || (s->alignment & (s->alignment - 1u)) ||
        s->offset < AOTX_CCIR_DATA || s->offset % s->alignment ||
        s->offset > end || s->bytes > end - s->offset ||
        aotx_ccir_u64(row + 40) != s->bytes || !aotx_ccir_zero(row + 54, 2u) ||
        !aotx_ccir_zero(row + 88, 40u)) return AOTX_CCIR_INVALID;
    if (s->bytes > limits->section_bytes) return AOTX_CCIR_LIMIT;
    if (aotx_ccir_u16(row + 52)) return AOTX_CCIR_UNSUPPORTED;
    return AOTX_CCIR_OK;
}
int aotx_ccir_profile(int fd, const aotx_ccir_view *view,
                      const unsigned char commit[AOTX_CCIR_COMMIT])
{
    unsigned char manifest[AOTX_CCIR_MANIFEST_BYTES];
    const aotx_ccir_section *known[8] = {NULL};
    uint32_t i;
    int rc, unsupported = 0;
    for (i = 0; i < view->count; i++) {
        const aotx_ccir_section *s = &view->sections[i];
        if (s->type <= AOTX_CCIR_TAIL || (s->flags & AOTX_CCIR_REQUIRED)) {
            if (s->type > AOTX_CCIR_REPLAY) { unsupported = 1; continue; }
            if (s->type != AOTX_CCIR_ASSET) {
                if (known[s->type]) return AOTX_CCIR_INVALID;
                known[s->type] = s;
            }
            if (s->schema != 1u && !(s->type <= AOTX_CCIR_TAIL && s->schema == 2u) &&
                !(s->type == AOTX_CCIR_RUNTIME && (s->schema >= 2u && s->schema <= 4u)) &&
                !(s->type == AOTX_CCIR_MANIFEST && s->schema == 3u)) unsupported = 1;
            if (s->flags != AOTX_CCIR_REQUIRED) return AOTX_CCIR_INVALID;
        }
    }
    if (!known[1] || !known[2]) return AOTX_CCIR_INVALID;
    if (unsupported) return AOTX_CCIR_UNSUPPORTED;
    if (known[1]->bytes != sizeof(manifest)) return AOTX_CCIR_INVALID;
    rc = aotx_ccir_pread(fd, manifest, sizeof(manifest), known[1]->offset);
    if (rc) return rc;
    if (memcmp(manifest, "AOTXDATA", 8u)) return AOTX_CCIR_INVALID;
    uint32_t schema = aotx_ccir_u32(manifest + 8);
    if ((schema < 1u || schema > 3u) || schema != known[1]->schema || aotx_ccir_u32(manifest + 12) != 1u ||
        aotx_ccir_u32(manifest + 16) != 256u || aotx_ccir_u32(manifest + 20) != known[2]->schema ||
        (schema == 1 && aotx_ccir_u64(manifest + 56))) return AOTX_CCIR_UNSUPPORTED;
    if (schema >= 2 && (!known[4] || known[3] ||
        memcmp(manifest + 56, known[4]->id, 16))) return AOTX_CCIR_INVALID;
    if (schema == 1 && known[4]) return AOTX_CCIR_INVALID;
    uint32_t reserved = schema == 1 ? 64 : schema == 2 ? 72 : 88;
    if (!aotx_ccir_zero(manifest + reserved, sizeof(manifest) - reserved) ||
        memcmp(manifest + 24, known[2]->id, 16u) ||
        memcmp(commit + 136, known[1]->id, 16u) ||
        memcmp(commit + 152, known[2]->id, 16u)) return AOTX_CCIR_INVALID;
    if (schema == 3) {
        if (!known[5] || !known[7]) return AOTX_CCIR_INVALID;
        rc = aotx_runtime_profile(fd, view, manifest + 72);
        if (rc) return rc;
    } else {
        for (i = 0; i < view->count; ++i)
            if (view->sections[i].type >= AOTX_CCIR_RUNTIME &&
                (view->sections[i].flags & AOTX_CCIR_REQUIRED)) return AOTX_CCIR_INVALID;
    }
    if (known[3]) {
        if (memcmp(manifest + 40, known[3]->id, 16u) ||
            memcmp(commit + 168, known[3]->id, 16u)) return AOTX_CCIR_INVALID;
    } else if (!aotx_ccir_zero(manifest + 40, 16u) ||
               !aotx_ccir_zero(commit + 168, 16u) ||
               view->meta.durable_sequence != view->meta.checkpoint_sequence)
        return AOTX_CCIR_INVALID;
    return AOTX_CCIR_OK;
}
