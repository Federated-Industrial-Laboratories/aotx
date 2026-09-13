/* Purpose: Validate the complete text runtime dependency index without execution.
 * Owns: Index framing and required asset reference checks.
 * Threading: One caller holds the file lease and checks all rows.
 * Lifetime: One generation read or publication. */
#include "disk/runtime/runtime.h"
#include "disk/runtime/replay.h"
#include "disk/ccir/internal.h"
#include "cuda/shared/profile.h"
#include <stdlib.h>
#include <string.h>

int aotx_runtime_name(const char *name) {
    size_t n = strnlen(name, AOTX_RUNTIME_NAME);
    if (!n || n == AOTX_RUNTIME_NAME || name[0] == '/' || name[n - 1] == '/') return 0;
    size_t start = 0;
    for (size_t i = 0; i <= n; ++i) {
        unsigned char c = (unsigned char)name[i];
        if (!c || c == '/') {
            size_t span = i - start;
            if (!span || (span == 1 && name[start] == '.') ||
                (span == 2 && name[start] == '.' && name[start + 1] == '.')) return 0;
            start = i + 1;
        } else if (c < 33 || c > 126 || c == '\\' || c == ':' || c == '"') return 0;
    }
    return 1;
}
int aotx_runtime_section(const aotx_ccir_view *view, const unsigned char id[16]) {
    for (uint32_t i = 0; i < view->count; ++i)
        if (!memcmp(view->sections[i].id, id, 16)) return (int)i;
    return -1;
}
void aotx_runtime_revision(const aotx_ccir_view *view, unsigned char digest[32]) {
    unsigned char source[64];
    memcpy(source, view->prologue_digest, 32); memcpy(source + 32, view->commit_digest, 32);
    aotx_ccir_hash(source, sizeof(source), digest);
}
static int text_zero(const unsigned char *p, size_t n) {
    const unsigned char *end = memchr(p, 0, n);
    return end && aotx_ccir_zero(end, n - (size_t)(end - p));
}
int aotx_runtime_index_read(int fd, const aotx_ccir_view *view, aotx_runtime_index *index) {
    const aotx_ccir_section *s = NULL;
    for (uint32_t i = 0; i < view->count; ++i) {
        if (view->sections[i].type != AOTX_CCIR_RUNTIME ||
            !(view->sections[i].flags & AOTX_CCIR_REQUIRED)) continue;
        if (s) return AOTX_CCIR_INVALID;
        s = view->sections + i;
    }
    if (!s || (s->schema != 1 && s->schema != 2) || s->flags != AOTX_CCIR_REQUIRED ||
        s->bytes < AOTX_RUNTIME_HEADER) return AOTX_CCIR_INVALID;
    unsigned char *h = index->header;
    int rc = aotx_ccir_pread(fd, h, AOTX_RUNTIME_HEADER, s->offset);
    if (rc) return rc;
    index->count = aotx_ccir_u32(h + 16);
    if (memcmp(h, "AOTXRT01", 8) || aotx_ccir_u32(h + 8) != 1 ||
        aotx_ccir_u32(h + 12) != AOTX_RUNTIME_ROW || !index->count ||
        index->count > AOTX_CCIR_SECTIONS ||
        s->bytes != AOTX_RUNTIME_HEADER + (uint64_t)index->count * AOTX_RUNTIME_ROW ||
        !aotx_ccir_zero(h + 52, 12) ||
        !text_zero(h + 64, 64) || !h[64] || !aotx_ccir_u32(h + 24) ||
        !aotx_ccir_u32(h + 28) || !aotx_ccir_u32(h + 32) ||
        !aotx_ccir_u32(h + 36) || !aotx_ccir_u64(h + 40)) return AOTX_CCIR_INVALID;
    if (aotx_ccir_u32(h + 20) & ~(AOTX_RUNTIME_AFFECT | AOTX_RUNTIME_VISION | AOTX_RUNTIME_AUDIO | AOTX_RUNTIME_SHARED) ||
        aotx_ccir_u32(h + 48) != AOTX_RUNTIME_ABI) return AOTX_CCIR_UNSUPPORTED;
    aotx_runtime_shared_profile shared;
    if (s->schema != ((aotx_ccir_u32(h + 20) & AOTX_RUNTIME_SHARED) ? 2 : 1))
        return AOTX_CCIR_UNSUPPORTED;
    rc = aotx_runtime_shared_read(h, &shared);
    if (rc) return rc;
    int replay = aotx_runtime_section(view, h + 128);
    if (replay < 0 || view->sections[replay].type != AOTX_CCIR_REPLAY ||
        view->sections[replay].schema != 1 || view->sections[replay].flags != AOTX_CCIR_REQUIRED)
        return AOTX_CCIR_INVALID;
    unsigned char replay_header[128];
    rc = aotx_runtime_replay_header(fd, view->sections + replay, replay_header);
    if (rc) return rc;
    if (aotx_ccir_u32(replay_header + 12) == 2) {
        const aotx_ccir_section *live = NULL;
        for (uint32_t i = 0; i < view->count; ++i)
            if (view->sections[i].type == AOTX_CCIR_LIVE &&
                (view->sections[i].flags & AOTX_CCIR_REQUIRED)) live = view->sections + i;
        unsigned char saved[128];
        if (!live || live->bytes < sizeof(saved)) return AOTX_CCIR_INVALID;
        rc = aotx_ccir_pread(fd, saved, sizeof(saved), live->offset);
        if (rc) return rc;
        if (memcmp(saved, "AOTXLCP1", 8) ||
            aotx_ccir_u64(saved + 64) != aotx_ccir_u64(replay_header + 64) ||
            aotx_ccir_u64(saved + 72) != aotx_ccir_u64(replay_header + 24)) return AOTX_CCIR_INVALID;
    }
    rc = aotx_ccir_pread(fd, index->rows, (size_t)index->count * AOTX_RUNTIME_ROW,
                         s->offset + AOTX_RUNTIME_HEADER);
    if (rc) return rc;
    uint32_t assets = 0;
    for (uint32_t i = 0; i < view->count; ++i) assets += view->sections[i].type == AOTX_CCIR_ASSET &&
        (view->sections[i].flags & AOTX_CCIR_REQUIRED);
    if (assets != index->count) return AOTX_CCIR_INVALID;
    for (uint32_t i = 0; i < index->count; ++i) {
        const unsigned char *p = index->rows[i];
        int at = aotx_runtime_section(view, p);
        if (at < 0 || !text_zero(p + 64, AOTX_RUNTIME_NAME) ||
            !aotx_runtime_name((const char *)p + 64) || !aotx_ccir_zero(p + 320, 64))
            return AOTX_CCIR_INVALID;
        const aotx_ccir_section *asset = view->sections + at;
        uint32_t kind = aotx_ccir_u32(p + 16);
        if ((kind != 1 && kind != 2) || aotx_ccir_u32(p + 20)) return AOTX_CCIR_UNSUPPORTED;
        if (asset->type != AOTX_CCIR_ASSET || asset->schema != 1 ||
            asset->flags != AOTX_CCIR_REQUIRED || asset->bytes != aotx_ccir_u64(p + 24) ||
            memcmp(asset->digest, p + 32, 32)) return AOTX_CCIR_INVALID;
        for (uint32_t j = 0; j < i; ++j)
            if (!memcmp(p, index->rows[j], 16) ||
                !strcmp((const char *)p + 64, (const char *)index->rows[j] + 64)) return AOTX_CCIR_INVALID;
    }
    return AOTX_CCIR_OK;
}
int aotx_runtime_shared_read(const unsigned char h[AOTX_RUNTIME_HEADER],
    aotx_runtime_shared_profile *p) {
    memset(p, 0, sizeof(*p));
    if (!(aotx_ccir_u32(h + 20) & AOTX_RUNTIME_SHARED))
        return aotx_ccir_zero(h + 144, 112) ? AOTX_CCIR_OK : AOTX_CCIR_INVALID;
    if (memcmp(h + 144, "AOTXSH01", 8)) return AOTX_CCIR_INVALID;
    if (aotx_ccir_u32(h + 152) != AOTX_RUNTIME_SHARED_SCHEMA ||
        aotx_ccir_u32(h + 156) != AOTX_RUNTIME_SHARED_BYTES) return AOTX_CCIR_UNSUPPORTED;
    if (!aotx_ccir_zero(h + 188, 68)) return AOTX_CCIR_INVALID;
    p->participants = aotx_ccir_u32(h + 160); p->spaces = aotx_ccir_u32(h + 164);
    p->conversations = aotx_ccir_u32(h + 168); p->members = aotx_ccir_u32(h + 172);
    p->receipts = aotx_ccir_u32(h + 176); p->command_bytes = aotx_ccir_u32(h + 180);
    p->result_bytes = aotx_ccir_u32(h + 184);
    return p->participants && p->spaces && p->conversations && p->members && p->receipts &&
        p->command_bytes && p->result_bytes ? AOTX_CCIR_OK : AOTX_CCIR_INVALID;
}
void aotx_runtime_shared_write(unsigned char h[AOTX_RUNTIME_HEADER],
    const aotx_runtime_shared_profile *p) {
    memset(h + 144, 0, 112);
    aotx_ccir_put(h + 20, aotx_ccir_u32(h + 20) | AOTX_RUNTIME_SHARED, 4);
    memcpy(h + 144, "AOTXSH01", 8);
    aotx_ccir_put(h + 152, AOTX_RUNTIME_SHARED_SCHEMA, 4);
    aotx_ccir_put(h + 156, AOTX_RUNTIME_SHARED_BYTES, 4);
    aotx_ccir_put(h + 160, p->participants, 4); aotx_ccir_put(h + 164, p->spaces, 4);
    aotx_ccir_put(h + 168, p->conversations, 4); aotx_ccir_put(h + 172, p->members, 4);
    aotx_ccir_put(h + 176, p->receipts, 4); aotx_ccir_put(h + 180, p->command_bytes, 4);
    aotx_ccir_put(h + 184, p->result_bytes, 4);
}
void aotx_runtime_shared_current(aotx_runtime_shared_profile *p) {
    p->participants = AOTX_SHARED_PARTICIPANTS; p->spaces = AOTX_SHARED_SPACES;
    p->conversations = AOTX_SHARED_CONVERSATIONS; p->members = AOTX_SHARED_MEMBERS;
    p->receipts = AOTX_SHARED_RECEIPTS; p->command_bytes = AOTX_SHARED_COMMAND_BYTES;
    p->result_bytes = AOTX_SHARED_RESULT_BYTES;
}
int aotx_runtime_shared_fits(const aotx_runtime_shared_profile *p) {
    return p->participants && p->participants <= AOTX_SHARED_PARTICIPANTS &&
        p->spaces && p->spaces <= AOTX_SHARED_SPACES &&
        p->conversations && p->conversations <= AOTX_SHARED_CONVERSATIONS &&
        p->members && p->members <= AOTX_SHARED_MEMBERS &&
        p->receipts && p->receipts <= AOTX_SHARED_RECEIPTS &&
        p->command_bytes == AOTX_SHARED_COMMAND_BYTES && p->result_bytes == AOTX_SHARED_RESULT_BYTES;
}
int aotx_runtime_profile(int fd, const aotx_ccir_view *view, const unsigned char id[16]) {
    int at = aotx_runtime_section(view, id);
    if (at < 0 || view->sections[at].type != AOTX_CCIR_RUNTIME ||
        (view->sections[at].schema != 1 && view->sections[at].schema != 2) ||
        view->sections[at].flags != AOTX_CCIR_REQUIRED)
        return AOTX_CCIR_INVALID;
    aotx_runtime_index *index = malloc(sizeof(*index));
    if (!index) return AOTX_CCIR_IO;
    int rc = aotx_runtime_index_read(fd, view, index);
    free(index);
    return rc;
}
