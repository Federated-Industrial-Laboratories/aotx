/* Purpose: Append complete section directories and publish durable roots.
 * Owns: Bounded payload copies and the ordered publication of a generation.
 * Threading: The caller holds the exclusive file lease.
 * Lifetime: One generation write. */
#include "disk/ccir/internal.h"
#include <limits.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static int reserve(uint64_t *at, uint64_t bytes, uint32_t alignment,
                   const aotx_ccir_limits *limits, uint64_t *offset)
{
    uint64_t pad = (alignment - *at % alignment) % alignment;
    if (*at > limits->file_bytes || pad > limits->file_bytes - *at)
        return AOTX_CCIR_LIMIT;
    *at += pad; *offset = *at;
    if (bytes > limits->file_bytes - *at) return AOTX_CCIR_LIMIT;
    *at += bytes;
    return AOTX_CCIR_OK;
}

static int inputs_check(const aotx_ccir_input *inputs, uint32_t count,
                        const aotx_ccir_view *old, aotx_ccir_view *next,
                        const aotx_ccir_limits *limits, uint64_t *at)
{
    uint32_t i, j;
    int rc;
    for (i = 0; i < count; i++) {
        const aotx_ccir_input *in = &inputs[i];
        aotx_ccir_section *s = &next->sections[i];
        *s = in->section;
        if (!s->type || !s->schema || !s->bytes ||
            (s->flags & ~AOTX_CCIR_REQUIRED) || aotx_ccir_zero(s->id, 16u) ||
            !s->alignment || s->alignment > AOTX_CCIR_PAGE ||
            (s->alignment & (s->alignment - 1u))) return AOTX_CCIR_INVALID;
        if (s->bytes > limits->section_bytes) return AOTX_CCIR_LIMIT;
        if ((s->flags & AOTX_CCIR_REQUIRED) &&
            (s->type > AOTX_CCIR_REPLAY || (s->schema != 1u &&
             !(s->type <= AOTX_CCIR_TAIL && s->schema == 2u) &&
             !(s->type == AOTX_CCIR_RUNTIME && s->schema == 2u) &&
             !(s->type == AOTX_CCIR_MANIFEST && s->schema == 3u))))
            return AOTX_CCIR_UNSUPPORTED;
        for (j = 0; j < i; j++)
            if (!memcmp(s->id, next->sections[j].id, 16u)) return AOTX_CCIR_INVALID;
        if (in->source == AOTX_CCIR_REUSE) {
            for (j = 0; j < old->count; j++)
                if (!memcmp(s->id, old->sections[j].id, 16u)) break;
            if (j == old->count) return AOTX_CCIR_INVALID;
            if (s->type != old->sections[j].type || s->schema != old->sections[j].schema ||
                s->flags != old->sections[j].flags || s->bytes != old->sections[j].bytes ||
                s->alignment != old->sections[j].alignment) return AOTX_CCIR_INVALID;
            *s = old->sections[j];
        } else {
            if (in->source == AOTX_CCIR_MEMORY) {
                if (!in->data || s->bytes > SIZE_MAX) return AOTX_CCIR_INVALID;
            } else if (in->source == AOTX_CCIR_FILE) {
                struct stat st;
                if (in->fd < 0 || fstat(in->fd, &st) || !S_ISREG(st.st_mode) ||
                    st.st_size < 0 || in->source_offset > (uint64_t)st.st_size ||
                    s->bytes > (uint64_t)st.st_size - in->source_offset)
                    return AOTX_CCIR_INVALID;
            } else return AOTX_CCIR_INVALID;
            rc = reserve(at, s->bytes, s->alignment, limits, &s->offset);
            if (rc) return rc;
        }
    }
    return AOTX_CCIR_OK;
}

static int payloads(int fd, const aotx_ccir_input *inputs, aotx_ccir_view *next)
{
    unsigned char block[AOTX_CCIR_CHUNK];
    uint32_t i;
    for (i = 0; i < next->count; i++) {
        const aotx_ccir_input *in = &inputs[i];
        aotx_ccir_section *s = &next->sections[i];
        aotx_sha256 hash;
        uint64_t at = 0u;
        if (in->source == AOTX_CCIR_REUSE) continue;
        aotx_sha256_init(&hash);
        while (at < s->bytes) {
            size_t take = s->bytes - at < sizeof(block) ?
                          (size_t)(s->bytes - at) : sizeof(block);
            const void *data = block;
            int rc;
            if (in->source == AOTX_CCIR_MEMORY)
                data = (const unsigned char *)in->data + (size_t)at;
            else {
                rc = aotx_ccir_pread(in->fd, block, take, in->source_offset + at);
                if (rc) return rc;
            }
            rc = aotx_ccir_pwrite(fd, data, take, s->offset + at);
            if (rc) return rc;
            aotx_sha256_update(&hash, data, take);
            at += take;
        }
        aotx_sha256_final(&hash, s->digest);
    }
    return AOTX_CCIR_OK;
}

static void commit_bytes(const aotx_ccir_view *old, const aotx_ccir_view *next,
                         const unsigned char *rows, unsigned char out[256])
{
    uint32_t i;
    memset(out, 0, AOTX_CCIR_COMMIT);
    memcpy(out, "AOTXCMT1", 8u);
    aotx_ccir_put(out + 8, next->generation, 8u);
    memcpy(out + 16, old->commit_digest, 32u);
    aotx_ccir_put(out + 48, next->meta.checkpoint_sequence, 8u);
    aotx_ccir_put(out + 56, next->meta.durable_sequence, 8u);
    aotx_ccir_put(out + 64, next->meta.source_tick, 8u);
    aotx_ccir_put(out + 72, next->directory_offset, 8u);
    aotx_ccir_put(out + 80, next->count, 4u);
    aotx_ccir_put(out + 84, AOTX_CCIR_ROW, 4u);
    aotx_ccir_put(out + 88, next->count * AOTX_CCIR_ROW, 8u);
    aotx_ccir_hash(rows, next->count * AOTX_CCIR_ROW, out + 96);
    aotx_ccir_put(out + 128, next->end, 8u);
    for (i = 0; i < next->count; i++) {
        const aotx_ccir_section *s = &next->sections[i];
        if (s->type <= AOTX_CCIR_TAIL)
            memcpy(out + 136 + (s->type - 1u) * 16u, s->id, 16u);
    }
}

int aotx_ccir_write_generation(int fd, const aotx_ccir_view *old,
                               const aotx_ccir_input *inputs, uint32_t count,
                               const aotx_ccir_meta *meta,
                               const aotx_ccir_limits *limits, aotx_ccir_view *published)
{
    unsigned char rows[AOTX_CCIR_SECTIONS * AOTX_CCIR_ROW];
    unsigned char commit[AOTX_CCIR_COMMIT], page[AOTX_CCIR_PAGE];
    aotx_ccir_view next = *old;
    struct stat st;
    uint64_t at;
    uint32_t i;
    int rc;
    if (!inputs || !meta || count < 2u ||
        meta->checkpoint_sequence > meta->durable_sequence ||
        meta->checkpoint_sequence < old->meta.checkpoint_sequence ||
        meta->durable_sequence < old->meta.durable_sequence ||
        meta->source_tick < old->meta.source_tick) return AOTX_CCIR_INVALID;
    if (count > limits->sections) return AOTX_CCIR_LIMIT;
    if (old->generation == UINT64_MAX) return AOTX_CCIR_LIMIT;
    if (fstat(fd, &st) || st.st_size < AOTX_CCIR_DATA) return AOTX_CCIR_IO;
    at = (uint64_t)st.st_size;
    next.count = count; next.generation = old->generation + 1u; next.meta = *meta;
    rc = inputs_check(inputs, count, old, &next, limits, &at);
    if (rc) return rc;
    rc = reserve(&at, (uint64_t)count * AOTX_CCIR_ROW, 128u, limits,
                   &next.directory_offset);
    if (rc) return rc;
    rc = reserve(&at, AOTX_CCIR_COMMIT, 128u, limits, &next.commit_offset);
    if (rc) return rc;
    next.end = at;
    rc = payloads(fd, inputs, &next);
    if (rc) return rc;
    for (i = 0; i < count; i++) aotx_ccir_encode_row(&next.sections[i], rows + i * 128u);
    commit_bytes(old, &next, rows, commit);
    rc = aotx_ccir_profile(fd, &next, commit);
    if (rc) return rc;
    rc = aotx_ccir_pwrite(fd, rows, count * AOTX_CCIR_ROW, next.directory_offset);
    if (rc || fsync(fd)) return AOTX_CCIR_IO;
    rc = aotx_ccir_pwrite(fd, commit, sizeof(commit), next.commit_offset);
    if (rc || fsync(fd)) return AOTX_CCIR_IO;
    memset(page, 0, sizeof(page));
    memcpy(page, "AOTXROOT", 8u);
    aotx_ccir_put(page + 8, 1u, 2u);
    aotx_ccir_put(page + 16, next.generation, 8u);
    aotx_ccir_put(page + 24, next.commit_offset, 8u);
    aotx_ccir_put(page + 32, AOTX_CCIR_COMMIT, 8u);
    aotx_ccir_put(page + 40, next.end, 8u);
    aotx_ccir_hash(commit, sizeof(commit), page + 48);
    memcpy(page + 80, old->prologue_digest, 32u);
    aotx_ccir_hash(page, AOTX_CCIR_HASH_OFFSET, page + AOTX_CCIR_HASH_OFFSET);
    i = old->generation ? 1u - old->root_slot : 0u;
    rc = aotx_ccir_pwrite(fd, page, sizeof(page), (i + 1u) * AOTX_CCIR_PAGE);
    if (rc || fsync(fd)) return AOTX_CCIR_IO;
    if (published) {
        next.fd = fd; next.root_slot = i; next.fallback = 0; next.trailing_bytes = 0;
        aotx_ccir_hash(commit, sizeof(commit), next.commit_digest);
        *published = next;
    }
    return AOTX_CCIR_OK;
}
