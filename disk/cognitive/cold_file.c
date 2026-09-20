/* Purpose: Frame cold payload catalogs and copy exact requested file extents.
 * Owns: File byte ranges and metadata identity; semantic checks stay on CUDA.
 * Threading: One catalog belongs to one bounded disk operation.
 * Lifetime: The caller retains the source descriptor throughout each operation. */
#include "disk/cognitive/cold_io.h"
#include "disk/ccir/internal.h"
#include <stdlib.h>
#include <string.h>

static int same_row(const unsigned char *a, const unsigned char *b) {
    return !memcmp(a, b, AOTX_CO_FLAGS) &&
        ((aotx_ccir_u32(a + AOTX_CO_FLAGS) ^ aotx_ccir_u32(b + AOTX_CO_FLAGS)) & ~AOTX_COG_COLD) == 0 &&
        !memcmp(a + AOTX_CO_ID, b + AOTX_CO_ID, AOTX_CO_OFFSET - AOTX_CO_ID) &&
        !memcmp(a + AOTX_CO_BYTES, b + AOTX_CO_BYTES, AOTX_COG_OBJECT - AOTX_CO_BYTES);
}
int aotx_cold_catalog_open(const aotx_ccir_view *v, aotx_cold_catalog *c) {
    memset(c, 0, sizeof(*c)); c->fd = v->fd;
    const aotx_ccir_section *s = NULL;
    for (uint32_t i = 0; i < v->count; ++i) if (v->sections[i].type == AOTX_CCIR_COLD) {
        if (s || v->sections[i].schema != 1 || v->sections[i].flags != AOTX_CCIR_REQUIRED) return AOTX_CCIR_INVALID;
        s = v->sections + i;
    }
    if (!s) return 0;
    unsigned char h[AOTX_COLD_EXTENT_HEADER];
    if (s->bytes < sizeof(h)) return AOTX_CCIR_INVALID;
    int rc = aotx_ccir_pread(v->fd, h, sizeof(h), s->offset);
    if (rc) return rc;
    c->count = aotx_ccir_u32(h + 12); c->bytes = aotx_ccir_u64(h + 24);
    uint64_t table = (uint64_t)c->count * AOTX_COLD_EXTENT_ROW;
    if (memcmp(h, "AOTXCOLD", 8) || aotx_ccir_u32(h + 8) != 1 ||
        aotx_ccir_u32(h + 16) != AOTX_COLD_EXTENT_ROW || aotx_ccir_u32(h + 20) ||
        memcmp(h + 32, v->lineage, 16) || !aotx_ccir_zero(h + 48, 16) ||
        c->count > AOTX_COG_OBJECTS || table > s->bytes - sizeof(h) ||
        c->bytes != s->bytes - sizeof(h) - table) return AOTX_CCIR_INVALID;
    c->payload = s->offset + sizeof(h) + table;
    c->rows = malloc(table ? (size_t)table : 1);
    if (!c->rows) return AOTX_CCIR_IO;
    rc = aotx_ccir_pread(v->fd, c->rows, (size_t)table, s->offset + sizeof(h));
    /* Bind these exact catalog bytes to the selected committed section. */
    aotx_sha256 hash; aotx_sha256_init(&hash);
    aotx_sha256_update(&hash, h, sizeof(h));
    if (!rc) aotx_sha256_update(&hash, c->rows, (size_t)table);
    unsigned char block[AOTX_COLD_COPY], digest[32];
    for (uint64_t at = 0; !rc && at < c->bytes;) {
        size_t take = c->bytes - at < sizeof(block) ? (size_t)(c->bytes - at) : sizeof(block);
        rc = aotx_ccir_pread(v->fd, block, take, c->payload + at);
        if (!rc) aotx_sha256_update(&hash, block, take);
        at += take;
    }
    aotx_sha256_final(&hash, digest);
    if (!rc && memcmp(digest, s->digest, sizeof(digest))) rc = AOTX_CCIR_INVALID;
    uint64_t used = 0;
    for (uint32_t i = 0; !rc && i < c->count; ++i) {
        const unsigned char *r = c->rows + i * AOTX_COLD_EXTENT_ROW;
        uint64_t n = aotx_ccir_u64(r + AOTX_CO_BYTES);
        if (!(aotx_ccir_u32(r + AOTX_CO_FLAGS) & AOTX_COG_COLD) ||
            aotx_ccir_u64(r + AOTX_CO_OFFSET) != used || !n || n > AOTX_COG_PAYLOAD ||
            n > c->bytes - used || memcmp(r + AOTX_CO_LINEAGE, v->lineage, 16)) rc = AOTX_CCIR_INVALID;
        else used += n;
        for (uint32_t j = 0; !rc && j < i; ++j) {
            const unsigned char *p = c->rows + j * AOTX_COLD_EXTENT_ROW;
            if (!memcmp(p + AOTX_CO_ID, r + AOTX_CO_ID, 16) &&
                aotx_ccir_u64(p + AOTX_CO_VERSION) == aotx_ccir_u64(r + AOTX_CO_VERSION)) rc = AOTX_CCIR_INVALID;
        }
    }
    if (!rc && used != c->bytes) rc = AOTX_CCIR_INVALID;
    if (rc) aotx_cold_catalog_close(c);
    return rc;
}
void aotx_cold_catalog_close(aotx_cold_catalog *c) {
    free(c->rows); c->rows = NULL; c->count = 0;
}
int aotx_cold_catalog_find(const aotx_cold_catalog *c, const unsigned char *row) {
    for (uint32_t i = 0; i < c->count; ++i)
        if (same_row(c->rows + i * AOTX_COLD_EXTENT_ROW, row)) return (int)i;
    return -1;
}
int aotx_cold_catalog_read(const aotx_cold_catalog *c, uint32_t i,
    uint64_t at, size_t bytes, unsigned char *out) {
    if (i >= c->count) return AOTX_CCIR_INVALID;
    const unsigned char *r = c->rows + i * AOTX_COLD_EXTENT_ROW;
    uint64_t n = aotx_ccir_u64(r + AOTX_CO_BYTES), offset = aotx_ccir_u64(r + AOTX_CO_OFFSET);
    if (at > n || bytes > n - at) return AOTX_CCIR_INVALID;
    return aotx_ccir_pread(c->fd, out, bytes, c->payload + offset + at);
}
int aotx_cold_profile(const aotx_ccir_view *v) {
    const aotx_ccir_section *state = NULL, *cold = NULL;
    for (uint32_t i = 0; i < v->count; ++i) {
        if (v->sections[i].type == AOTX_CCIR_CHECKPOINT) state = v->sections + i;
        if (v->sections[i].type == AOTX_CCIR_COLD) cold = v->sections + i;
    }
    if (!state || (state->schema != 3 && !cold)) return 0;
    if (state->schema == 3 && !cold) return AOTX_CCIR_INVALID;
    aotx_cold_catalog c;
    int rc = aotx_cold_catalog_open(v, &c);
    if (rc) return rc;
    unsigned char h[AOTX_COG_HEADER], row[AOTX_COG_OBJECT];
    rc = state->bytes < sizeof(h) ? AOTX_CCIR_INVALID : aotx_ccir_pread(v->fd, h, sizeof(h), state->offset);
    uint32_t count = !rc ? aotx_ccir_u32(h + 20) : 0, found = 0;
    if (!rc && (count > AOTX_COG_OBJECTS || state->bytes < sizeof(h) + (uint64_t)count * sizeof(row))) rc = AOTX_CCIR_INVALID;
    for (uint32_t i = 0; !rc && i < count; ++i) {
        rc = aotx_ccir_pread(v->fd, row, sizeof(row), state->offset + sizeof(h) + (uint64_t)i * sizeof(row));
        if (!rc && (aotx_ccir_u32(row + AOTX_CO_FLAGS) & AOTX_COG_COLD)) {
            if (state->schema != 3 || aotx_ccir_u64(row + AOTX_CO_OFFSET) || aotx_cold_catalog_find(&c, row) < 0)
                rc = AOTX_CCIR_INVALID;
            ++found;
        }
    }
    if (!rc && found != c.count) rc = AOTX_CCIR_INVALID;
    aotx_cold_catalog_close(&c);
    return rc;
}

int aotx_cold_section_build(const aotx_ccir_view *v, const unsigned char *image,
    uint64_t bytes, aotx_ccir_input *in, FILE **temporary) {
    *temporary = NULL; memset(in, 0, sizeof(*in));
    if (bytes < AOTX_COG_HEADER) return AOTX_CCIR_INVALID;
    uint32_t count = aotx_ccir_u32(image + 20), cold_count = 0;
    if (count > AOTX_COG_OBJECTS || bytes < AOTX_COG_HEADER + (uint64_t)count * AOTX_COG_OBJECT) return AOTX_CCIR_INVALID;
    for (uint32_t i = 0; i < count; ++i)
        cold_count += !!(aotx_ccir_u32(image + AOTX_COG_HEADER + i * AOTX_COG_OBJECT + AOTX_CO_FLAGS) & AOTX_COG_COLD);
    aotx_cold_catalog c;
    int rc = aotx_cold_catalog_open(v, &c);
    if (rc) return rc;
    int same = cold_count == c.count;
    for (uint32_t i = 0; same && i < count; ++i) {
        const unsigned char *r = image + AOTX_COG_HEADER + i * AOTX_COG_OBJECT;
        if ((aotx_ccir_u32(r + AOTX_CO_FLAGS) & AOTX_COG_COLD) && aotx_cold_catalog_find(&c, r) < 0) same = 0;
    }
    for (uint32_t i = 0; same && i < v->count; ++i) if (v->sections[i].type == AOTX_CCIR_COLD) {
        in->section = v->sections[i]; in->source = AOTX_CCIR_REUSE;
        aotx_cold_catalog_close(&c); return 0;
    }
    const aotx_ccir_section *state = NULL;
    for (uint32_t i = 0; i < v->count; ++i) if (v->sections[i].type == AOTX_CCIR_CHECKPOINT) state = v->sections + i;
    unsigned char head[AOTX_COG_HEADER];
    if (!state || state->bytes < sizeof(head)) rc = AOTX_CCIR_INVALID;
    if (!rc) rc = aotx_ccir_pread(v->fd, head, sizeof(head), state->offset);
    if (!rc && cold_count) {
        unsigned char digest[32];
        rc = aotx_ccir_hash_fd(v->fd, state->offset, state->bytes, digest);
        if (!rc && memcmp(digest, state->digest, 32)) rc = AOTX_CCIR_INVALID;
    }
    uint32_t old_count = !rc ? aotx_ccir_u32(head + 20) : 0;
    uint64_t old_payload = !rc ? aotx_ccir_u64(head + 72) : 0;
    if (!rc && (old_count > AOTX_COG_OBJECTS || old_payload != sizeof(head) + (uint64_t)old_count * AOTX_COG_OBJECT ||
        old_payload > state->bytes)) rc = AOTX_CCIR_INVALID;
    unsigned char *old = !rc ? malloc((size_t)old_count * AOTX_COG_OBJECT + 1) : NULL;
    if (!rc && !old) rc = AOTX_CCIR_IO;
    if (!rc) rc = aotx_ccir_pread(v->fd, old, (size_t)old_count * AOTX_COG_OBJECT, state->offset + sizeof(head));
    FILE *file = !rc ? tmpfile() : NULL;
    if (!rc && !file) rc = AOTX_CCIR_IO;
    uint64_t payload = AOTX_COLD_EXTENT_HEADER + (uint64_t)cold_count * AOTX_COLD_EXTENT_ROW, used = 0;
    uint32_t at = 0;
    aotx_ccir_limits limits; aotx_ccir_default_limits(&limits);
    unsigned char buffer[AOTX_COLD_COPY];
    for (uint32_t i = 0; !rc && i < count; ++i) {
        const unsigned char *r = image + AOTX_COG_HEADER + i * AOTX_COG_OBJECT;
        if (!(aotx_ccir_u32(r + AOTX_CO_FLAGS) & AOTX_COG_COLD)) continue;
        uint64_t n = aotx_ccir_u64(r + AOTX_CO_BYTES), source = 0;
        if (!n || n > AOTX_COG_PAYLOAD || payload > limits.section_bytes ||
            used > limits.section_bytes - payload || n > limits.section_bytes - payload - used) { rc = AOTX_CCIR_LIMIT; break; }
        int found = aotx_cold_catalog_find(&c, r);
        if (found < 0) {
            int matched = 0;
            for (uint32_t j = 0; j < old_count; ++j) {
                const unsigned char *p = old + j * AOTX_COG_OBJECT;
                uint64_t offset = aotx_ccir_u64(p + AOTX_CO_OFFSET);
                if (same_row(p, r) && !(aotx_ccir_u32(p + AOTX_CO_FLAGS) & AOTX_COG_COLD) &&
                    offset <= state->bytes - old_payload && n <= state->bytes - old_payload - offset) {
                    source = state->offset + old_payload + offset; matched = 1; break;
                }
            }
            if (!matched) { rc = AOTX_CCIR_CHANGED; break; }
        }
        aotx_sha256 hash; aotx_sha256_init(&hash);
        for (uint64_t pos = 0; !rc && pos < n;) {
            size_t take = n - pos < sizeof(buffer) ? (size_t)(n - pos) : sizeof(buffer);
            rc = found >= 0 ? aotx_cold_catalog_read(&c, (uint32_t)found, pos, take, buffer) :
                aotx_ccir_pread(v->fd, buffer, take, source + pos);
            if (!rc) {
                aotx_sha256_update(&hash, buffer, take);
                rc = aotx_ccir_pwrite(fileno(file), buffer, take, payload + used + pos);
            }
            pos += take;
        }
        memcpy(buffer, r, AOTX_COG_OBJECT); aotx_ccir_put(buffer + AOTX_CO_OFFSET, used, 8);
        aotx_sha256_final(&hash, buffer + AOTX_COG_OBJECT);
        if (!rc && found >= 0 && memcmp(buffer + AOTX_COG_OBJECT,
            c.rows + (uint32_t)found * AOTX_COLD_EXTENT_ROW + AOTX_COG_OBJECT, 32)) rc = AOTX_CCIR_INVALID;
        if (!rc) rc = aotx_ccir_pwrite(fileno(file), buffer, AOTX_COLD_EXTENT_ROW,
            AOTX_COLD_EXTENT_HEADER + (uint64_t)at++ * AOTX_COLD_EXTENT_ROW);
        used += n;
    }
    if (!rc) {
        memset(buffer, 0, AOTX_COLD_EXTENT_HEADER); memcpy(buffer, "AOTXCOLD", 8);
        aotx_ccir_put(buffer + 8, 1, 4); aotx_ccir_put(buffer + 12, cold_count, 4);
        aotx_ccir_put(buffer + 16, AOTX_COLD_EXTENT_ROW, 4); aotx_ccir_put(buffer + 24, used, 8);
        memcpy(buffer + 32, image + 48, 16);
        rc = aotx_ccir_pwrite(fileno(file), buffer, AOTX_COLD_EXTENT_HEADER, 0);
    }
    free(old); aotx_cold_catalog_close(&c);
    if (rc) { if (file) fclose(file); return rc; }
    in->section.type = AOTX_CCIR_COLD; in->section.schema = 1; in->section.flags = AOTX_CCIR_REQUIRED;
    in->section.id[0] = AOTX_CCIR_COLD; in->section.alignment = 8; in->section.bytes = payload + used;
    for (uint32_t candidate = 0; candidate <= AOTX_CCIR_SECTIONS; ++candidate) {
        aotx_ccir_put(in->section.id + 1, candidate, 4);
        int used_id = 0;
        for (uint32_t i = 0; i < v->count; ++i)
            if (v->sections[i].type != AOTX_CCIR_COLD && !memcmp(in->section.id, v->sections[i].id, 16)) used_id = 1;
        if (!used_id) break;
    }
    for (uint32_t i = 0; i < v->count; ++i)
        if (v->sections[i].type == AOTX_CCIR_COLD) memcpy(in->section.id, v->sections[i].id, 16);
    in->source = AOTX_CCIR_FILE; in->fd = fileno(file); *temporary = file;
    return 0;
}
