/* Purpose: Read and write explicit little-endian device state fields.
 * Owns: Bounded byte access helpers; no allocation or persistent state.
 * Launch shape: Called by threads that hold validated buffer ranges.
 * Lifetime: One state operation. */
#ifndef AOTX_COGNITIVE_CODEC_CUH
#define AOTX_COGNITIVE_CODEC_CUH
#include "cognitive/state.cuh"

__device__ inline uint16_t aotx_cog_u16(const unsigned char *p) {
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}
__device__ inline uint32_t aotx_cog_u32(const unsigned char *p) {
    return (uint32_t)aotx_cog_u16(p) | ((uint32_t)aotx_cog_u16(p + 2) << 16);
}
__device__ inline uint64_t aotx_cog_u64(const unsigned char *p) {
    return (uint64_t)aotx_cog_u32(p) | ((uint64_t)aotx_cog_u32(p + 4) << 32);
}
__device__ inline void aotx_cog_put(unsigned char *p, uint64_t value, unsigned bytes) {
    for (unsigned i = 0; i < bytes; ++i) p[i] = (unsigned char)(value >> (8 * i));
}
__device__ inline bool aotx_cog_zero(const unsigned char *p, unsigned bytes) {
    for (unsigned i = 0; i < bytes; ++i) if (p[i]) return false;
    return true;
}
__device__ inline bool aotx_cog_equal(const unsigned char *a, const unsigned char *b,
                                     unsigned bytes = 16) {
    for (unsigned i = 0; i < bytes; ++i) if (a[i] != b[i]) return false;
    return true;
}
__device__ inline bool aotx_cog_scaled(uint32_t x) {
    return x <= AOTX_COG_SCALE || x == AOTX_COG_UNKNOWN;
}
__device__ inline uint32_t aotx_cog_header(const unsigned char *p, uint64_t bytes, bool tail) {
    const char *magic = tail ? "AOTXLOG1" : "AOTXOBJ1";
    if (bytes < AOTX_COG_HEADER) return AOTX_COG_FORMAT;
    for (unsigned i = 0; i < 8; ++i)
        if (p[i] != (unsigned char)magic[i]) return AOTX_COG_FORMAT;
    uint32_t n = aotx_cog_u32(p + 20);
    uint64_t payload = aotx_cog_u64(p + 24);
    if (n > AOTX_COG_OBJECTS || payload > AOTX_COG_PAYLOAD) return AOTX_COG_CAPACITY;
    uint64_t offset = AOTX_COG_HEADER + (uint64_t)n * AOTX_COG_OBJECT;
    if (aotx_cog_u32(p + 8) != 1 || aotx_cog_u32(p + 12) != AOTX_COG_HEADER ||
        aotx_cog_u32(p + 16) != AOTX_COG_OBJECT || aotx_cog_u64(p + 64) != AOTX_COG_HEADER ||
        aotx_cog_u64(p + 72) != offset || bytes != offset + payload ||
        aotx_cog_u64(p + 80) != bytes || aotx_cog_u32(p + 88) != 1 ||
        !aotx_cog_zero(p + 92, 36) || aotx_cog_zero(p + 48, 16)) return AOTX_COG_FORMAT;
    if (tail && (!n || !aotx_cog_u64(p + 32) ||
                 aotx_cog_u64(p + 32) > UINT64_MAX - (n - 1))) return AOTX_COG_SEQUENCE;
    return AOTX_COG_OK;
}
__device__ inline int aotx_cog_find(const aotx_cognitive_store *s,
                                   const unsigned char *id, uint64_t version) {
    for (uint32_t j = 0; j < s->count; ++j)
        if (aotx_cog_equal(id, s->objects[j] + AOTX_CO_ID) &&
            aotx_cog_u64(s->objects[j] + AOTX_CO_VERSION) == version) return (int)j;
    return -1;
}
/* A derived object cannot make its source visible to more principals. */
__device__ inline bool aotx_cog_scope(const unsigned char *child, const unsigned char *source) {
    uint32_t scope = aotx_cog_u32(source + AOTX_CO_SCOPE);
    if (scope == AOTX_COG_INSTANCE) return true;
    if (aotx_cog_u32(child + AOTX_CO_SCOPE) != scope) return false;
    return scope == AOTX_COG_PRIVATE ? aotx_cog_equal(child + AOTX_CO_OWNER, source + AOTX_CO_OWNER)
                                    : aotx_cog_equal(child + AOTX_CO_ROOM, source + AOTX_CO_ROOM);
}
__device__ inline uint32_t aotx_cog_reference(const aotx_cognitive_store *s,
    const unsigned char *r, const unsigned char *id, uint64_t version, uint16_t kind = 0) {
    if (aotx_cog_zero(id, 16)) return version ? AOTX_COG_REFERENCE : AOTX_COG_OK;
    int j = aotx_cog_find(s, id, version);
    if (j < 0) return AOTX_COG_REFERENCE;
    const unsigned char *source = s->objects[j];
    if (aotx_cog_u64(source + AOTX_CO_UPDATED) >= aotx_cog_u64(r + AOTX_CO_UPDATED) ||
        (aotx_cog_u32(source + AOTX_CO_FLAGS) & AOTX_COG_TOMBSTONE) ||
        (kind && aotx_cog_u16(source + AOTX_CO_KIND) != kind)) return AOTX_COG_REFERENCE;
    return aotx_cog_scope(r, source) ? AOTX_COG_OK : AOTX_COG_SCOPE;
}
#endif
