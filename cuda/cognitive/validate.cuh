/* Purpose: Validate one object against a complete staged object batch.
 * Owns: Version, provenance, scope and payload admission rules.
 * Launch shape: One 64-thread block strides the configured object table.
 * Lifetime: Before a staged store replaces live state. */
#ifndef AOTX_COGNITIVE_VALIDATE_CUH
#define AOTX_COGNITIVE_VALIDATE_CUH
#include "cognitive/payload.cuh"

__device__ inline uint32_t aotx_cog_validate(const aotx_cognitive_store *s, uint32_t i) {
    const unsigned char *r = s->objects[i];
    uint16_t kind = aotx_cog_u16(r + AOTX_CO_KIND);
    uint32_t flags = aotx_cog_u32(r + AOTX_CO_FLAGS), scope = aotx_cog_u32(r + AOTX_CO_SCOPE);
    uint32_t source = aotx_cog_u32(r + AOTX_CO_SOURCE_KIND);
    uint64_t version = aotx_cog_u64(r + AOTX_CO_VERSION);
    uint64_t created = aotx_cog_u64(r + AOTX_CO_CREATED), updated = aotx_cog_u64(r + AOTX_CO_UPDATED);
    uint64_t offset = aotx_cog_u64(r + AOTX_CO_OFFSET), bytes = aotx_cog_u64(r + AOTX_CO_BYTES);
    if (aotx_cog_u16(r) != 1 || kind < AOTX_COG_EVENT || kind > AOTX_COG_REVIEW ||
        flags & ~(AOTX_COG_TOMBSTONE | AOTX_COG_PROTECTED | AOTX_COG_COLD) ||
        !aotx_cog_equal(r + AOTX_CO_LINEAGE, s->lineage) || aotx_cog_zero(r + AOTX_CO_ID, 16) ||
        aotx_cog_zero(r + AOTX_CO_OWNER, 16) || !version || !created || created > updated ||
        version > updated || updated > s->sequence || aotx_cog_u32(r + AOTX_CO_EVIDENCE) > 3 ||
        aotx_cog_u32(r + AOTX_CO_RETENTION) > 2 || !aotx_cog_scaled(aotx_cog_u32(r + AOTX_CO_IMPORTANCE)) ||
        !aotx_cog_zero(r + 196, 4) || !aotx_cog_zero(r + 240, 16) ||
        !aotx_cog_u64(r + AOTX_CO_POLICY) ||
        (aotx_cog_cold(r) ? (!s->tiered || offset || !bytes || bytes > AOTX_COG_PAYLOAD) :
            (offset > s->bytes || bytes > s->bytes - offset)) ||
        (!bytes && offset)) return AOTX_COG_FORMAT;
    if (scope > AOTX_COG_INSTANCE ||
        (scope == AOTX_COG_ROOM ? aotx_cog_zero(r + AOTX_CO_ROOM, 16)
                              : !aotx_cog_zero(r + AOTX_CO_ROOM, 16))) return AOTX_COG_SCOPE;
    if (source < AOTX_COG_AUTHORED || source > AOTX_COG_INFERRED ||
        (source == AOTX_COG_INFERRED && aotx_cog_zero(r + AOTX_CO_SOURCE, 16))) return AOTX_COG_SOURCE;
    int prior = -1;
    for (uint32_t j = 0; j < s->count; ++j) {
        if (j == i) continue;
        const unsigned char *other = s->objects[j];
        if (aotx_cog_u64(other + AOTX_CO_UPDATED) == updated) return AOTX_COG_SEQUENCE;
        uint64_t start = aotx_cog_u64(other + AOTX_CO_OFFSET), length = aotx_cog_u64(other + AOTX_CO_BYTES);
        if (!aotx_cog_cold(other) && (start > s->bytes || length > s->bytes - start)) return AOTX_COG_FORMAT;
        if (!aotx_cog_cold(r) && !aotx_cog_cold(other) && bytes && length &&
            offset < start + length && start < offset + bytes) return AOTX_COG_FORMAT;
        if (!aotx_cog_equal(r + AOTX_CO_ID, other + AOTX_CO_ID)) continue;
        uint64_t v = aotx_cog_u64(other + AOTX_CO_VERSION);
        if (v == version) return AOTX_COG_VERSION;
        if (s->pressure_percent ? v < version && (prior < 0 ||
            v > aotx_cog_u64(s->objects[prior] + AOTX_CO_VERSION)) : v == version - 1) prior = (int)j;
    }
    bool rooted = s->pressure_percent && updated <= s->root_sequence;
    if (created != updated && (version == 1 ||
        ((kind == AOTX_COG_EVENT || kind == AOTX_COG_MEDIA || kind == AOTX_COG_COMPONENT) &&
         !(flags & AOTX_COG_TOMBSTONE)))) return AOTX_COG_VERSION;
    if (s->pressure_percent && created == updated && version != 1 && version != updated)
        return AOTX_COG_VERSION;
    if (s->pressure_percent && !rooted && version != updated) return AOTX_COG_VERSION;
    if (s->pressure_percent ? created == updated : version == 1) {
        if (created != updated || prior >= 0 || flags & AOTX_COG_TOMBSTONE) return AOTX_COG_VERSION;
    } else if (prior < 0) {
        if (!rooted) return AOTX_COG_VERSION;
    } else {
        const unsigned char *p = s->objects[prior];
        if (aotx_cog_u64(p + AOTX_CO_CREATED) != created ||
            aotx_cog_u64(p + AOTX_CO_UPDATED) >= updated ||
            aotx_cog_u16(p + AOTX_CO_KIND) != kind ||
            !aotx_cog_equal(p + AOTX_CO_OWNER, r + AOTX_CO_OWNER) ||
            !aotx_cog_equal(p + AOTX_CO_SUBJECT, r + AOTX_CO_SUBJECT)) return AOTX_COG_VERSION;
        if (aotx_cog_u32(p + AOTX_CO_SOURCE_KIND) != source ||
            !aotx_cog_equal(p + AOTX_CO_SOURCE, r + AOTX_CO_SOURCE) ||
            aotx_cog_u64(p + AOTX_CO_SOURCE_VERSION) != aotx_cog_u64(r + AOTX_CO_SOURCE_VERSION))
            return AOTX_COG_SOURCE;
        if (!aotx_cog_scope(r, p)) return AOTX_COG_SCOPE;
        uint32_t oldflags = aotx_cog_u32(p + AOTX_CO_FLAGS);
        if (oldflags & AOTX_COG_TOMBSTONE ||
            ((oldflags & AOTX_COG_PROTECTED) && (!(flags & AOTX_COG_PROTECTED) || (flags & AOTX_COG_TOMBSTONE))))
            return AOTX_COG_DENIED;
    }
    uint32_t status = aotx_cog_reference(s, r, r + AOTX_CO_SOURCE, aotx_cog_u64(r + AOTX_CO_SOURCE_VERSION));
    if (status) return status;
    if (!aotx_cog_zero(r + AOTX_CO_SOURCE, 16)) {
        int j = aotx_cog_find(s, r + AOTX_CO_SOURCE, aotx_cog_u64(r + AOTX_CO_SOURCE_VERSION));
        const unsigned char *p = s->objects[j];
        if (aotx_cog_u32(p + AOTX_CO_SOURCE_KIND) == AOTX_COG_INFERRED && source != AOTX_COG_INFERRED)
            return AOTX_COG_SOURCE;
        if (kind == AOTX_COG_APPRAISAL &&
            !aotx_cog_equal(r + AOTX_CO_SUBJECT, p + AOTX_CO_SUBJECT)) return AOTX_COG_SOURCE;
    }
    status = aotx_cog_reference(s, r, r + AOTX_CO_SUPERSEDES, aotx_cog_u64(r + AOTX_CO_SUPER_VERSION));
    if (status) return status;
    if (!aotx_cog_zero(r + AOTX_CO_SUPERSEDES, 16)) {
        int j = aotx_cog_find(s, r + AOTX_CO_SUPERSEDES, aotx_cog_u64(r + AOTX_CO_SUPER_VERSION));
        if (aotx_cog_u16(s->objects[j] + AOTX_CO_KIND) != kind ||
            !aotx_cog_equal(s->objects[j] + AOTX_CO_SUBJECT, r + AOTX_CO_SUBJECT)) return AOTX_COG_SOURCE;
    }
    status = aotx_cog_reference(s, r, r + AOTX_CO_EMBEDDING,
                              aotx_cog_u64(r + AOTX_CO_EMBED_VERSION), AOTX_COG_COMPONENT);
    return status ? status : aotx_cog_payload(s, r);
}
#endif
