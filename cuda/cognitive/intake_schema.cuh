/* Purpose: Validate inferred text against its exact immutable source span.
 * Owns: Payload and correction admission; no model interpretation.
 * Launch shape: One validation call for each staged object.
 * Lifetime: Live updates, snapshots and recovered memory. */
#ifndef AOTX_COGNITIVE_INTAKE_SCHEMA_CUH
#define AOTX_COGNITIVE_INTAKE_SCHEMA_CUH
#include "cognitive/intake.h"
#include "cognitive/memory_schema.cuh"

__device__ inline uint32_t aotx_intake_kind(uint32_t kind) {
    return kind == AOTX_INTAKE_PARTICIPANT ? AOTX_COG_IDENTITY :
        kind == AOTX_INTAKE_TASK ? AOTX_COG_CUE : AOTX_COG_ASSERTION;
}
__device__ inline bool aotx_intake_payload(const unsigned char *p, uint64_t bytes) {
    return bytes >= AOTX_INTAKE_PAYLOAD && aotx_cog_equal(p, (const unsigned char *)"AOTXMEM3", 8);
}
/* Later history does not change admission at an earlier sequence. */
__device__ inline bool aotx_intake_current(const aotx_cognitive_store *s,
    const unsigned char *r, const unsigned char *old) {
    uint64_t cut = aotx_cog_u64(r + AOTX_CO_UPDATED);
    uint64_t version = aotx_cog_u64(old + AOTX_CO_VERSION);
    for (uint32_t i = 0; i < s->count; ++i) {
        const unsigned char *p = s->objects[i];
        if (aotx_cog_u64(p + AOTX_CO_UPDATED) >= cut) continue;
        if (aotx_cog_equal(p + AOTX_CO_ID, old + AOTX_CO_ID) &&
            aotx_cog_u64(p + AOTX_CO_VERSION) > version) return false;
        if (!aotx_cog_equal(p + AOTX_CO_ID, r + AOTX_CO_ID) &&
            aotx_cog_equal(p + AOTX_CO_SUPERSEDES, old + AOTX_CO_ID) &&
            aotx_cog_u64(p + AOTX_CO_SUPER_VERSION) == version) return false;
    }
    return true;
}
__device__ inline uint32_t aotx_intake_schema(const aotx_cognitive_store *s,
    const unsigned char *r, const unsigned char *p, uint64_t bytes) {
    if (bytes < AOTX_INTAKE_PAYLOAD || aotx_cog_u32(p + 8) != 3 ||
        aotx_cog_zero(p + 24, 32) || aotx_cog_zero(p + 56, 32) || !aotx_cog_zero(p + 88, 8) ||
        aotx_cog_u32(r + AOTX_CO_SOURCE_KIND) != AOTX_COG_INFERRED ||
        !aotx_cog_zero(r + AOTX_CO_SUBJECT, 16)) return AOTX_COG_FORMAT;
    uint32_t kind = aotx_cog_u32(p + 16), start = aotx_cog_u32(p + 20), length = aotx_cog_u32(p + 12);
    if (!kind || kind > AOTX_INTAKE_CORRECTION || aotx_cog_u16(r + AOTX_CO_KIND) != aotx_intake_kind(kind) ||
        !length || length > 2048 || bytes != AOTX_INTAKE_PAYLOAD + length ||
        !aotx_recall_utf8(p + AOTX_INTAKE_PAYLOAD, length)) return AOTX_COG_FORMAT;
    int source = aotx_cog_find(s, r + AOTX_CO_SOURCE, aotx_cog_u64(r + AOTX_CO_SOURCE_VERSION));
    if (source < 0) return AOTX_COG_SOURCE;
    const unsigned char *event = s->objects[source];
    uint64_t offset = aotx_cog_u64(event + AOTX_CO_OFFSET), size = aotx_cog_u64(event + AOTX_CO_BYTES);
    if (aotx_cog_u16(event + AOTX_CO_KIND) != AOTX_COG_EVENT || size < 32 ||
        offset > s->bytes || size > s->bytes - offset) return AOTX_COG_SOURCE;
    const unsigned char *text = s->payload + offset;
    uint32_t extent = aotx_cog_u32(text + 12);
    if (!aotx_cog_equal(text, (const unsigned char *)"AOTXMEM1", 8) || aotx_cog_u32(text + 8) != 1 ||
        size != 32ull + extent || !aotx_cog_zero(text + 16, 16) || !aotx_recall_utf8(text + 32, extent) ||
        start > extent || length > extent - start ||
        !aotx_cog_equal(text + 32 + start, p + AOTX_INTAKE_PAYLOAD, length)) return AOTX_COG_SOURCE;
    bool correction = !aotx_cog_zero(r + AOTX_CO_SUPERSEDES, 16);
    if (correction != (kind == AOTX_INTAKE_CORRECTION)) return AOTX_COG_REFERENCE;
    if (correction) {
        int prior = aotx_cog_find(s, r + AOTX_CO_SUPERSEDES, aotx_cog_u64(r + AOTX_CO_SUPER_VERSION));
        if (prior < 0) return AOTX_COG_REFERENCE;
        const unsigned char *old = s->objects[prior];
        uint64_t at = aotx_cog_u64(old + AOTX_CO_OFFSET), n = aotx_cog_u64(old + AOTX_CO_BYTES);
        if (at > s->bytes || n > s->bytes - at || !aotx_intake_payload(s->payload + at, n) ||
            aotx_cog_u32(s->payload + at + 16) < AOTX_INTAKE_ASSERTION ||
            aotx_cog_u32(old + AOTX_CO_SOURCE_KIND) != AOTX_COG_INFERRED ||
            aotx_cog_u32(old + AOTX_CO_FLAGS) & AOTX_COG_PROTECTED ||
            !aotx_cog_equal(old + AOTX_CO_OWNER, r + AOTX_CO_OWNER, 32) ||
            aotx_cog_u32(old + AOTX_CO_SCOPE) != aotx_cog_u32(r + AOTX_CO_SCOPE)) return AOTX_COG_DENIED;
        if (!aotx_intake_current(s, r, old)) return AOTX_COG_STALE;
    }
    return AOTX_COG_OK;
}
#endif
