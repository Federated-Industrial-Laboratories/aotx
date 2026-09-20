/* Purpose: Validate exact source, configuration and relationship evidence references.
 * Owns: Versioned payload admission; no model output or persistent mutation.
 * Launch shape: One validation helper for each object in a batch.
 * Lifetime: Staged writes, checkpoint import and recorded recovery. */
#ifndef AOTX_APPRAISAL_SCHEMA_CUH
#define AOTX_APPRAISAL_SCHEMA_CUH
#include "appraisal/format.h"
#include "cognitive/memory_schema.cuh"
#include "cognitive/recall.h"
#include "cognitive/intake_schema.cuh"

__device__ inline bool aotx_appraisal_magic(const unsigned char *p, uint64_t bytes, const char *magic) {
    return bytes >= 8 && aotx_cog_equal(p, (const unsigned char *)magic, 8);
}
__device__ inline const unsigned char *aotx_appraisal_reference(const unsigned char *r,
    const unsigned char *p, uint64_t bytes, uint32_t which = 0) {
    if (aotx_appraisal_magic(p, bytes, "AOTXAPQ1") && bytes == AOTX_APPRAISAL_QUEUE_BYTES)
        return !which ? p + 16 : aotx_cog_zero(p + 128, 16) ? 0 : p + 128;
    if (which) return 0;
    if (aotx_appraisal_magic(p, bytes, "AOTXREL1") && bytes == AOTX_APPRAISAL_RELATION_BYTES) return p + 136;
    if (aotx_cog_u16(r + AOTX_CO_KIND) == AOTX_COG_APPRAISAL &&
        bytes == AOTX_APPRAISAL_ASSESS_BYTES && aotx_cog_u32(p) == 2) return p + 96;
    return 0;
}
__device__ inline const unsigned char *aotx_appraisal_source(const aotx_cognitive_store *s,
    const unsigned char *r, uint32_t *length) {
    int index = aotx_cog_find(s, r + AOTX_CO_SOURCE, aotx_cog_u64(r + AOTX_CO_SOURCE_VERSION));
    if (index < 0) return 0;
    const unsigned char *event = s->objects[index];
    if (aotx_cog_cold(event)) return 0;
    uint64_t bytes = aotx_cog_u64(event + AOTX_CO_BYTES), offset = aotx_cog_u64(event + AOTX_CO_OFFSET);
    if (aotx_cog_u16(event + AOTX_CO_KIND) != AOTX_COG_EVENT || bytes < 32 ||
        offset > s->bytes || bytes > s->bytes - offset ||
        !aotx_cog_equal(event + AOTX_CO_SUBJECT, r + AOTX_CO_SUBJECT) ||
        !aotx_cog_equal(event + AOTX_CO_OWNER, r + AOTX_CO_OWNER, 32) ||
        aotx_cog_u32(event + AOTX_CO_SCOPE) != aotx_cog_u32(r + AOTX_CO_SCOPE) ||
        aotx_cog_u32(event + AOTX_CO_SOURCE_KIND) == AOTX_COG_INFERRED) return 0;
    const unsigned char *p = s->payload + offset;
    *length = aotx_cog_u32(p + 12);
    if (!aotx_appraisal_magic(p, bytes, "AOTXMEM1") || aotx_cog_u32(p + 8) != 1 ||
        !*length || *length > AOTX_RECALL_TEXT || bytes != 32ull + *length ||
        !aotx_cog_zero(p + 16, 16) || !aotx_recall_utf8(p + 32, *length)) return 0;
    return p + 32;
}
__device__ inline bool aotx_appraisal_span(const unsigned char *source, uint32_t bytes,
    uint32_t start, uint32_t length) {
    if (!length) return !start;
    if (start > bytes || length > bytes - start || !aotx_recall_utf8(source + start, length)) return false;
    uint32_t found = 0;
    for (uint32_t i = 0; i <= bytes - length; ++i)
        found += aotx_cog_equal(source + i, source + start, length);
    return found == 1;
}
__device__ inline uint32_t aotx_appraisal_config_schema(const unsigned char *r,
    const unsigned char *p, uint64_t bytes) {
    if (bytes != AOTX_APPRAISAL_CONFIG_BYTES) return AOTX_COG_FORMAT;
    uint32_t flags = aotx_cog_u32(p + 12);
    if (bytes != AOTX_APPRAISAL_CONFIG_BYTES || aotx_cog_u32(p + 8) != 1 || flags & ~7u ||
        aotx_cog_u16(r + AOTX_CO_KIND) != AOTX_COG_POLICY ||
        aotx_cog_u32(r + AOTX_CO_SOURCE_KIND) != AOTX_COG_AUTHORED ||
        aotx_cog_u32(r + AOTX_CO_SCOPE) != AOTX_COG_INSTANCE ||
        !aotx_cog_zero(r + AOTX_CO_SOURCE, 40) ||
        !aotx_cog_u32(p + 16) || !aotx_cog_u32(p + 20) || aotx_cog_u32(p + 20) > AOTX_INTAKE_REPLY ||
        !aotx_cog_u32(p + 24) || !aotx_cog_u32(p + 36) || aotx_cog_u32(p + 36) > AOTX_RECALL_BATCH ||
        aotx_cog_u32(p + 28) > AOTX_COG_SCALE || aotx_cog_u32(p + 32) > AOTX_COG_SCALE ||
        aotx_cog_zero(p + 40, 32) || !aotx_cog_zero(p + 72, 24)) return AOTX_COG_FORMAT;
    return AOTX_COG_OK;
}
__device__ inline uint32_t aotx_appraisal_queue_schema(const aotx_cognitive_store *s,
    const unsigned char *r, const unsigned char *p, uint64_t bytes) {
    if (bytes != AOTX_APPRAISAL_QUEUE_BYTES || aotx_cog_u32(p + 8) != 1 ||
        aotx_cog_u32(p + 12) > AOTX_APPRAISAL_INTERRUPTED || aotx_cog_u32(p + 56) > AOTX_COG_UNAVAILABLE ||
        aotx_cog_u32(p + 60) || !aotx_cog_zero(p + 152, 8) || aotx_cog_u16(r + AOTX_CO_KIND) != AOTX_COG_POLICY ||
        aotx_cog_u32(r + AOTX_CO_SOURCE_KIND) != AOTX_COG_INFERRED ||
        aotx_cog_zero(r + AOTX_CO_SUBJECT, 16) || !aotx_cog_zero(r + AOTX_CO_SUPERSEDES, 24) ||
        !aotx_cog_zero(r + AOTX_CO_EMBEDDING, 24) || aotx_cog_u32(r + AOTX_CO_FLAGS) & AOTX_COG_PROTECTED)
        return AOTX_COG_FORMAT;
    uint32_t status = aotx_cog_reference(s, r, p + 16, aotx_cog_u64(p + 32), AOTX_COG_POLICY);
    int config = aotx_cog_find(s, p + 16, aotx_cog_u64(p + 32));
    if (status || config < 0) return status ? status : AOTX_COG_REFERENCE;
    const unsigned char *c = s->objects[config];
    const unsigned char *cp = s->payload + aotx_cog_u64(c + AOTX_CO_OFFSET);
    if (!aotx_appraisal_magic(cp, aotx_cog_u64(c + AOTX_CO_BYTES), "AOTXAPC1") ||
        aotx_cog_u64(c + AOTX_CO_BYTES) != AOTX_APPRAISAL_CONFIG_BYTES ||
        !aotx_cog_equal(cp + 40, p + 64, 32) || !(aotx_cog_u32(cp + 12) & AOTX_APPRAISAL_WRITE)) return AOTX_COG_SOURCE;
    uint32_t length;
    if (!aotx_appraisal_source(s, r, &length)) return AOTX_COG_SOURCE;
    status = aotx_cog_reference(s, r, p + 128, aotx_cog_u64(p + 144), AOTX_COG_CUE);
    if (status) return status;
    if (!aotx_cog_zero(p + 128, 16)) {
        int task = aotx_cog_find(s, p + 128, aotx_cog_u64(p + 144));
        const unsigned char *t = s->objects[task], *tp = s->payload + aotx_cog_u64(t + AOTX_CO_OFFSET);
        if (aotx_cog_cold(t)) return AOTX_COG_UNAVAILABLE;
        uint64_t n = aotx_cog_u64(t + AOTX_CO_BYTES);
        if (!aotx_cog_equal(p + 40, p + 128) || aotx_cog_u32(t + AOTX_CO_SOURCE_KIND) != AOTX_COG_AUTHORED ||
            !aotx_appraisal_magic(tp, n, "AOTXMEM1") || n < 32 ||
            aotx_cog_u32(tp + 8) != 1 || !aotx_cog_zero(tp + 16, 16) ||
            !aotx_cog_u32(tp + 12) || n != 32ull + aotx_cog_u32(tp + 12) ||
            !aotx_recall_utf8(tp + 32, aotx_cog_u32(tp + 12))) return AOTX_COG_SOURCE;
    }
    uint32_t state = aotx_cog_u32(p + 12);
    if (state == AOTX_APPRAISAL_PENDING && (aotx_cog_u32(p + 56) || !aotx_cog_zero(p + 96, 32))) return AOTX_COG_FORMAT;
    if (state == AOTX_APPRAISAL_COMPLETE && (aotx_cog_u32(p + 56) || aotx_cog_zero(p + 96, 32))) return AOTX_COG_FORMAT;
    if (state >= AOTX_APPRAISAL_REFUSED && !aotx_cog_u32(p + 56)) return AOTX_COG_FORMAT;
    return AOTX_COG_OK;
}
__device__ inline uint32_t aotx_appraisal_evidence_schema(const aotx_cognitive_store *s,
    const unsigned char *r, const unsigned char *p, uint64_t bytes, bool relation) {
    if (bytes != (relation ? AOTX_APPRAISAL_RELATION_BYTES : AOTX_APPRAISAL_ASSESS_BYTES) ||
        aotx_cog_u32(r + AOTX_CO_SOURCE_KIND) != AOTX_COG_INFERRED ||
        aotx_cog_zero(r + AOTX_CO_SUBJECT, 16)) return AOTX_COG_FORMAT;
    const unsigned char *ref = p + (relation ? 136 : 96);
    uint32_t status = aotx_cog_reference(s, r, ref, aotx_cog_u64(ref + 16), AOTX_COG_POLICY);
    int queue = aotx_cog_find(s, ref, aotx_cog_u64(ref + 16));
    if (status || queue < 0) return status ? status : AOTX_COG_REFERENCE;
    const unsigned char *qr = s->objects[queue], *q = s->payload + aotx_cog_u64(qr + AOTX_CO_OFFSET);
    if (!aotx_appraisal_magic(q, aotx_cog_u64(qr + AOTX_CO_BYTES), "AOTXAPQ1") ||
        aotx_cog_u64(qr + AOTX_CO_BYTES) != AOTX_APPRAISAL_QUEUE_BYTES ||
        aotx_cog_u32(q + 12) != AOTX_APPRAISAL_COMPLETE || aotx_cog_u32(q + 56) ||
        !aotx_cog_equal(qr + AOTX_CO_SOURCE, r + AOTX_CO_SOURCE, 40) ||
        !aotx_cog_equal(q + 64, p + (relation ? 72 : 32), 64)) return AOTX_COG_SOURCE;
    uint32_t length, at = relation ? 48 : 120;
    const unsigned char *source = aotx_appraisal_source(s, r, &length);
    if (!source || !aotx_appraisal_span(source, length, aotx_cog_u32(p + at), aotx_cog_u32(p + at + 4))) return AOTX_COG_SOURCE;
    if (!aotx_cog_zero(r + AOTX_CO_SUPERSEDES, 16)) {
        int prior = aotx_cog_find(s, r + AOTX_CO_SUPERSEDES, aotx_cog_u64(r + AOTX_CO_SUPER_VERSION));
        if (prior < 0) return AOTX_COG_REFERENCE;
        const unsigned char *old = s->objects[prior];
        if (aotx_cog_u32(old + AOTX_CO_SOURCE_KIND) != AOTX_COG_INFERRED ||
            aotx_cog_u32(old + AOTX_CO_FLAGS) & AOTX_COG_PROTECTED ||
            !aotx_cog_equal(old + AOTX_CO_OWNER, r + AOTX_CO_OWNER, 32) ||
            aotx_cog_u32(old + AOTX_CO_SCOPE) != aotx_cog_u32(r + AOTX_CO_SCOPE) ||
            !aotx_intake_current(s, r, old)) return AOTX_COG_DENIED;
    }
    if (relation) {
        if (aotx_cog_u16(r + AOTX_CO_KIND) != AOTX_COG_RELATIONSHIP || aotx_cog_u32(p + 8) != 1 ||
            aotx_cog_u32(p + 12) != 1 || !aotx_cog_zero(p + 160, 32) ||
            !aotx_cog_equal(p + 32, q + 40)) return AOTX_COG_FORMAT;
        for (uint32_t j = 16; j < 32; j += 4) if (!aotx_cog_scaled(aotx_cog_u32(p + j))) return AOTX_COG_FORMAT;
        for (uint32_t j = 56; j <= 64; j += 8)
            if (!aotx_appraisal_span(source, length, aotx_cog_u32(p + j), aotx_cog_u32(p + j + 4))) return AOTX_COG_SOURCE;
        if ((!aotx_cog_u32(p + 60) || aotx_cog_zero(p + 32, 16)) &&
            (aotx_cog_u32(p + 24) != AOTX_COG_UNKNOWN || aotx_cog_u32(p + 28) != AOTX_COG_UNKNOWN)) return AOTX_COG_SOURCE;
        if (aotx_cog_u32(p + 24) != AOTX_COG_UNKNOWN || aotx_cog_u32(p + 28) != AOTX_COG_UNKNOWN) {
            int task = aotx_cog_find(s, q + 128, aotx_cog_u64(q + 144));
            if (task < 0) return AOTX_COG_SOURCE;
            const unsigned char *t = s->objects[task], *tp = s->payload + aotx_cog_u64(t + AOTX_CO_OFFSET);
            if (aotx_cog_cold(t)) return AOTX_COG_UNAVAILABLE;
            uint32_t text = aotx_cog_u32(tp + 12), start = aotx_cog_u32(p + 56);
            if (text != aotx_cog_u32(p + 60) || start > length || text > length - start ||
                !aotx_cog_equal(source + start, tp + 32, text)) return AOTX_COG_SOURCE;
        }
        if (!aotx_cog_u32(p + 52)) {
            for (uint32_t j = 16; j < 32; j += 4) if (aotx_cog_u32(p + j) != AOTX_COG_UNKNOWN) return AOTX_COG_SOURCE;
            if (aotx_cog_u32(p + 60) || aotx_cog_u32(p + 68)) return AOTX_COG_SOURCE;
        }
    } else if (!aotx_cog_u32(p + 124)) {
        for (uint32_t j = 4; j <= 20; j += 4)
            if (aotx_cog_u32(p + j) != (j == 16 ? 0 : AOTX_COG_UNKNOWN)) return AOTX_COG_SOURCE;
    }
    return AOTX_COG_OK;
}
#endif
