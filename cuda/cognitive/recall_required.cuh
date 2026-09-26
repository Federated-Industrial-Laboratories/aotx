/* Purpose: Validate complete required sets and recorded appraisal dependencies.
 * Owns: Exact selection checks; no similarity search or persistent mutation.
 * Launch shape: One checking thread per query or replay row.
 * Lifetime: One current store cut through publication. */
#ifndef AOTX_COGNITIVE_RECALL_REQUIRED_CUH
#define AOTX_COGNITIVE_RECALL_REQUIRED_CUH
#include "appraisal/recall.cuh"
#include "reflection/evidence.cuh"
#include "cognitive/recall_score.cuh"

__device__ inline bool aotx_recall_prefix_has(const aotx_recall_result *out, uint32_t index, uint32_t count) {
    for (uint32_t j = 0; j < count; ++j) if (out->index[j] == index) return true;
    return false;
}
__device__ inline uint32_t aotx_recall_next_required(const aotx_cognitive_store *s,
    const unsigned char *q, const aotx_recall_result *out, uint32_t count) {
    uint32_t best = UINT32_MAX;
    if (!(aotx_context_flags(q) & AOTX_RECALL_TASKS)) return best;
    for (uint32_t j = 0; j < s->count; ++j) {
        const unsigned char *r = s->objects[j];
        if (!aotx_recall_obligatory(s, q, r) || aotx_recall_prefix_has(out, j, count) ||
            aotx_recall_match(s, q, r + AOTX_CO_ID, aotx_cog_u64(r + AOTX_CO_VERSION)).status) continue;
        if (best == UINT32_MAX || aotx_recall_before(r, s->objects[best])) best = j;
    }
    return best;
}
__device__ inline uint32_t aotx_recall_expect(const aotx_recall_result *out, uint32_t index, uint32_t *at) {
    if (aotx_recall_prefix_has(out, index, *at)) return AOTX_COG_OK;
    if (*at == out->count || out->index[*at] != index) return AOTX_COG_REFERENCE;
    ++*at; return AOTX_COG_OK;
}
static __device__ __noinline__ uint32_t aotx_recall_required_set(const aotx_cognitive_store *s,
    const unsigned char *q, const aotx_recall_result *out, uint32_t *count) {
    *count = 0;
    for (uint32_t group = 0; group < 3; ++group) {
        if (group == 1) {
            for (;;) {
                uint32_t best = aotx_recall_next_required(s, q, out, *count);
                if (best == UINT32_MAX) break;
                uint32_t status = aotx_recall_expect(out, best, count);
                if (status) return status;
                if (aotx_review_kind(s, s->objects[best])) {
                    uint32_t group[AOTX_REVIEW_REFERENCES];
                    if (!aotx_review_group(s, q, best, group)) return AOTX_COG_REFERENCE;
                    for (uint32_t j = 0; j < AOTX_REVIEW_REFERENCES; ++j) {
                        status = aotx_recall_expect(out, group[j], count);
                        if (status) return status;
                    }
                }
            }
            continue;
        }
        uint32_t pins = aotx_cog_u32(q + (group ? 144 : 140));
        const unsigned char *refs = q + (group ? 4448 : 4256);
        for (uint32_t j = 0; j < pins; ++j) {
            const unsigned char *ref = refs + j * 24;
            aotx_cognitive_match m = aotx_recall_match(s, q, ref, aotx_cog_u64(ref + 16));
            if (m.status) return m.status;
            uint32_t status = aotx_recall_expect(out, m.index, count);
            if (status) return status;
        }
    }
    return AOTX_COG_OK;
}
static __device__ __noinline__ bool aotx_recall_pair(const aotx_cognitive_store *s,
    const aotx_recall_result *out, uint32_t index) {
    const unsigned char *r = s->objects[index];
    for (uint32_t j = 0; j < out->count; ++j) {
        const unsigned char *other = s->objects[out->index[j]];
        if (aotx_cog_u16(r + AOTX_CO_KIND) == AOTX_COG_APPRAISAL ?
            aotx_cog_u16(other + AOTX_CO_KIND) != AOTX_COG_APPRAISAL && aotx_recall_assesses(s, other, r) :
            aotx_recall_assesses(s, r, other)) return true;
    }
    return false;
}
static __device__ __noinline__ uint32_t aotx_recall_checked_index(const aotx_cognitive_store *s,
    const unsigned char *q, const unsigned char *entry, uint32_t *index) {
    aotx_cognitive_match m = aotx_recall_match(s, q, entry, aotx_cog_u64(entry + 16));
    if (m.status) return m.status;
    if (aotx_recall_text(s, s->objects[m.index]) <= 0) return AOTX_COG_FORMAT;
    if (!aotx_recall_applicable(s, q, s->objects[m.index])) return AOTX_COG_DENIED;
    *index = m.index; return AOTX_COG_OK;
}
static __device__ __noinline__ bool aotx_recall_automatic_selection(const aotx_cognitive_store *s,
    const unsigned char *q, const aotx_recall_result *out, uint32_t index) {
    uint32_t group[4];
    if (!aotx_appraisal_recall_group(s, q, index, group)) return false;
    for (uint32_t j = 0; j < 4; ++j)
        if (!aotx_recall_prefix_has(out, group[j], out->count)) return false;
    for (uint32_t j = 0; j < out->count; ++j) {
        const unsigned char *r = s->objects[out->index[j]];
        if (!aotx_recall_assesses(s, r, s->objects[index])) continue;
        double score = 0;
        if (!aotx_recall_score(s, q, r, &score) && score >=
            (double)aotx_cog_u32(q + AOTX_RECALL_EXTENSION + 36) / AOTX_COG_SCALE) return true;
    }
    return false;
}
static __device__ __noinline__ bool aotx_recall_automatic_parent(const aotx_cognitive_store *s,
    const unsigned char *q, const aotx_recall_result *out, uint32_t index) {
    for (uint32_t j = 0; j < out->count; ++j) {
        uint32_t assessment = out->index[j], group[4];
        if (!aotx_appraisal_recall_group(s, q, assessment, group)) continue;
        if (group[2] == index || group[3] == index) return true;
    }
    return false;
}
static __device__ __noinline__ bool aotx_recall_review_support(const aotx_cognitive_store *s,
    const unsigned char *q, const aotx_recall_result *out, uint32_t index, uint32_t required) {
    for (uint32_t j = 0; j < required; ++j) {
        uint32_t group[AOTX_REVIEW_REFERENCES];
        if (!aotx_review_group(s, q, out->index[j], group)) continue;
        bool present = false, complete = true;
        for (uint32_t k = 0; k < AOTX_REVIEW_REFERENCES; ++k) {
            present |= group[k] == index;
            complete &= aotx_recall_prefix_has(out, group[k], required);
        }
        if (present && complete) return true;
    }
    return false;
}
__device__ __forceinline__ uint32_t aotx_recall_selection_check(const aotx_cognitive_store *s,
    const unsigned char *q, aotx_recall_result *out, uint32_t *required) {
    const unsigned char *p = out->selection;
    if (out->count > AOTX_RECALL_LIMIT || out->count > aotx_cog_u32(q + 132) ||
        aotx_cog_u32(p) != 1 || aotx_cog_u32(p + 4) != out->count || !aotx_cog_zero(p + 8, 8) ||
        !aotx_cog_zero(p + 16 + out->count * 32, (AOTX_RECALL_LIMIT - out->count) * 32)) return AOTX_COG_FORMAT;
    for (uint32_t j = 0; j < out->count; ++j) {
        const unsigned char *entry = p + 16 + j * 32;
        if (aotx_cog_u32(entry + 24) != 1 || aotx_cog_u32(entry + 28)) return AOTX_COG_FORMAT;
        for (uint32_t k = 0; k < j; ++k)
            if (aotx_cog_equal(entry, p + 16 + k * 32)) return AOTX_COG_REFERENCE;
        uint32_t status = aotx_recall_checked_index(s, q, entry, out->index + j);
        if (status) return status;
    }
    uint32_t status = aotx_recall_required_set(s, q, out, required);
    if (status) return status;
    for (uint32_t j = 0; j < out->count; ++j) {
        const unsigned char *r = s->objects[out->index[j]];
        uint32_t automatic = aotx_appraisal_recall_kind(s, r);
        if (j < *required && automatic &&
            aotx_recall_review_support(s, q, out, out->index[j], *required)) continue;
        if (automatic >= 2) {
            if (j < *required || !aotx_recall_automatic_parent(s, q, out, out->index[j])) return AOTX_COG_REFERENCE;
            continue;
        }
        if (aotx_cog_u16(r + AOTX_CO_KIND) != AOTX_COG_APPRAISAL) continue;
        uint32_t intensity = aotx_appraisal_recall_intensity(s, q, out->index[j]);
        if (j < *required || !(aotx_context_flags(q) & AOTX_RECALL_APPRAISE) ||
            !aotx_cog_u32(q + AOTX_RECALL_EXTENSION + 40) || !intensity || intensity == AOTX_COG_UNKNOWN ||
            !aotx_recall_pair(s, out, out->index[j]) ||
            (automatic == 1 && !aotx_recall_automatic_selection(s, q, out, out->index[j]))) return AOTX_COG_REFERENCE;
    }
    return AOTX_COG_OK;
}
#endif
