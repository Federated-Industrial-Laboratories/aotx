/* Purpose: Check exact current references and their transitive visibility.
 * Owns: Shared lookup rules; no state mutation or authentication.
 * Launch shape: Device functions called for each request in a batch.
 * Lifetime: One quiescent admitted store operation. */
#ifndef AOTX_COGNITIVE_LOOKUP_CUH
#define AOTX_COGNITIVE_LOOKUP_CUH
#include "cognitive/codec.cuh"
#include "appraisal/schema.cuh"

__device__ inline int aotx_cog_latest(const aotx_cognitive_store *live, const unsigned char *id) {
    if (aotx_cog_zero(id, 16)) return -1;
    int found = -1;
    uint64_t version = 0;
    for (uint32_t j = 0; j < live->count; ++j) {
        const unsigned char *r = live->objects[j];
        uint64_t v = aotx_cog_u64(r + AOTX_CO_VERSION);
        if (aotx_cog_equal(id, r + AOTX_CO_ID) && v > version) { found = (int)j; version = v; }
    }
    return found;
}
__device__ inline bool aotx_cog_visible(const unsigned char *r,
    const aotx_cognitive_query *q, bool evidence, uint64_t cut) {
    uint32_t scope = aotx_cog_u32(r + AOTX_CO_SCOPE);
    uint64_t expiry = aotx_cog_u64(r + AOTX_CO_EXPIRY);
    bool visible = scope == AOTX_COG_INSTANCE || (scope == AOTX_COG_PRIVATE
        ? aotx_cog_equal(q->principal, r + AOTX_CO_OWNER) : aotx_cog_equal(q->room, r + AOTX_CO_ROOM));
    return visible && (!evidence || aotx_cog_u32(r + AOTX_CO_EVIDENCE) != 3) && !(aotx_cog_u32(r + AOTX_CO_FLAGS) & AOTX_COG_TOMBSTONE) &&
           (!expiry || expiry > cut);
}
__device__ inline bool aotx_cog_superseded(const aotx_cognitive_store *live, const unsigned char *r) {
    for (uint32_t j = 0; j < live->count; ++j) {
        const unsigned char *p = live->objects[j];
        if (aotx_cog_equal(p + AOTX_CO_SUPERSEDES, r + AOTX_CO_ID) &&
            aotx_cog_u64(p + AOTX_CO_SUPER_VERSION) == aotx_cog_u64(r + AOTX_CO_VERSION)) return true;
    }
    return false;
}
__device__ inline void aotx_cog_mark(uint32_t *need, int index) {
    if (index >= 0) need[index / 32] |= 1u << (index % 32);
}
/* Historical sources remain readable only while their current scope permits access. */
__device__ inline uint32_t aotx_cog_dependencies_scratch(const aotx_cognitive_store *live,
    const aotx_cognitive_query *q, int first, bool evidence, uint64_t cut, uint32_t *need, uint32_t *done,
    bool resident = true) {
    bool unavailable = false;
    for (uint32_t j = 0; j < AOTX_COG_WORDS; ++j) need[j] = done[j] = 0;
    aotx_cog_mark(need, first);
    for (uint32_t pass = 0; pass < live->count; ++pass) {
        bool progress = false;
        for (uint32_t j = 0; j < live->count; ++j) {
            uint32_t bit = 1u << (j % 32);
            if (!(need[j / 32] & bit) || (done[j / 32] & bit)) continue;
            done[j / 32] |= bit; progress = true;
            const unsigned char *r = live->objects[j];
            int current = aotx_cog_latest(live, r + AOTX_CO_ID);
            if (current < 0 || !aotx_cog_visible(live->objects[current], q, evidence, cut)) return AOTX_COG_DENIED;
            aotx_cog_mark(need, aotx_cog_find(live, r + AOTX_CO_SOURCE, aotx_cog_u64(r + AOTX_CO_SOURCE_VERSION)));
            aotx_cog_mark(need, aotx_cog_find(live, r + AOTX_CO_SUPERSEDES, aotx_cog_u64(r + AOTX_CO_SUPER_VERSION)));
            aotx_cog_mark(need, aotx_cog_find(live, r + AOTX_CO_EMBEDDING, aotx_cog_u64(r + AOTX_CO_EMBED_VERSION)));
            if (aotx_cog_cold(r)) { unavailable = true; continue; }
            for (uint32_t k = 0; k < 2; ++k) {
                const unsigned char *extra = aotx_appraisal_reference(r,
                    live->payload + aotx_cog_u64(r + AOTX_CO_OFFSET), aotx_cog_u64(r + AOTX_CO_BYTES), k);
                if (extra) aotx_cog_mark(need, aotx_cog_find(live, extra, aotx_cog_u64(extra + 16)));
            }
            if (aotx_cog_u16(r + AOTX_CO_KIND) == AOTX_COG_SELECTION) {
                const unsigned char *p = live->payload + aotx_cog_u64(r + AOTX_CO_OFFSET);
                for (uint32_t k = 0; k < aotx_cog_u32(p + 4); ++k) {
                    const unsigned char *entry = p + 16 + k * 32;
                    int selected = aotx_cog_latest(live, entry);
                    if (selected < 0 || !aotx_cog_visible(live->objects[selected], q, evidence, cut)) return AOTX_COG_DENIED;
                    if (aotx_cog_u64(live->objects[selected] + AOTX_CO_VERSION) != aotx_cog_u64(entry + 16) ||
                        aotx_cog_superseded(live, live->objects[selected]))
                        return AOTX_COG_STALE;
                    aotx_cog_mark(need, selected);
                }
            }
        }
        if (!progress) break;
    }
    return resident && unavailable ? AOTX_COG_UNAVAILABLE : AOTX_COG_OK;
}

__device__ inline aotx_cognitive_match aotx_cog_resolve_scratch(
    const aotx_cognitive_store *live, const aotx_cognitive_query *q, bool evidence, uint64_t cut,
    uint32_t *need, uint32_t *done) {
    aotx_cognitive_match answer = {AOTX_COG_MISSING, UINT32_MAX, 0};
    for (uint32_t j = 0; j < live->count; ++j) {
        const unsigned char *r = live->objects[j];
        uint64_t v = aotx_cog_u64(r + AOTX_CO_VERSION);
        if (aotx_cog_equal(q->id, r + AOTX_CO_ID) && v > answer.version) {
            answer.index = j; answer.version = v;
        }
    }
    if (answer.index != UINT32_MAX) {
        const unsigned char *r = live->objects[answer.index];
        if (!aotx_cog_visible(r, q, evidence, cut)) answer = {AOTX_COG_DENIED, UINT32_MAX, 0};
        else if (q->version != answer.version || aotx_cog_superseded(live, r))
            answer = {AOTX_COG_STALE, UINT32_MAX, 0};
        else {
            answer.status = aotx_cog_dependencies_scratch(live, q, (int)answer.index, evidence, cut, need, done);
            if (answer.status) { answer.index = UINT32_MAX; answer.version = 0; }
        }
    }
    return answer;
}
__device__ inline uint32_t aotx_cog_dependencies(const aotx_cognitive_store *live,
    const aotx_cognitive_query *q, int first, bool evidence, uint64_t cut) {
    uint32_t need[AOTX_COG_WORDS], done[AOTX_COG_WORDS];
    return aotx_cog_dependencies_scratch(live, q, first, evidence, cut, need, done);
}
__device__ inline aotx_cognitive_match aotx_cog_resolve_one(
    const aotx_cognitive_store *live, const aotx_cognitive_query *q, bool evidence, uint64_t cut) {
    uint32_t need[AOTX_COG_WORDS], done[AOTX_COG_WORDS];
    return aotx_cog_resolve_scratch(live, q, evidence, cut, need, done);
}
__device__ inline aotx_cognitive_match aotx_cog_resolve_one(
    const aotx_cognitive_store *live, const aotx_cognitive_query *q, bool evidence = false) {
    return aotx_cog_resolve_one(live, q, evidence, live->sequence);
}
#endif
