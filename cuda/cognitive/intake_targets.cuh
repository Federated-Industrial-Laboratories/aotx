/* Purpose: Select and validate exact assertion targets apart from reply context.
 * Owns: Bounded target references and source-labelled target text.
 * Launch shape: One serial target allocation per source in a batch.
 * Lifetime: Pre-write selection through the recorded interpretation decision. */
#ifndef AOTX_COGNITIVE_INTAKE_TARGETS_CUH
#define AOTX_COGNITIVE_INTAKE_TARGETS_CUH
#include "cognitive/intake.cuh"
#include "cognitive/recall_labels.cuh"

__device__ inline bool aotx_intake_source_mode(uint32_t row) {
    return aotx_context_sources(aotx_live.requests + 64 + row * AOTX_RECALL_QUERY);
}
__device__ inline uint32_t aotx_intake_target_count(uint32_t row) {
    return aotx_intake_source_mode(row) ? aotx_intake.rows[row].target_count : aotx_live.results[row].count;
}
__device__ inline uint32_t aotx_intake_stride(void) {
    return aotx_live.intake_sources ? AOTX_LIVE_INTAKE_SOURCE_ROW : AOTX_LIVE_INTAKE_ROW;
}
__device__ inline const unsigned char *aotx_intake_target_ref(uint32_t row, uint32_t target) {
    return (aotx_intake_source_mode(row) ? aotx_intake.rows[row].targets :
        aotx_live.results[row].selection) + 16 + (target - 1) * 32;
}
__device__ inline uint32_t aotx_intake_target_index(uint32_t row, uint32_t target) {
    return aotx_intake_source_mode(row) ? aotx_intake.rows[row].target_index[target - 1] :
        aotx_live.results[row].index[target - 1];
}
__device__ inline uint32_t aotx_intake_target_check(uint32_t row, uint32_t index, const unsigned char *entry) {
    const unsigned char *q = aotx_live.requests + 64 + row * AOTX_RECALL_QUERY;
    if (index >= aotx_live_store.count) return AOTX_COG_REFERENCE;
    const unsigned char *old = aotx_live_store.objects[index];
    if (!aotx_cog_equal(entry, old + AOTX_CO_ID) ||
        aotx_cog_u64(entry + 16) != aotx_cog_u64(old + AOTX_CO_VERSION) ||
        aotx_cog_latest(&aotx_live_store, entry) != (int)index ||
        aotx_cog_superseded(&aotx_live_store, old)) return AOTX_COG_STALE;
    const unsigned char *p = aotx_live_store.payload + aotx_cog_u64(old + AOTX_CO_OFFSET);
    if (aotx_cog_cold(old)) return AOTX_COG_UNAVAILABLE;
    if (!aotx_intake_payload(p, aotx_cog_u64(old + AOTX_CO_BYTES)) ||
        aotx_cog_u32(p + 16) < AOTX_INTAKE_ASSERTION ||
        aotx_cog_u16(old + AOTX_CO_KIND) != AOTX_COG_ASSERTION ||
        aotx_cog_u32(old + AOTX_CO_SOURCE_KIND) != AOTX_COG_INFERRED ||
        aotx_cog_u32(old + AOTX_CO_FLAGS) & AOTX_COG_PROTECTED ||
        !aotx_cog_equal(old + AOTX_CO_OWNER, q + 16, 32) ||
        aotx_cog_u32(old + AOTX_CO_SCOPE) != aotx_cog_u32(q + 152)) return AOTX_COG_DENIED;
    return AOTX_COG_OK;
}
__device__ inline uint32_t aotx_intake_target(uint32_t row, uint32_t target) {
    uint32_t count = aotx_intake_target_count(row);
    if (!target || target > count) return AOTX_COG_REFERENCE;
    return aotx_intake_target_check(row, aotx_intake_target_index(row, target), aotx_intake_target_ref(row, target));
}
__device__ inline uint32_t aotx_intake_target_text(uint32_t index, uint32_t number,
    unsigned char *out, uint32_t at, uint32_t cap) {
    at = aotx_recall_number(out, at, cap, number);
    at = aotx_recall_word(out, at, cap, ": ");
    uint32_t source = aotx_recall_source_index(&aotx_live_store, index);
    const unsigned char *event = aotx_live_store.objects[source];
    const unsigned char *actor = aotx_recall_source_actor(&aotx_live_store, source);
    at = aotx_recall_word(out, at, cap, "[source_ref=");
    at = aotx_recall_hex(out, at, cap, event + AOTX_CO_ID);
    at = aotx_recall_word(out, at, cap, "@");
    at = aotx_recall_number(out, at, cap, aotx_cog_u64(event + AOTX_CO_VERSION));
    at = aotx_recall_word(out, at, cap, " source_actor=");
    at = actor ? aotx_recall_hex(out, at, cap, actor) : aotx_recall_word(out, at, cap, "unknown");
    at = aotx_recall_word(out, at, cap, "]\n");
    const unsigned char *r = aotx_live_store.objects[index];
    const unsigned char *p = aotx_live_store.payload + aotx_cog_u64(r + AOTX_CO_OFFSET);
    at = aotx_recall_run(out, at, cap, p + AOTX_INTAKE_PAYLOAD, aotx_cog_u32(p + 12));
    return aotx_recall_word(out, at, cap, "\n");
}
/* Every admitted interpretation names an exact source event. */
__device__ inline bool aotx_intake_target_source(const unsigned char *r, uint32_t group) {
    const unsigned char *source = aotx_live_store.objects[group];
    return aotx_cog_u16(source + AOTX_CO_KIND) == AOTX_COG_EVENT &&
        aotx_cog_equal(r + AOTX_CO_SOURCE, source + AOTX_CO_ID) &&
        aotx_cog_u64(r + AOTX_CO_SOURCE_VERSION) == aotx_cog_u64(source + AOTX_CO_VERSION);
}
/* Source order follows the exact pre-write selection; each source takes one turn. */
static __device__ __noinline__ uint32_t aotx_intake_targets_prepare(uint32_t row, uint32_t capacity) {
    aotx_intake_row *out = aotx_intake.rows + row;
    out->target_count = out->target_bytes = 0; out->target_capacity = capacity;
    for (uint32_t j = 0; j < AOTX_INTAKE_TARGETS; ++j) out->targets[j] = 0;
    if (!aotx_intake_source_mode(row)) return AOTX_COG_OK;
    aotx_cog_put(out->targets, 1, 4);
    if (!capacity) return AOTX_COG_OK;
    uint32_t *groups = out->target_groups, *previous = out->target_previous, count = 0;
    const aotx_recall_result *selected = aotx_live.results + row;
    const unsigned char *q = aotx_live.requests + 64 + row * AOTX_RECALL_QUERY;
    for (uint32_t j = 0; j < selected->count; ++j) {
        uint32_t source = aotx_recall_source_index(&aotx_live_store, selected->index[j]);
        bool seen = false;
        for (uint32_t k = 0; k < count; ++k) if (groups[k] == source) seen = true;
        if (!seen) { groups[count] = source; previous[count++] = UINT32_MAX; }
    }
    for (uint32_t pass = 0; pass < AOTX_RECALL_LIMIT && out->target_count < AOTX_RECALL_LIMIT; ++pass) {
        bool added = false;
        for (uint32_t group = 0; group < count && out->target_count < AOTX_RECALL_LIMIT; ++group) {
            uint32_t best = UINT32_MAX;
            for (uint32_t j = 0; j < aotx_live_store.count; ++j) {
                const unsigned char *r = aotx_live_store.objects[j];
                if (aotx_cog_u16(r + AOTX_CO_KIND) != AOTX_COG_ASSERTION ||
                    aotx_cog_u64(r + AOTX_CO_BYTES) < AOTX_INTAKE_PAYLOAD ||
                    !aotx_intake_target_source(r, groups[group]) ||
                    (previous[group] != UINT32_MAX && !aotx_recall_before(aotx_live_store.objects[previous[group]], r))) continue;
                unsigned char ref[24];
                for (uint32_t k = 0; k < 16; ++k) ref[k] = r[AOTX_CO_ID + k];
                aotx_cog_put(ref + 16, aotx_cog_u64(r + AOTX_CO_VERSION), 8);
                if (aotx_intake_target_check(row, j, ref) ||
                    aotx_recall_match_scratch(&aotx_live_store, q, ref, aotx_cog_u64(ref + 16),
                        aotx_live_store.sequence, out->target_need, out->target_done).status ||
                    aotx_recall_text(&aotx_live_store, r) <= 0) continue;
                if (aotx_intake_target_text(j, out->target_count + 1, 0, out->target_bytes,
                    capacity) > capacity) continue;
                if (best == UINT32_MAX || aotx_recall_before(r, aotx_live_store.objects[best])) best = j;
            }
            if (best == UINT32_MAX) continue;
            previous[group] = best; added = true;
            const unsigned char *r = aotx_live_store.objects[best];
            unsigned char *ref = out->targets + 16 + out->target_count * 32;
            for (uint32_t k = 0; k < 16; ++k) ref[k] = r[AOTX_CO_ID + k];
            aotx_cog_put(ref + 16, aotx_cog_u64(r + AOTX_CO_VERSION), 8); aotx_cog_put(ref + 24, 1, 4);
            out->target_index[out->target_count++] = best;
            out->target_bytes = aotx_intake_target_text(best, out->target_count, 0,
                out->target_bytes, capacity);
        }
        if (!added) break;
    }
    aotx_cog_put(out->targets, 1, 4); aotx_cog_put(out->targets + 4, out->target_count, 4);
    return AOTX_COG_OK;
}
#endif
