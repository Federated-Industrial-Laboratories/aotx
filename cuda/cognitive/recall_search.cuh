/* Purpose: Select required, focused and semantic memory in bounded device batches.
 * Owns: Per-request scores and ordered selections; no persistent mutation.
 * Launch shape: One 64-thread block per request, with candidates spread across threads.
 * Lifetime: One quiescent admitted store and request batch. */
#ifndef AOTX_COGNITIVE_RECALL_SEARCH_CUH
#define AOTX_COGNITIVE_RECALL_SEARCH_CUH
#include "cognitive/recall_context.cuh"
#include "cognitive/recall_appraisal.cuh"
#include "cognitive/recall_score.cuh"

static __device__ bool aotx_recall_has(const aotx_recall_result *out, uint32_t index) {
    for (uint32_t j = 0; j < out->count; ++j) if (out->index[j] == index) return true;
    return false;
}
static __device__ uint32_t aotx_recall_add(const aotx_cognitive_store *s, const unsigned char *q,
    uint32_t index, uint32_t reason, uint32_t *used, aotx_recall_result *out) {
    if (aotx_recall_has(out, index)) return AOTX_COG_OK;
    if (!aotx_recall_applicable(s, q, s->objects[index])) return AOTX_COG_DENIED;
    if (aotx_cog_u16(s->objects[index] + AOTX_CO_KIND) == AOTX_COG_APPRAISAL && reason != AOTX_RECALL_ASSESSMENT)
        return AOTX_COG_FORMAT;
    uint32_t cap = aotx_cog_u32(q + 136);
    uint32_t after = aotx_recall_one(s, index, reason, 0, *used, cap);
    if (out->count == aotx_cog_u32(q + 132) || after > cap) return AOTX_COG_CAPACITY;
    const unsigned char *r = s->objects[index];
    unsigned char *entry = out->selection + 16 + out->count * 32;
    for (uint32_t j = 0; j < 16; ++j) entry[j] = r[AOTX_CO_ID + j];
    aotx_cog_put(entry + 16, aotx_cog_u64(r + AOTX_CO_VERSION), 8);
    aotx_cog_put(entry + 24, 1, 4);
    out->index[out->count] = index; out->reason[out->count++] = reason; *used = after;
    return AOTX_COG_OK;
}

__device__ __forceinline__ void aotx_recall_search_block(const aotx_cognitive_store *live,
    const unsigned char *requests, uint64_t bytes, aotx_recall_result *results,
    aotx_recall_scratch *scratch, uint32_t count) {
    uint32_t n = blockIdx.x;
    if (n >= count || count > AOTX_RECALL_BATCH) return;
    aotx_recall_result *out = results + n;
    aotx_recall_clear(out);
    double *scores = scratch[n].scores;
    uint32_t *states = scratch[n].states;
    const unsigned char *q = requests + AOTX_RECALL_HEADER + (uint64_t)n * AOTX_RECALL_QUERY;
    if (!threadIdx.x) {
        out->status = aotx_recall_envelope(live, requests, bytes, count);
        if (!out->status) out->status = aotx_recall_query_check(q);
        if (!out->status) {
            aotx_cognitive_query access = {};
            for (uint32_t j = 0; j < 16; ++j) { access.principal[j] = q[16 + j]; access.room[j] = q[32 + j]; }
            for (uint32_t j = 0; j < live->count; ++j) {
                const unsigned char *r = live->objects[j];
                uint32_t scope = aotx_cog_u32(r + AOTX_CO_SCOPE);
                if (!aotx_cog_cold(r) || !aotx_recall_kind(aotx_cog_u16(r + AOTX_CO_KIND)) ||
                    (scope != AOTX_COG_INSTANCE && scope != aotx_cog_u32(q + 152)) ||
                    !aotx_cog_visible(r, &access, true, live->sequence) ||
                    aotx_cog_latest(live, r + AOTX_CO_ID) != (int)j || aotx_cog_superseded(live, r)) continue;
                uint32_t status = aotx_cog_dependencies(live, &access, (int)j, true, live->sequence);
                if (status == AOTX_COG_UNAVAILABLE) { out->status = status; break; }
            }
        }
        if (!out->status) {
            out->cut = live->sequence; out->searches = 1;
            for (uint32_t j = 0; j < 16; ++j) { out->request_id[j] = q[j]; out->selection_id[j] = q[48 + j]; }
        }
    }
    __syncthreads();
    if (out->status) return;
    for (uint32_t j = threadIdx.x; j < live->count; j += blockDim.x) {
        const unsigned char *r = live->objects[j]; states[j] = AOTX_COG_MISSING; scores[j] = -2;
        if (aotx_cog_cold(r)) continue;
        if (!aotx_recall_kind(aotx_cog_u16(r + AOTX_CO_KIND)) || !aotx_recall_applicable(live, q, r)) continue;
        if (aotx_recall_match(live, q, r + AOTX_CO_ID, aotx_cog_u64(r + AOTX_CO_VERSION)).status) continue;
        if (aotx_cog_u16(r + AOTX_CO_KIND) == AOTX_COG_APPRAISAL) {
            if (aotx_context_flags(q) & AOTX_RECALL_APPRAISE) {
                uint32_t value = aotx_appraisal_recall_intensity(live, q, j);
                if (value && value != AOTX_COG_UNKNOWN) { states[j] = AOTX_RECALL_APPRAISAL_STATE; scores[j] = value; }
            }
            continue;
        }
        if (aotx_appraisal_recall_kind(live, r) >= 2) continue;
        if (aotx_recall_obligatory(live, q, r)) { states[j] = AOTX_RECALL_CUE_STATE; continue; }
        int length = aotx_recall_text(live, r);
        if (length < 0) { states[j] = AOTX_COG_FORMAT; continue; }
        if (length > 0) states[j] = aotx_recall_score(live, q, r, scores + j);
    }
    __syncthreads();
    for (uint32_t j = threadIdx.x; j < live->count; j += blockDim.x)
        aotx_recall_appraisal_score(live, q, j, scratch + n);
    __syncthreads();
    for (uint32_t j = threadIdx.x; j < live->count; j += blockDim.x)
        if (scratch[n].appraisals[j] == UINT32_MAX - 1) { states[j] = AOTX_COG_MISSING; scratch[n].appraisals[j] = UINT32_MAX; }
    __syncthreads();
    if (threadIdx.x) return;
    uint32_t compatible = 0, incompatible = 0;
    uint32_t used = aotx_recall_context_label(q, 0, 0, aotx_cog_u32(q + 136));
    if (used > aotx_cog_u32(q + 136)) { aotx_recall_refuse(out, AOTX_COG_CAPACITY); return; }
    for (uint32_t j = 0; j < live->count; ++j) {
        if (states[j] == AOTX_COG_OK) ++compatible;
        else if (states[j] == AOTX_COG_SOURCE) ++incompatible;
        else if (states[j] != AOTX_COG_MISSING && states[j] != AOTX_RECALL_CUE_STATE && states[j] != AOTX_RECALL_APPRAISAL_STATE) { aotx_recall_refuse(out, states[j]); return; }
    }
    if (incompatible && !compatible) { aotx_recall_refuse(out, AOTX_COG_SOURCE); return; }
    for (uint32_t group = 0; group < 3; ++group) {
        if (group == 1) {
            for (;;) {
                uint32_t best = aotx_recall_next_required(live, q, out, out->count);
                if (best == UINT32_MAX) break;
                uint32_t status = aotx_recall_add(live, q, best, AOTX_RECALL_OBLIGATION, &used, out);
                if (status) { aotx_recall_refuse(out, status); return; }
            }
            continue;
        }
        uint32_t pins = aotx_cog_u32(q + (group ? 144 : 140));
        const unsigned char *refs = q + (group ? 4448 : 4256);
        for (uint32_t j = 0; j < pins; ++j) {
            const unsigned char *ref = refs + j * 24;
            aotx_cognitive_match m = aotx_recall_match(live, q, ref, aotx_cog_u64(ref + 16));
            uint32_t status = m.status;
            if (!status && aotx_recall_text(live, live->objects[m.index]) <= 0) status = AOTX_COG_FORMAT;
            if (!status) status = aotx_recall_add(live, q, m.index, group ? AOTX_RECALL_FOCUS : AOTX_RECALL_REQUIRED, &used, out);
            if (status) { aotx_recall_refuse(out, status); return; }
        }
    }
    for (uint32_t pass = 0; pass < live->count && out->count < aotx_cog_u32(q + 132); ++pass) {
        uint32_t best = UINT32_MAX;
        for (uint32_t j = 0; j < live->count; ++j) {
            if (states[j] != AOTX_COG_OK) continue;
            uint32_t appraisal = scratch[n].appraisals[j];
            if (aotx_recall_has(out, j) && (appraisal == UINT32_MAX || aotx_recall_has(out, appraisal) ||
                aotx_appraisal_recall_kind(live, live->objects[appraisal]) != 1)) continue;
            if (best == UINT32_MAX || scores[j] > scores[best] ||
                (scores[j] == scores[best] && aotx_recall_before(live->objects[j], live->objects[best]))) best = j;
        }
        if (best == UINT32_MAX) break;
        states[best] = AOTX_COG_MISSING;
        uint32_t appraisal = scratch[n].appraisals[best];
        if (appraisal == UINT32_MAX) { aotx_recall_add(live, q, best, AOTX_RECALL_SEMANTIC, &used, out); continue; }
        uint32_t bundle[5] = {best, appraisal, 0, 0, 0}, total = 2;
        if (aotx_appraisal_recall_kind(live, live->objects[appraisal]) == 1) {
            if (!aotx_appraisal_recall_group(live, q, appraisal, bundle + 1)) continue;
            total = 5;
        }
        uint32_t cap = aotx_cog_u32(q + 136), after = used, needed = 0;
        for (uint32_t j = 0; j < total; ++j) {
            bool present = aotx_recall_has(out, bundle[j]);
            for (uint32_t k = 0; k < j; ++k) if (bundle[k] == bundle[j]) present = true;
            if (present) continue;
            ++needed;
            uint32_t reason = aotx_appraisal_recall_kind(live, live->objects[bundle[j]]) ||
                bundle[j] == appraisal ? AOTX_RECALL_ASSESSMENT : AOTX_RECALL_SIGNIFICANT;
            after = aotx_recall_one(live, bundle[j], reason, 0, after, cap);
        }
        if (needed > aotx_cog_u32(q + 132) - out->count || after > cap) continue;
        for (uint32_t j = 0; j < total; ++j) {
            uint32_t reason = aotx_appraisal_recall_kind(live, live->objects[bundle[j]]) ||
                bundle[j] == appraisal ? AOTX_RECALL_ASSESSMENT : AOTX_RECALL_SIGNIFICANT;
            aotx_recall_add(live, q, bundle[j], reason, &used, out);
        }
    }
    aotx_cog_put(out->selection, 1, 4); aotx_cog_put(out->selection + 4, out->count, 4);
    uint32_t status = aotx_recall_render(live, q, out);
    if (status) aotx_recall_refuse(out, status);
}

#endif
