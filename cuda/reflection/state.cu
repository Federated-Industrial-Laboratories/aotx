/* Purpose: Nominate current, supported task evidence in source revision order.
 * Owns: Cached eligibility and queries; no model or public output is produced.
 * Launch shape: Ordered metadata thread over the bounded device store.
 * Lifetime: One immutable store cut between publication and admission. */
#include "reflection/state.cuh"
#include "reflection/evidence.cuh"
#include "reflection/encode.cuh"
#include "cognitive/maintenance.cuh"

__device__ aotx_review_state aotx_review;
static __device__ unsigned char aotx_review_probe[AOTX_RECALL_QUERY];
/* Eligibility and request admission run on the ordered metadata thread. */
static __device__ uint32_t aotx_review_need[AOTX_COG_WORDS], aotx_review_done[AOTX_COG_WORDS];
static __device__ uint32_t aotx_review_probe_group[AOTX_REVIEW_REFERENCES];
__device__ bool aotx_review_query(const aotx_cognitive_store *s, uint32_t index, unsigned char *q) {
    const unsigned char *r = s->objects[index];
    if (aotx_appraisal_recall_kind(s, r) != 1 || aotx_cog_zero(r + AOTX_CO_SUBJECT, 16)) return false;
    const unsigned char *p = s->payload + aotx_cog_u64(r + AOTX_CO_OFFSET);
    int queue = aotx_cog_find(s, p + 96, aotx_cog_u64(p + 112));
    if (queue < 0 || aotx_cog_cold(s->objects[queue]) ||
        aotx_cog_u64(s->objects[queue] + AOTX_CO_BYTES) != AOTX_APPRAISAL_QUEUE_BYTES) return false;
    const unsigned char *qp = s->payload + aotx_cog_u64(s->objects[queue] + AOTX_CO_OFFSET);
    for (uint32_t j = 0; j < AOTX_RECALL_QUERY; ++j) q[j] = 0;
    for (uint32_t j = 0; j < 16; ++j) {
        q[16 + j] = r[AOTX_CO_OWNER + j]; q[32 + j] = r[AOTX_CO_ROOM + j];
        q[AOTX_RECALL_EXTENSION + 16 + j] = qp[40 + j];
        q[AOTX_RECALL_EXTENSION + 48 + j] = r[AOTX_CO_SUBJECT + j];
    }
    aotx_cog_put(q + 152, aotx_cog_u32(r + AOTX_CO_SCOPE), 4);
    for (uint32_t j = 0; j < 8; ++j) q[AOTX_RECALL_EXTENSION + j] = "AOTXCTX1"[j];
    aotx_cog_put(q + AOTX_RECALL_EXTENSION + 8, 1, 4);
    aotx_cog_put(q + AOTX_RECALL_EXTENSION + 12, AOTX_RECALL_TASKS | AOTX_RECALL_APPRAISE, 4);
    aotx_cog_put(q + AOTX_RECALL_EXTENSION + 32, 1, 4);
    aotx_cog_put(q + AOTX_RECALL_EXTENSION + 44, 1, 4);
    return aotx_review_evidence_scratch(s, q, index, aotx_review_probe_group,
        aotx_review_need, aotx_review_done);
}
__device__ uint32_t aotx_review_pending(void) {
    if (!aotx_review.enabled || aotx_review.active || !aotx_live.ready) return 0;
    const aotx_cognitive_store *s = &aotx_live_store;
    if (aotx_review.blocked_sequence == s->sequence && aotx_review.blocked_root == s->root_sequence &&
        aotx_review.blocked_bytes == s->bytes && aotx_review.blocked_count == s->count) return 0;
    if (aotx_review.observed == s->sequence && aotx_review.root == s->root_sequence &&
        aotx_review.bytes == s->bytes && aotx_review.observed_count == s->count &&
        aotx_review.maintenance == aotx_maintenance.passes) return aotx_review.pending;
    aotx_review.observed = s->sequence; aotx_review.root = s->root_sequence;
    aotx_review.bytes = s->bytes; aotx_review.observed_count = s->count;
    aotx_review.maintenance = aotx_maintenance.passes; aotx_review.pending = 0;
    for (uint32_t i = 0; i < s->count; ++i) {
        const unsigned char *r = s->objects[i];
        uint64_t revision = aotx_cog_u64(r + AOTX_CO_UPDATED);
        if (revision <= aotx_review.frontier || !aotx_review_query(s, i, aotx_review_probe)) continue;
        unsigned char id[16]; aotx_review_id(id, revision, 1);
        if (aotx_cog_latest(s, id) >= 0) continue;
        uint32_t at = 0;
        while (at < aotx_review.pending && aotx_cog_u64(s->objects[aotx_review.indices[at]] + AOTX_CO_UPDATED) < revision) ++at;
        if (at == AOTX_REVIEW_BATCH) continue;
        if (aotx_review.pending < AOTX_REVIEW_BATCH) ++aotx_review.pending;
        for (uint32_t j = aotx_review.pending - 1; j > at; --j) aotx_review.indices[j] = aotx_review.indices[j - 1];
        aotx_review.indices[at] = i;
    }
    return aotx_review.pending;
}
