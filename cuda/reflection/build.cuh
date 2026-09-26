/* Purpose: Construct a complete validated review tail from exact source rows.
 * Owns: Shared temporary indices and caller-owned output bytes.
 * Launch shape: One 64-thread block processes the admitted source batch.
 * Lifetime: One immutable store cut before atomic publication. */
#ifndef AOTX_REFLECTION_BUILD_CUH
#define AOTX_REFLECTION_BUILD_CUH
#include "reflection/evidence.cuh"
#include "reflection/encode.cuh"

__device__ inline void aotx_review_build_block(const aotx_cognitive_store *s,
    const unsigned char *queries, const uint32_t *assessments, uint32_t count,
    unsigned char *tail, aotx_cognitive_result *result) {
    __shared__ uint32_t groups[AOTX_REVIEW_BATCH][AOTX_REVIEW_REFERENCES], status;
    uint32_t row = threadIdx.x;
    if (!row) status = !count || count > AOTX_REVIEW_BATCH ? AOTX_COG_FORMAT :
        2 * count > AOTX_COG_OBJECTS - s->count || count * AOTX_REVIEW_PAYLOAD > AOTX_COG_PAYLOAD - s->bytes ||
        2 * count > UINT64_MAX - s->sequence || s->tick == UINT64_MAX ? AOTX_COG_CAPACITY : 0;
    __syncthreads();
    if (!status && row < count) {
        uint32_t index = assessments[row];
        uint32_t error = aotx_review_evidence(s, queries + row * AOTX_RECALL_QUERY, index, groups[row]) ? 0 : AOTX_COG_SOURCE;
        if (!error) {
            unsigned char id[16];
            for (uint32_t k = 0; k < 2; ++k) {
                aotx_review_id(id, aotx_cog_u64(s->objects[index] + AOTX_CO_UPDATED), k);
                if (aotx_cog_latest(s, id) >= 0) error = AOTX_COG_REFERENCE;
            }
            for (uint32_t j = 0; j < row; ++j) if (assessments[j] == index) error = AOTX_COG_REFERENCE;
        }
        if (error) atomicCAS(&status, 0u, error);
    }
    __syncthreads();
    uint32_t bytes = AOTX_COG_HEADER + count * (2 * AOTX_COG_OBJECT + AOTX_REVIEW_PAYLOAD);
    if (!status) {
        for (uint32_t j = row; j < bytes; j += blockDim.x) tail[j] = 0;
        __syncthreads();
        if (!row) aotx_review_tail(s, tail, count);
        if (row < count) aotx_review_encode(s, groups[row], tail, row, count);
    }
    __syncthreads();
    if (!row) {
        *result = {}; result->status = status;
        if (!status) { result->applied = 2 * count; result->bytes = bytes; }
    }
}
#endif
