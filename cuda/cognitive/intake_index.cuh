/* Purpose: Index exact source substrings for constrained interpretation output.
 * Owns: Sparse suffix states, byte transitions and occurrence counts.
 * Launch shape: One block per source with parallel clearing and serial construction.
 * Lifetime: One leased interpretation batch; no persistent memory. */
#ifndef AOTX_COGNITIVE_INTAKE_INDEX_CUH
#define AOTX_COGNITIVE_INTAKE_INDEX_CUH
#include "cognitive/intake.cuh"
#define AOTX_INTAKE_NODES (2u * AOTX_RECALL_TEXT + 1u)
#define AOTX_INTAKE_EDGES (3u * AOTX_RECALL_TEXT + 1u)
#define AOTX_INTAKE_HASH (8u * AOTX_RECALL_TEXT)
typedef struct aotx_intake_node { uint32_t length, link, edge, ends, position; } aotx_intake_node;
typedef struct aotx_intake_edge { uint32_t key, target, next; } aotx_intake_edge;
typedef struct aotx_intake_index_row {
    uint32_t ready, nodes, edges, eligible, bytes;
    unsigned char source[AOTX_RECALL_TEXT];
    aotx_intake_node node[AOTX_INTAKE_NODES];
    aotx_intake_edge edge[AOTX_INTAKE_EDGES];
    uint32_t hash[AOTX_INTAKE_HASH], order[AOTX_INTAKE_NODES], sizes[AOTX_RECALL_TEXT + 1];
} aotx_intake_index_row;
extern __device__ aotx_intake_index_row aotx_intake_index_rows[AOTX_RECALL_BATCH];
__device__ __forceinline__ uint32_t aotx_intake_edge_at(const aotx_intake_index_row *s, uint32_t from, uint32_t byte) {
    uint32_t key = (from << 8) | byte;
    uint32_t mixed = key ^ (key >> 16); mixed *= 0x7feb352du;
    mixed ^= mixed >> 15; mixed *= 0x846ca68bu; mixed ^= mixed >> 16;
    uint32_t at = mixed & (AOTX_INTAKE_HASH - 1);
    for (uint32_t j = 0; j < AOTX_INTAKE_HASH; ++j) {
        uint32_t found = s->hash[at];
        if (!found || s->edge[found - 1].key == key) return at;
        at = (at + 1) & (AOTX_INTAKE_HASH - 1);
    }
    return UINT32_MAX;
}
__device__ __forceinline__ uint32_t aotx_intake_next(const aotx_intake_index_row *s, uint32_t from, uint32_t byte) {
    uint32_t at = aotx_intake_edge_at(s, from, byte);
    return at == UINT32_MAX || !s->hash[at] ? UINT32_MAX : s->edge[s->hash[at] - 1].target;
}
__global__ void aotx_intake_index(void);
#endif
