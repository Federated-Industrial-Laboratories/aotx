/* Purpose: Index exact source substrings for constrained interpretation output.
 * Owns: Sparse suffix states, byte transitions and occurrence counts.
 * Launch shape: One block per source with parallel clearing and serial construction.
 * Lifetime: One leased interpretation batch; no persistent memory. */
#ifndef AOTX_COGNITIVE_INTAKE_INDEX_CUH
#define AOTX_COGNITIVE_INTAKE_INDEX_CUH
#include "cognitive/intake.cuh"
#define AOTX_INTAKE_PROJECTION (AOTX_RECALL_TEXT + AOTX_INTAKE_ITEMS - 1u)
#define AOTX_INTAKE_NODES (2u * AOTX_INTAKE_PROJECTION + 1u)
#define AOTX_INTAKE_EDGES (3u * AOTX_INTAKE_PROJECTION + 1u)
#define AOTX_INTAKE_CONSUMED 8192u
#define AOTX_INTAKE_HASH (16u * AOTX_RECALL_TEXT)
typedef struct aotx_intake_node { uint32_t length, link, edge, ends, position, last; } aotx_intake_node;
typedef struct aotx_intake_edge { uint32_t key, target, next; } aotx_intake_edge;
typedef struct aotx_intake_index_row {
    uint32_t ready, nodes, edges, eligible, bytes, mode, row;
    unsigned char source[AOTX_INTAKE_PROJECTION];
    aotx_intake_node node[AOTX_INTAKE_NODES];
    aotx_intake_edge edge[AOTX_INTAKE_EDGES];
    uint32_t hash[AOTX_INTAKE_HASH], order[AOTX_INTAKE_NODES], sizes[AOTX_INTAKE_PROJECTION + 1];
} aotx_intake_index_row;
extern __device__ aotx_intake_index_row aotx_intake_index_rows[AOTX_RECALL_BATCH];
extern __device__ aotx_intake_index_row aotx_intake_filtered_rows[AOTX_RECALL_BATCH];
extern __device__ uint32_t aotx_intake_completion[AOTX_RECALL_BATCH][AOTX_INTAKE_NODES];
extern __device__ uint32_t aotx_intake_consumed[AOTX_RECALL_BATCH][AOTX_INTAKE_CONSUMED];
__device__ bool aotx_intake_consume(uint32_t row, const aotx_intake_item *item);
__device__ __forceinline__ uint32_t aotx_intake_consumed_at(uint32_t row, uint32_t node, uint32_t length) {
    uint32_t key = ((node << 12) | length) + 1, mixed = key * 0x9e3779b9u;
    uint32_t at = (mixed ^ (mixed >> 16)) & (AOTX_INTAKE_CONSUMED - 1);
    for (uint32_t i = 0; i < AOTX_INTAKE_CONSUMED; ++i) {
        uint32_t stored = aotx_intake_consumed[row][at] & 0x0fffffffu;
        if (!stored || stored == key) return at;
        at = (at + 1) & (AOTX_INTAKE_CONSUMED - 1);
    }
    return UINT32_MAX;
}
__device__ __forceinline__ bool aotx_intake_excluded(uint32_t row, uint32_t node, uint32_t length,
    uint32_t kind, bool dead) {
    uint32_t at = aotx_intake_consumed_at(row, node, length);
    uint32_t bit = 28 + (kind - 1) * 2 + (dead ? 1 : 0);
    return at == UINT32_MAX || (aotx_intake_consumed[row][at] & (1u << bit));
}
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
__device__ __forceinline__ bool aotx_intake_optional(const aotx_intake_index_row *s,
    uint32_t node, uint32_t length, uint32_t kind) {
    if (kind < 1 || kind > 2 || length > aotx_intake_completion[s->row][node] ||
        aotx_intake_excluded(s->row, node, length, kind, true)) return false;
    if (length) return true;
    for (uint32_t edge = s->node[0].edge; edge != UINT32_MAX; edge = s->edge[edge].next) {
        uint32_t byte = s->edge[edge].key & 255, child = s->edge[edge].target;
        if ((byte >= 32 && byte < 127) || byte == 9 || byte == 10 || (byte >= 194 && byte <= 244))
            if (aotx_intake_completion[s->row][child] && !aotx_intake_excluded(s->row, child, 1, kind, true)) return true;
    }
    return false;
}
__global__ void aotx_intake_index(void);
#endif
