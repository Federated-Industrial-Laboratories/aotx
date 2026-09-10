/* Purpose: Build bounded exact-substring indexes for internal generation.
 * Owns: One sparse suffix automaton and eligible target mask per source.
 * Launch shape: One block per source; threads clear the sparse table together.
 * Lifetime: One admitted input through its complete internal model pass. */
#include "cognitive/intake_index.cuh"
#include "cognitive/intake_parse.cuh"
#include "sched/sched.cuh"

__device__ aotx_intake_index_row aotx_intake_index_rows[AOTX_RECALL_BATCH];
static __device__ bool aotx_intake_edge_set(aotx_intake_index_row *s, uint32_t from, uint32_t byte, uint32_t target) {
    uint32_t at = aotx_intake_edge_at(s, from, byte);
    if (at == UINT32_MAX) return false;
    if (s->hash[at]) { s->edge[s->hash[at] - 1].target = target; return true; }
    if (s->edges == AOTX_INTAKE_EDGES) return false;
    uint32_t index = s->edges++;
    s->edge[index] = {(from << 8) | byte, target, s->node[from].edge};
    s->node[from].edge = index; s->hash[at] = index + 1;
    return true;
}
static __device__ uint32_t aotx_intake_node_add(aotx_intake_index_row *s, uint32_t length, uint32_t position, uint32_t ends) {
    if (s->nodes == AOTX_INTAKE_NODES) return UINT32_MAX;
    uint32_t at = s->nodes++;
    s->node[at] = {length, UINT32_MAX, UINT32_MAX, ends, position}; return at;
}
static __device__ bool aotx_intake_index_build(aotx_intake_index_row *s, const unsigned char *source, uint32_t bytes) {
    s->nodes = s->edges = s->eligible = 0; s->bytes = bytes;
    for (uint32_t i = 0; i < bytes; ++i) s->source[i] = source[i];
    uint32_t last = aotx_intake_node_add(s, 0, 0, 0);
    for (uint32_t i = 0; i < bytes; ++i) {
        uint32_t byte = source[i], cur = aotx_intake_node_add(s, s->node[last].length + 1, i, 1), p = last;
        if (cur == UINT32_MAX) return false;
        while (p != UINT32_MAX && aotx_intake_next(s, p, byte) == UINT32_MAX) {
            if (!aotx_intake_edge_set(s, p, byte, cur)) return false;
            p = s->node[p].link;
        }
        if (p == UINT32_MAX) s->node[cur].link = 0;
        else {
            uint32_t q = aotx_intake_next(s, p, byte);
            if (s->node[p].length + 1 == s->node[q].length) s->node[cur].link = q;
            else {
                uint32_t clone = aotx_intake_node_add(s, s->node[p].length + 1, s->node[q].position, 0);
                if (clone == UINT32_MAX) return false;
                s->node[clone].link = s->node[q].link;
                for (uint32_t e = s->node[q].edge; e != UINT32_MAX; e = s->edge[e].next)
                    if (!aotx_intake_edge_set(s, clone, s->edge[e].key & 255, s->edge[e].target)) return false;
                while (p != UINT32_MAX && aotx_intake_next(s, p, byte) == q) {
                    if (!aotx_intake_edge_set(s, p, byte, clone)) return false;
                    p = s->node[p].link;
                }
                s->node[q].link = s->node[cur].link = clone;
            }
        }
        last = cur;
    }
    for (uint32_t i = 0; i <= bytes; ++i) s->sizes[i] = 0;
    for (uint32_t i = 0; i < s->nodes; ++i) ++s->sizes[s->node[i].length];
    for (uint32_t i = 1; i <= bytes; ++i) s->sizes[i] += s->sizes[i - 1];
    for (uint32_t i = 0; i < s->nodes; ++i) s->order[--s->sizes[s->node[i].length]] = i;
    for (uint32_t i = s->nodes; i > 1; --i) {
        uint32_t node = s->order[i - 1]; s->node[s->node[node].link].ends += s->node[node].ends;
    }
    return true;
}
__global__ void aotx_intake_index(void) {
    uint32_t i = blockIdx.x;
    if (aotx_sched.held || aotx_seam.replaying || aotx_live.phase != AOTX_INTAKE_RUN || i >= aotx_live.count ||
        aotx_intake.rows[i].state != 1) return;
    aotx_intake_index_row *s = aotx_intake_index_rows + i;
    for (uint32_t j = threadIdx.x; j < AOTX_INTAKE_HASH; j += blockDim.x) s->hash[j] = 0;
    __syncthreads();
    if (threadIdx.x) return;
    const unsigned char *q = aotx_live.requests + 64 + i * AOTX_RECALL_QUERY;
    s->ready = aotx_intake_index_build(s, q + 4640, aotx_cog_u32(q + 148));
    if (!s->ready) { aotx_intake.rows[i].status = AOTX_COG_CAPACITY; return; }
    for (uint32_t j = 1; j <= aotx_live.results[i].count; ++j)
        if (!aotx_intake_target(i, j)) s->eligible |= 1u << j;
}
