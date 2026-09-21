/* Purpose: Build bounded exact-substring indexes for internal generation.
 * Owns: One sparse suffix automaton and eligible target mask per source.
 * Launch shape: One block per source; threads clear the sparse table together.
 * Lifetime: One admitted input through its complete internal model pass. */
#include "cognitive/intake_index.cuh"
#include "cognitive/intake_parse.cuh"
#include "appraisal/parse.cuh"
#include "sched/sched.cuh"

__device__ aotx_intake_index_row aotx_intake_index_rows[AOTX_RECALL_BATCH];
__device__ aotx_intake_index_row aotx_intake_filtered_rows[AOTX_RECALL_BATCH];
__device__ uint32_t aotx_intake_consumed[AOTX_RECALL_BATCH][AOTX_INTAKE_CONSUMED];
__device__ uint32_t aotx_intake_completion[AOTX_RECALL_BATCH][AOTX_INTAKE_NODES];
__device__ __noinline__ uint32_t aotx_intake_span_capacity(uint32_t row) {
    const aotx_intake_row *r = aotx_intake.rows + row;
    const unsigned char *q = aotx_live.requests + 64 + row * AOTX_RECALL_QUERY;
    uint32_t count = r->phase == 1 ? r->source_count : r->first_count;
    uint32_t bytes = count ? 1 + count * (r->phase == 1 ? 17 : 10) : 2;
    for (uint32_t j = 0; j < count; ++j)
        bytes += aotx_source_escaped(q + 4640 + r->statements[j].start, r->statements[j].length);
    return bytes > AOTX_INTAKE_REPLY ? AOTX_COG_CAPACITY : AOTX_COG_OK;
}
__device__ __noinline__ uint32_t aotx_intake_spans(uint32_t row) {
    aotx_intake_row *r = aotx_intake.rows + row;
    const unsigned char *q = aotx_live.requests + 64 + row * AOTX_RECALL_QUERY;
    if (!aotx_source_split(q + 4640, aotx_cog_u32(q + 148), aotx_intake_index_rows[row].sizes,
        r->statements, &r->source_count, true)) return AOTX_COG_CAPACITY;
    return aotx_intake_span_capacity(row);
}
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
    s->node[at] = {length, UINT32_MAX, UINT32_MAX, ends, position, position}; return at;
}
static __device__ bool aotx_intake_index_build(aotx_intake_index_row *s, const unsigned char *source, uint32_t bytes) {
    if (bytes > AOTX_INTAKE_PROJECTION) return false;
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
        uint32_t node = s->order[i - 1], parent = s->node[node].link;
        s->node[parent].ends += s->node[node].ends;
        s->node[parent].last = max(s->node[parent].last, s->node[node].last);
    }
    return true;
}
static __device__ bool aotx_intake_complete_index(uint32_t row, aotx_intake_index_row *s,
    aotx_intake_index_row *filtered) {
    /* The filtered sort order is unused after its occurrence counts are complete. */
    uint32_t *inside = filtered->order, *complete = aotx_intake_completion[row];
    for (uint32_t i = 0; i < s->nodes; ++i) inside[i] = 0;
    const aotx_intake_row *r = aotx_intake.rows + row;
    for (uint32_t j = 0; j < r->first_count; ++j) {
        const aotx_intake_span *span = r->statements + j;
        uint32_t node = 0;
        for (uint32_t k = 0; k < span->length; ++k) {
            node = aotx_intake_next(s, node, s->source[span->start + k]);
            if (node == UINT32_MAX) return false;
            inside[node] = max(inside[node], k + 1);
        }
    }
    for (uint32_t i = s->nodes; i > 1; --i) {
        uint32_t node = s->order[i - 1], parent = s->node[node].link;
        inside[parent] = max(inside[parent], min(s->node[parent].length, inside[node]));
    }
    for (uint32_t i = s->nodes; i; --i) {
        uint32_t node = s->order[i - 1];
        uint32_t limit = s->node[node].ends == 1 ? inside[node] : 0;
        for (uint32_t edge = s->node[node].edge; edge != UINT32_MAX; edge = s->edge[edge].next) {
            uint32_t next = complete[s->edge[edge].target];
            if (next) limit = max(limit, min(s->node[node].length, next - 1));
        }
        complete[node] = limit;
    }
    return true;
}
__global__ void aotx_intake_index(void) {
    uint32_t i = blockIdx.x;
    if (aotx_sched.held || aotx_seam.replaying || aotx_live.phase != AOTX_INTAKE_RUN || i >= aotx_live.count ||
        aotx_intake.rows[i].state != 1) return;
    aotx_intake_index_row *s = aotx_intake_index_rows + i;
    aotx_intake_index_row *filtered = aotx_intake_filtered_rows + i;
    for (uint32_t j = threadIdx.x; j < AOTX_INTAKE_HASH; j += blockDim.x) s->hash[j] = filtered->hash[j] = 0;
    for (uint32_t j = threadIdx.x; j < AOTX_INTAKE_CONSUMED; j += blockDim.x) aotx_intake_consumed[i][j] = 0;
    __syncthreads();
    if (threadIdx.x) return;
    const unsigned char *q = aotx_live.requests + 64 + i * AOTX_RECALL_QUERY;
    s->mode = aotx_appraisal.active || !aotx_intake_source_mode(i) ? 0 : aotx_intake.rows[i].phase;
    s->row = i; filtered->ready = 0;
    s->ready = aotx_intake_index_build(s, q + 4640, aotx_cog_u32(q + 148));
    if (s->ready && s->mode == 2) {
        const aotx_intake_row *r = aotx_intake.rows + i;
        uint32_t bytes = 0;
        for (uint32_t j = 0; j < r->first_count; ++j) {
            const aotx_intake_span *span = r->statements + j;
            if (j) filtered->source[bytes++] = 0;
            if (span->length > AOTX_INTAKE_PROJECTION - bytes) { s->ready = 0; break; }
            for (uint32_t k = 0; k < span->length; ++k) filtered->source[bytes++] = q[4640 + span->start + k];
        }
        if (s->ready) s->ready = filtered->ready = aotx_intake_index_build(filtered, filtered->source, bytes);
        if (s->ready) s->ready = aotx_intake_complete_index(i, s, filtered);
    }
    if (!s->ready) { aotx_intake.rows[i].status = AOTX_COG_CAPACITY; return; }
    if (aotx_appraisal.active) {
        uint32_t bytes = 0;
        if (aotx_appraisal_task_source(i, &bytes)) s->eligible |= 1u;
    }
    uint32_t count = aotx_appraisal.active ? aotx_appraisal.rows[i].prior_count : aotx_intake_target_count(i);
    for (uint32_t j = 1; j <= count && j <= AOTX_RECALL_LIMIT; ++j)
        if (!(aotx_appraisal.active ? aotx_appraisal_target(i, j) : aotx_intake_target(i, j))) s->eligible |= 1u << j;
}
