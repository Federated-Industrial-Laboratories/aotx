/* Purpose: Recover recorded request batches and render exact saved selections.
 * Owns: Request/context output only; no vector search or state mutation.
 * Launch shape: One block for inventory; one 64-thread block per replayed request.
 * Lifetime: One quiescent admitted store. */
#include "cognitive/recall_context.cuh"

__global__ void aotx_recall_requests(const aotx_cognitive_store *live,
    unsigned char *requests, aotx_cognitive_result *result) {
    if (blockIdx.x) return;
    for (uint32_t j = threadIdx.x; j < AOTX_RECALL_REQUESTS; j += blockDim.x) requests[j] = 0;
    __syncthreads();
    if (threadIdx.x) return;
    uint32_t count = 0;
    uint64_t previous = 0;
    *result = {AOTX_COG_MISSING, 0, 0, live->sequence};
    for (uint32_t pass = 0; pass < live->count; ++pass) {
        int best = -1;
        uint64_t next = UINT64_MAX;
        for (uint32_t j = 0; j < live->count; ++j) {
            const unsigned char *r = live->objects[j];
            uint64_t sequence = aotx_cog_u64(r + AOTX_CO_UPDATED);
            if (sequence <= previous || aotx_cog_latest(live, r + AOTX_CO_ID) != (int)j || aotx_recall_source(live, r) < 0) continue;
            if (best < 0 || sequence < next) { best = (int)j; next = sequence; }
        }
        if (best < 0) break;
        if (count == AOTX_RECALL_BATCH) { result->status = AOTX_COG_CAPACITY; return; }
        const unsigned char *r = live->objects[best];
        int source = aotx_recall_source(live, r);
        const unsigned char *p = live->payload + aotx_cog_u64(live->objects[source] + AOTX_CO_OFFSET) + 16;
        if (!aotx_cog_equal(r + AOTX_CO_ID, p + 48)) { result->status = AOTX_COG_REFERENCE; return; }
        unsigned char *q = requests + AOTX_RECALL_HEADER + count * AOTX_RECALL_QUERY;
        for (uint32_t j = 0; j < AOTX_RECALL_QUERY; ++j) q[j] = p[j];
        ++count; previous = next;
    }
    if (!count) return;
    for (uint32_t j = 0; j < 8; ++j) requests[j] = "AOTXREQ1"[j];
    aotx_cog_put(requests + 8, count, 4); aotx_cog_put(requests + 12, 1, 4);
    for (uint32_t j = 0; j < 16; ++j) requests[16 + j] = live->lineage[j];
    aotx_cog_put(requests + 32, live->sequence, 8); aotx_cog_put(requests + 40, AOTX_RECALL_QUERY, 4);
    *result = {AOTX_COG_OK, count, AOTX_RECALL_HEADER + (uint64_t)count * AOTX_RECALL_QUERY, live->sequence};
}
__global__ void aotx_recall_replay(const aotx_cognitive_store *live,
    const unsigned char *requests, uint64_t bytes, aotx_recall_result *results, uint32_t count) {
    uint32_t n = blockIdx.x;
    if (n >= count || count > AOTX_RECALL_BATCH) return;
    aotx_recall_result *out = results + n;
    aotx_recall_clear(out);
    if (threadIdx.x) return;
    uint32_t status = aotx_recall_envelope(live, requests, bytes, count);
    const unsigned char *q = requests + AOTX_RECALL_HEADER + (uint64_t)n * AOTX_RECALL_QUERY;
    if (!status) status = aotx_recall_query_check(q);
    if (status) { out->status = status; return; }
    for (uint32_t j = 0; j < 16; ++j) { out->request_id[j] = q[j]; out->selection_id[j] = q[48 + j]; }
    int selected = aotx_cog_latest(live, q + 48);
    if (selected < 0) { out->status = AOTX_COG_MISSING; return; }
    const unsigned char *r = live->objects[selected];
    aotx_cognitive_match m = aotx_recall_match(live, q, q + 48, aotx_cog_u64(r + AOTX_CO_VERSION));
    if (m.status) { out->status = m.status; return; }
    int source = aotx_recall_source(live, r);
    if (source < 0) { out->status = AOTX_COG_REFERENCE; return; }
    const unsigned char *event = live->objects[source];
    const unsigned char *p = live->payload + aotx_cog_u64(event + AOTX_CO_OFFSET);
    uint64_t cut = aotx_cog_u64(p + 8);
    if (!aotx_cog_equal(p + 16, q, AOTX_RECALL_QUERY) || !aotx_cog_equal(event + AOTX_CO_ID, q) ||
        cut >= aotx_cog_u64(event + AOTX_CO_UPDATED) || !aotx_cog_equal(r + AOTX_CO_OWNER, q + 16) ||
        aotx_cog_u32(r + AOTX_CO_SCOPE) != aotx_cog_u32(q + 152) || !aotx_cog_equal(r + AOTX_CO_ROOM, q + 32)) {
        out->status = AOTX_COG_REFERENCE; return;
    }
    uint64_t selected_bytes = aotx_cog_u64(r + AOTX_CO_BYTES);
    const unsigned char *selection = live->payload + aotx_cog_u64(r + AOTX_CO_OFFSET);
    uint32_t selected_count = aotx_cog_u32(selection + 4);
    if (selected_bytes > AOTX_RECALL_SELECTION || selected_count > aotx_cog_u32(q + 132)) {
        out->status = AOTX_COG_CAPACITY; return;
    }
    out->cut = cut; out->count = selected_count;
    for (uint32_t j = 0; j < selected_bytes; ++j) out->selection[j] = selection[j];
    status = aotx_recall_render(live, q, out);
    if (status) aotx_recall_refuse(out, status);
}
