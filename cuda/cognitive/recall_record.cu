/* Purpose: Encode one complete batch of recall requests and selected references.
 * Owns: A caller-provided tail image; publication remains the state module's work.
 * Launch shape: One 64-thread block for a bounded request batch.
 * Lifetime: Search output and requests remain immutable through record and apply. */
#include "cognitive/recall_context.cuh"

static __device__ void aotx_recall_object(unsigned char *r, const aotx_cognitive_store *s,
    const unsigned char *q, uint32_t kind, uint64_t seq, uint64_t offset, uint64_t bytes) {
    aotx_cog_put(r, 1, 2); aotx_cog_put(r + AOTX_CO_KIND, kind, 2);
    for (uint32_t j = 0; j < 16; ++j) {
        r[AOTX_CO_ID + j] = q[(kind == AOTX_COG_EVENT ? 0 : 48) + j];
        r[AOTX_CO_LINEAGE + j] = s->lineage[j]; r[AOTX_CO_OWNER + j] = q[16 + j]; r[AOTX_CO_ROOM + j] = q[32 + j];
        if (kind == AOTX_COG_SELECTION) r[AOTX_CO_SOURCE + j] = q[j];
    }
    aotx_cog_put(r + AOTX_CO_VERSION, s->pressure_percent ? seq : 1, 8);
    aotx_cog_put(r + AOTX_CO_CREATED, seq, 8); aotx_cog_put(r + AOTX_CO_UPDATED, seq, 8);
    aotx_cog_put(r + AOTX_CO_OFFSET, offset, 8); aotx_cog_put(r + AOTX_CO_BYTES, bytes, 8);
    aotx_cog_put(r + AOTX_CO_SCOPE, aotx_cog_u32(q + 152), 4);
    aotx_cog_put(r + AOTX_CO_SOURCE_KIND, kind == AOTX_COG_EVENT ? AOTX_COG_AUTHORED : AOTX_COG_INFERRED, 4);
    aotx_cog_put(r + AOTX_CO_IMPORTANCE, AOTX_COG_UNKNOWN, 4); aotx_cog_put(r + AOTX_CO_POLICY, 1, 8);
    if (kind == AOTX_COG_SELECTION) aotx_cog_put(r + AOTX_CO_SOURCE_VERSION, s->pressure_percent ? seq - 1 : 1, 8);
}
__global__ void aotx_recall_record(const aotx_cognitive_store *live,
    const unsigned char *requests, uint64_t bytes, const aotx_recall_result *results,
    uint32_t count, unsigned char *tail, aotx_cognitive_result *result) {
    if (blockIdx.x) return;
    __shared__ uint32_t status;
    __shared__ uint64_t payload, total;
    if (!threadIdx.x) {
        status = aotx_recall_envelope(live, requests, bytes, count); payload = 0; total = 0;
        *result = {status, 0, 0, live->sequence};
        if (!status && (count * 2 > AOTX_COG_OBJECTS - live->count || live->tick == UINT64_MAX ||
            live->sequence > UINT64_MAX - count * 2)) status = AOTX_COG_CAPACITY;
        uint32_t saved = 0;
        for (uint32_t j = 0; !status && j < live->count; ++j) {
            const unsigned char *r = live->objects[j];
            if (aotx_recall_source(live, r) >= 0 && aotx_cog_latest(live, r + AOTX_CO_ID) == (int)j) ++saved;
        }
        if (!status && saved + count > AOTX_RECALL_BATCH) status = AOTX_COG_CAPACITY;
        for (uint32_t i = 0; !status && i < count; ++i) {
            const unsigned char *q = requests + AOTX_RECALL_HEADER + i * AOTX_RECALL_QUERY;
            const aotx_recall_result *out = results + i;
            status = aotx_recall_query_check(q);
            if (!status) status = out->status;
            if (!status && (out->count > AOTX_RECALL_LIMIT || out->count > aotx_cog_u32(q + 132) ||
                out->cut != live->sequence || out->searches != 1 || !aotx_cog_equal(out->request_id, q) ||
                !aotx_cog_equal(out->selection_id, q + 48) || aotx_cog_u32(out->selection) != 1 ||
                aotx_cog_u32(out->selection + 4) != out->count)) status = AOTX_COG_FORMAT;
            if (!status && (aotx_cog_latest(live, q) >= 0 || aotx_cog_latest(live, q + 48) >= 0)) status = AOTX_COG_VERSION;
            for (uint32_t k = 0; !status && k < i; ++k) {
                const unsigned char *prior = requests + AOTX_RECALL_HEADER + k * AOTX_RECALL_QUERY;
                if (aotx_cog_equal(q, prior) || aotx_cog_equal(q, prior + 48) ||
                    aotx_cog_equal(q + 48, prior) || aotx_cog_equal(q + 48, prior + 48)) status = AOTX_COG_VERSION;
            }
            for (uint32_t k = 0; !status && k < out->count; ++k) {
                const unsigned char *entry = out->selection + 16 + k * 32;
                aotx_cognitive_match m = aotx_recall_match(live, q, entry, aotx_cog_u64(entry + 16), live->sequence + count * 2);
                status = m.status;
                if (!status && (aotx_recall_text(live, live->objects[m.index]) <= 0 ||
                    aotx_cog_u32(entry + 24) != 1 || aotx_cog_u32(entry + 28))) status = AOTX_COG_FORMAT;
            }
            payload += 16 + AOTX_RECALL_QUERY + 16 + (uint64_t)out->count * 32;
        }
        if (!status && payload > AOTX_COG_PAYLOAD - live->bytes) status = AOTX_COG_CAPACITY;
        if (!status) total = AOTX_COG_HEADER + count * 2 * AOTX_COG_OBJECT + payload;
        result->status = status;
    }
    __syncthreads();
    if (status) return;
    for (uint64_t j = threadIdx.x; j < total; j += blockDim.x) tail[j] = 0;
    __syncthreads();
    if (threadIdx.x) return;
    for (uint32_t j = 0; j < 8; ++j) tail[j] = "AOTXLOG1"[j];
    aotx_cog_put(tail + 8, 1, 4); aotx_cog_put(tail + 12, AOTX_COG_HEADER, 4);
    aotx_cog_put(tail + 16, AOTX_COG_OBJECT, 4); aotx_cog_put(tail + 20, count * 2, 4);
    aotx_cog_put(tail + 24, payload, 8); aotx_cog_put(tail + 32, live->sequence + 1, 8);
    aotx_cog_put(tail + 40, live->tick + 1, 8);
    for (uint32_t j = 0; j < 16; ++j) tail[48 + j] = live->lineage[j];
    aotx_cog_put(tail + 64, AOTX_COG_HEADER, 8);
    uint64_t base = AOTX_COG_HEADER + count * 2 * AOTX_COG_OBJECT;
    aotx_cog_put(tail + 72, base, 8); aotx_cog_put(tail + 80, total, 8); aotx_cog_put(tail + 88, 1, 4);
    aotx_cog_policy_write(tail, live);
    uint64_t offset = 0;
    for (uint32_t i = 0; i < count; ++i) {
        const unsigned char *q = requests + AOTX_RECALL_HEADER + i * AOTX_RECALL_QUERY;
        const aotx_recall_result *out = results + i;
        unsigned char *r = tail + AOTX_COG_HEADER + i * 2 * AOTX_COG_OBJECT;
        aotx_recall_object(r, live, q, AOTX_COG_EVENT, live->sequence + i * 2 + 1, offset, 16 + AOTX_RECALL_QUERY);
        for (uint32_t j = 0; j < 8; ++j) tail[base + offset + j] = "AOTXQUE1"[j];
        aotx_cog_put(tail + base + offset + 8, live->sequence, 8);
        for (uint32_t j = 0; j < AOTX_RECALL_QUERY; ++j) tail[base + offset + 16 + j] = q[j];
        offset += 16 + AOTX_RECALL_QUERY;
        uint32_t selected = 16 + out->count * 32;
        aotx_recall_object(r + AOTX_COG_OBJECT, live, q, AOTX_COG_SELECTION, live->sequence + i * 2 + 2, offset, selected);
        for (uint32_t j = 0; j < selected; ++j) tail[base + offset + j] = out->selection[j];
        offset += selected;
    }
    *result = {AOTX_COG_OK, count * 2, total, live->sequence};
}
