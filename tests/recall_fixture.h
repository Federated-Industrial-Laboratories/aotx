/* Purpose: Build independent prepared text and vector fixtures for device checks.
 * Owns: Test images, requests and host result comparisons.
 * Launch shape: Distinct batches at N=1 and N=64.
 * Lifetime: One test process without model weights. */
#ifndef AOTX_RECALL_FIXTURE_H
#define AOTX_RECALL_FIXTURE_H
#include "cognitive_fixture.h"
#include "cognitive/recall.cuh"
#include <algorithm>
#include <cmath>
#include <limits>

static void aotx_float_put(unsigned char *p, float value) {
    uint32_t bits; memcpy(&bits, &value, 4); aotx_put(p, bits, 4);
}
static aotx_row aotx_memory_row(unsigned owner, unsigned kind, uint64_t id, uint64_t seq, unsigned scope = 0) {
    auto r = aotx_object(owner, kind, id, seq);
    aotx_put(r.data() + AOTX_CO_SCOPE, scope, 4); memset(r.data() + AOTX_CO_ROOM, 0, 16);
    if (scope == AOTX_COG_ROOM) aotx_id(r.data() + AOTX_CO_ROOM, 2000 + owner);
    return r;
}
static aotx_bytes aotx_memory_text(const std::string &text) {
    aotx_bytes p(32 + text.size(), 0); memcpy(p.data(), "AOTXMEM1", 8);
    aotx_put(p.data() + 8, 1, 4); aotx_put(p.data() + 12, text.size(), 4);
    memcpy(p.data() + 32, text.data(), text.size()); return p;
}
static aotx_bytes aotx_memory_vector(float x, float y, float z) {
    aotx_bytes p(140, 0); memcpy(p.data(), "AOTXVEC1", 8);
    aotx_put(p.data() + 8, 1, 4); aotx_put(p.data() + 12, 3, 4);
    aotx_put(p.data() + 16, 4, 4); aotx_put(p.data() + 20, 1, 4);
    memset(p.data() + 24, 0x12, 32); memset(p.data() + 56, 0x34, 32); memset(p.data() + 88, 0x56, 32);
    aotx_float_put(p.data() + 128, x); aotx_float_put(p.data() + 132, y); aotx_float_put(p.data() + 136, z); return p;
}
static aotx_fixture aotx_memory_corpus(unsigned n, bool ranking = false, unsigned scope = 0) {
    aotx_fixture f;
    unsigned vectors = ranking ? 3 : n;
    const float values[3][3] = {{1,4,0}, {2,1,0}, {3,0,5}};
    for (unsigned i = 0; i < vectors; ++i) {
        auto r = aotx_memory_row(i, AOTX_COG_COMPONENT, 900000 + i, f.rows.size() + 1, 2);
        f.add(r, ranking ? aotx_memory_vector(values[i][0], values[i][1], values[i][2]) : aotx_memory_vector(2 + i, 3, 1));
    }
    for (unsigned i = 0; i < n; ++i) for (unsigned j = 0; j < (ranking ? 3u : 1u); ++j) {
        auto r = aotx_memory_row(i, AOTX_COG_ASSERTION, 10000 + i * 3 + j, f.rows.size() + 1, scope);
        aotx_id(r.data() + AOTX_CO_EMBEDDING, 900000 + (ranking ? j : i)); aotx_put(r.data() + AOTX_CO_EMBED_VERSION, 1);
        f.add(r, aotx_memory_text("fact " + std::to_string(i) + " item " + std::to_string(j)));
    }
    return f;
}
static unsigned char *aotx_query_at(aotx_bytes &q, unsigned i) { return q.data() + 64 + (size_t)i * AOTX_RECALL_QUERY; }
static aotx_bytes aotx_memory_queries(unsigned n, uint64_t sequence, unsigned scope = 0) {
    aotx_bytes q(64 + (size_t)n * AOTX_RECALL_QUERY, 0); memcpy(q.data(), "AOTXREQ1", 8);
    aotx_put(q.data() + 8, n, 4); aotx_put(q.data() + 12, 1, 4); aotx_id(q.data() + 16, 9000);
    aotx_put(q.data() + 32, sequence); aotx_put(q.data() + 40, AOTX_RECALL_QUERY, 4);
    for (unsigned i = 0; i < n; ++i) {
        auto p = aotx_query_at(q, i); aotx_id(p, 100000 + i); aotx_id(p + 16, 1000 + i); aotx_id(p + 48, 200000 + i);
        if (scope == 1) aotx_id(p + 32, 2000 + i);
        memset(p + 64, 0x12, 32); memset(p + 96, 0x34, 32);
        aotx_put(p + 128, 3, 4); aotx_put(p + 132, 3, 4); aotx_put(p + 136, AOTX_RECALL_BUDGET, 4);
        aotx_put(p + 152, scope, 4);
        aotx_float_put(p + 160, 2 + i * 0.01f); aotx_float_put(p + 164, 3 - (i % 7) * 0.1f); aotx_float_put(p + 168, 0.2f);
        std::string text = "request " + std::to_string(i); aotx_put(p + 148, text.size(), 4); memcpy(p + 4640, text.data(), text.size());
    }
    return q;
}
static void aotx_pin(unsigned char *q, unsigned group, unsigned at, uint64_t id, uint64_t version = 1) {
    unsigned char *p = q + (group ? 4448 : 4256) + at * 24; aotx_id(p, id); aotx_put(p + 16, version);
    aotx_put(q + (group ? 144 : 140), at + 1, 4);
}
struct aotx_recall_device : aotx_device {
    unsigned char *requests;
    aotx_recall_result *rows;
    aotx_recall_device() {
        AOTX_CUDA(cudaMalloc(&requests, AOTX_RECALL_REQUESTS));
        AOTX_CUDA(cudaMalloc(&rows, AOTX_RECALL_BATCH * sizeof(*rows)));
    }
    ~aotx_recall_device() { cudaFree(rows); cudaFree(requests); }
    std::vector<aotx_recall_result> search(const aotx_bytes &q, unsigned n, bool replay = false) {
        AOTX_CUDA(cudaMemcpy(requests, q.data(), q.size(), cudaMemcpyHostToDevice));
        if (replay) aotx_recall_replay<<<n,64>>>(live, requests, q.size(), rows, n);
        else aotx_recall_search<<<n,64>>>(live, requests, q.size(), rows, n);
        AOTX_CUDA(cudaGetLastError()); std::vector<aotx_recall_result> out(n);
        AOTX_CUDA(cudaMemcpy(out.data(), rows, n * sizeof(*rows), cudaMemcpyDeviceToHost)); return out;
    }
    aotx_cognitive_result record(const aotx_bytes &q, unsigned n, bool apply = false) {
        aotx_recall_record<<<1,64>>>(live, requests, q.size(), rows, n, image, result);
        auto r = finish();
        if (!r.status && apply) { aotx_cognitive_apply<<<1,64>>>(live, stage, image, r.bytes, result); r = finish(); }
        return r;
    }
    aotx_bytes saved_queries(unsigned n) {
        aotx_recall_requests<<<1,64>>>(live, requests, result); auto r = finish();
        aotx_check(!r.status && r.applied == n, "recorded request count"); aotx_bytes q(r.bytes);
        AOTX_CUDA(cudaMemcpy(q.data(), requests, q.size(), cudaMemcpyDeviceToHost)); return q;
    }
};
static uint64_t aotx_selected(const aotx_recall_result &r, unsigned i) { return aotx_get(r.selection + 16 + i * 32); }
static std::string aotx_context(const aotx_recall_result &r) { return std::string((const char *)r.context, r.context_bytes); }
static void aotx_status_rows(const std::vector<aotx_recall_result> &rows, unsigned status, const char *what) {
    for (const auto &r : rows) {
        if (r.status != status) fprintf(stderr,"%s expected=%u actual=%u\n",what,status,r.status);
        aotx_check(r.status == status, what);
        if (status) aotx_check(!r.count && !r.context_bytes && !r.selection[0] && !r.context[0], "refusal has no partial context");
    }
}
#endif
