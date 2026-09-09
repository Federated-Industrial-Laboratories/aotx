/* Purpose: Build independent typed state fixtures and drive the device API.
 * Owns: Test buffers, expected byte images and check tallies.
 * Launch shape: State batches contain one or 64 distinct objects.
 * Lifetime: One test process. */
#ifndef AOTX_COGNITIVE_FIXTURE_H
#define AOTX_COGNITIVE_FIXTURE_H
#include <cuda_runtime.h>
#include "cognitive/state.cuh"
#include <array>
#include <vector>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static unsigned aotx_checks, aotx_failures;
static void aotx_check(bool good, const char *what) {
    ++aotx_checks;
    if (!good) { ++aotx_failures; fprintf(stderr, "FAIL: %s\n", what); }
}
#define AOTX_CUDA(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA: %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(2); } } while (0)
using aotx_bytes = std::vector<unsigned char>;
using aotx_row = std::array<unsigned char, AOTX_COG_OBJECT>;
static void aotx_put(unsigned char *p, uint64_t value, unsigned bytes = 8) {
    for (unsigned i = 0; i < bytes; ++i) p[i] = (unsigned char)(value >> (i * 8));
}
static uint64_t aotx_get(const unsigned char *p, unsigned bytes = 8) {
    uint64_t value = 0;
    for (unsigned i = 0; i < bytes; ++i) value |= (uint64_t)p[i] << (i * 8);
    return value;
}
static void aotx_id(unsigned char *p, uint64_t value) {
    memset(p, 0, 16); aotx_put(p, value); p[15] = 0xa7;
}
static aotx_row aotx_object(unsigned i, unsigned kind, uint64_t id, uint64_t sequence) {
    aotx_row r = {};
    aotx_put(r.data(), 1, 2); aotx_put(r.data() + AOTX_CO_KIND, kind, 2);
    aotx_id(r.data() + AOTX_CO_ID, id); aotx_id(r.data() + AOTX_CO_LINEAGE, 9000);
    aotx_put(r.data() + AOTX_CO_VERSION, 1);
    aotx_put(r.data() + AOTX_CO_CREATED, sequence); aotx_put(r.data() + AOTX_CO_UPDATED, sequence);
    aotx_id(r.data() + AOTX_CO_OWNER, 1000 + i); aotx_id(r.data() + AOTX_CO_SUBJECT, 3000 + i);
    aotx_put(r.data() + AOTX_CO_SCOPE, i % 3, 4);
    if (i % 3 == AOTX_COG_ROOM) aotx_id(r.data() + AOTX_CO_ROOM, 2000 + i);
    aotx_put(r.data() + AOTX_CO_SOURCE_KIND, AOTX_COG_REPORTED, 4);
    aotx_put(r.data() + AOTX_CO_IMPORTANCE, AOTX_COG_UNKNOWN, 4);
    aotx_put(r.data() + AOTX_CO_POLICY, 1);
    return r;
}
struct aotx_fixture {
    std::vector<aotx_row> rows;
    std::vector<aotx_bytes> payloads;
    void add(aotx_row row, aotx_bytes payload) { rows.push_back(row); payloads.push_back(payload); }
    aotx_bytes wire(bool tail, uint64_t sequence, uint64_t tick = 5) const {
        size_t payload = 0;
        for (const auto &p : payloads) payload += p.size();
        size_t start = AOTX_COG_HEADER + rows.size() * AOTX_COG_OBJECT;
        aotx_bytes out(start + payload, 0);
        memcpy(out.data(), tail ? "AOTXLOG1" : "AOTXOBJ1", 8);
        aotx_put(out.data() + 8, 1, 4); aotx_put(out.data() + 12, AOTX_COG_HEADER, 4);
        aotx_put(out.data() + 16, AOTX_COG_OBJECT, 4); aotx_put(out.data() + 20, rows.size(), 4);
        aotx_put(out.data() + 24, payload); aotx_put(out.data() + 32, sequence);
        aotx_put(out.data() + 40, tick); aotx_id(out.data() + 48, 9000);
        aotx_put(out.data() + 64, AOTX_COG_HEADER); aotx_put(out.data() + 72, start);
        aotx_put(out.data() + 80, out.size()); aotx_put(out.data() + 88, 1, 4);
        size_t offset = 0;
        for (size_t i = 0; i < rows.size(); ++i) {
            unsigned char *r = out.data() + AOTX_COG_HEADER + i * AOTX_COG_OBJECT;
            memcpy(r, rows[i].data(), AOTX_COG_OBJECT);
            aotx_put(r + AOTX_CO_OFFSET, payloads[i].empty() ? 0 : offset);
            aotx_put(r + AOTX_CO_BYTES, payloads[i].size());
            if (!payloads[i].empty()) memcpy(out.data() + start + offset, payloads[i].data(), payloads[i].size());
            offset += payloads[i].size();
        }
        return out;
    }
    void append(const aotx_fixture &other) {
        rows.insert(rows.end(), other.rows.begin(), other.rows.end());
        payloads.insert(payloads.end(), other.payloads.begin(), other.payloads.end());
    }
};
static aotx_fixture aotx_initial(unsigned n) {
    aotx_fixture f;
    for (unsigned i = 0; i < n; ++i) {
        std::string text = "entity constraint " + std::to_string(3000 + i);
        f.add(aotx_object(i, AOTX_COG_ASSERTION, i + 1, i + 1), aotx_bytes(text.begin(), text.end()));
    }
    return f;
}
static aotx_fixture aotx_appraisals(unsigned n) {
    aotx_fixture f;
    for (unsigned i = 0; i < n; ++i) {
        auto r = aotx_object(i, AOTX_COG_APPRAISAL, 101 + i, n + i + 1);
        aotx_id(r.data() + AOTX_CO_SOURCE, i + 1); aotx_put(r.data() + AOTX_CO_SOURCE_VERSION, 1);
        aotx_put(r.data() + AOTX_CO_SOURCE_KIND, AOTX_COG_INFERRED, 4);
        aotx_bytes p(32, 0);
        aotx_put(p.data(), 1, 4); aotx_put(p.data() + 4, 700000 + i, 4);
        aotx_put(p.data() + 8, 800000 - i, 4); aotx_put(p.data() + 12, i ? 400000 + i : AOTX_COG_UNKNOWN, 4);
        aotx_put(p.data() + 16, 4, 4); aotx_put(p.data() + 20, 600000 + i, 4);
        aotx_put(p.data() + 24, 1, 4); f.add(r, p);
    }
    return f;
}
static aotx_fixture aotx_selections(unsigned n) {
    aotx_fixture f;
    for (unsigned i = 0; i < n; ++i) {
        auto r = aotx_object(i, AOTX_COG_SELECTION, 201 + i, 2 * n + i + 1);
        aotx_bytes p(80, 0); aotx_put(p.data(), 1, 4); aotx_put(p.data() + 4, 2, 4);
        aotx_id(p.data() + 16, 101 + i); aotx_put(p.data() + 32, 1); aotx_put(p.data() + 40, 1, 4);
        aotx_id(p.data() + 48, 1 + i); aotx_put(p.data() + 64, 1); aotx_put(p.data() + 72, 1, 4);
        f.add(r, p);
    }
    return f;
}
struct aotx_device {
    aotx_cognitive_store *live = nullptr, *stage = nullptr;
    aotx_cognitive_result *result = nullptr;
    unsigned char *image = nullptr;
    aotx_device() {
        AOTX_CUDA(cudaMalloc(&live, sizeof(*live))); AOTX_CUDA(cudaMalloc(&stage, sizeof(*stage)));
        AOTX_CUDA(cudaMalloc(&result, sizeof(*result))); AOTX_CUDA(cudaMalloc(&image, AOTX_COG_IMAGE));
        AOTX_CUDA(cudaMemset(live, 0, sizeof(*live)));
    }
    ~aotx_device() { cudaFree(image); cudaFree(result); cudaFree(stage); cudaFree(live); }
    aotx_cognitive_result finish() {
        AOTX_CUDA(cudaGetLastError()); aotx_cognitive_result r;
        AOTX_CUDA(cudaMemcpy(&r, result, sizeof(r), cudaMemcpyDeviceToHost)); return r;
    }
    aotx_cognitive_result load(const aotx_bytes &bytes, bool tail = false) {
        AOTX_CUDA(cudaMemcpy(image, bytes.data(), bytes.size(), cudaMemcpyHostToDevice));
        if (tail) aotx_cognitive_apply<<<1, 64>>>(live, stage, image, bytes.size(), result);
        else aotx_cognitive_restore<<<1, 64>>>(live, stage, image, bytes.size(), result);
        return finish();
    }
    aotx_bytes checkpoint(uint64_t capacity = AOTX_COG_IMAGE) {
        aotx_cognitive_checkpoint<<<1, 64>>>(live, image, capacity, result);
        auto r = finish(); aotx_check(!r.status, "checkpoint status");
        aotx_bytes bytes(r.bytes);
        AOTX_CUDA(cudaMemcpy(bytes.data(), image, bytes.size(), cudaMemcpyDeviceToHost)); return bytes;
    }
    void rejects(const aotx_bytes &bad, bool tail, uint32_t status, const char *name) {
        auto before = checkpoint(); auto r = load(bad, tail);
        if (r.status != status) fprintf(stderr, "%s: expected=%u actual=%u\n", name, status, r.status);
        aotx_check(r.status == status && !r.applied, name);
        aotx_check(checkpoint() == before, "rejected batch leaves exact live state");
    }
    std::vector<aotx_cognitive_match> resolve(unsigned n, uint64_t first, uint64_t version,
                                             bool wrong = false) {
        std::vector<aotx_cognitive_query> q(n);
        std::vector<aotx_cognitive_match> out(n);
        for (unsigned i = 0; i < n; ++i) {
            aotx_id(q[i].id, first + i); aotx_id(q[i].principal, 1000 + i + (wrong ? 5000 : 0));
            aotx_id(q[i].room, 2000 + i + (wrong ? 5000 : 0)); q[i].version = version;
        }
        aotx_cognitive_query *dq; aotx_cognitive_match *dm;
        AOTX_CUDA(cudaMalloc(&dq, n * sizeof(*dq))); AOTX_CUDA(cudaMalloc(&dm, n * sizeof(*dm)));
        AOTX_CUDA(cudaMemcpy(dq, q.data(), n * sizeof(*dq), cudaMemcpyHostToDevice));
        aotx_cognitive_resolve<<<(n + 63) / 64, 64>>>(live, dq, dm, n);
        AOTX_CUDA(cudaGetLastError()); AOTX_CUDA(cudaMemcpy(out.data(), dm, n * sizeof(*dm), cudaMemcpyDeviceToHost));
        cudaFree(dq); cudaFree(dm); return out;
    }
};
#endif
