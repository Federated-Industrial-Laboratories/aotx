/* Purpose: Form independent retention inputs and inspect their recorded results.
 * Owns: Host fixtures and exact byte comparisons for batched device tests.
 * Launch shape: One and 64 distinct bindings through the real live kernels.
 * Lifetime: One test process without model weights. */
#ifndef AOTX_TEST_RETAIN_FIXTURE_H
#define AOTX_TEST_RETAIN_FIXTURE_H
#include "live_fixture.h"

static aotx_bytes aotx_retain_query(unsigned n, uint64_t cut, unsigned turn, bool focus = false,
    unsigned scope = 0, unsigned width = 3, unsigned length = 0) {
    auto p = aotx_live_query_bytes(n, cut, turn, scope);
    for (unsigned i = 0; i < n; ++i) {
        auto r = p.data() + 64 + i * AOTX_LIVE_QUERY_ROW, q = r + 64;
        aotx_put(r + 4, focus, 4); aotx_put(q + 140, 0, 4); memset(q + 4256, 0, 192);
        aotx_put(q + 132, 16, 4); aotx_put(q + 136, AOTX_RECALL_BUDGET, 4);
        aotx_put(q + 128, width, 4);
        for (unsigned j = 0; j < width; ++j) aotx_float_put(q + 160 + j * 4, 1 + i + j * 0.01f);
        if (length) {
            std::string text = "retained source " + std::to_string(i) + " "; text.resize(length, 'a' + i % 26);
            memset(q + 4640, 0, AOTX_RECALL_TEXT); memcpy(q + 4640, text.data(), text.size());
            aotx_put(q + 148, length, 4);
        }
    }
    return p;
}
static aotx_bytes aotx_retain_bytes(unsigned n, uint64_t cut, unsigned turn, unsigned retention = 0, bool focus = true) {
    auto p = aotx_live_envelope("AOTXRTN1", n, cut, 160);
    for (unsigned i = 0; i < n; ++i) {
        auto r = p.data() + 64 + i * 160;
        aotx_put(r, i, 4); aotx_put(r + 4, focus, 4); aotx_id(r + 8, 8000 + i); aotx_put(r + 24, turn);
        aotx_id(r + 32, 100000 + turn * 64 + i); aotx_id(r + 48, 300000 + turn * 64 + i);
        aotx_id(r + 64, 400000 + turn * 64 + i); aotx_put(r + 104, 1); aotx_id(r + 112, 3000 + i);
        aotx_put(r + 128, 500000 + i, 4); aotx_put(r + 132, retention, 4);
    }
    return p;
}
static aotx_bytes aotx_retain_store(void) {
    aotx_bytes p(sizeof(aotx_cognitive_store));
    AOTX_CUDA(cudaMemcpyFromSymbol(p.data(), aotx_live_store, p.size())); return p;
}
static aotx_bytes aotx_retain_result(const aotx_live_records &records, unsigned op = 9) {
    aotx_bytes p;
    for (const auto &r : records) {
        auto h = (const aotx_record_header *)r.data(); auto b = r.data() + 64;
        if (h->type != 33 || aotx_get(b + 4, 4) != op) continue;
        aotx_check(aotx_get(b + 28, 4) == p.size(), "result parts have exact consecutive offsets");
        p.insert(p.end(), b + 32, b + h->body_len);
        aotx_check(p.size() <= aotx_get(b + 24, 4), "result fits declared bytes");
    }
    return p;
}
static void aotx_retain_open(aotx_live_device &d, unsigned n, unsigned scope = 0) {
    aotx_fixture empty; d.send(aotx_live_load_bytes(empty.wire(false, 0)), 1);
    d.send(aotx_live_binding_bytes(n, 0, scope), 3);
    aotx_check(!d.state().status && d.state().ready, "empty store and bindings load");
}
static void aotx_retain_mutate(aotx_live_records &records, size_t offset) {
    for (auto &r : records) {
        auto b = r.data() + 64;
        if (aotx_get(b + 4, 4) != 9) continue;
        size_t start = aotx_get(b + 28, 4), bytes = ((aotx_record_header *)r.data())->body_len - 32;
        if (offset >= start && offset < start + bytes) { b[32 + offset - start] ^= 1; return; }
    }
    aotx_check(false, "mutation reaches a recorded result byte");
}
#endif
