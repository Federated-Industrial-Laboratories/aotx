/* Purpose: Drive text preparation with independent token and vector service fixtures.
 * Owns: Test model metadata and deterministic service outputs; no model weight claim.
 * Launch shape: Distinct successful batches at one and all profile slots.
 * Lifetime: One device test process and its exact replay. */
#ifndef AOTX_TEST_TEXT_FIXTURE_H
#define AOTX_TEST_TEXT_FIXTURE_H
#include "live_fixture.h"
#include "model/load.cuh"

static const unsigned char aotx_test_processor[32] = {
    0x7d,0x12,0xaf,0x1d,0x2c,0xd1,0xe5,0x19,0x4d,0xef,0x98,0x3d,0x1f,0xd8,0x07,0x3d,
    0x1c,0x36,0xc4,0x44,0xee,0xa3,0x9c,0x2d,0xcf,0x9b,0xbe,0x39,0x4e,0x75,0x89,0x2d
};
static aotx_fixture aotx_text_corpus(unsigned n) {
    auto f = aotx_memory_corpus(n);
    for (unsigned i = 0; i < n; ++i) {
        memcpy(f.payloads[i].data() + 56, aotx_test_processor, 32);
    }
    return f;
}
static aotx_bytes aotx_text_input(unsigned n, uint64_t seq, uint64_t ordinal = 1) {
    auto p = aotx_live_query_bytes(n, seq, ordinal);
    memcpy(p.data(), "AOTXTXT1", 8);
    for (unsigned i = 0; i < n; ++i) {
        auto q = p.data() + 128 + i * AOTX_LIVE_QUERY_ROW;
        memset(q + 64, 0, 68); memset(q + 160, 0, 4096);
        memset(q + 4256, 0, 8 * 24); aotx_put(q + 140, 0, 4);
    }
    return p;
}
__global__ void aotx_text_setup(unsigned n) {
    unsigned i = threadIdx.x;
    if (!i) {
        aotx_tool_embed.ready = 1; aotx_tool_embed.role = AOTX_MODEL_EMBEDDING; aotx_tool_embed.width = 3;
        aotx_model[AOTX_MODEL_EMBEDDING].hidden = 3; aotx_model[AOTX_MODEL_EMBEDDING].layers = 1;
        aotx_kvl_make(&aotx_model_space[AOTX_MODEL_EMBEDDING].shape, 1, 1, 8);
        auto *r = aotx_model_load.resident + AOTX_MODEL_EMBEDDING;
        r->active = 1; r->slot = AOTX_MODEL_EMBEDDING;
        for (unsigned j = 0; j < 32; ++j) r->body.digest[j] = 0x12;
    }
    if (i < n) aotx_tool_embed.state[i] = AOTX_TOOL_EMBED_NONE;
}
/* Token output is independent test data. The actual model run is checked by the boot test. */
__global__ void aotx_text_token_fixture(unsigned n, unsigned failure) {
    unsigned i = threadIdx.x;
    if (!i) aotx_kv.served = aotx_kv.made;
    if (i >= n) return;
    aotx_kv.count[i] = failure == 2 ? 0 : 16;
    if (aotx_live_text_pending(i)) {
        unsigned count = aotx_tool_gear.length[i];
        if (i + 1 == n && failure == 1) count = 0;
        if (i + 1 == n && failure == 4) count = AOTX_TOOL_TOKENS + 1;
        aotx_tool_gear.count[i] = count;
        aotx_tool_gear.piece_count[i] = 1;
        aotx_tool_gear.chunk[i * AOTX_TOOL_TOKEN_STRIDE] = count + (failure == 6 && i + 1 == n);
        for (unsigned j = 0; j < count && j < AOTX_TOOL_TOKEN_STRIDE; ++j)
            aotx_tool_gear.id[i * AOTX_TOOL_TOKEN_STRIDE + j] = aotx_tool_gear.text[i * AOTX_TOOL_TEXT_BYTES + j];
    }
}
__global__ void aotx_text_vector_fixture(unsigned n, unsigned failure) {
    unsigned i = threadIdx.x;
    if (i >= n || !aotx_live_text_pending(i) || aotx_tool_embed.state[i] != AOTX_TOOL_EMBED_RUN) return;
    if (!i && failure == 5) aotx_kv.made = aotx_kv.served + AOTX_KV_QUEUE_MAX;
    unsigned p = aotx_tool_embed.place[i];
    aotx_tool_embed.vector[p * 3] = 2 + i;
    aotx_tool_embed.vector[p * 3 + 1] = 3;
    aotx_tool_embed.vector[p * 3 + 2] = failure == 3 && i + 1 == n ? __uint_as_float(0x7fc00000) : 1;
}
static unsigned aotx_text_n, aotx_text_failure;
static void aotx_text_service(bool replay) {
    if (replay || aotx_text_failure == 7) return;
#ifdef AOTX_AFFECT
    aotx_tool_fill<<<AOTX_SLOTS,AOTX_QUALITY_FILL_THREADS>>>();
#else
    aotx_tool_fill<<<1,AOTX_SLOTS>>>();
#endif
    aotx_text_token_fixture<<<1,64>>>(aotx_text_n, aotx_text_failure);
    aotx_tool_plan<<<1,64>>>(0);
    aotx_text_vector_fixture<<<1,64>>>(aotx_text_n, aotx_text_failure);
    aotx_tool_step<<<1,64>>>(0);
}
struct aotx_text_device : aotx_live_device {
    explicit aotx_text_device(unsigned n) : aotx_live_device(n) {
        AOTX_LIVE_CLEAR(aotx_tool_embed); AOTX_LIVE_CLEAR(aotx_tool_gear);
        AOTX_LIVE_CLEAR(aotx_model_load); AOTX_LIVE_CLEAR(aotx_requests); AOTX_LIVE_CLEAR(aotx_tool_done);
        AOTX_LIVE_CLEAR(aotx_kv);
        aotx_text_n = n; aotx_text_failure = 0;
        aotx_text_setup<<<1,64>>>(n); AOTX_CUDA(cudaDeviceSynchronize());
    }
    aotx_live_records process_text(const aotx_live_records &r, bool replay = false, bool finish = true) {
        return process(r, replay, finish, aotx_text_service);
    }
    aotx_live_records text(const aotx_bytes &p) {
        return process_text(aotx_live_parts(p, AOTX_LIVE_TEXT, next_id++));
    }
    uint64_t encoded() {
        unsigned long long n = 0;
        AOTX_CUDA(cudaMemcpyFromSymbol(&n, aotx_live, sizeof(n), offsetof(aotx_live_state, encoded))); return n;
    }
};
static aotx_bytes aotx_text_choice(const aotx_live_records &records) {
    aotx_bytes out;
    for (const auto &r : records) {
        auto p = r.data() + 64;
        if (aotx_get(p + 4, 4) != AOTX_LIVE_TEXT_CHOICE) continue;
        unsigned size = aotx_get(p + 24, 4), at = aotx_get(p + 28, 4);
        auto h = (const aotx_record_header *)r.data();
        if (out.empty()) out.resize(size);
        aotx_check(size == out.size() && at + h->body_len - 32 <= out.size(), "recorded text choice framing");
        memcpy(out.data() + at, p + 32, h->body_len - 32);
    }
    return out;
}
#endif
