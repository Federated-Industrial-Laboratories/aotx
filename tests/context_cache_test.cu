/* Purpose: Check completed cognitive cache release and partial page-request progress.
 * Owns: Independent page-service fixtures and exact unchanged state checks.
 * Launch shape: Distinct batches of one and 64 conversation slots.
 * Lifetime: One device test process without model weights or mapped cache pages. */
#include "live_fixture.h"
#include "model/decode_state.cuh"
#include <memory>
#ifdef AOTX_AFFECT
#include "quality/quality.cuh"
#endif

__global__ void aotx_context_cache_setup(unsigned n, unsigned mode) {
    unsigned i = threadIdx.x;
    if (!i) {
        aotx_live.phase = AOTX_LIVE_IDLE; aotx_sched.held = mode == 12;
        aotx_kv.made = mode == 11 ? AOTX_KV_QUEUE_MAX : 0; aotx_kv.served = 0;
        aotx_kv.refused = 0;
    }
    if (i >= n) return;
    aotx_live_bindings[i].active = mode != 1;
    aotx_agents.agent[i].state = mode == 2 ? AOTX_AGENT_STATE_RUN : AOTX_AGENT_STATE_IDLE;
    aotx_agents.agent[i].task = mode == 3 ? i : ~0u;
    aotx_agent_gear[i].has_message = mode == 4; aotx_say.slot[i].wanted = mode == 5;
    aotx_seqs.slot[i].state = mode == 6 ? AOTX_SEQ_STATE_PREFILL : AOTX_SEQ_STATE_DONE;
    aotx_seqs.slot[i].prompt = 100 + i; aotx_seqs.slot[i].sampled = 1 + i;
    aotx_seqs.tokens[i][0] = 1000 + i; aotx_model_seen[i] = 100 + i;
    aotx_tool_embed.state[i] = mode == 7 ? AOTX_TOOL_EMBED_RUN : AOTX_TOOL_EMBED_NONE;
    aotx_kv.count[i] = mode == 10 || mode == 13 ? 0 : 2 + i % 3;
    aotx_seq_asked[i] = mode == 10 ? 0 : 4 + i % 3;
#ifdef AOTX_AFFECT
    aotx_quality_state[i].ended = mode == 8; aotx_quality_state[i].pending = mode == 9;
#endif
}
__global__ void aotx_context_cache_drain(void) { aotx_kv.served = aotx_kv.made; }
static void aotx_context_release(unsigned n) {
    aotx_live_device d(n);
    for (unsigned mode = 0; mode < 14; ++mode) {
        aotx_context_cache_setup<<<1,64>>>(n, mode); AOTX_CUDA(cudaDeviceSynchronize());
        aotx_seq_table seqs; AOTX_CUDA(cudaMemcpyFromSymbol(&seqs, aotx_seqs, sizeof(seqs)));
        aotx_live_stage<<<1,64>>>(); AOTX_CUDA(cudaDeviceSynchronize());
        bool release = mode == 0 || mode == 13;
#ifndef AOTX_AFFECT
        release = release || mode == 8 || mode == 9;
#endif
        aotx_kv_table kv; AOTX_CUDA(cudaMemcpyFromSymbol(&kv, aotx_kv, sizeof(kv)));
        aotx_check(kv.made == (mode == 11 ? AOTX_KV_QUEUE_MAX : release ? n : 0), "cache release respects all consumers and queue pressure");
        std::vector<unsigned> seen(n, 0), asked(n), model(n);
        if (release) for (unsigned j = 0; j < n; ++j) {
            aotx_check(kv.queue[j].agent < n && !kv.queue[j].pages, "release requests name only current batch slots");
            if (kv.queue[j].agent < n) ++seen[kv.queue[j].agent];
        }
        AOTX_CUDA(cudaMemcpyFromSymbol(asked.data(), aotx_seq_asked, n * 4));
        AOTX_CUDA(cudaMemcpyFromSymbol(model.data(), aotx_model_seen, n * 4));
        for (unsigned i = 0; i < n; ++i) {
            aotx_check(kv.count[i] == (release || mode == 10 ? 0 : 2 + i % 3) &&
                asked[i] == (release || mode == 10 ? 0 : 4 + i % 3), "only eligible completed cache requests are cleared");
            aotx_check(!release || seen[i] == 1, "each eligible slot releases exactly once");
            aotx_check(model[i] == 100 + i, "cache release preserves the sequence processing position");
        }
        auto after = std::make_unique<aotx_seq_table>(); AOTX_CUDA(cudaMemcpyFromSymbol(after.get(), aotx_seqs, sizeof(*after)));
        aotx_check(!memcmp(&seqs, after.get(), sizeof(seqs)), "sequence tokens, reply counts and completion metadata remain exact");
        if (release) {
            aotx_live_stage<<<1,64>>>(); AOTX_CUDA(cudaDeviceSynchronize());
            unsigned made; AOTX_CUDA(cudaMemcpyFromSymbol(&made, aotx_kv, 4, offsetof(aotx_kv_table, made)));
            aotx_check(made == n, "completed caches do not emit repeated release requests");
        }
        if (mode == 11) {
            aotx_context_cache_drain<<<1,1>>>(); aotx_live_stage<<<1,64>>>(); AOTX_CUDA(cudaDeviceSynchronize());
            AOTX_CUDA(cudaMemcpyFromSymbol(&kv, aotx_kv, sizeof(kv)));
            aotx_check(kv.made == AOTX_KV_QUEUE_MAX + n, "completed caches retry after a full release queue is served");
            for (unsigned i = 0; i < n; ++i) aotx_check(!kv.count[i], "release retry clears each completed cache count");
        }
    }
}
__global__ void aotx_context_credit_setup(unsigned n, unsigned mode, unsigned pattern) {
    unsigned i = threadIdx.x;
    if (!i) {
        aotx_kv.made = mode == 1; aotx_kv.served = 0;
        aotx_kv.mapped_pages = mode ? AOTX_KV_PAGES - 512 : AOTX_KV_PAGES;
        aotx_kvl_make(&aotx_model_space[0].shape, 28, 8, 128);
    }
    if (i >= n) return;
    aotx_kv.count[i] = 2; aotx_seq_asked[i] = 4 + ((i >> (4 - 2 * pattern)) & 3);
    aotx_seqs.slot[i].page_limit = AOTX_KV_PAGES_EACH;
}
__global__ void aotx_context_credit_try(unsigned n, unsigned pattern, unsigned *result) {
    unsigned i = threadIdx.x; if (i >= n) return;
    /* Each context is distinct. Each pair of slot bits sets the page count in one pattern. */
    unsigned shift = 2 * pattern;
    unsigned order = ((i << shift) | (i >> (6 - shift))) & 63;
    result[i] = aotx_seq_pages(i, 0, 49 + order);
}
__global__ void aotx_context_credit_service(void) {
    if (threadIdx.x) return;
    for (unsigned j = aotx_kv.served; j < aotx_kv.made; ++j) {
        const auto &r = aotx_kv.queue[j & (AOTX_KV_QUEUE_MAX - 1)];
        aotx_kv.count[r.agent] += r.pages; aotx_kv.mapped_pages += r.pages;
    }
    aotx_kv.served = aotx_kv.made;
}
static void aotx_context_credit(unsigned n) {
    aotx_live_device d(n); unsigned *result; AOTX_CUDA(cudaMalloc(&result, n * 4));
    auto kv = std::make_unique<aotx_kv_table>();
    for (unsigned pattern = 0; pattern < 3; ++pattern) for (unsigned mode = 0; mode < 3; ++mode) {
        aotx_context_credit_setup<<<1,64>>>(n, mode, pattern);
        std::vector<unsigned> seen(n, 0), asked(n);
        unsigned rounds = mode == 2 ? n + 1 : 2;
        for (unsigned j = 0; j < rounds; ++j) {
            aotx_context_credit_try<<<1,64>>>(n, pattern, result);
            if (mode == 2) {
                AOTX_CUDA(cudaDeviceSynchronize());
                AOTX_CUDA(cudaMemcpyFromSymbol(kv.get(), aotx_kv, sizeof(*kv)));
                for (unsigned at = kv->served; at < kv->made; ++at) {
                    const auto &r = kv->queue[at & (AOTX_KV_QUEUE_MAX - 1)];
                    aotx_check(r.agent < n, "each credit request names a current batch slot before service");
                    if (r.agent >= n) continue;
                    unsigned need = 4 + ((r.agent >> (4 - 2 * pattern)) & 3);
                    aotx_check(need > kv->count[r.agent] && r.pages == need - kv->count[r.agent],
                        "each request carries the page deficit of its named slot before service");
                    ++seen[r.agent];
                }
                aotx_context_credit_service<<<1,1>>>();
            }
        }
        AOTX_CUDA(cudaDeviceSynchronize());
        AOTX_CUDA(cudaMemcpyFromSymbol(kv.get(), aotx_kv, sizeof(*kv))); std::vector<unsigned> done(n);
        AOTX_CUDA(cudaMemcpy(done.data(), result, n * 4, cudaMemcpyDeviceToHost));
        AOTX_CUDA(cudaMemcpyFromSymbol(asked.data(), aotx_seq_asked, n * 4));
        if (mode < 2) aotx_check(kv->made == mode, "a full pool or an outstanding queue produces no duplicate request");
        else aotx_check(kv->made == n && kv->served == n, "served partial requests retry once per slot when credit is available");
        for (unsigned i = 0; i < n; ++i) {
            unsigned need = 4 + ((i >> (4 - 2 * pattern)) & 3);
            aotx_check(done[i] == (unsigned)(mode == 2) && kv->count[i] == (mode == 2 ? need : 2) && asked[i] == need,
                "all distinct waiting sequences make progress with returned page credit");
            aotx_check(seen[i] == (unsigned)(mode == 2), "each waiting slot owns exactly one served credit request");
        }
    }
    cudaFree(result);
}
int main() {
    for (unsigned n : {1u, 64u}) { aotx_context_release(n); aotx_context_credit(n); }
    printf("context cache: %u checks, %u failed\n", aotx_checks, aotx_failures);
    return aotx_failures || aotx_checks < 2000 ? 1 : 0;
}
