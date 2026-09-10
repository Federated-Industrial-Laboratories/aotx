/* Purpose: Verify complete source embeddings under finite page-pool contention.
 * Owns: An independent page service and token-position expectations.
 * Launch shape: N=1 and N=64 sources with multiple forward passes per source.
 * Lifetime: A complete text batch through page cleanup and atomic publication. */
#include "text_fixture.h"

__global__ void aotx_text_paging_tokens(unsigned n) {
    unsigned i = threadIdx.x;
    if (!i) aotx_kvl_make(&aotx_model_space[AOTX_MODEL_EMBEDDING].shape, 28, 8, 128);
    if (i >= n || !aotx_live_text_pending(i)) return;
    aotx_tool_gear.count[i] = 1121 + (i % 2);
    aotx_tool_gear.piece_count[i] = 1;
    aotx_tool_gear.chunk[i * AOTX_TOOL_TOKEN_STRIDE] = aotx_tool_gear.count[i];
    for (unsigned j = 0; j < aotx_tool_gear.count[i]; ++j)
        aotx_tool_gear.id[i * AOTX_TOOL_TOKEN_STRIDE + j] = 10000 * i + j;
}
static void aotx_text_paging_serve(std::vector<unsigned> &held, unsigned capacity) {
    aotx_kv_table kv; AOTX_CUDA(cudaMemcpyFromSymbol(&kv, aotx_kv, sizeof(kv)));
    unsigned used = 0; for (auto n : held) used += n;
    for (unsigned at = kv.served; at != kv.made; ++at) {
        auto entry = kv.queue[at & (AOTX_KV_QUEUE_MAX - 1)];
        if (!entry.pages) { used -= held[entry.agent]; held[entry.agent] = 0; }
        else {
            unsigned add = std::min(entry.pages, capacity - used);
            held[entry.agent] += add; used += add;
        }
    }
    for (unsigned i = 0; i < held.size(); ++i) kv.count[i] = held[i];
    kv.mapped_pages = AOTX_KV_PAGES - (capacity - used); kv.served = kv.made;
    AOTX_CUDA(cudaMemcpyToSymbol(aotx_kv, &kv, sizeof(kv)));
}
static void aotx_text_paging(unsigned n, bool unavailable) {
    auto f = aotx_text_corpus(n); aotx_text_device d(n);
    d.send(aotx_live_load_bytes(f.wire(false, 2 * n)), AOTX_LIVE_LOAD);
    d.send(aotx_live_binding_bytes(n, 2 * n), AOTX_LIVE_BIND);
    auto input = aotx_text_input(n, 2 * n);
    for (unsigned i = 0; i < n; ++i) {
        auto q = input.data() + 128 + i * AOTX_LIVE_QUERY_ROW;
        aotx_put(q + 148, AOTX_RECALL_TEXT, 4); memset(q + 4640, 'a' + i % 26, AOTX_RECALL_TEXT);
    }
    d.process_text(aotx_live_parts(input, AOTX_LIVE_TEXT, d.next_id++), false, false);
    aotx_text_paging_tokens<<<1,64>>>(n); AOTX_CUDA(cudaDeviceSynchronize());
    aotx_kvl_shape shape; aotx_kvl_make(&shape, 28, 8, 128);
    unsigned need = aotx_kvl_pages(&shape, 1122);
    unsigned capacity = unavailable ? need - 1 : need * (n == 1 ? 1 : 12) + need / 2;
    std::vector<unsigned> held(n, 0), complete(n, 0);
    unsigned ticks = 0, last_first = 0;
    for (; ticks < 500 && d.state().phase == AOTX_LIVE_ENCODING; ++ticks) {
        aotx_text_paging_serve(held, capacity);
        aotx_tool_plan<<<1,64>>>(0); AOTX_CUDA(cudaDeviceSynchronize());
        aotx_tool_embed_batch embed; AOTX_CUDA(cudaMemcpyFromSymbol(&embed, aotx_tool_embed, sizeof(embed)));
        for (unsigned i = 0; i < n; ++i) if (embed.state[i] == AOTX_TOOL_EMBED_RUN) {
            unsigned place = embed.place[i], start = embed.offset[place], end = embed.offset[place + 1];
            aotx_check(held[i] >= need && start < end && end <= AOTX_MODEL_MAX_TOKENS,
                "each pass has a complete page lease and bounded token rows");
            if (!complete[i]) last_first = ticks;
            for (unsigned j = start; j < end; ++j)
                aotx_check(embed.ids[j] == (int)(10000 * i + complete[i] + j - start),
                    "every source token reaches its exact complete-source position once");
            complete[i] += end - start;
        }
        aotx_text_vector_fixture<<<1,64>>>(n, 0);
        aotx_tool_step<<<1,64>>>(0); aotx_live_prepare<<<1,64>>>(); AOTX_CUDA(cudaDeviceSynchronize());
    }
    aotx_text_paging_serve(held, capacity);
    aotx_check(ticks < 500, "finite contention completes or refuses within the batch deadline");
    for (auto &b : d.bindings(n)) aotx_check(!b.ordinal, "completed embeddings do not publish a partial input batch");
    for (unsigned i = 0; i < n; ++i) {
        aotx_check(!held[i], "every completed or refused source returns its pages");
        aotx_check(complete[i] == (unavailable ? 0u : 1121 + i % 2), "successful sources encode their complete token extent");
    }
    aotx_check(unavailable || n == 1 || last_first > AOTX_TOOL_ASK_LIMIT,
        "valid page contention can exceed the ordinary tool retry bound");
    d.process_text({});
    aotx_check(unavailable ? d.state().status == AOTX_COG_CAPACITY : !d.state().status,
        "only complete successful embedding batches publish input");
    for (auto &b : d.bindings(n)) aotx_check(b.ordinal == !unavailable, "page waits preserve input ordinals");
}
int main(void) {
    int cards = 0; if (cudaGetDeviceCount(&cards) != cudaSuccess || !cards) return 77;
    for (unsigned n : {1u, 64u}) for (bool unavailable : {false, true}) aotx_text_paging(n, unavailable);
    printf("text paging: %u checks, %u failures\n", aotx_checks, aotx_failures);
    return aotx_failures ? 1 : 0;
}
