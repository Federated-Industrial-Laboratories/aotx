/* Purpose: Verify text page release and publication while the page queue is full.
 * Owns: Explicit page ownership and queue observations without a service reset.
 * Launch shape: Distinct successful and refused batches at one and all profile slots.
 * Lifetime: One text request from admission through complete page release. */
#ifndef AOTX_TEST_TEXT_CLEANUP_H
#define AOTX_TEST_TEXT_CLEANUP_H
#include "text_fixture.h"

__global__ void aotx_text_hold_queue(unsigned n) {
    unsigned i = threadIdx.x;
    if (i < n) aotx_kv.count[i] = 2 + i % 5;
    if (i) return;
    aotx_kv.served = aotx_kv.made;
    for (unsigned j = 0; j < AOTX_KV_QUEUE_MAX; ++j) {
        auto *p = aotx_kv.queue + ((aotx_kv.made + j) & (AOTX_KV_QUEUE_MAX - 1));
        p->agent = (j + 17) % AOTX_SLOTS; p->pages = 2 + j % 3;
    }
    aotx_kv.made += AOTX_KV_QUEUE_MAX;
}
__global__ void aotx_text_allow_queue(unsigned count) {
    if (!threadIdx.x) aotx_kv.served += count;
}
static aotx_kv_table aotx_text_pages() {
    aotx_kv_table p; AOTX_CUDA(cudaMemcpyFromSymbol(&p, aotx_kv, sizeof(p))); return p;
}
static void aotx_text_no_publication(aotx_text_device &d, unsigned n,
                                      const std::vector<aotx_live_binding> &before, uint64_t tail) {
    aotx_check(d.state().phase == AOTX_LIVE_ENCODING && !d.state().searches,
               "full page queue blocks search and publication");
    aotx_check(d.seam().dev.tail == tail, "blocked cleanup writes no decision record");
    auto rows = d.bindings(n);
    aotx_say_state say; AOTX_CUDA(cudaMemcpyFromSymbol(&say, aotx_say, sizeof(say)));
    for (unsigned i = 0; i < n; ++i) {
        aotx_check(!memcmp(&rows[i], &before[i], sizeof(rows[i])), "blocked cleanup preserves each binding");
        aotx_check(!say.slot[i].wanted, "blocked cleanup queues no language prompt");
    }
}
static void aotx_text_cleanup(unsigned n, unsigned failure) {
    auto f = aotx_text_corpus(n); aotx_text_device d(n);
    d.send(aotx_live_load_bytes(f.wire(false, 2 * n)), AOTX_LIVE_LOAD);
    d.send(aotx_live_binding_bytes(n, 2 * n), AOTX_LIVE_BIND);
    auto before = d.bindings(n);
    d.process_text(aotx_live_parts(aotx_text_input(n, 2 * n), AOTX_LIVE_TEXT, d.next_id++), false, false);
    aotx_check(d.state().phase == AOTX_LIVE_ENCODING && !d.encoded(), "cleanup fixture stops before the embedding service");
    aotx_text_failure = failure;
    for (unsigned tick = 0; tick < 64; ++tick) {
        aotx_text_service(false); AOTX_CUDA(cudaDeviceSynchronize());
        if (d.encoded() == n - (failure == 1)) break;
    }
    aotx_check(d.encoded() == n - (failure == 1), "cleanup observes each completed embedding row");
    aotx_text_hold_queue<<<1,64>>>(n); AOTX_CUDA(cudaDeviceSynchronize());
    auto full = aotx_text_pages(); auto tail = d.seam().dev.tail;
    /* No service fixture runs again. Only the code under test can release these pages. */
    for (unsigned tick = 0; tick < 2; ++tick) {
        aotx_live_prepare<<<1,64>>>(); aotx_live_search<<<64,64>>>();
        aotx_live_decide<<<1,64>>>(); aotx_live_commit<<<1,64>>>(); aotx_agent_step<<<1,AOTX_SLOTS>>>(0);
        AOTX_CUDA(cudaDeviceSynchronize());
        aotx_text_no_publication(d, n, before, tail);
        auto held = aotx_text_pages();
        aotx_check(held.made == full.made && held.served == full.served &&
                   !memcmp(held.queue, full.queue, sizeof(full.queue)), "full queue preserves all earlier requests");
        for (unsigned i = 0; i < n; ++i)
            aotx_check(held.count[i] == 2 + i % 5, "failed release retains the pages of its own slot");
    }
    aotx_text_allow_queue<<<1,1>>>(1); aotx_live_prepare<<<1,64>>>(); AOTX_CUDA(cudaDeviceSynchronize());
    auto partial = aotx_text_pages();
    aotx_check(partial.made == full.made + 1, "one free queue entry accepts exactly one release");
    const auto &first = partial.queue[full.made & (AOTX_KV_QUEUE_MAX - 1)];
    aotx_check(first.agent == 0 && first.pages == 0, "first release names the first text slot");
    for (unsigned i = 0; i < n; ++i)
        aotx_check(partial.count[i] == (i ? 2 + i % 5 : 0), "partial cleanup changes only the released slot");
    if (n > 1) aotx_text_no_publication(d, n, before, tail);
    aotx_text_allow_queue<<<1,1>>>(partial.made - partial.served);
    aotx_live_prepare<<<1,64>>>(); AOTX_CUDA(cudaDeviceSynchronize());
    auto released = aotx_text_pages();
    aotx_check(released.made == full.made + n, "every text slot queues one release without duplication");
    aotx_tool_embed_batch embed; AOTX_CUDA(cudaMemcpyFromSymbol(&embed, aotx_tool_embed, sizeof(embed)));
    for (unsigned i = 0; i < n; ++i) {
        const auto &entry = released.queue[(full.made + i) & (AOTX_KV_QUEUE_MAX - 1)];
        aotx_check(entry.agent == i && entry.pages == 0, "release request retains the exact slot identity");
        aotx_check(!released.count[i] && embed.state[i] == AOTX_TOOL_EMBED_NONE,
                   "terminal text cleanup retains no pages or embedding lease");
    }
    aotx_check(d.state().phase == AOTX_LIVE_SEARCH, "complete release permits the decision stage");
    auto records = d.process({});
    auto choice = aotx_text_choice(records);
    aotx_check(!choice.empty() && d.state().phase == AOTX_LIVE_IDLE, "cleanup ends with one complete decision");
    auto rows = d.bindings(n);
    if (failure) {
        aotx_check(d.state().status && !d.state().searches && choice.size() == 64,
                   "failed encoding publishes only a refusal after cleanup");
        for (unsigned i = 0; i < n; ++i)
            aotx_check(!memcmp(&rows[i], &before[i], sizeof(rows[i])), "refused cleanup preserves every binding");
    } else {
        aotx_check(!d.state().status && d.state().searches == n, "successful cleanup permits one search per row");
        d.prompt(n);
        for (const auto &b : rows) aotx_check(b.ordinal == 1, "successful cleanup publishes each request once");
    }
    auto terminal = aotx_text_pages();
    aotx_check(terminal.made == released.made, "decision publication does not duplicate page release");
    for (unsigned i = 0; i < n; ++i) aotx_check(!terminal.count[i], "publication retains no temporary pages");
}
#endif
