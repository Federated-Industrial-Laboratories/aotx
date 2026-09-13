/* Purpose: Verify creator decisions against real memory maintenance and recovery.
 * Owns: Distinct scoped memories, complete journals and exact state comparisons.
 * Launch shape: The live policy graph and maintenance kernels at N=1 and N=64.
 * Lifetime: One test process with finite native graphs. */
#include "policy_fixture.h"

static unsigned aotx_policy_batch_size;
static void aotx_policy_live_hook(bool replay) {
    if (replay) aotx_live_test_idle<<<1,64>>>(aotx_policy_batch_size);
    aotx_policy_active_graph->tick();
}
static aotx_bytes aotx_policy_corpus(unsigned n) {
    aotx_fixture seed;
    unsigned fill = AOTX_COG_OBJECTS / 2;
    for (unsigned i = 0; i < fill; ++i)
        seed.add(aotx_memory_row(i % n, AOTX_COG_EVENT, 3000000 + i, i + 1),
            aotx_bytes(32, (unsigned char)(i % 251 + 1)));
    auto corpus = aotx_memory_corpus(n);
    for (auto &r : corpus.rows) {
        aotx_put(r.data() + AOTX_CO_CREATED, aotx_get(r.data() + AOTX_CO_CREATED) + fill);
        aotx_put(r.data() + AOTX_CO_UPDATED, aotx_get(r.data() + AOTX_CO_UPDATED) + fill);
    }
    seed.append(corpus);
    auto image = seed.wire(false, seed.rows.size());
    aotx_put(image.data() + 8, 2, 4);
    aotx_put(image.data() + 96, seed.rows.size());
    aotx_put(image.data() + 112, n, 4); aotx_put(image.data() + 116, 1, 4);
    aotx_put(image.data() + 120, 1, 4); aotx_put(image.data() + 124, 95, 4);
    return image;
}
static void aotx_policy_collect(aotx_live_device &d, aotx_live_records &journal) {
    unsigned limit = AOTX_POLICY_EVENT_BYTES / ((AOTX_BODY_BYTES - AOTX_POLICY_PART) * AOTX_POLICY_EMIT) + 8;
    for (unsigned tick = 0; tick < limit; ++tick) {
        aotx_maint_append(journal, d.process({}, false, true, aotx_policy_live_hook));
        auto state = aotx_policy_read_state();
        if (!state->pending && state->source == aotx_maint_store()->sequence &&
            state->root == aotx_maint_store()->root_sequence) return;
    }
    aotx_check(false, "policy and memory settle within their finite publication bound");
}
static void aotx_policy_live(unsigned n, unsigned mode, unsigned stride) {
    aotx_policy_batch_size = n;
    aotx_policy_asset asset(mode, stride);
    aotx_live_records journal;
    std::unique_ptr<aotx_policy_state> expected;
    std::unique_ptr<aotx_cognitive_store> store;
    std::vector<aotx_live_binding> bindings;
    uint64_t hash = 0;
    {
        aotx_live_device d(n); AOTX_LIVE_CLEAR(aotx_checkpoint);
        aotx_checkpoint_test_clear<<<1,AOTX_SLOTS>>>();
        asset.open(); aotx_policy_graph graph; aotx_policy_active_graph = &graph;
        auto image = aotx_policy_corpus(n);
        aotx_maint_append(journal, d.send(aotx_live_load_bytes(image), AOTX_LIVE_LOAD));
        aotx_check(!d.state().status && d.state().ready, "the prepared maintenance image passes complete admission");
        if (d.state().status) exit(1);
        uint64_t source = aotx_maint_store()->sequence;
        aotx_maint_append(journal, d.send(aotx_live_binding_bytes(n, source), AOTX_LIVE_BIND));
        aotx_maint_append(journal, d.send(aotx_live_query_bytes(n, source, 1), AOTX_LIVE_QUERY));
        d.prompt(n); d.idle(n);
        auto before = aotx_maint_store();
        aotx_check(before->count > AOTX_COG_OBJECTS * 40 / 100 && before->count < AOTX_COG_OBJECTS * 95 / 100,
            "resident use reaches the creator threshold below the store threshold");
        for (unsigned guard : {1u, 2u, 3u}) {
            aotx_policy_test_control<<<1,1>>>(guard); graph.tick();
            auto quiet = aotx_policy_read_state();
            aotx_check(!quiet->calls && !quiet->decision && !quiet->candidate[0],
                "pause, scheduler hold and foreground work prevent native entry");
        }
        aotx_policy_test_control<<<1,1>>>(0);
        aotx_policy_collect(d, journal);
        store = aotx_maint_store(); expected = aotx_policy_read_state(); bindings = d.bindings(n);
        aotx_check(!d.state().status && store->count == 2 * n && store->bytes < before->bytes,
            "the creator proposal reclaims unneeded memory and retains bound dependencies");
        aotx_check(expected->decision == 1 && expected->calls == 1 && !expected->pending && !expected->status,
            "one pressure decision reaches accepted state without repeating the unchanged source");
        aotx_check(aotx_get(expected->current) == 1 && aotx_get(expected->current + 8) == before->sequence,
            "native private state records its own accepted maintenance proposal");
        for (unsigned i = 0; i < 20; ++i) graph.tick();
        auto quiet = aotx_policy_read_state();
        aotx_check(quiet->calls == expected->calls && quiet->decision == expected->decision,
            "unchanged idle ticks perform no creator work");
        hash = d.seam().apply.state_hash;
        printf("policy live n=%u mode=%u state=%u decisions=%llu objects=%u maximum_ns=%llu records=%zu\n",
            n, mode, stride, (unsigned long long)expected->decision, store->count,
            (unsigned long long)expected->maximum_ns, journal.size());
    }
    aotx_policy_close();
    {
        aotx_live_device d(n); AOTX_LIVE_CLEAR(aotx_checkpoint);
        aotx_checkpoint_test_clear<<<1,AOTX_SLOTS>>>();
        asset.open(); aotx_policy_graph graph; aotx_policy_active_graph = &graph;
        d.process(journal, true, true, aotx_policy_live_hook);
        auto actual = aotx_policy_read_state(); auto memory = aotx_maint_store(); auto b = d.bindings(n);
        aotx_check(!actual->fatal && actual->decision == expected->decision && actual->source == expected->source &&
            actual->root == expected->root && !memcmp(actual->current, expected->current, stride),
            "recorded fragments restore the exact accepted private state and observation");
        aotx_check(!actual->calls && !actual->candidate[0], "replay does not invoke the creator entry");
        aotx_check(!memcmp(store.get(), memory.get(), sizeof(*store)) && d.seam().apply.state_hash == hash,
            "the full maintained store and authoritative hash survive replay exactly");
        for (unsigned i = 0; i < n; ++i) {
            aotx_check(!b[i].choice.searches && bindings[i].choice.searches == 1,
                "recorded recall performs no new vector search");
            auto logical = bindings[i]; logical.choice.searches = 0;
            aotx_check(!memcmp(&b[i], &logical, sizeof(b[i])),
                "each independently scoped conversation binding survives recovery with its recorded selection");
        }
        aotx_maint_append(journal, d.send(aotx_live_query_bytes(n, memory->sequence, 2), AOTX_LIVE_QUERY));
        auto prompts = d.prompt(n);
        for (unsigned i = 0; i < n; ++i) aotx_check(prompts[i].find("fact " + std::to_string(i)) != std::string::npos,
            "fresh recovered work uses the correct retained memory");
    }
    aotx_policy_close();
}
int main() {
    for (unsigned n : {1u, 64u}) {
        aotx_policy_live(n, AOTX_POLICY_RULES, 16);
        aotx_policy_live(n, AOTX_POLICY_NATIVE, AOTX_POLICY_STATE_BYTES);
    }
    printf("policy live: %u checks, %u failures\n", aotx_checks, aotx_failures);
    return aotx_failures ? 1 : 0;
}
