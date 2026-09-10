/* Purpose: Check contextual recall through live input, retention and exact recovery.
 * Owns: Distinct conversation batches, complete refusals and altered decision checks.
 * Launch shape: One and 64 bindings through real live kernels and prompt consumers.
 * Lifetime: One device test process; text service output is independent fixture data. */
#include "context_fixture.h"
#include "retain_fixture.h"
#include "text_fixture.h"

static void aotx_context_no_queue(unsigned n) {
    for (unsigned i = 0; i < n; ++i) {
        uint32_t queued;
        AOTX_CUDA(cudaMemcpyFromSymbol(&queued, aotx_agent_gear, sizeof(queued),
            i * sizeof(aotx_agent_work) + offsetof(aotx_agent_work, has_message)));
        aotx_check(!queued, "refusal queues no input for any binding");
    }
}

static aotx_bytes aotx_context_live_query(unsigned n, uint64_t cut, unsigned turn,
    unsigned scope, bool text, bool unknown = false) {
    auto p = aotx_live_query_bytes(n, cut, turn, scope);
    auto prepared = aotx_context_queries(n, cut, scope, text ? 1 : 3);
    if (text) memcpy(p.data(), "AOTXTXT1", 8);
    for (unsigned i = 0; i < n; ++i) {
        auto q = p.data() + 128 + i * AOTX_LIVE_QUERY_ROW;
        memcpy(q, aotx_query_at(prepared, i), AOTX_RECALL_QUERY);
        aotx_id(q, 100000 + turn * 64 + i); aotx_id(q + 48, 200000 + turn * 64 + i);
        if (unknown) aotx_id(q + AOTX_RECALL_EXTENSION + 48, 7000 + i);
        if (text) { memset(q + 64, 0, 68); memset(q + 160, 0, 4096); aotx_put(q + 132, 2, 4); }
    }
    return p;
}
static void aotx_context_live_rows(aotx_live_device &d, unsigned n, const aotx_bytes &q, bool text, bool unknown) {
    auto bindings = d.bindings(n); auto prompts = d.prompt(n);
    for (unsigned i = 0; i < n; ++i) {
        const auto &b = bindings[i];
        aotx_check(!memcmp(b.query + AOTX_RECALL_EXTENSION, q.data() + 128 + i * AOTX_LIVE_QUERY_ROW + AOTX_RECALL_EXTENSION,
            AOTX_RECALL_QUERY - AOTX_RECALL_EXTENSION), "live preparation preserves exact task and participant controls");
        aotx_check(b.choice.count >= 1 && b.choice.reason[0] == AOTX_RECALL_OBLIGATION, "required context reaches the live binding");
        aotx_check(prompts[i].find("ask unknown requirements " + std::to_string(i)) != std::string::npos,
            "the generic task cue reaches its own real agent prompt");
        aotx_check((prompts[i].find("constraint " + std::to_string(i) + "\n") != std::string::npos) == !unknown,
            "a newcomer never inherits the known subject constraint");
        if (!text && !unknown) aotx_check(prompts[i].find("benefit=" + std::to_string(700000 + i)) != std::string::npos &&
            prompts[i].find("harm=" + std::to_string(900000 - i)) != std::string::npos, "live prompt preserves separate mixed values");
    }
}
static void aotx_context_live_roundtrip(unsigned n, unsigned scope, bool text) {
    auto corpus = aotx_context_corpus(n, scope); auto cut = corpus.rows.size();
    if (text) for (unsigned i = 0; i < n; ++i) for (unsigned j = 0; j < 3; ++j)
        memcpy(corpus.payloads[i * 12 + j].data() + 56, aotx_test_processor, 32);
    auto first_q = aotx_context_live_query(n, cut, 1, scope, text);
    auto next_q = aotx_context_live_query(n, cut + 3 * n, 2, scope, text, true);
    aotx_live_records start, first, second; aotx_bytes expected;
    std::vector<aotx_live_binding> bindings;
    {
        aotx_text_device d(n);
        start = d.send(aotx_live_load_bytes(corpus.wire(false, cut)), 1);
        auto bind = aotx_live_binding_bytes(n, cut, scope);
        for (unsigned i = 0; i < n; ++i) aotx_put(bind.data() + 124 + i * 64, 1, 4);
        auto bound = d.send(bind, 3); start.insert(start.end(), bound.begin(), bound.end());
        aotx_check(!d.state().status, "contextual store and automatic bindings are admitted");
        first = text ? d.text(first_q) : d.send(first_q, 4);
        aotx_check(!d.state().status && d.state().searches == n, "contextual automatic batch searches once per row");
        aotx_context_live_rows(d, n, first_q, text, false); d.idle(n);
        second = text ? d.text(next_q) : d.send(next_q, 4);
        aotx_check(!d.state().status, "newcomer input admits through the same binding");
        aotx_context_live_rows(d, n, next_q, text, true);
        expected = aotx_retain_store(); bindings = d.bindings(n);
        auto state = (const aotx_cognitive_store *)expected.data();
        aotx_check(state->count == cut + 6 * n && state->sequence == cut + 6 * n,
            "two inputs retain exact source objects without appraisal mutation");
        aotx_check(d.encoded() == (text ? 2 * n : 0), "only text input uses encoding");
        d.idle(n); d.process(second, true);
        aotx_check(d.state().fatal && aotx_retain_store() == expected, "duplicate decisions cannot repeat memory or significance");
    }
    {
        aotx_text_device d(n); d.process_text(start, true); d.process_text(first, true); d.idle(n); d.process_text(second, true);
        aotx_check(!d.state().fatal && !d.encoded() && !d.state().searches && d.state().replays == 2 * n,
            "recorded contextual recovery performs no encoding or search");
        aotx_check(aotx_retain_store() == expected, "all source and appraisal bytes recover exactly");
        auto restored = d.bindings(n);
        for (unsigned i = 0; i < n; ++i) {
            bindings[i].choice.searches = 0;
            aotx_check(!memcmp(&bindings[i], &restored[i], sizeof(bindings[i])), "exact contextual binding and reason recovery");
        }
        aotx_context_live_rows(d, n, next_q, text, true);
    }
    if (scope || text) return;
    auto result = aotx_retain_result(first, 10);
    for (unsigned mode = 0; mode < 3; ++mode) {
        aotx_text_device d(n); d.process(start, true); auto before = aotx_retain_store(); auto original = d.bindings(n);
        d.process(aotx_live_parts(first_q, 4, 3), true); auto bad = result;
        size_t row = 64 + (n - 1) * AOTX_LIVE_AUTO_ROW;
        if (mode == 0) bad[row + 64 + AOTX_RECALL_EXTENSION + 48] ^= 1;
        if (mode == 1) aotx_id(bad.data() + row + 64 + AOTX_RECALL_QUERY + 16 + 3 * 32, aotx_context_id(n - 1, 10));
        if (mode == 2) aotx_id(bad.data() + row + 64 + AOTX_RECALL_QUERY + 16, aotx_context_id(n - 1, 6));
        d.process(aotx_live_parts(bad, 10, 3), true);
        aotx_check(d.state().fatal && aotx_retain_store() == before, "changed participants, sources or obligations cannot publish retention");
        aotx_context_no_queue(n);
        auto now = d.bindings(n);
        for (unsigned i = 0; i < n; ++i) aotx_check(!memcmp(&now[i], &original[i], sizeof(now[i])), "a bad final row refuses every binding");
    }
    {
        aotx_text_device d(n); d.process(start, true); auto before = aotx_retain_store(); auto original = d.bindings(n);
        aotx_put(first_q.data() + 128 + (n - 1) * AOTX_LIVE_QUERY_ROW + 132, 1, 4);
        auto records = d.send(first_q, 4); auto refused = aotx_retain_result(records, 10);
        aotx_check(d.state().status == AOTX_COG_CAPACITY && refused.size() == 64 && aotx_retain_store() == before,
            "a required-set overflow refuses the complete input without memory publication");
        aotx_context_no_queue(n);
        auto now = d.bindings(n);
        for (unsigned i = 0; i < n; ++i) aotx_check(!memcmp(&now[i], &original[i], sizeof(now[i])), "pressure changes no ordinal, focus or choice");
    }
    {
        aotx_text_device d(n); auto changed = corpus;
        aotx_put(changed.rows[(n - 1) * 12 + 9].data() + AOTX_CO_EXPIRY, cut + 3 * n);
        d.send(aotx_live_load_bytes(changed.wire(false, cut)), 1);
        auto bind = aotx_live_binding_bytes(n, cut);
        for (unsigned i = 0; i < n; ++i) aotx_put(bind.data() + 124 + i * 64, 1, 4);
        d.send(bind, 3); aotx_check(!d.state().status, "prospective appraisal expiry setup");
        auto before = aotx_retain_store(); auto bound = d.bindings(n);
        d.send(aotx_context_live_query(n, cut, 1, 0, false), 4);
        aotx_check(d.state().status == AOTX_COG_DENIED && aotx_retain_store() == before,
            "an appraisal that expires at the resulting cut refuses the whole automatic input");
        aotx_context_no_queue(n); auto now = d.bindings(n);
        for (unsigned i = 0; i < n; ++i) aotx_check(!memcmp(&bound[i], &now[i], sizeof(now[i])), "expiry refusal preserves exact bindings");
    }
}
int main(int argc, char **argv) {
    bool bounded = argc == 2 && !strcmp(argv[1], "--recovery");
    if (argc > 1 && !bounded) return 2;
    for (unsigned n : {1u, 64u}) {
        aotx_context_live_roundtrip(n, 0, false);
        if (!bounded) { aotx_context_live_roundtrip(n, 1, false); aotx_context_live_roundtrip(n, 2, false); aotx_context_live_roundtrip(n, 0, true); }
    }
    printf("context live: %u checks, %u failed\n", aotx_checks, aotx_failures);
    return aotx_failures || aotx_checks < 1000 ? 1 : 0;
}
