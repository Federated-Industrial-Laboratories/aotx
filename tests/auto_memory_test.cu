/* Purpose: Verify automatic memory admission, exact sources and atomic recovery.
 * Owns: Distinct input, scope, collision and result-mutation expectations.
 * Launch shape: Real live kernels at N=1 and N=64.
 * Lifetime: One bounded device process without model weights. */
#include "retain_fixture.h"
#include "text_fixture.h"
#include "model/decode_state.cuh"

static std::vector<uint32_t> aotx_auto_queues(unsigned n);

static aotx_bytes aotx_auto_bind(unsigned n, uint64_t cut, unsigned scope = 0, bool mixed = false) {
    auto data = aotx_live_binding_bytes(n, cut, scope);
    for (unsigned i = 0; i < n; ++i) aotx_put(data.data() + 124 + i * 64, !mixed || !(i % 2), 4);
    return data;
}
static unsigned aotx_auto_count(unsigned n, bool mixed) { return mixed ? (n + 1) / 2 : n; }
static void aotx_auto_same(const std::vector<aotx_live_binding> &a, const std::vector<aotx_live_binding> &b) {
    aotx_check(a.size() == b.size(), "binding counts match");
    for (unsigned i = 0; i < a.size(); ++i)
        aotx_check(!memcmp(&a[i], &b[i], sizeof(a[i])), "every binding byte remains unchanged");
}
static void aotx_auto_recovered(std::vector<aotx_live_binding> a, std::vector<aotx_live_binding> b) {
    for (unsigned i = 0; i < a.size(); ++i) {
        aotx_check(a[i].choice.searches == 1 && !b[i].choice.searches, "only live admission performs a search");
        a[i].choice.searches = 0;
    }
    aotx_auto_same(a, b);
}
static void aotx_auto_roundtrip(unsigned n, unsigned scope, bool mixed) {
    unsigned count = aotx_auto_count(n, mixed);
    auto query = aotx_retain_query(n, 0, 1, false, scope, 3, 80);
    aotx_live_records start, first, second;
    aotx_bytes expected_store;
    std::vector<aotx_live_binding> expected_bindings;
    std::vector<std::string> expected_prompt;
    {
        aotx_live_device d(n); aotx_fixture empty;
        start = d.send(aotx_live_load_bytes(empty.wire(false, 0)), 1);
        auto bound = d.send(aotx_auto_bind(n, 0, scope, mixed), 3); start.insert(start.end(), bound.begin(), bound.end());
        aotx_check(!d.state().status, "automatic and manual bindings admitted together");
        auto old = aotx_retain_store(); auto bindings = d.bindings(n);
        first = d.process(aotx_live_parts(query, 4, d.next_id++), false, false);
        aotx_check(d.state().phase == AOTX_LIVE_WRITE, "combined decision spans ticks");
        if (d.state().phase == AOTX_LIVE_WRITE) {
            aotx_check(aotx_retain_store() == old, "no early automatic store publication");
            aotx_check(aotx_auto_queues(n) == std::vector<uint32_t>(n, 0), "no input is queued before complete recording");
            aotx_auto_same(bindings, d.bindings(n));
            auto rest = d.process({}); first.insert(first.end(), rest.begin(), rest.end());
        }
        aotx_check(!d.state().status && d.state().accepted == n + 2, "one accepted input per row");
        auto result = aotx_retain_result(first, 10); auto stored = aotx_retain_store();
        if (d.state().status || result.size() <= 64 + n * AOTX_LIVE_AUTO_ROW) {
            aotx_check(false, "successful admission has complete combined rows and tail"); return;
        }
        auto s = (const aotx_cognitive_store *)stored.data();
        aotx_check(s->count == 3 * count && s->sequence == 3 * count, "only selected bindings retain three objects");
        aotx_check(result.size() > 64 + n * AOTX_LIVE_AUTO_ROW && !memcmp(result.data(), "AOTXACH1", 8), "complete combined result");
        auto current = d.bindings(n); auto prompts = d.prompt(n);
        for (unsigned i = 0, retained = 0; i < n; ++i) {
            bool enabled = !mixed || !(i % 2);
            const auto &b = current[i]; const auto q = query.data() + 128 + i * AOTX_LIVE_QUERY_ROW;
            auto out = result.data() + 64 + i * AOTX_LIVE_AUTO_ROW + AOTX_LIVE_TEXT_CHOICE_ROW;
            aotx_check(b.auto_retain == enabled && b.ordinal == 1 && b.choice.cut == s->sequence,
                "mode ordinal and prompt cut agree after mutation");
            aotx_check(!memcmp(b.query, q, AOTX_RECALL_QUERY), "automatic retention keeps exact prepared query");
            aotx_check(b.focus_count == (unsigned)enabled, "only automatic bindings gain focus");
            aotx_check(prompts[i].find("retained source " + std::to_string(i)) != std::string::npos,
                "own accepted input reaches the actual prompt");
            if (!enabled) {
                aotx_bytes zero(AOTX_LIVE_RETAINED_ROW, 0);
                aotx_check(!memcmp(out, zero.data(), zero.size()), "manual result has no retention metadata"); continue;
            }
            aotx_check(!memcmp(out + 32, q, 16) && !memcmp(out + 48, b.focus[0], 16), "event and retained ID belong to this input");
            for (unsigned k = 0; k < 3; ++k) {
                auto row = s->objects[retained * 3 + k]; auto payload = s->payload + aotx_get(row + AOTX_CO_OFFSET);
                aotx_check(!memcmp(row + AOTX_CO_OWNER, q + 16, 16) && aotx_get(row + AOTX_CO_SCOPE, 4) == scope,
                    "automatic object preserves caller authority");
                aotx_check(!aotx_get(row + AOTX_CO_EVIDENCE, 4) && aotx_get(row + AOTX_CO_IMPORTANCE, 4) == UINT32_MAX &&
                    !aotx_get(row + AOTX_CO_RETENTION, 4), "automatic metadata does not invent evidence or importance");
                if (k == 1) aotx_check(!memcmp(payload + 128, q + 160, 12) && !memcmp(payload + 88, q, 16), "exact vector and source dependency");
                else aotx_check(aotx_get(payload + 12, 4) == 80 && !memcmp(payload + 32, q + 4640, 80), "exact retained input text");
            }
            ++retained;
        }
        d.idle(n);
        auto next = aotx_retain_query(n, s->sequence, 2, true, scope);
        for (unsigned i = 0; i < n; ++i) aotx_put(next.data() + 128 + i * AOTX_LIVE_QUERY_ROW + 132, 1, 4);
        second = d.send(next, 4); aotx_check(!d.state().status, "next automatic input recalls prior focus");
        expected_store = aotx_retain_store(); expected_bindings = d.bindings(n); expected_prompt = d.prompt(n);
        for (unsigned i = 0; i < n; ++i) if (!mixed || !(i % 2))
            aotx_check(expected_bindings[i].choice.count == 1 &&
                !memcmp(expected_bindings[i].choice.selection + 16, current[i].focus[0], 24), "own previous input is the selected memory");
    }
    {
        aotx_live_device d(n); d.process(start, true); d.process(first, true); d.idle(n); d.process(second, true);
        aotx_check(!d.state().fatal && !d.state().searches && d.state().replays == 2 * n, "automatic recovery performs zero searches");
        aotx_check(aotx_retain_store() == expected_store, "automatic recovery preserves every store byte");
        aotx_auto_recovered(expected_bindings, d.bindings(n));
        auto prompt = d.prompt(n); for (unsigned i = 0; i < n; ++i) aotx_check(prompt[i] == expected_prompt[i], "exact prompt recovery");
    }
    if (scope || mixed) return;
    auto original = aotx_retain_result(first, 10);
    for (unsigned mode = 0; mode < 6; ++mode) {
        aotx_live_device d(n); d.process(start, true); auto before = aotx_retain_store(); auto bound = d.bindings(n);
        d.process(aotx_live_parts(query, 4, 3), true);
        auto changed = original; unsigned at = 64 + (n - 1) * AOTX_LIVE_AUTO_ROW;
        if (mode < 4) {
            unsigned offsets[] = {0, 64 + 4640, AOTX_LIVE_TEXT_CHOICE_ROW + 48, AOTX_LIVE_TEXT_CHOICE_ROW + 192};
            changed[at + offsets[mode]] ^= 1;
        } else if (mode == 4) changed.back() ^= 1;
        auto parts = aotx_live_parts(changed, 10, 3);
        if (mode == 5) parts.pop_back();
        d.process(parts, true);
        if (mode == 5) { unsigned *out; AOTX_CUDA(cudaMalloc(&out, 4)); aotx_live_test_end<<<1,1>>>(out); AOTX_CUDA(cudaDeviceSynchronize()); cudaFree(out); }
        aotx_check(d.state().fatal && aotx_retain_store() == before, "altered or partial automatic result never publishes memory");
        aotx_check(aotx_auto_queues(n) == std::vector<uint32_t>(n, 0), "altered or partial result queues no input");
        aotx_auto_same(bound, d.bindings(n));
    }
    {
        aotx_live_device d(n); d.process(start, true); d.process(first, true); d.idle(n);
        auto before = aotx_retain_store(); auto bound = d.bindings(n);
        d.process(aotx_live_parts(original, 10, 3), true);
        aotx_check(d.state().fatal && aotx_retain_store() == before, "duplicate result does not retain twice");
        aotx_check(aotx_auto_queues(n) == std::vector<uint32_t>(n, 0), "duplicate result queues no extra input");
        aotx_auto_same(bound, d.bindings(n));
    }
}
static void aotx_auto_text(unsigned n) {
    aotx_live_records start, records; aotx_bytes expected;
    std::vector<aotx_live_binding> bindings;
    {
        aotx_text_device d(n); auto f = aotx_text_corpus(n); auto cut = 2 * n;
        start = d.send(aotx_live_load_bytes(f.wire(false, cut)), 1);
        auto bind = d.send(aotx_auto_bind(n, cut), 3); start.insert(start.end(), bind.begin(), bind.end());
        records = d.text(aotx_text_input(n, cut));
        aotx_check(!d.state().status && d.encoded() == n && d.state().searches == n, "automatic text encodes and searches once per row");
        expected = aotx_retain_store(); bindings = d.bindings(n); auto s = (const aotx_cognitive_store *)expected.data();
        aotx_check(s->count == 5 * n, "text input creates its own three objects");
        for (unsigned i = 0; i < n; ++i) {
            auto row = s->objects[2 * n + 3 * i + 1]; auto payload = s->payload + aotx_get(row + AOTX_CO_OFFSET);
            aotx_check(!memcmp(payload + 128, bindings[i].query + 160, 12), "automatic text retains its actual prepared vector");
        }
        d.prompt(n);
    }
    {
        aotx_text_device d(n); d.process_text(start, true); d.process_text(records, true);
        aotx_check(!d.state().fatal && !d.encoded() && !d.state().searches, "automatic text recovery does not encode or search");
        aotx_check(aotx_retain_store() == expected, "automatic text mutation recovers exactly");
        aotx_auto_recovered(bindings, d.bindings(n)); d.prompt(n);
    }
}
#include "auto_pressure.h"
#include "auto_decode.h"

int main(int argc, char **argv) {
    bool recovery = argc == 2 && !strcmp(argv[1], "--recovery");
    if (argc > 1 && !recovery) return 2;
    for (unsigned n : {1u, 64u}) {
        aotx_auto_decode(n);
        if (recovery) { aotx_auto_roundtrip(n, 0, false); continue; }
        aotx_auto_roundtrip(n, 0, false); aotx_auto_roundtrip(n, 1, false);
        aotx_auto_roundtrip(n, 2, false); aotx_auto_roundtrip(n, 0, true); aotx_auto_text(n);
        aotx_auto_pressure(n); aotx_auto_admission(n); aotx_auto_turnover(n); aotx_auto_text_failure(n);
    }
    printf("automatic memory: %u checks, %u failed\n", aotx_checks, aotx_failures);
    return aotx_failures || aotx_checks < (recovery ? 10000u : 48000u) ? 1 : 0;
}
