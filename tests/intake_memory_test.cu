/* Purpose: Verify atomic interpretation admission, exact provenance and recorded recovery.
 * Owns: Distinct expected sources, scopes, malformed outputs and mutation controls.
 * Launch shape: N=1 and N=64 through live CUDA consumers.
 * Lifetime: One test process with independent model-response fixtures. */
#include "intake_fixture.h"

static void aotx_intake_roundtrip(unsigned n, unsigned scope, bool mixed) {
    aotx_live_records start, records;
    aotx_bytes expected;
    std::vector<aotx_live_binding> bindings;
    std::vector<std::string> prompts;
    {
        aotx_intake_device d(n); aotx_fixture empty;
        start = d.send(aotx_live_load_bytes(empty.wire(false, 0)), AOTX_LIVE_LOAD);
        auto b = d.send(aotx_intake_bind(n, scope, mixed), AOTX_LIVE_BIND); start.insert(start.end(), b.begin(), b.end());
        aotx_check(!d.state().status, "semantic binding mode is admitted");
        records = d.intake(aotx_intake_query(n, 0, 1, scope), aotx_intake_initial(n));
        auto choice = aotx_retain_result(records, AOTX_INTAKE_CHOICE);
        aotx_check(!d.state().status && choice.size() > 64, "semantic decision is complete");
        if (d.state().status || choice.size() <= 64) return;
        expected = aotx_retain_store(); bindings = d.bindings(n); prompts = d.prompt(n);
        auto *s = (const aotx_cognitive_store *)expected.data();
        unsigned retained = 0, interpreted = 0;
        for (unsigned i = 0; i < n; ++i) {
            unsigned mode = mixed ? i % 3 : 2;
            retained += mode != 0; interpreted += mode == 2 ? 3 : 0;
            aotx_check(bindings[i].ordinal == 1 && bindings[i].auto_retain == mode, "per-binding mode and ordinal are exact");
            auto meta = choice.data() + 64 + i * AOTX_LIVE_INTAKE_ROW + AOTX_LIVE_AUTO_ROW;
            aotx_check(aotx_get(meta + 72, 4) == (mode == 2 ? 3u : 0u), "only selected bindings create interpretations");
            aotx_check(prompts[i].find("Iris" + std::to_string(i) + " will cook.") != std::string::npos, "own source reaches the actual prompt");
            for (unsigned j = 0; j < AOTX_RECALL_SELECTION; ++j)
                aotx_check(meta[AOTX_INTAKE_META + AOTX_INTAKE_REPLY + j] == bindings[i].choice.selection[j], "effective selection is recorded exactly");
        }
        aotx_check(s->count == 3 * retained + interpreted && s->sequence == s->count, "all and only declared objects are published");
        for (unsigned i = 3 * retained; i < s->count; ++i) {
            const auto *r = s->objects[i], *p = s->payload + aotx_get(r + AOTX_CO_OFFSET);
            aotx_check(!memcmp(p, "AOTXMEM3", 8) && aotx_get(r + AOTX_CO_SOURCE_KIND, 4) == AOTX_COG_INFERRED, "interpretation stays inferred");
            aotx_check(!aotx_get(r + AOTX_CO_EVIDENCE, 4) && aotx_get(r + AOTX_CO_IMPORTANCE, 4) == UINT32_MAX, "inference grants no evidence or importance");
            aotx_bytes zero(16, 0); aotx_check(!memcmp(r + AOTX_CO_SUBJECT, zero.data(), 16), "a mention grants no authenticated subject");
        }
    }
    {
        aotx_intake_device d(n); d.process(start, true); d.process(records, true);
        aotx_check(!d.state().fatal && d.state().replays == n && d.state().searches == 0, "recovery uses recorded choices without search");
        aotx_check(aotx_retain_store() == expected, "every recovered store byte matches");
        auto actual = d.bindings(n);
        for (unsigned i = 0; i < n; ++i) {
            bindings[i].choice.searches = 0;
            aotx_check(!memcmp(&bindings[i], &actual[i], sizeof(actual[i])), "every recovered binding byte matches");
        }
        aotx_check(d.prompt(n) == prompts, "recovered prompts match exactly");
        unsigned long long calls = 1;
        AOTX_CUDA(cudaMemcpyFromSymbol(&calls, aotx_intake, sizeof(calls), offsetof(aotx_intake_state, calls)));
        aotx_check(!calls, "recovery starts no interpretation sequence");
    }
    auto choice = aotx_retain_result(records, AOTX_INTAKE_CHOICE);
    for (size_t offset : {size_t(64 + AOTX_LIVE_AUTO_ROW + 8), size_t(64 + AOTX_LIVE_AUTO_ROW + 40),
        size_t(64 + AOTX_LIVE_AUTO_ROW + 72), size_t(64 + AOTX_LIVE_AUTO_ROW + AOTX_INTAKE_META + 6), choice.size() - 1}) {
        aotx_intake_device d(n); d.process(start, true);
        auto q = aotx_intake_query(n, 0, 1, scope); d.process(aotx_live_parts(q, AOTX_LIVE_QUERY, 3), true);
        auto changed = choice; changed[offset] ^= 1;
        d.process(aotx_live_parts(changed, AOTX_INTAKE_CHOICE, 3), true);
        aotx_check(d.state().fatal && !d.state().replays, "changed semantic decisions refuse whole-batch recovery");
        for (auto &b : d.bindings(n)) aotx_check(!b.ordinal, "rejected decisions publish no binding");
    }
}
static void aotx_intake_bad_output(unsigned n) {
    const char *bad[] = {"", "text", "[", "[[0,\"Iris0\",0]]", "[[1,\"absent\",0]]",
        "[[3,\"Iris0 will cook.\",0],]", "[[3,\"Iris0 will cook.\",0,1]]", "[[4,\"Iris0\",1]]",
        "[[1,\"Iris0\",0],[1,\"Iris0\",0]]", "[[1,\"\\ud800\",0]]", "[[1,\"\\u0000\",0]]",
        "[[1,\"Iris0\",4294967296]]", "[[1,\"Iris0\",00]]"};
    for (const char *value : bad) {
        aotx_intake_device d(n); aotx_fixture empty;
        d.send(aotx_live_load_bytes(empty.wire(false, 0)), 1); d.send(aotx_intake_bind(n), 3);
        auto before = aotx_retain_store(); auto replies = aotx_intake_initial(n); std::string failed = value;
        for (size_t at = 0; (at = failed.find("Iris0", at)) != std::string::npos; at += 6)
            failed.replace(at, 5, "Iris" + std::to_string(n - 1));
        replies[n - 1] = failed;
        d.intake(aotx_intake_query(n, 0, 1), replies);
        aotx_check(d.state().status && aotx_retain_store() == before, "bad output leaves every memory byte unchanged");
        for (auto &b : d.bindings(n)) aotx_check(!b.ordinal && !b.focus_count, "bad output leaves all bindings unchanged");
    }
}
int main(void) {
    int cards = 0; if (cudaGetDeviceCount(&cards) != cudaSuccess || !cards) return 77;
    for (unsigned n : {1u, 64u}) {
        for (unsigned scope : {0u, 1u, 2u}) aotx_intake_roundtrip(n, scope, false);
        if (n > 1) aotx_intake_roundtrip(n, 0, true);
        aotx_intake_bad_output(n);
    }
    printf("semantic memory: %u checks, %u failures\n", aotx_checks, aotx_failures);
    return aotx_failures ? 1 : 0;
}
