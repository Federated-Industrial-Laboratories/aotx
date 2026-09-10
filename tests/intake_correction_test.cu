/* Purpose: Verify exact inferred corrections and their authority limits.
 * Owns: Independent target identities, source versions and refusal controls.
 * Launch shape: N=1 and N=64 through combined admission and recorded replay.
 * Lifetime: Original assertions, corrected assertions and an exact recovered store. */
#include "intake_fixture.h"

static aotx_bytes aotx_intake_distinct(unsigned n, uint64_t cut, unsigned turn, unsigned scope) {
    auto p = aotx_intake_query(n, cut, turn, scope);
    for (unsigned i = 0; i < n; ++i) {
        auto q = p.data() + 128 + i * AOTX_LIVE_QUERY_ROW;
        aotx_put(q + 128, 64, 4); memset(q + 160, 0, 4096); aotx_float_put(q + 160 + 4 * i, 1);
    }
    return p;
}
static void aotx_intake_correction(unsigned n, unsigned scope, unsigned bad) {
    aotx_live_records start, first, records;
    aotx_bytes expected, prior;
    std::vector<aotx_live_binding> bindings;
    std::vector<std::array<unsigned char,16>> targets(n);
    {
        aotx_intake_device d(n); aotx_fixture empty;
        start = d.send(aotx_live_load_bytes(empty.wire(false, 0)), 1);
        auto b = d.send(aotx_intake_bind(n, scope), 3); start.insert(start.end(), b.begin(), b.end());
        first = d.intake(aotx_intake_distinct(n, 0, 1, scope), aotx_intake_initial(n));
        aotx_check(!d.state().status, "original inferred assertions are admitted");
        if (d.state().status) return;
        prior = aotx_retain_store(); auto *s = (aotx_cognitive_store *)prior.data();
        for (unsigned i = 0; i < n; ++i) {
            auto r = s->objects[3 * n + 3 * i + 2];
            memcpy(targets[i].data(), r + AOTX_CO_ID, 16);
        }
        d.idle(n); aotx_intake_targets = targets;
        auto q = aotx_intake_distinct(n, s->sequence, 2, scope);
        std::vector<std::string> replies;
        for (unsigned i = 0; i < n; ++i)
            replies.push_back("[[4,\"Iris" + std::to_string(i) + " will not cook.\",@]]");
        unsigned last = n - 1; auto r = s->objects[3 * n + 3 * last + 2];
        auto row = q.data() + 128 + last * AOTX_LIVE_QUERY_ROW;
        if (bad == 1) { aotx_put(row + 140, 1, 4); memcpy(row + 4256, targets[last].data(), 16); aotx_put(row + 4272, 1); }
        if (bad == 2) { aotx_put(row + 144, 1, 4); memcpy(row + 4448, targets[last].data(), 16); aotx_put(row + 4464, 1); }
        if (bad == 3) aotx_put(r + AOTX_CO_FLAGS, AOTX_COG_PROTECTED, 4);
        if (bad == 4) aotx_id(r + AOTX_CO_OWNER, 990000);
        if (bad == 5) aotx_put(r + AOTX_CO_SOURCE_KIND, AOTX_COG_AUTHORED, 4);
        if (bad >= 3) AOTX_CUDA(cudaMemcpyToSymbol(aotx_live_store, s, sizeof(*s)));
        records = d.intake(q, replies); bindings = d.bindings(n); expected = aotx_retain_store();
        if (bad) {
            aotx_check(d.state().status && expected == prior, "a forbidden correction leaves every memory byte unchanged");
            for (auto &binding : bindings) aotx_check(binding.ordinal == 1, "atomic refusal keeps every prior ordinal");
            return;
        }
        aotx_check(!d.state().status, "same-owner optional assertions can be corrected");
        if (d.state().status) return;
        auto *after = (const aotx_cognitive_store *)expected.data();
        aotx_check(after->count == 10 * n && after->sequence == 10 * n, "one correction adds exactly its source vector working and assertion");
        for (unsigned i = 0; i < n; ++i) {
            auto r = after->objects[9 * n + i], p = after->payload + aotx_get(r + AOTX_CO_OFFSET);
            aotx_check(!memcmp(r + AOTX_CO_SUPERSEDES, targets[i].data(), 16) && aotx_get(r + AOTX_CO_SUPER_VERSION) == 1,
                "correction names exactly its own original assertion version");
            aotx_check(aotx_get(p + 16, 4) == 4 && std::string((const char *)p + 96, aotx_get(p + 12, 4)) ==
                "Iris" + std::to_string(i) + " will not cook.", "correction preserves the complete negation span");
            for (unsigned j = 0; j < bindings[i].choice.count; ++j)
                aotx_check(memcmp(bindings[i].choice.selection + 16 + 32 * j, targets[i].data(), 16) != 0,
                    "effective answer context excludes the superseded assertion");
        }
        aotx_check(!memcmp(after->objects, s->objects, 6 * n * AOTX_COG_OBJECT) &&
            !memcmp(after->payload, s->payload, s->bytes), "all original source and assertion bytes remain exact");
    }
    {
        aotx_intake_device d(n); d.process(start, true); d.process(first, true); d.idle(n); d.process(records, true);
        aotx_check(!d.state().fatal && !d.state().searches && aotx_retain_store() == expected,
            "correction replay restores every accepted byte without search");
        auto actual = d.bindings(n);
        for (unsigned i = 0; i < n; ++i) {
            bindings[i].choice.searches = 0;
            aotx_check(!memcmp(&bindings[i], &actual[i], sizeof(actual[i])), "correction replay restores exact effective bindings");
        }
    }
}
int main(void) {
    int cards = 0; if (cudaGetDeviceCount(&cards) != cudaSuccess || !cards) return 77;
    for (unsigned n : {1u, 64u}) {
        for (unsigned scope : {0u, 1u, 2u}) aotx_intake_correction(n, scope, 0);
        for (unsigned bad : {1u, 2u, 3u, 5u}) aotx_intake_correction(n, 0, bad);
        aotx_intake_correction(n, 2, 4);
    }
    printf("inferred corrections: %u checks, %u failures\n", aotx_checks, aotx_failures);
    return aotx_failures ? 1 : 0;
}
