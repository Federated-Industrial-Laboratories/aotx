/* Purpose: Verify exact GPU retention, bounded working sets and atomic replay.
 * Owns: Independent source, scope, pressure and malformed-result expectations.
 * Launch shape: Distinct batches at N=1 and N=64 through live graph nodes.
 * Lifetime: One bounded device test process without model weights. */
#include "retain_vector.h"
#include "retain_supersession.h"

static void aotx_retain_roundtrip(unsigned n, unsigned scope, unsigned width, unsigned length) {
    aotx_live_records load, bind, query, retained, next;
    aotx_bytes expected_store;
    std::vector<aotx_live_binding> expected_bindings;
    std::vector<std::string> prompts;
    {
        aotx_live_device d(n); aotx_fixture empty;
        load = d.send(aotx_live_load_bytes(empty.wire(false, 0)), 1);
        bind = d.send(aotx_live_binding_bytes(n, 0, scope), 3);
        auto q = aotx_retain_query(n, 0, 1, false, scope, width, length);
        query = d.send(q, 4); aotx_check(!d.state().status, "accepted source query"); d.idle(n);
        auto request = aotx_retain_bytes(n, 0, 1);
        auto before = aotx_retain_store(); auto search = d.state().searches;
        auto parts = aotx_live_parts(request, 8, d.next_id++);
        retained = d.process(parts, false, false);
        if (n == 64) {
            aotx_check(d.state().phase == AOTX_LIVE_WRITE, "large mutation waits for all result fragments");
            aotx_check(aotx_retain_store() == before, "store remains unchanged before full record publication");
            for (const auto &b : d.bindings(n)) aotx_check(!b.focus_count, "no early focus publication");
            auto more = d.process({}); retained.insert(retained.end(), more.begin(), more.end());
        }
        aotx_check(!d.state().status && d.state().searches == search, "retention uses no recall search");
        auto result = aotx_retain_result(retained);
        aotx_check(result.size() >= 64 && !memcmp(result.data(), "AOTXRCH1", 8), "typed retained result magic");
        aotx_check(aotx_get(result.data() + 8, 4) == n && !aotx_get(result.data() + 44, 4), "all retained rows succeed");
        expected_store = aotx_retain_store(); auto s = (const aotx_cognitive_store *)expected_store.data();
        aotx_check(s->count == 3 * n && s->sequence == 3 * n, "three typed immutable objects per input");
        for (unsigned i = 0; i < n; ++i) {
            auto raw = q.data() + 128 + i * AOTX_LIVE_QUERY_ROW;
            auto out = result.data() + 64 + i * 384;
            aotx_check(!memcmp(out, request.data() + 64 + i * 160, 160), "result keeps exact metadata");
            aotx_check(aotx_get(out + 160, 4) == 1 && aotx_get(out + 192) == 300064 + i,
                "result records the exact working reference");
            for (unsigned k = 0; k < 3; ++k) {
                const auto r = s->objects[i * 3 + k]; auto p = s->payload + aotx_get(r + AOTX_CO_OFFSET);
                aotx_check(aotx_get(r + AOTX_CO_KIND, 2) == (k == 0 ? 1u : k == 1 ? 9u : 7u), "object kinds retain provenance");
                aotx_check(!memcmp(r + AOTX_CO_OWNER, raw + 16, 16) && aotx_get(r + AOTX_CO_SCOPE, 4) == scope,
                    "object authority comes from accepted binding");
                aotx_check(!aotx_get(r + AOTX_CO_EVIDENCE, 4) && aotx_get(r + AOTX_CO_IMPORTANCE, 4) == 500000 + i,
                    "importance never becomes evidence");
                aotx_check(aotx_get(r + AOTX_CO_SOURCE_KIND, 4) == (k == 1 ? 4u : 3u), "reported text and inferred vector remain distinct");
                if (k) aotx_check(!memcmp(r + AOTX_CO_SOURCE, raw, 16) && aotx_get(r + AOTX_CO_SOURCE_VERSION) == 1,
                    "derived records name the exact reported event");
                if (k == 1) {
                    aotx_check(!memcmp(p, "AOTXVEC2", 8) && aotx_get(p + 8, 4) == 2, "source-referenced vector schema");
                    aotx_check(!memcmp(p + 24, raw + 64, 64) && !memcmp(p + 88, raw, 16), "vector space and source identity are exact");
                    aotx_check(!memcmp(p + 128, raw + 160, width * 4), "retained vector bytes equal accepted vector");
                } else aotx_check(aotx_get(p + 12, 4) == aotx_get(raw + 148, 4) &&
                    !memcmp(p + 32, raw + 4640, aotx_get(raw + 148, 4)), "retained text is exact input");
            }
        }
        auto next_query = aotx_retain_query(n, 3 * n, 2, true, scope, width);
        for (unsigned i = 0; i < n; ++i) aotx_put(next_query.data() + 128 + i * AOTX_LIVE_QUERY_ROW + 132, 1, 4);
        next = d.send(next_query, 4);
        aotx_check(!d.state().status, "GPU working set feeds the next query");
        prompts = d.prompt(n); expected_bindings = d.bindings(n);
        for (unsigned i = 0; i < n; ++i) {
            const auto &b = expected_bindings[i];
            aotx_check(b.focus_count == 1 && b.choice.count == 1 && aotx_selected(b.choice, 0) == 300064 + i,
                "exact working memory selected without caller references");
            aotx_check(prompts[i].find("[memory id=") != std::string::npos && prompts[i].find("AUDIT_OLD") == std::string::npos,
                "wrapped prompt uses stored memory without audit history");
        }
    }
    {
        aotx_live_device d(n); d.process(load, true); d.process(bind, true); d.process(query, true); d.idle(n);
        d.process(retained, true);
        aotx_check(!d.state().fatal && aotx_retain_store() == expected_store, "restore reproduces every store byte");
        d.process(next, true); auto rows = d.bindings(n); auto actual = d.prompt(n);
        for (unsigned i = 0; i < n; ++i) {
            aotx_check(rows[i].focus_count == expected_bindings[i].focus_count &&
                !memcmp(rows[i].focus, expected_bindings[i].focus, sizeof(rows[i].focus)), "exact working set survives replay");
            aotx_check(actual[i] == prompts[i], "retained source produces exact restored prompt");
        }
        aotx_check(!d.state().fatal && !d.state().searches, "replay performs no recall search");
    }
    for (unsigned mode = 0; mode < 6; ++mode) {
        aotx_live_device d(n); d.process(load, true); d.process(bind, true); d.process(query, true); d.idle(n);
        auto before = aotx_retain_store(); auto bad = retained;
        if (mode == 0) aotx_retain_mutate(bad, 64 + (n - 1) * 384 + 192);
        if (mode == 1) aotx_retain_mutate(bad, 64 + n * 384 + 128 + 96);
        if (mode == 2) aotx_retain_mutate(bad, aotx_retain_result(retained).size() - 1);
        if (mode == 3) bad.pop_back();
        if (mode == 4) bad.erase(std::remove_if(bad.begin(), bad.end(), [](const auto &r) { return aotx_get(r.data() + 68, 4) == 9; }), bad.end());
        if (mode == 5) {
            d.process(retained, true); before = aotx_retain_store();
            bad.erase(std::remove_if(bad.begin(), bad.end(), [](const auto &r) { return aotx_get(r.data() + 68, 4) != 9; }), bad.end());
        }
        d.process(bad, true);
        unsigned *ok; AOTX_CUDA(cudaMallocManaged(&ok, sizeof(*ok)));
        aotx_live_test_end<<<1,1>>>(ok); AOTX_CUDA(cudaDeviceSynchronize());
        aotx_check(!*ok && d.state().fatal, "altered, missing, partial and repeated results fail restore"); cudaFree(ok);
        aotx_check(aotx_retain_store() == before, "failed replay cannot publish a partial mutation");
    }
}

static void aotx_retain_refusals(unsigned n) {
    aotx_live_device d(n); aotx_retain_open(d, n);
    d.send(aotx_retain_query(n, 0, 1), 4);
    auto good = aotx_retain_bytes(n, 0, 1), before = aotx_retain_store();
    d.send(good, 8); aotx_check(d.state().status == 11, "retention refuses an active input"); d.idle(n);
    const unsigned expected[] = {7,6,11,3,1,10,1,3,1,3};
    for (unsigned mode = 0; mode < 10; ++mode) {
        auto bad = good; auto r = bad.data() + 64 + (n - 1) * 160;
        if (mode == 0) r[32] ^= 1;
        if (mode == 1) aotx_put(r + 24, 2);
        if (mode == 2) r[8] ^= 1;
        if (mode == 3) memcpy(r + 64, r + 48, 16);
        if (mode == 4) aotx_put(r + 128, 1000001, 4);
        if (mode == 5) aotx_put(r + 136, n * 3);
        if (mode == 6) r[159] = 1;
        if (mode == 7) aotx_put(r + 96, 1);
        if (mode == 8) aotx_put(r + 4, 2, 4);
        if (mode == 9) memset(r + 48, 0, 16);
        auto records = d.send(bad, 8);
        if (d.state().status != expected[mode]) fprintf(stderr, "retention n=%u case=%u expected=%u actual=%u\n", n, mode, expected[mode], d.state().status);
        aotx_check(d.state().status == expected[mode], "bad last metadata row refuses the batch");
        aotx_check(aotx_retain_store() == before, "metadata refusal leaves every store byte unchanged");
        for (const auto &b : d.bindings(n)) aotx_check(!b.focus_count && b.ordinal == 1, "metadata refusal preserves bindings");
        auto result = aotx_retain_result(records);
        aotx_check(result.size() == 64 && aotx_get(result.data() + 44, 4) == expected[mode], "complete refusal is recorded");
    }
    d.send(good, 8); aotx_check(!d.state().status, "valid input succeeds after refusals");
    aotx_put(good.data() + 32, 3 * n); d.send(good, 8);
    aotx_check(d.state().status == (n * 6 > AOTX_COG_OBJECTS ? 2u : 5u), "same accepted event cannot be retained twice");
}

int main(int argc, char **argv) {
    bool small = argc == 2 && !strcmp(argv[1], "--bounded");
    bool replacement = argc == 2 && !strcmp(argv[1], "--supersession");
    for (unsigned n : {1u, 64u}) {
        aotx_retain_supersession(n);
        if (replacement) continue;
        aotx_retain_roundtrip(n, 0, small ? 3 : 1024, small ? 0 : 2048);
        aotx_retain_refusals(n);
        aotx_retain_pressure(n);
        aotx_retain_vectors(n);
        if (!small) for (unsigned scope : {1u, 2u}) aotx_retain_roundtrip(n, scope, 3, 0);
    }
    if (!replacement) aotx_retain_turnover();
    printf("retain memory: %u checks, %u failures\n", aotx_checks, aotx_failures);
    return aotx_failures ? 1 : 0;
}
