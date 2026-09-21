/* Purpose: Verify complete source spans, UTF-8 transport and inferred payload imports.
 * Owns: Independently encoded quotes and damaged source and provenance controls.
 * Launch shape: Distinct N=1 and N=64 requests through the real CUDA consumers.
 * Lifetime: Source admission, checkpoint import and malformed batch refusal. */
#include "intake_fixture.h"

static aotx_fixture aotx_intake_image(const aotx_bytes &bytes) {
    auto s = (const aotx_cognitive_store *)bytes.data(); aotx_fixture f;
    for (unsigned i = 0; i < s->count; ++i) {
        aotx_row r; memcpy(r.data(), s->objects[i], r.size());
        auto p = s->payload + aotx_get(r.data() + AOTX_CO_OFFSET);
        f.add(r, aotx_bytes(p, p + aotx_get(r.data() + AOTX_CO_BYTES)));
    }
    return f;
}
static void aotx_intake_span_cases(unsigned n) {
    aotx_fixture packed;
    {
        aotx_intake_device d(n); aotx_fixture empty;
        d.send(aotx_live_load_bytes(empty.wire(false, 0)), 1); d.send(aotx_intake_bind(n), 3);
        auto input = aotx_intake_query(n, 0, 1); std::vector<std::string> replies;
        for (unsigned i = 0; i < n; ++i) {
            auto q = input.data() + 128 + i * AOTX_LIVE_QUERY_ROW;
            std::string quote = "Ren\xc3\xa9" + std::to_string(i) + " may not cook \xf0\x9f\x8d\xb2.\n\"Check\" \\ end";
            unsigned start = 207;
            while ((128 + i * AOTX_LIVE_QUERY_ROW + 4640 + start + 3) % AOTX_LIVE_DATA != 159) ++start;
            std::string source(start, ' '); source += quote;
            memset(q + 4640, 0, AOTX_RECALL_TEXT); memcpy(q + 4640, source.data(), source.size()); aotx_put(q + 148, source.size(), 4);
            replies.push_back(" [ [ 3 , \"Ren\\u00e9" + std::to_string(i) +
                " may not cook \\ud83c\\udf72.\\n\\\"Check\\\" \\\\ end\", 0 ] ] \n");
        }
        d.intake(input, replies);
        aotx_check(!d.state().status, "split UTF-8 characters and JSON escapes retain complete source spans");
        if (d.state().status) return;
        auto bytes = aotx_retain_store(); auto s = (const aotx_cognitive_store *)bytes.data();
        aotx_check(s->count == 4 * n, "each source produces exactly one inferred assertion");
        for (unsigned i = 0; i < n; ++i) {
            auto r = s->objects[3 * n + i], p = s->payload + aotx_get(r + AOTX_CO_OFFSET);
            auto q = input.data() + 128 + i * AOTX_LIVE_QUERY_ROW;
            unsigned start = aotx_get(p + 20, 4), length = aotx_get(p + 12, 4);
            aotx_check(start >= 207 && start + length == aotx_get(q + 148, 4) &&
                !memcmp(p + 96, q + 4640 + start, length), "byte offsets preserve uncertainty negation and all UTF-8 bytes");
        }
        packed = aotx_intake_image(bytes);
    }
    for (unsigned bad = 0; bad < 12; ++bad) {
        aotx_device d; auto f = packed; unsigned last = f.rows.size() - 1;
        auto &r = f.rows[last]; auto &p = f.payloads[last];
        if (bad == 1) p[20] ^= 1;
        if (bad == 2) p[96] ^= 1;
        if (bad == 3) aotx_put(r.data() + AOTX_CO_SOURCE_VERSION, 2);
        if (bad == 4) p[88] = 1;
        if (bad == 5) aotx_put(p.data() + 16, 2, 4);
        if (bad == 6) aotx_id(r.data() + AOTX_CO_SUBJECT, 44);
        if (bad == 7) memset(p.data() + 24, 0, 32);
        if (bad == 8) memset(p.data() + 56, 0, 32);
        if (bad == 9) aotx_put(r.data() + AOTX_CO_SOURCE_KIND, AOTX_COG_REPORTED, 4);
        if (bad == 10) aotx_put(r.data() + AOTX_CO_SCOPE, AOTX_COG_INSTANCE, 4);
        if (bad == 11) aotx_put(p.data() + 20, UINT32_MAX, 4);
        auto result = d.load(f.wire(false, f.rows.size()));
        aotx_check(bad ? result.status && !result.applied : !result.status,
            "import validates source extent UTF-8 provenance model and visibility");
    }
    for (unsigned bad = 0; bad < 4; ++bad) {
        aotx_intake_device d(n); aotx_fixture empty;
        d.send(aotx_live_load_bytes(empty.wire(false, 0)), 1); d.send(aotx_intake_bind(n), 3);
        auto input = aotx_intake_query(n, 0, 1); auto replies = aotx_intake_initial(n);
        auto q = input.data() + 128 + (n - 1) * AOTX_LIVE_QUERY_ROW;
        std::string repeated = "Iris Iris";
        if (!bad) { memset(q + 4640, 0, 2048); memcpy(q + 4640, repeated.data(), repeated.size()); aotx_put(q + 148, repeated.size(), 4); replies[n - 1] = "[[1,\"Iris\",0]]"; }
        auto parts = aotx_live_parts(input, 4, d.next_id++);
        if (bad == 1) parts.erase(parts.begin() + 4);
        if (bad == 2) parts.insert(parts.begin() + 4, parts[3]);
        if (bad == 3) std::swap(parts[3], parts[4]);
        auto before = aotx_retain_store();
        if (!bad) d.intake(input, replies); else d.process(parts);
        aotx_check((bad ? d.state().refused != 0 : d.state().status != 0) && aotx_retain_store() == before, "ambiguous spans and damaged transport publish no source or inference");
        for (auto &b : d.bindings(n)) aotx_check(!b.ordinal, "failed span admission leaves every conversation unchanged");
    }
}
int main(void) {
    int cards = 0; if (cudaGetDeviceCount(&cards) != cudaSuccess || !cards) return 77;
    for (unsigned n : {1u, 64u}) aotx_intake_span_cases(n);
    printf("inferred source spans: %u checks, %u failures\n", aotx_checks, aotx_failures);
    return aotx_failures ? 1 : 0;
}
