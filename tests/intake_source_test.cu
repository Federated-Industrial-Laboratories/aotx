/* Purpose: Verify exact correction targets outside source-diverse reply context.
 * Owns: Independent target quotes, scope controls and recorded table mutations.
 * Launch shape: N=1 and N=64 through model-output parsing and exact replay.
 * Lifetime: One loaded source corpus and one atomic interpreted input. */
#include "intake_fixture.h"
#include "source_fixture.h"
#include "source_mask_fixture.h"
#include <chrono>

static aotx_bytes aotx_source_live_query(unsigned n, uint64_t cut, bool mixed, unsigned scope, unsigned limit) {
    auto p = aotx_intake_query(n, cut, 1, scope);
    for (unsigned i = 0; i < n; ++i) {
        auto q = p.data() + 128 + i * AOTX_LIVE_QUERY_ROW;
        aotx_float_put(q + 160, 1); aotx_float_put(q + 164, 0); aotx_float_put(q + 168, 0);
        aotx_put(q + 132, mixed && i % 2 ? 16 : limit, 4);
        if (!mixed || !(i % 2)) aotx_source_query(q, 8000 + i);
    }
    return p;
}
static void aotx_source_change(aotx_live_records &records, size_t offset) {
    for (auto &record : records) {
        auto h = (aotx_record_header *)record.data(); auto b = record.data() + 64;
        if (h->type != 33 || aotx_get(b + 4, 4) != 14) continue;
        auto first = aotx_get(b + 28, 4);
        if (offset >= first && offset < first + h->body_len - 32) { b[32 + offset - first] ^= 1; return; }
    }
    aotx_check(false, "mutation reaches an exact semantic decision byte");
}
static void aotx_source_intake(unsigned n, unsigned scope, bool mixed) {
    auto f = aotx_source_corpus(n, scope); aotx_live_records start, records;
    aotx_bytes expected, choice; std::vector<aotx_live_binding> bindings;
    {
        aotx_intake_device d(n);
        start = d.send(aotx_live_load_bytes(f.wire(false, f.rows.size())), 1);
        auto bind = aotx_intake_bind(n, scope); aotx_put(bind.data() + 32, f.rows.size());
        auto b = d.send(bind, 3); start.insert(start.end(), b.begin(), b.end());
        auto query = aotx_source_live_query(n, f.rows.size(), mixed, scope, 1);
        std::vector<std::string> replies;
        for (unsigned i = 0; i < n; ++i) {
            std::string quote = mixed && i % 2 ? "will cook" : "Iris" + std::to_string(i) + " will cook.";
            replies.push_back("[[4,\"" + quote + "\",@]]");
        }
        aotx_intake_targets.resize(n);
        for (unsigned i = 0; i < n; ++i) aotx_id(aotx_intake_targets[i].data(), aotx_source_id(i, 0) + 2);
        records = d.intake(query, replies); expected = aotx_retain_store(); bindings = d.bindings(n);
        aotx_check(!d.state().status && !d.state().fatal, "exact nonselected inferred targets can be corrected");
        if (d.state().status || d.state().fatal) return;
        choice = aotx_retain_result(records, 14);
        aotx_check(choice.size() >= 64 + n * AOTX_LIVE_INTAKE_SOURCE_ROW && !memcmp(choice.data(), "AOTXICH2", 8) &&
            aotx_get(choice.data() + 40, 4) == AOTX_LIVE_INTAKE_SOURCE_ROW, "new framing records the complete target table for every row");
        auto s = (const aotx_cognitive_store *)expected.data();
        for (unsigned i = 0; i < n; ++i) {
            bool modern = !mixed || !(i % 2); auto row = choice.data() + 64 + i * AOTX_LIVE_INTAKE_SOURCE_ROW;
            auto table = row + AOTX_LIVE_INTAKE_ROW; auto pre = row + 64 + AOTX_RECALL_QUERY;
            aotx_check(aotx_get(row + AOTX_LIVE_AUTO_ROW, 4) == (modern ? 2u : 1u), "mixed rows retain their own processor contract");
            if (modern) {
                aotx_check(aotx_get(pre + 4, 4) == 1 && aotx_get(pre + 16) == aotx_source_id(i, 0) + 30,
                    "reply preselection contains the complete working source and no inferred assertion");
                unsigned targets = aotx_get(table + 4, 4);
                aotx_check(aotx_get(table, 4) == 1 && targets > 1 && targets <= 16 &&
                    aotx_get(table + 16) == aotx_source_id(i, 0) + 2,
                    "the separate table includes the exact first eligible assertion in stable source order");
                auto text = aotx_context(bindings[i].choice);
                aotx_check(text.find("source_actor=" + aotx_source_hex(7000 + i * 3)) != std::string::npos,
                    "another actor in the same scope remains correctly attributed");
            } else {
                bool zero = true; for (unsigned j = 0; j < AOTX_INTAKE_TARGETS + AOTX_INTAKE_FIRST; ++j) zero &= table[j] == 0;
                aotx_check(zero, "old rows have no extra target authority");
            }
            auto r = s->objects[f.rows.size() + 3 * n + i];
            aotx_check(!memcmp(r + AOTX_CO_SUPERSEDES, aotx_intake_targets[i].data(), 16),
                "correction resolves its recorded index to the exact old assertion");
            auto event = s->objects[f.rows.size() + 3 * i];
            aotx_check(aotx_get(event + AOTX_CO_SUBJECT) == (modern ? 8000 + i : 1000 + i),
                "current event actor is separate from owner in the new query mode");
        }
    }
    {
        aotx_intake_device d(n); d.process(start, true); d.process(records, true);
        aotx_check(!d.state().fatal && !d.state().searches && aotx_retain_store() == expected,
            "source choice replay validates the target table and restores every state byte without search");
        auto actual = d.bindings(n);
        for (unsigned i = 0; i < n; ++i) { bindings[i].choice.searches = 0;
            aotx_check(!memcmp(&actual[i], &bindings[i], sizeof(actual[i])), "source choice replay restores exact prompt bindings"); }
    }
    if (!mixed) for (unsigned mode = 0; mode < 5; ++mode) {
        aotx_intake_device d(n); d.process(start, true); auto before = aotx_retain_store(); auto bad = records;
        size_t offset = 64 + (n - 1) * AOTX_LIVE_INTAKE_SOURCE_ROW +
            (mode == 0 ? AOTX_LIVE_INTAKE_ROW + 16 : mode == 1 ? AOTX_LIVE_INTAKE_ROW + 48 : mode == 2 ? AOTX_LIVE_AUTO_ROW + 40 :
             mode == 3 ? AOTX_LIVE_AUTO_ROW + 76 : AOTX_LIVE_AUTO_ROW + 8);
        aotx_source_change(bad, offset); d.process(bad, true);
        aotx_check(d.state().fatal && aotx_retain_store() == before,
            "a changed target, unused target, processor, model role or model digest refuses the entire replay batch");
    }
}
static void aotx_source_target_bounds(unsigned n, unsigned mode) {
    auto f = aotx_source_corpus(n);
    for (auto &r : f.rows) {
        if (mode == 2 && aotx_get(r.data() + AOTX_CO_KIND, 2) == AOTX_COG_EVENT)
            memset(r.data() + AOTX_CO_SUBJECT, 0, 16);
        if (aotx_get(r.data() + AOTX_CO_KIND, 2) != AOTX_COG_ASSERTION) continue;
        if (mode == 1) aotx_put(r.data() + AOTX_CO_FLAGS, AOTX_COG_PROTECTED, 4);
    }
    aotx_intake_device d(n); d.send(aotx_live_load_bytes(f.wire(false, f.rows.size())), 1);
    auto b = aotx_intake_bind(n); aotx_put(b.data() + 32, f.rows.size()); d.send(b, 3);
    auto q = aotx_source_live_query(n, f.rows.size(), false, 0, 3);
    if (mode == 2) for (unsigned i = 0; i < n; ++i)
        memset(q.data() + 128 + i * AOTX_LIVE_QUERY_ROW + AOTX_RECALL_ACTOR, 0, 16);
    d.process(aotx_live_parts(q, 4, d.next_id++), false, false);
    aotx_check(d.state().phase == AOTX_INTAKE_RUN, "statement prompt is ready before model output");
    std::vector<std::string> first;
    for (unsigned i = 0; i < n; ++i) first.push_back("[[\"Iris" + std::to_string(i) + " will cook.\",\"statement\"]]");
    aotx_intake_fixture_first_upload(first); aotx_intake_fixture_statements<<<1,64>>>(n); AOTX_CUDA(cudaDeviceSynchronize());
    std::vector<aotx_intake_row> rows(n);
    AOTX_CUDA(cudaMemcpyFromSymbol(rows.data(), aotx_intake, rows.size() * sizeof(rows[0]), offsetof(aotx_intake_state, rows)));
    aotx_say_state say; AOTX_CUDA(cudaMemcpyFromSymbol(&say, aotx_say, sizeof(say)));
    std::vector<std::string> prompts(n);
    for (unsigned i = 0; i < n; ++i) prompts[i].assign((const char *)say.prompt[i], say.slot[i].length);
    for (unsigned i = 0; i < n; ++i) {
        auto &r = rows[i];
        aotx_check(r.target_count <= 16 && r.target_bytes <= 4096, "target reference and rendered byte limits are independent and bounded");
        if (mode == 1) aotx_check(!r.target_count && !r.target_bytes, "protected assertions confer no correction target authority");
        else {
            aotx_check(r.target_count == 16 && r.target_bytes <= r.target_capacity,
                "compact source-labelled targets fill the row bound within the reserved prompt budget");
            for (unsigned group = 0; group < 3; ++group) aotx_check(aotx_get(r.targets + 16 + group * 32) == aotx_source_id(i, group) + 2,
                "every selected source contributes an assertion before repeated source targets");
            aotx_check(prompts[i].find("source_actor=" + (mode == 2 ? std::string("unknown") : aotx_source_hex(8000 + i))) != std::string::npos &&
                prompts[i].find("source_actor=" + (mode == 2 ? std::string("unknown") : aotx_source_hex(7000 + i * 3))) != std::string::npos,
                "the extraction prompt has both current and historical source actors");
        }
    }
}
__global__ void aotx_source_wrap_pressure(void) {
    if (threadIdx.x) return;
    for (unsigned j = 0; j < AOTX_WRAP_SPANS; ++j) aotx_model_wrap[AOTX_MODEL_LANGUAGE].length[j] = 255;
    aotx_model_wrap[AOTX_MODEL_LANGUAGE].prefix_length = 255;
}
static void aotx_source_capacity(unsigned n) {
    auto f = aotx_source_corpus(n);
    for (unsigned j = 0; j < f.rows.size(); ++j) {
        auto &r = f.rows[j];
        if (aotx_get(r.data() + AOTX_CO_KIND, 2) != AOTX_COG_ASSERTION) continue;
        for (unsigned source = 0; source < f.rows.size(); ++source) {
            if (memcmp(r.data() + AOTX_CO_SOURCE, f.rows[source].data() + AOTX_CO_ID, 16)) continue;
            auto &p = f.payloads[j]; const auto &raw = f.payloads[source];
            p.resize(AOTX_INTAKE_PAYLOAD + raw.size() - 32);
            aotx_put(p.data() + 12, raw.size() - 32, 4); aotx_put(p.data() + 20, 0, 4);
            memcpy(p.data() + AOTX_INTAKE_PAYLOAD, raw.data() + 32, raw.size() - 32); break;
        }
    }
    aotx_live_records start, records; aotx_bytes expected;
    {
        aotx_intake_device d(n); start = d.send(aotx_live_load_bytes(f.wire(false, f.rows.size())), 1);
        auto b = aotx_intake_bind(n); aotx_put(b.data() + 32, f.rows.size());
        auto binding = d.send(b, 3); start.insert(start.end(), binding.begin(), binding.end());
        auto q = aotx_source_live_query(n, f.rows.size(), false, 0, 3);
        for (unsigned i = 0; i < n; ++i) {
            auto raw = q.data() + 128 + i * AOTX_LIVE_QUERY_ROW;
            memset(raw + 4640, 'a' + i % 26, AOTX_RECALL_TEXT); aotx_put(raw + 148, AOTX_RECALL_TEXT, 4);
        }
        std::vector<std::string> replies;
        for (unsigned i = 0; i < n; ++i) replies.push_back("[[3,\"" + std::string(AOTX_RECALL_TEXT, 'a' + i % 26) + "\",0]]");
        records = d.intake(q, replies);
        aotx_check(!d.state().status, "full current sources leave a bounded correction table and still fit the extraction prompt");
        if (d.state().status) return;
        expected = aotx_retain_store(); std::vector<aotx_intake_row> rows(n);
        AOTX_CUDA(cudaMemcpyFromSymbol(rows.data(), aotx_intake, rows.size() * sizeof(rows[0]), offsetof(aotx_intake_state, rows)));
        for (const auto &row : rows) aotx_check(row.target_count && row.target_count < 16 &&
            row.target_capacity < 4096 && row.target_bytes <= row.target_capacity,
            "the exact source and fixed frame reduce target capacity before selection");
    }
    {
        aotx_intake_device d(n); d.process(start, true); d.process(records, true);
        aotx_check(!d.state().fatal && aotx_retain_store() == expected, "capacity-dependent target tables replay canonically");
    }
    {
        aotx_intake_device d(n); d.process(start, true); auto before = aotx_retain_store();
        aotx_source_wrap_pressure<<<1,1>>>(); d.process(records, true);
        aotx_check(d.state().fatal && aotx_retain_store() == before, "a changed wrapper that changes target capacity refuses exact replay");
    }
}
static void aotx_source_mask(unsigned n) {
    auto f = aotx_source_corpus(n); aotx_intake_device d(n);
    d.send(aotx_live_load_bytes(f.wire(false, f.rows.size())), 1);
    auto bind = aotx_intake_bind(n); aotx_put(bind.data() + 32, f.rows.size()); d.send(bind, 3);
    auto q = aotx_source_live_query(n, f.rows.size(), false, 0, 1);
    std::vector<std::string> prefixes, first;
    for (unsigned i = 0; i < n; ++i) {
        std::string quote = "Person" + std::to_string(i) + " will not cook lentils tonight.";
        std::string source = "Correction: " + quote + " The earlier cooking plan changed.";
        auto raw = q.data() + 128 + i * AOTX_LIVE_QUERY_ROW;
        memset(raw + 4640, 0, AOTX_RECALL_TEXT); memcpy(raw + 4640, source.data(), source.size());
        aotx_put(raw + 148, source.size(), 4);
        prefixes.push_back("[[4,\"Correction: " + quote + "\",");
        first.push_back("[[\"Correction: " + quote + "\",\"statement\"],"
            "[\"The earlier cooking plan changed.\",\"statement\"]]");
    }
    d.process(aotx_live_parts(q, 4, d.next_id++), false, false);
    aotx_check(d.state().phase == AOTX_INTAKE_RUN, "the target mask uses the live generation lease");
    aotx_intake_fixture_first_upload(first); aotx_intake_fixture_statements<<<1,64>>>(n); AOTX_CUDA(cudaDeviceSynchronize());
    aotx_source_mask_check(prefixes, 16);
}
int main(int argc, char **argv) {
    const char *cases[] = {"private", "room", "mixed", "capacity", "targets", "protected", "unknown", "mask"};
    unsigned only = argc > 1 ? !strcmp(argv[1], "1") ? 1 : !strcmp(argv[1], "64") ? 64 : 0 : 0;
    bool valid = argc <= 3 && (argc == 1 || only);
    if (argc == 3) { bool found = false; for (auto name : cases) found |= !strcmp(argv[2], name); valid &= found; }
    if (!valid) { fprintf(stderr, "usage: aotx_intake_source_test [1|64] [private|room|mixed|capacity|targets|protected|unknown|mask]\n"); return 2; }
    int cards = 0; if (cudaGetDeviceCount(&cards) != cudaSuccess || !cards) return 77;
    for (unsigned n : {1u, 64u}) {
        if (only && only != n) continue;
        for (unsigned which = 0; which < 8; ++which) {
            if (argc == 3 && strcmp(argv[2], cases[which])) continue;
            printf("source interpretation N=%u case=%s start\n", n, cases[which]); fflush(stdout);
            auto start = std::chrono::steady_clock::now();
            if (which < 3) aotx_source_intake(n, which == 1, which == 2);
            else if (which == 3) aotx_source_capacity(n);
            else if (which < 7) aotx_source_target_bounds(n, which - 4);
            else aotx_source_mask(n);
            double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            printf("source interpretation N=%u case=%s: %u checks, %u failures, %.3f seconds\n",
                n, cases[which], aotx_checks, aotx_failures, seconds); fflush(stdout);
        }
        printf("source interpretation N=%u: %u checks, %u failures\n", n, aotx_checks, aotx_failures);
    }
    return aotx_failures ? 1 : 0;
}
