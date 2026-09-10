/* Purpose: Check consequential recall, appraisal pairing and current corrections.
 * Owns: Independent ranking, scope, pressure and replay assertions.
 * Launch shape: Distinct batches at N=1 and N=64 through real device kernels.
 * Lifetime: One test process; no language model is loaded. */
#include "context_fixture.h"
#include "cognitive/recall_context.cuh"

__global__ void aotx_context_selection_test(const aotx_cognitive_store *s, const unsigned char *queries,
    aotx_recall_result *rows, unsigned n) {
    unsigned i = threadIdx.x; if (i >= n) return;
    uint32_t status = aotx_recall_render(s, queries + 64 + i * AOTX_RECALL_QUERY, rows + i);
    if (status) aotx_recall_refuse(rows + i, status);
}
static std::vector<aotx_recall_result> aotx_context_validate(aotx_recall_device &d, std::vector<aotx_recall_result> rows) {
    AOTX_CUDA(cudaMemcpy(d.rows, rows.data(), rows.size() * sizeof(rows[0]), cudaMemcpyHostToDevice));
    aotx_context_selection_test<<<1,64>>>(d.live, d.requests, d.rows, rows.size()); AOTX_CUDA(cudaGetLastError());
    AOTX_CUDA(cudaMemcpy(rows.data(), d.rows, rows.size() * sizeof(rows[0]), cudaMemcpyDeviceToHost)); return rows;
}
static void aotx_context_ranking(unsigned n, unsigned scope) {
    auto f = aotx_context_corpus(n, scope); auto q = aotx_context_queries(n, f.rows.size(), scope);
    aotx_recall_device d; aotx_check(!d.load(f.wire(false, f.rows.size())).status, "context corpus admission");
    auto before = d.checkpoint(); auto rows = d.search(q, n); aotx_context_expected(rows, n);
    aotx_check(d.checkpoint() == before, "recall and appraisal coupling make no memory writes");
    for (unsigned i = 0; i < n; ++i) {
        auto p = aotx_query_at(q, i);
        double score[2]; float query[3]; memcpy(query, p + 160, sizeof(query));
        for (unsigned j = 0; j < 2; ++j) {
            float v[3]; memcpy(v, f.payloads[i * 12 + j].data() + 128, sizeof(v));
            double dot = 0, a = 0, b = 0;
            for (unsigned k = 0; k < 3; ++k) { dot += (double)v[k] * query[k]; a += (double)v[k] * v[k]; b += (double)query[k] * query[k]; }
            score[j] = dot / (std::sqrt(a) * std::sqrt(b));
        }
        aotx_check(score[0] > score[1] && score[1] + 0.8 * (900000 - i) / 1000000.0 > score[0] + 0.08,
            "independent arithmetic establishes the nontrivial ranking reversal");
        aotx_put(p + AOTX_RECALL_EXTENSION + 40, 0, 4);
    }
    rows = d.search(q, n); aotx_status_rows(rows, 0, "zero boost retains the semantic path");
    for (unsigned i = 0; i < n; ++i) {
        aotx_check(rows[i].count == 4 && aotx_selected(rows[i], 2) == aotx_context_id(i, 6) &&
            aotx_selected(rows[i], 3) == aotx_context_id(i, 7), "zero boost has no appraisal dependency or ranking effect");
        aotx_context_extend(aotx_query_at(q, i), i); aotx_put(aotx_query_at(q, i) + AOTX_RECALL_EXTENSION + 36, 990000, 4);
    }
    rows = d.search(q, n); aotx_status_rows(rows, 0, "semantic floor is independent of appraisal intensity");
    for (unsigned i = 0; i < n; ++i) {
        aotx_check(rows[i].count == 4 && aotx_context_has(rows[i], aotx_context_id(i, 6)) &&
            !aotx_context_has(rows[i], aotx_context_id(i, 7)), "strong but insufficiently relevant memory cannot enter");
        aotx_context_extend(aotx_query_at(q, i), i, 3, true);
    }
    rows = d.search(q, n); aotx_status_rows(rows, 0, "unknown participant can use the generic cue");
    for (unsigned i = 0; i < n; ++i) {
        aotx_check(rows[i].count == 1 && aotx_selected(rows[i], 0) == aotx_context_id(i, 5),
            "new participant does not inherit a person's claims or episodes");
        aotx_put(aotx_query_at(q, i) + AOTX_RECALL_EXTENSION + 32, 0, 4);
        memset(aotx_query_at(q, i) + AOTX_RECALL_EXTENSION + 48, 0, 16);
    }
    rows = d.search(q, n);
    for (unsigned i = 0; i < n; ++i) aotx_check(!rows[i].status && rows[i].count == 1 &&
        aotx_context(rows[i]).find("participants=unknown") != std::string::npos, "absent participants stay explicitly unknown");
    q = aotx_context_queries(n, f.rows.size(), scope);
    for (unsigned i = 0; i < n; ++i) aotx_id(aotx_query_at(q, i) + AOTX_RECALL_EXTENSION + 16, 90000 + i);
    rows = d.search(q, n);
    for (unsigned i = 0; i < n; ++i) aotx_check(!rows[i].status && !rows[i].count, "an unrelated task does not activate a cue");
    q = aotx_context_queries(n, f.rows.size(), scope, 0); rows = d.search(q, n);
    for (unsigned i = 0; i < n; ++i) aotx_check(!rows[i].status && !rows[i].count, "extension OFF does not activate contextual memories");
    q = aotx_context_queries(n, f.rows.size(), scope);
    for (unsigned i = 0; i < n; ++i) { aotx_pin(aotx_query_at(q, i), 1, 0, aotx_context_id(i, 6)); aotx_put(aotx_query_at(q, i) + 132, 5, 4); }
    rows = d.search(q, n);
    for (unsigned i = 0; i < n; ++i) aotx_check(!rows[i].status && rows[i].count == 5 && rows[i].reason[2] == 2 &&
        aotx_selected(rows[i], 0) == aotx_context_id(i, 4) && aotx_selected(rows[i], 2) == aotx_context_id(i, 6),
        "typed obligations precede optional focus without discarding it");
    q = aotx_context_queries(n, f.rows.size(), scope);
    for (unsigned i = 0; i < n; ++i) aotx_put(aotx_query_at(q, i) + 132, 1, 4);
    aotx_status_rows(d.search(q, n), AOTX_COG_CAPACITY, "all obligations must fit the object budget");
    q = aotx_context_queries(n, f.rows.size(), scope);
    for (unsigned i = 0; i < n; ++i) aotx_put(aotx_query_at(q, i) + 136, 64, 4);
    aotx_status_rows(d.search(q, n), AOTX_COG_CAPACITY, "all obligations and context must fit the byte budget");
    aotx_check(d.checkpoint() == before, "pressure and unknown contexts do not strengthen or alter sources");
}
static void aotx_context_bad_queries(unsigned n) {
    auto f = aotx_context_corpus(n); auto q = aotx_context_queries(n, f.rows.size()); aotx_recall_device d;
    aotx_check(!d.load(f.wire(false, f.rows.size())).status, "bad-query setup");
    for (unsigned mode = 0; mode < 14; ++mode) {
        auto bad = q;
        for (unsigned i = 0; i < n; ++i) {
            auto c = aotx_query_at(bad, i) + AOTX_RECALL_EXTENSION;
            if (mode == 0) c[0] ^= 1;
            if (mode == 1) aotx_put(c + 8, 2, 4);
            if (mode == 2) aotx_put(c + 12, 4, 4);
            if (mode == 3) memset(c + 16, 0, 16);
            if (mode == 4) aotx_put(c + 32, 65, 4);
            if (mode == 5) { aotx_put(c + 32, 2, 4); memcpy(c + 64, c + 48, 16); }
            if (mode == 6) memset(c + 48, 0, 16);
            if (mode == 7) c[1072] = 1;
            if (mode == 8) aotx_put(c + 36, 1000001, 4);
            if (mode == 9) aotx_put(c + 40, 1000001, 4);
            if (mode == 10) aotx_put(c + 44, 2, 4);
            if (mode == 11) aotx_put(c + 12, 1, 4);
            if (mode == 12) aotx_put(c + 12, 2, 4);
            if (mode == 13) c[1503] = 1;
        }
        aotx_status_rows(d.search(bad, n), AOTX_COG_FORMAT, "malformed context refuses every affected query");
    }
    for (unsigned i = 0; i < n; ++i) {
        auto c = aotx_query_at(q, i) + AOTX_RECALL_EXTENSION; aotx_put(c + 32, 64, 4);
        for (unsigned j = 0; j < 64; ++j) aotx_id(c + 48 + j * 16, j ? 800000 + i * 64 + j : 3000 + i);
    }
    auto rows = d.search(q, n); aotx_context_expected(rows, n);
}
static void aotx_context_bad_payloads(unsigned n) {
    auto f = aotx_context_corpus(n); aotx_recall_device d;
    aotx_check(!d.load(f.wire(false, f.rows.size())).status, "bad-payload setup");
    for (unsigned mode = 0; mode < 8; ++mode) {
        auto bad = f; auto &p = bad.payloads[(n - 1) * 12 + 4]; auto &r = bad.rows[(n - 1) * 12 + 4];
        if (mode == 0) aotx_put(p.data() + 8, 1, 4);
        if (mode == 1) aotx_put(p.data() + 32, 2, 4);
        if (mode == 2) memset(p.data() + 16, 0, 16);
        if (mode == 3) memset(r.data() + AOTX_CO_SUBJECT, 0, 16);
        if (mode == 4) memset(r.data() + AOTX_CO_SOURCE, 0, 24);
        if (mode == 5) p[64] = 0xc0;
        if (mode == 6) p.resize(8);
        if (mode == 7) p = aotx_context_text(n - 1, true, std::string(2049, 'x'));
        d.rejects(bad.wire(false, bad.rows.size()), false, AOTX_COG_FORMAT, "bad contextual object refuses the complete store");
    }
}
static void aotx_context_recovery(unsigned n) {
    auto f = aotx_context_corpus(n); auto q = aotx_context_queries(n, f.rows.size()); aotx_bytes saved;
    std::vector<aotx_recall_result> original;
    {
        aotx_recall_device d; aotx_check(!d.load(f.wire(false, f.rows.size())).status, "context recovery setup");
        original = d.search(q, n); aotx_context_expected(original, n);
        auto bad = original; auto &last = bad.back();
        aotx_id(last.selection + 16 + 3 * 32, aotx_context_id(n - 1, 10));
        auto checked = aotx_context_validate(d, bad);
        aotx_check(checked.back().status == AOTX_COG_REFERENCE && !checked.back().count, "unrelated recorded appraisal cannot support a selected memory");
        bad = original; auto &missing = bad.back();
        memmove(missing.selection + 16, missing.selection + 48, 3 * 32);
        memset(missing.selection + 16 + 3 * 32, 0, 32); missing.count = 3; aotx_put(missing.selection + 4, 3, 4);
        checked = aotx_context_validate(d, bad);
        aotx_check(checked.back().status == AOTX_COG_REFERENCE && !checked.back().count, "replay cannot omit a required contextual claim");
        d.search(q, n); auto result = d.record(q, n, true);
        aotx_check(!result.status && result.applied == 2 * n, "exact typed selection and context are recorded atomically"); saved = d.checkpoint();
    }
    aotx_recall_device d; aotx_check(!d.load(saved).status, "contextual selection restores on a fresh device");
    auto replay_q = d.saved_queries(n); auto rows = d.search(replay_q, n, true); aotx_context_expected(rows, n);
    for (unsigned i = 0; i < n; ++i) {
        aotx_check(!rows[i].searches && rows[i].cut == original[i].cut && aotx_context(rows[i]) == aotx_context(original[i]),
            "exact context replays without ranking or appraisal computation");
        aotx_check(!memcmp(aotx_query_at(replay_q, i), aotx_query_at(q, i), AOTX_RECALL_QUERY), "complete context descriptor survives record and restore");
    }
    aotx_check(d.checkpoint() == saved, "repeated appraisal recall adds no evidence or state");
}

#include "context_state_cases.h"
#include "context_edge_cases.h"

int main() {
    for (unsigned n : {1u, 64u}) {
        for (unsigned scope = 0; scope < 3; ++scope) aotx_context_ranking(n, scope);
        aotx_context_bad_queries(n); aotx_context_bad_payloads(n); aotx_context_recovery(n);
        aotx_context_corrections(n); aotx_context_duplicates(n); aotx_context_withdrawal(n); aotx_context_intentions(n);
        aotx_context_edges(n); aotx_context_shared_source(n); aotx_context_required_capacity(n);
        printf("context memory N=%u complete\n", n);
    }
    printf("context memory: %u checks, %u failed\n", aotx_checks, aotx_failures);
    return aotx_failures || aotx_checks < 3000 ? 1 : 0;
}
