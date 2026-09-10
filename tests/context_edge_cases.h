/* Purpose: Check appraisal boundaries, shared event support and full required sets.
 * Owns: Independent source, access, budget and capacity expectations.
 * Launch shape: One and 64 distinct query rows.
 * Lifetime: One contextual memory test process. */
#ifndef AOTX_TEST_CONTEXT_EDGE_CASES_H
#define AOTX_TEST_CONTEXT_EDGE_CASES_H

static void aotx_context_edges(unsigned n) {
    auto f = aotx_context_corpus(n); auto q = aotx_context_queries(n, f.rows.size()); aotx_recall_device d;
    aotx_check(!d.load(f.wire(false, f.rows.size())).status, "budget boundary setup");
    auto full = d.search(q, n);
    for (unsigned i = 0; i < n; ++i) aotx_put(aotx_query_at(q, i) + 136,
        full[i].context_bytes - 8 - aotx_get(aotx_query_at(q, i) + 148, 4), 4);
    aotx_context_expected(d.search(q, n), n);
    for (unsigned i = 0; i < n; ++i) {
        auto p = aotx_query_at(q, i); aotx_put(p + 136, aotx_get(p + 136, 4) - 1, 4);
    }
    auto rows = d.search(q, n);
    for (unsigned i = 0; i < n; ++i) aotx_check(!rows[i].status &&
        !aotx_context_has(rows[i], aotx_context_id(i, 9)) && !aotx_context_has(rows[i], aotx_context_id(i, 7)),
        "one byte below the complete pair budget admits neither member of that pair");
    q = aotx_context_queries(n, f.rows.size());
    for (unsigned i = 0; i < n; ++i) aotx_put(aotx_query_at(q, i) + 132, 3, 4);
    rows = d.search(q, n);
    for (unsigned i = 0; i < n; ++i) aotx_check(!rows[i].status && rows[i].count == 2,
        "one spare object slot cannot split an appraisal pair");
    for (unsigned scope = 0; scope < 2; ++scope) {
        f = aotx_context_corpus(n, scope); aotx_check(!d.load(f.wire(false, f.rows.size())).status, "access boundary setup");
        q = aotx_context_queries(n, f.rows.size(), scope);
        for (unsigned i = 0; i < n; ++i) aotx_id(aotx_query_at(q, i) + (scope ? 32 : 16), 80000 + i);
        rows = d.search(q, n);
        for (unsigned i = 0; i < n; ++i) aotx_check(!rows[i].status && !rows[i].count, "task matches never override private or room access");
    }
}
static void aotx_context_shared_source(unsigned n) {
    auto f = aotx_context_corpus(n);
    for (unsigned i = 0; i < n; ++i) {
        for (unsigned part : {6u, 7u}) {
            aotx_put(f.rows[i * 12 + part].data() + AOTX_CO_KIND, AOTX_COG_WORKING, 2);
            f.payloads[i * 12 + part] = aotx_memory_text("working episode " + std::to_string(i) + " part " + std::to_string(part));
        }
        aotx_context_source(f.rows[i * 12 + 9], i, 3);
    }
    aotx_recall_device d; aotx_check(!d.load(f.wire(false, f.rows.size())).status, "shared event appraisal setup");
    auto q = aotx_context_queries(n, f.rows.size(), 0, 2);
    for (unsigned i = 0; i < n; ++i) aotx_put(aotx_query_at(q, i) + 132, 3, 4);
    auto rows = d.search(q, n);
    for (unsigned i = 0; i < n; ++i) aotx_check(!rows[i].status && rows[i].count == 3 &&
        aotx_selected(rows[i], 0) == aotx_context_id(i, 6) && aotx_selected(rows[i], 1) == aotx_context_id(i, 9) &&
        aotx_selected(rows[i], 2) == aotx_context_id(i, 7) && rows[i].reason[2] == AOTX_RECALL_SIGNIFICANT,
        "one immutable event appraisal can support two working memories without a duplicate selection");
    aotx_check(!d.record(q, n, true).status, "shared-source selection records");
    auto saved = d.checkpoint(); aotx_recall_device restored;
    aotx_check(!restored.load(saved).status, "shared-source selection restores");
    auto replay = restored.search(restored.saved_queries(n), n, true);
    for (unsigned i = 0; i < n; ++i) aotx_check(!replay[i].status && !replay[i].searches &&
        aotx_context(replay[i]) == aotx_context(rows[i]), "nonadjacent shared appraisal support replays exactly");
    for (unsigned i = 0; i < n; ++i) {
        aotx_put(f.payloads[i * 12 + 9].data() + 4, AOTX_COG_UNKNOWN, 4);
        aotx_put(f.payloads[i * 12 + 9].data() + 8, AOTX_COG_UNKNOWN, 4);
        aotx_put(f.payloads[i * 12 + 11].data() + 4, 0, 4); aotx_put(f.payloads[i * 12 + 11].data() + 8, 0, 4);
    }
    aotx_check(!d.load(f.wire(false, f.rows.size())).status, "unknown and zero appraisal setup"); rows = d.search(q, n);
    for (unsigned i = 0; i < n; ++i) aotx_check(!rows[i].status && rows[i].count == 2 && rows[i].reason[0] == 3 && rows[i].reason[1] == 3,
        "unknown and known zero values create no appraisal boost or invented confidence");
}
static void aotx_context_required_capacity(unsigned n) {
    auto f = aotx_context_corpus(n);
    for (unsigned i = 0; i < n; ++i) for (unsigned part = 32; part < 42; ++part) {
        auto r = aotx_memory_row(i, AOTX_COG_CUE, aotx_context_id(i, part), f.rows.size() + 1);
        aotx_context_source(r, i, 3); f.add(r, aotx_context_text(i, true, "required cue"));
    }
    aotx_recall_device d; aotx_check(!d.load(f.wire(false, f.rows.size())).status, "large required-set setup");
    auto q = aotx_context_queries(n, f.rows.size(), 0, 1);
    for (unsigned i = 0; i < n; ++i) aotx_put(aotx_query_at(q, i) + 132, 12, 4);
    auto rows = d.search(q, n);
    for (unsigned i = 0; i < n; ++i) aotx_check(!rows[i].status && rows[i].count == 12 &&
        aotx_context_has(rows[i], aotx_context_id(i, 41)), "all twelve applicable obligations survive without a small cue table");
    for (unsigned i = 0; i < n; ++i) aotx_put(aotx_query_at(q, i) + 132, 11, 4);
    aotx_status_rows(d.search(q, n), AOTX_COG_CAPACITY, "the twelfth required cue causes complete refusal at eleven slots");
}
#endif
