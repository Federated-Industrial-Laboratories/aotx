/* Purpose: Check correction, source withdrawal and completed contextual intentions.
 * Owns: Independent update batches and exact before/after state checks.
 * Launch shape: One and 64 distinct subjects through the public state API.
 * Lifetime: One contextual recall test process. */
#ifndef AOTX_TEST_CONTEXT_STATE_CASES_H
#define AOTX_TEST_CONTEXT_STATE_CASES_H

static void aotx_context_corrections(unsigned n) {
    auto f = aotx_context_corpus(n); auto q = aotx_context_queries(n, f.rows.size()); aotx_recall_device d;
    aotx_check(!d.load(f.wire(false, f.rows.size())).status, "correction setup");
    d.search(q, n); aotx_check(!d.record(q, n, true).status, "old choices exist before correction");
    auto saved = d.checkpoint(); auto old_q = d.saved_queries(n); uint64_t cut = aotx_get(saved.data() + 32);
    aotx_fixture change;
    for (unsigned i = 0; i < n; ++i) {
        auto event = aotx_memory_row(i, AOTX_COG_EVENT, aotx_context_id(i, 16), cut + change.rows.size() + 1);
        change.add(event, aotx_memory_text("correction report " + std::to_string(i)));
        auto claim = aotx_memory_row(i, AOTX_COG_ASSERTION, aotx_context_id(i, 17), cut + change.rows.size() + 1);
        aotx_context_source(claim, i, 16); aotx_id(claim.data() + AOTX_CO_SUPERSEDES, aotx_context_id(i, 4));
        aotx_put(claim.data() + AOTX_CO_SUPER_VERSION, 1);
        change.add(claim, aotx_context_text(i, true, "corrected constraint " + std::to_string(i)));
    }
    aotx_check(!d.load(change.wire(true, cut + 1, aotx_get(saved.data() + 40) + 1), true).status, "new evidence supersedes the exact old claim");
    aotx_put(old_q.data() + 32, cut + 2 * n);
    aotx_status_rows(d.search(old_q, n, true), AOTX_COG_STALE, "recorded obsolete claims cannot become current after correction");
    q = aotx_context_queries(n, cut + 2 * n); auto rows = d.search(q, n); aotx_status_rows(rows, 0, "corrected required set");
    for (unsigned i = 0; i < n; ++i) {
        aotx_check(aotx_context_has(rows[i], aotx_context_id(i, 17)) && !aotx_context_has(rows[i], aotx_context_id(i, 4)),
            "only the corrected claim is current for its original subject");
        aotx_check(aotx_context(rows[i]).find("corrected constraint " + std::to_string(i)) != std::string::npos,
            "corrected source text reaches the context");
    }
    saved = d.checkpoint(); cut = aotx_get(saved.data() + 32); change = {};
    for (unsigned i = 0; i < n; ++i) {
        auto r = aotx_memory_row(i, AOTX_COG_APPRAISAL, aotx_context_id(i, 18), cut + i + 1);
        aotx_context_source(r, i, 7); aotx_id(r.data() + AOTX_CO_SUPERSEDES, aotx_context_id(i, 9));
        aotx_put(r.data() + AOTX_CO_SUPER_VERSION, 1); aotx_put(r.data() + AOTX_CO_SOURCE_KIND, AOTX_COG_INFERRED, 4);
        change.add(r, aotx_context_appraisal(0, 0, i));
    }
    aotx_check(!d.load(change.wire(true, cut + 1, aotx_get(saved.data() + 40) + 1), true).status, "revised appraisal retains source and supersession");
    q = aotx_context_queries(n, cut + n); rows = d.search(q, n); aotx_status_rows(rows, 0, "appraisal correction changes optional recall");
    for (unsigned i = 0; i < n; ++i) aotx_check(aotx_context_has(rows[i], aotx_context_id(i, 6)) &&
        aotx_context_has(rows[i], aotx_context_id(i, 11)) && !aotx_context_has(rows[i], aotx_context_id(i, 9)) &&
        aotx_context_has(rows[i], aotx_context_id(i, 17)), "revised significance changes the episode while the corrected obligation remains");
    auto after = d.checkpoint(); std::string bytes(after.begin(), after.end());
    for (unsigned i = 0; i < n; ++i) aotx_check(bytes.find("reported source " + std::to_string(i)) != std::string::npos &&
        bytes.find("correction report " + std::to_string(i)) != std::string::npos, "original and corrective source events both survive");
}
static void aotx_context_duplicates(unsigned n) {
    auto f = aotx_context_corpus(n); auto q = aotx_context_queries(n, f.rows.size()); aotx_recall_device d;
    aotx_check(!d.load(f.wire(false, f.rows.size())).status, "duplicate assessment setup");
    auto original = d.search(q, n); aotx_fixture more;
    for (unsigned i = 0; i < n; ++i) {
        auto r = aotx_memory_row(i, AOTX_COG_APPRAISAL, aotx_context_id(i, 24), f.rows.size() + i + 1);
        aotx_context_source(r, i, 7); aotx_put(r.data() + AOTX_CO_SOURCE_KIND, AOTX_COG_INFERRED, 4);
        more.add(r, f.payloads[i * 12 + 9]);
    }
    auto bad = more; aotx_put(bad.rows.back().data() + AOTX_CO_SOURCE_KIND, AOTX_COG_REPORTED, 4);
    d.rejects(bad.wire(true, f.rows.size() + 1, 6), true, AOTX_COG_SOURCE, "an appraisal cannot promote an inferred source to a report");
    aotx_check(!d.load(more.wire(true, f.rows.size() + 1, 6), true).status, "distinct duplicate assessment records are admitted");
    aotx_put(q.data() + 32, f.rows.size() + n); auto rows = d.search(q, n); aotx_context_expected(rows, n);
    for (unsigned i = 0; i < n; ++i) aotx_check(aotx_context(rows[i]) == aotx_context(original[i]) &&
        !memcmp(rows[i].selection, original[i].selection, AOTX_RECALL_SELECTION), "duplicate assessments do not accumulate priority or evidence");
    auto stable = d.checkpoint(); d.search(q, n); d.search(q, n);
    aotx_check(d.checkpoint() == stable, "repeated retrieval changes no assessment or source bytes");
}
static void aotx_context_withdrawal(unsigned n) {
    auto f = aotx_context_corpus(n); auto q = aotx_context_queries(n, f.rows.size()); aotx_recall_device d;
    aotx_check(!d.load(f.wire(false, f.rows.size())).status, "source withdrawal setup");
    aotx_fixture change;
    for (unsigned i = 0; i < n; ++i) {
        auto r = f.rows[i * 12 + 3]; aotx_put(r.data() + AOTX_CO_VERSION, 2);
        aotx_put(r.data() + AOTX_CO_UPDATED, f.rows.size() + i + 1); aotx_put(r.data() + AOTX_CO_FLAGS, AOTX_COG_TOMBSTONE, 4);
        change.add(r, {});
    }
    aotx_check(!d.load(change.wire(true, f.rows.size() + 1, 6), true).status, "source withdrawal records an exact tombstone version");
    aotx_put(q.data() + 32, f.rows.size() + n); auto rows = d.search(q, n);
    for (unsigned i = 0; i < n; ++i) {
        aotx_check(!rows[i].status && !rows[i].count, "withdrawn source cannot leak through a claim, cue or appraisal");
        aotx_pin(aotx_query_at(q, i), 0, 0, aotx_context_id(i, 4));
    }
    aotx_status_rows(d.search(q, n), AOTX_COG_DENIED, "an explicit pin cannot bypass withdrawn source access");
}
static void aotx_context_intentions(unsigned n) {
    auto f = aotx_context_corpus(n); auto cut = f.rows.size();
    for (unsigned i = 0; i < n; ++i) {
        auto r = aotx_memory_row(i, AOTX_COG_INTENTION, aotx_context_id(i, 20), f.rows.size() + 1);
        aotx_context_source(r, i, 3); aotx_put(r.data() + AOTX_CO_RETENTION, 2, 4);
        f.add(r, aotx_context_text(i, true, "pending task " + std::to_string(i)));
    }
    aotx_recall_device d; aotx_check(!d.load(f.wire(false, f.rows.size())).status, "pending intention setup");
    auto q = aotx_context_queries(n, f.rows.size());
    for (unsigned i = 0; i < n; ++i) aotx_put(aotx_query_at(q, i) + 132, 5, 4);
    auto rows = d.search(q, n);
    for (unsigned i = 0; i < n; ++i) aotx_check(!rows[i].status && aotx_context_has(rows[i], aotx_context_id(i, 20)), "pending intention enters the required set");
    aotx_fixture completed;
    for (unsigned i = 0; i < n; ++i) {
        auto r = f.rows[cut + i]; aotx_put(r.data() + AOTX_CO_VERSION, 2);
        aotx_put(r.data() + AOTX_CO_UPDATED, f.rows.size() + i + 1); aotx_put(r.data() + AOTX_CO_RETENTION, 0, 4);
        completed.add(r, f.payloads[cut + i]);
    }
    aotx_check(!d.load(completed.wire(true, f.rows.size() + 1, 6), true).status, "completion updates the existing intention provenance");
    aotx_put(q.data() + 32, f.rows.size() + n); rows = d.search(q, n);
    for (unsigned i = 0; i < n; ++i) aotx_check(!rows[i].status && !aotx_context_has(rows[i], aotx_context_id(i, 20)), "completed intention is not nominated again");
    auto saved = d.checkpoint(); aotx_recall_device restored;
    aotx_check(!restored.load(saved).status, "completed intention checkpoint restores"); rows = restored.search(q, n);
    for (unsigned i = 0; i < n; ++i) aotx_check(!rows[i].status && !aotx_context_has(rows[i], aotx_context_id(i, 20)), "completion persists after restore");
}
#endif
