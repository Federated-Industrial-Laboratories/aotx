/* Purpose: Verify source diversity, exact actor labels and old selection replay.
 * Owns: Independent reference order, byte limits and access refusal controls.
 * Launch shape: N=1 and N=64 through the maintained device recall API.
 * Lifetime: One complete source corpus through current and recorded queries. */
#include "source_fixture.h"

static unsigned aotx_source_history(const aotx_recall_result &row, const unsigned char *q) {
    return row.context_bytes - 8 - aotx_get(q + 148, 4);
}
static void aotx_source_recall(unsigned n, unsigned scope, bool modern) {
    auto f = aotx_source_corpus(n, scope); aotx_recall_device d;
    aotx_check(!d.load(f.wire(false, f.rows.size())).status, "complete source corpus loads");
    auto q = aotx_source_queries(n, f.rows.size(), scope, modern); auto rows = d.search(q, n);
    aotx_status_rows(rows, 0, "source recall succeeds");
    for (unsigned i = 0; i < n; ++i) {
        aotx_check(rows[i].count == 6, "bounded selection has six references");
        auto text = aotx_context(rows[i]);
        for (unsigned j = 0; j < (modern ? 3u : 6u); ++j)
            aotx_check(aotx_selected(rows[i], j) == aotx_source_id(i, modern ? j : 0) + (modern ? 30 : 2 + j),
                "distinct sources precede repeated representations and old queries retain object order");
        if (modern) {
            aotx_check(text.find("uses CUDA. Their peer is Iris.") != std::string::npos,
                "complete working source retains both facts");
            for (unsigned group = 0; group < 3; ++group)
                aotx_check(text.find("source_ref=" + aotx_source_hex(aotx_source_id(i, group)) + "@1 source_actor=" +
                    aotx_source_hex(7000 + i * 3 + group)) != std::string::npos,
                    "source actor comes from the exact event rather than owner or working subject");
            aotx_check(text.find("source_actor=" + aotx_source_hex(8000 + i)) == std::string::npos,
                "current actor does not replace historical actors");
        } else aotx_check(text.find("source_ref=") == std::string::npos && text.find("source_actor=") == std::string::npos,
            "old context has no added source labels");
    }
    aotx_check(!d.record(q, n, true).status, "recorded source selections apply");
    auto saved = d.saved_queries(n); auto replay = d.search(saved, n, true);
    for (unsigned i = 0; i < n; ++i) {
        aotx_check(!memcmp(aotx_query_at(saved, i), aotx_query_at(q, i), AOTX_RECALL_QUERY), "recorded query bytes are exact");
        aotx_check(!replay[i].status && !replay[i].searches && replay[i].context_bytes == rows[i].context_bytes &&
            !memcmp(replay[i].context, rows[i].context, rows[i].context_bytes) &&
            !memcmp(replay[i].selection, rows[i].selection, AOTX_RECALL_SELECTION), "recorded context and selected references replay exactly");
    }
}
static void aotx_source_limits(unsigned n) {
    auto f = aotx_source_corpus(n); aotx_recall_device d;
    aotx_check(!d.load(f.wire(false, f.rows.size())).status, "limit corpus loads");
    auto q = aotx_source_queries(n, f.rows.size());
    for (unsigned i = 0; i < n; ++i) { auto p = aotx_query_at(q, i); aotx_put(p + 132, 3, 4);
        aotx_pin(p, 0, 0, aotx_source_id(i, 0) + 2); aotx_pin(p, 1, 0, aotx_source_id(i, 0) + 3); }
    auto rows = d.search(q, n);
    for (unsigned i = 0; i < n; ++i) {
        aotx_check(!rows[i].status && rows[i].count == 3 && aotx_selected(rows[i], 0) == aotx_source_id(i, 0) + 2 &&
            aotx_selected(rows[i], 1) == aotx_source_id(i, 0) + 3 && aotx_selected(rows[i], 2) == aotx_source_id(i, 1) + 30,
            "required and focus pins remain exact while another source gets the remaining row");
        aotx_put(aotx_query_at(q, i) + 136, aotx_source_history(rows[i], aotx_query_at(q, i)), 4);
    }
    auto exact = d.search(q, n);
    for (unsigned i = 0; i < n; ++i) {
        aotx_check(!exact[i].status && aotx_context(exact[i]) == aotx_context(rows[i]), "all labels pay the exact context byte bound");
        aotx_put(aotx_query_at(q, i) + 136, aotx_source_history(rows[i], aotx_query_at(q, i)) - 1, 4);
    }
    auto short_rows = d.search(q, n);
    for (unsigned i = 0; i < n; ++i) {
        auto p = aotx_query_at(q, i); unsigned cap = aotx_get(p + 136, 4);
        bool bounded = !short_rows[i].status && short_rows[i].count >= 2 && aotx_source_history(short_rows[i], p) <= cap &&
            aotx_selected(short_rows[i], 0) == aotx_source_id(i, 0) + 2 && aotx_selected(short_rows[i], 1) == aotx_source_id(i, 0) + 3;
        if (!bounded) fprintf(stderr, "boundary row=%u status=%u count=%u context=%u input=%u cap=%u previous=%u\n",
            i, short_rows[i].status, short_rows[i].count, short_rows[i].context_bytes,
            (unsigned)aotx_get(p + 148, 4), cap, aotx_source_history(rows[i], p));
        aotx_check(bounded, "one fewer byte cannot exceed the bound or remove mandatory pins");
    }
    q = aotx_source_queries(n, f.rows.size());
    for (unsigned i = 0; i < n; ++i) aotx_id(aotx_query_at(q, i) + 16, 990000 + i);
    rows = d.search(q, n);
    for (auto &r : rows) aotx_check(!r.status && !r.count, "foreign private source groups remain inaccessible");
    for (unsigned mode = 0; mode < 4; ++mode) {
        q = aotx_source_queries(n, f.rows.size());
        for (unsigned i = 0; i < n; ++i) {
            auto p = aotx_query_at(q, i);
            if (mode == 0) p[AOTX_RECALL_EXTENSION + 7] = '1';
            if (mode == 1) aotx_put(p + AOTX_RECALL_EXTENSION + 44, 1, 4);
            if (mode == 2) p[AOTX_RECALL_ACTOR + 16] = 1;
            if (mode == 3) memset(p + AOTX_RECALL_ACTOR, 0, 16);
        }
        rows = d.search(q, n); aotx_status_rows(rows, mode == 3 ? 0 : AOTX_COG_FORMAT,
            "query version and reserved bytes are checked and unknown actors are allowed");
    }
}
static void aotx_source_oversized(unsigned n) {
    auto f = aotx_source_corpus(n, 0, 1900); aotx_recall_device d;
    aotx_check(!d.load(f.wire(false, f.rows.size())).status, "large complete sources load");
    auto q = aotx_source_queries(n, f.rows.size());
    for (unsigned i = 0; i < n; ++i) { auto p = aotx_query_at(q, i); aotx_put(p + 132, 3, 4); aotx_put(p + 136, 1200, 4); }
    auto rows = d.search(q, n);
    for (unsigned i = 0; i < n; ++i) {
        aotx_check(!rows[i].status && rows[i].count == 3 && aotx_source_history(rows[i], aotx_query_at(q, i)) <= 1200,
            "oversized complete sources use fitting bounded representations");
        for (unsigned j = 0; j < 3; ++j) aotx_check(aotx_selected(rows[i], j) == aotx_source_id(i, j) + 2,
            "an oversized working row does not let its fragments exclude other sources");
    }
}
static void aotx_source_standalone(unsigned n) {
    auto f = aotx_memory_corpus(n); aotx_recall_device d;
    aotx_check(!d.load(f.wire(false, f.rows.size())).status, "standalone prepared rows load");
    auto q = aotx_memory_queries(n, f.rows.size());
    for (unsigned i = 0; i < n; ++i) aotx_source_query(aotx_query_at(q, i));
    auto rows = d.search(q, n);
    for (unsigned i = 0; i < n; ++i) aotx_check(!rows[i].status && rows[i].count == 1 &&
        aotx_context(rows[i]).find("source_ref=" + aotx_source_hex(10000 + i * 3) + "@1 source_actor=unknown") != std::string::npos,
        "a standalone prepared row forms its own group and cannot authenticate its subject");
}
static void aotx_source_partial(unsigned n, bool equal_length) {
    auto f = aotx_source_corpus(n); aotx_recall_device d;
    for (unsigned j = 0; j < f.rows.size(); ++j) {
        auto &r = f.rows[j];
        if (aotx_get(r.data() + AOTX_CO_KIND, 2) != AOTX_COG_WORKING) continue;
        unsigned owner = aotx_get(r.data() + AOTX_CO_OWNER) - 1000;
        if (aotx_get(r.data() + AOTX_CO_SOURCE) != aotx_source_id(owner, 0)) continue;
        if (equal_length) f.payloads[j][32] = 'X';
        else f.payloads[j] = aotx_memory_text("Their peer is Iris.");
    }
    aotx_check(!d.load(f.wire(false, f.rows.size())).status, "prepared partial working text remains admissible");
    auto q = aotx_source_queries(n, f.rows.size());
    for (unsigned i = 0; i < n; ++i) aotx_put(aotx_query_at(q, i) + 132, 3, 4);
    auto rows = d.search(q, n);
    for (unsigned i = 0; i < n; ++i) {
        aotx_check(!rows[i].status && rows[i].count == 3 && aotx_selected(rows[i], 0) == aotx_source_id(i, 0) + 2 &&
            aotx_selected(rows[i], 1) == aotx_source_id(i, 1) + 30 && aotx_selected(rows[i], 2) == aotx_source_id(i, 2) + 30,
            "only the complete exact source text receives working preference");
        aotx_pin(aotx_query_at(q, i), 0, 0, aotx_source_id(i, 0) + 30);
    }
    rows = d.search(q, n);
    for (unsigned i = 0; i < n; ++i) aotx_check(!rows[i].status && aotx_selected(rows[i], 0) == aotx_source_id(i, 0) + 30,
        "partial working text retains ordinary required-reference access");
}
int main(int argc, char **argv) {
    unsigned only = argc == 2 ? !strcmp(argv[1], "1") ? 1 : !strcmp(argv[1], "64") ? 64 : 0 : 0;
    if (argc > 2 || (argc == 2 && !only)) { fprintf(stderr, "usage: aotx_source_recall_test [1|64]\n"); return 2; }
    int cards = 0; if (cudaGetDeviceCount(&cards) != cudaSuccess || !cards) return 77;
    for (unsigned n : {1u, 64u}) {
        if (only && only != n) continue;
        for (unsigned scope : {0u, 1u}) for (bool modern : {false, true}) {
            printf("source recall N=%u scope=%u sources=%u start\n", n, scope, modern); fflush(stdout);
            aotx_source_recall(n, scope, modern);
        }
        printf("source recall N=%u bounds start\n", n); fflush(stdout);
        aotx_source_limits(n); aotx_source_oversized(n); aotx_source_standalone(n);
        for (bool equal_length : {false, true}) aotx_source_partial(n, equal_length);
        printf("source recall N=%u: %u checks, %u failures\n", n, aotx_checks, aotx_failures); fflush(stdout);
    }
    return aotx_failures ? 1 : 0;
}
