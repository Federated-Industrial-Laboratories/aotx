/* Purpose: Check automatic recall groups, corrections and recorded dependencies.
 * Owns: Real device selection, scope, fit and replay assertions.
 * Launch shape: Distinct batches at N=1 and N=64.
 * Lifetime: One test process without model weights or persistent files. */
#include "appraisal_recall_fixture.h"
#include "cognitive/recall_context.cuh"
#include "appraisal/recall_config.cuh"
#include <chrono>

__global__ void aotx_ar_check_rows(const aotx_cognitive_store *s, const unsigned char *q,
    aotx_recall_result *rows, unsigned n) {
    unsigned i = threadIdx.x; if (i >= n) return;
    uint32_t status = aotx_recall_render(s, q + 64 + i * AOTX_RECALL_QUERY, rows + i);
    if (status) aotx_recall_refuse(rows + i, status);
}
__global__ void aotx_ar_defaults(const aotx_cognitive_store *s, unsigned char *q, unsigned n) {
    unsigned i = threadIdx.x; if (i >= n) return;
    aotx_appraisal_recall_defaults(s, aotx_appraisal_recall_config(s), q + 64 + i * AOTX_RECALL_QUERY);
}
static std::vector<aotx_recall_result> aotx_ar_validate(aotx_recall_device &d, std::vector<aotx_recall_result> rows) {
    AOTX_CUDA(cudaMemcpy(d.rows, rows.data(), rows.size() * sizeof(rows[0]), cudaMemcpyHostToDevice));
    aotx_ar_check_rows<<<1,64>>>(d.live, d.requests, d.rows, rows.size()); AOTX_CUDA(cudaGetLastError());
    AOTX_CUDA(cudaMemcpy(rows.data(), d.rows, rows.size() * sizeof(rows[0]), cudaMemcpyDeviceToHost)); return rows;
}
static aotx_bytes aotx_ar_apply_defaults(aotx_recall_device &d, aotx_bytes q, unsigned n) {
    AOTX_CUDA(cudaMemcpy(d.requests, q.data(), q.size(), cudaMemcpyHostToDevice));
    aotx_ar_defaults<<<1,64>>>(d.live, d.requests, n); AOTX_CUDA(cudaGetLastError());
    AOTX_CUDA(cudaMemcpy(q.data(), d.requests, q.size(), cudaMemcpyDeviceToHost)); return q;
}
static void aotx_ar_remove(aotx_recall_result &r, unsigned at) {
    memmove(r.selection + 16 + at * 32, r.selection + 48 + at * 32, (r.count - at - 1) * 32);
    memset(r.selection + 16 + (--r.count) * 32, 0, 32); aotx_put(r.selection + 4, r.count, 4);
}
static unsigned aotx_ar_find(const aotx_fixture &f, uint64_t id) {
    for (unsigned j = 0; j < f.rows.size(); ++j) if (aotx_get(f.rows[j].data() + AOTX_CO_ID) == id) return j;
    fprintf(stderr, "missing fixture ID\n"); exit(2);
}
static void aotx_ar_selection(unsigned n, unsigned scope, bool corrected, bool sources = false) {
    auto f = aotx_ar_corpus(n, corrected, scope); auto q = aotx_ar_queries(n, f.rows.size(), scope);
    if (sources) for (unsigned i = 0; i < n; ++i) {
        auto p = aotx_query_at(q, i), c = p + AOTX_RECALL_EXTENSION;
        memcpy(c, "AOTXCTX2", 8); aotx_put(c + 8, 2, 4); aotx_put(c + 44, 2, 4);
        aotx_id(p + AOTX_RECALL_ACTOR, 88000 + i);
    }
    aotx_recall_device d; aotx_check(!d.load(f.wire(false, f.rows.size())).status, "automatic source corpus admission");
    auto before = d.checkpoint(); auto rows = d.search(q, n); aotx_ar_expected(rows, n, corrected);
    aotx_check(d.checkpoint() == before, "automatic recall creates no exposure or memory writes");
    auto scoped = q;
    for (unsigned i = 0; i < n; ++i) {
        auto c = aotx_query_at(scoped, i) + AOTX_RECALL_EXTENSION;
        aotx_put(c + 32, 0, 4); memset(c + 48, 0, 16);
    }
    aotx_ar_expected(d.search(scoped, n), n, corrected);
    d.search(q, n);
    for (unsigned component = 1; component < 6; ++component) {
        auto bad = rows; for (auto &r : bad) if (r.count == 6) aotx_ar_remove(r, component);
        auto checked = aotx_ar_validate(d, bad);
        aotx_status_rows(checked, AOTX_COG_REFERENCE, "recorded source groups cannot omit any component");
    }
    for (unsigned mode = 0; mode < 6; ++mode) {
        auto changed = q;
        for (unsigned i = 0; i < n; ++i) {
            auto p = aotx_query_at(changed, i), c = p + AOTX_RECALL_EXTENSION;
            if (mode == 0) aotx_id(c + 16, 90000 + i);
            if (mode == 1) aotx_id(c + 48, 90000 + i);
            if (mode == 2) { aotx_put(c + 12, 2, 4); memset(c + 16, 0, 20); memset(c + 48, 0, 16); }
            if (mode == 3) aotx_put(c + 36, 990000, 4);
            if (mode == 4) aotx_put(p + 132, 5, 4);
            if (mode == 5) aotx_put(p + 136, 600, 4);
        }
        auto actual = d.search(changed, n); aotx_status_rows(actual, 0, "wrong context or source pressure retains valid base recall");
        for (unsigned i = 0; i < n; ++i) {
            unsigned base = corrected ? 12 : 3;
            aotx_check(!aotx_context_has(actual[i], aotx_ar_id(i, base + 3)) &&
                !aotx_context_has(actual[i], aotx_ar_id(i, base + 4)) &&
                !aotx_context_has(actual[i], aotx_ar_id(i, base + 2)), "no partial automatic evidence group is selected");
        }
        if (mode < 4) for (const auto &r : aotx_ar_validate(d, rows))
            aotx_check(r.status && !r.count && !r.context_bytes,
                "recorded groups independently check the task, subject and cosine floor");
    }
    d.search(q, n); auto result = d.record(q, n, true);
    aotx_check(!result.status && result.applied == 2 * n, "automatic selected groups are recorded atomically");
    auto saved = d.checkpoint(); aotx_check(!d.load(saved).status, "automatic selected groups restore");
    auto recovered = d.saved_queries(n); auto replay = d.search(recovered, n, true); aotx_ar_expected(replay, n, corrected);
    for (unsigned i = 0; i < n; ++i) aotx_check(!replay[i].searches && aotx_context(replay[i]) == aotx_context(rows[i]),
        "recorded source groups restore exact context without a new search");
    aotx_check(d.checkpoint() == saved, "recorded recall preserves the single source exposure");
}
static void aotx_ar_unknown(unsigned n) {
    auto f = aotx_ar_corpus(n);
    for (unsigned i = 0; i < n; ++i) {
        auto &a = f.payloads[aotx_ar_find(f, aotx_ar_id(i, 6))], &r = f.payloads[aotx_ar_find(f, aotx_ar_id(i, 7))];
        for (unsigned at = 4; at <= 20; at += 4) aotx_put(a.data() + at, at == 16 ? 0 : AOTX_COG_UNKNOWN, 4);
        memset(a.data() + 120, 0, 8);
        for (unsigned at = 16; at < 32; at += 4) aotx_put(r.data() + at, AOTX_COG_UNKNOWN, 4);
        memset(r.data() + 48, 0, 24);
    }
    aotx_recall_device d; aotx_check(!d.load(f.wire(false, f.rows.size())).status, "unknown third-party appraisal admission");
    auto rows = d.search(aotx_ar_queries(n, f.rows.size()), n); aotx_status_rows(rows, 0, "unknown appraisal recall");
    for (unsigned i = 0; i < n; ++i) aotx_check(!aotx_context_has(rows[i], aotx_ar_id(i, 6)) &&
        !aotx_context_has(rows[i], aotx_ar_id(i, 7)), "mechanical familiarity and unknown values do not invent priority");
}
static void aotx_ar_config_checks(unsigned n) {
    auto f = aotx_ar_corpus(n); aotx_recall_device d;
    aotx_check(!d.load(f.wire(false, f.rows.size())).status, "query default corpus admission");
    auto tasks = aotx_ar_queries(n, f.rows.size(), 0, 1); auto injected = aotx_ar_apply_defaults(d, tasks, n);
    for (unsigned i = 0; i < n; ++i) {
        auto a = aotx_query_at(tasks, i), b = aotx_query_at(injected, i); auto c = b + AOTX_RECALL_EXTENSION;
        aotx_check(aotx_get(c + 12, 4) == 3 && aotx_get(c + 36, 4) == 600000 && aotx_get(c + 40, 4) == 800000,
            "current config supplies recall flags, floor and priority");
        aotx_check(!memcmp(a, b, AOTX_RECALL_EXTENSION) && !memcmp(a + AOTX_RECALL_EXTENSION + 16, c + 16, 20) &&
            !memcmp(a + AOTX_RECALL_EXTENSION + 48, c + 48, AOTX_RECALL_QUERY - AOTX_RECALL_EXTENSION - 48),
            "default injection preserves task, participants and all other query bytes");
    }
    aotx_ar_expected(d.search(injected, n), n, false);
    auto scoped = tasks;
    for (unsigned i = 0; i < n; ++i) {
        auto c = aotx_query_at(scoped, i) + AOTX_RECALL_EXTENSION;
        aotx_put(c + 32, 0, 4); memset(c + 48, 0, 16);
    }
    aotx_ar_expected(d.search(aotx_ar_apply_defaults(d, scoped, n), n), n, false);
    auto explicit_q = injected;
    for (unsigned i = 0; i < n; ++i) aotx_put(aotx_query_at(explicit_q, i) + AOTX_RECALL_EXTENSION + 40, 0, 4);
    aotx_check(aotx_ar_apply_defaults(d, explicit_q, n) == explicit_q, "explicit zero appraisal priority overrides defaults");
    auto source_q = tasks;
    for (unsigned i = 0; i < n; ++i) {
        auto p = aotx_query_at(source_q, i), c = p + AOTX_RECALL_EXTENSION;
        memcpy(c, "AOTXCTX2", 8); aotx_put(c + 8, 2, 4); aotx_put(c + 44, 2, 4);
        aotx_id(p + AOTX_RECALL_ACTOR, 88000 + i);
    }
    auto source_defaults = aotx_ar_apply_defaults(d, source_q, n);
    for (unsigned i = 0; i < n; ++i) {
        auto p = aotx_query_at(source_defaults, i);
        aotx_check(aotx_get(p + AOTX_RECALL_ACTOR) == 88000 + i &&
            aotx_get(p + AOTX_RECALL_EXTENSION + 8, 4) == 2, "appraisal defaults preserve the explicit actor and query version");
    }
    aotx_ar_expected(d.search(source_defaults, n), n, false);
    auto config = f.rows[0]; aotx_put(config.data() + AOTX_CO_VERSION, 2); aotx_put(config.data() + AOTX_CO_UPDATED, f.rows.size() + 1);
    f.add(config, aotx_ar_config(0)); aotx_check(!d.load(f.wire(false, f.rows.size())).status, "latest disabled config admission");
    aotx_check(aotx_ar_apply_defaults(d, tasks, n) == tasks, "disabled latest config cannot fall back to an enabled predecessor");
}
static void aotx_ar_duplicate(unsigned n) {
    auto f = aotx_ar_corpus(n);
    for (unsigned i = 0; i < n; ++i) {
        unsigned old = aotx_ar_find(f, aotx_ar_id(i, 7)); auto r = f.rows[old];
        aotx_id(r.data() + AOTX_CO_ID, aotx_ar_id(i, 20));
        aotx_put(r.data() + AOTX_CO_CREATED, f.rows.size() + 1); aotx_put(r.data() + AOTX_CO_UPDATED, f.rows.size() + 1);
        f.add(r, f.payloads[old]);
    }
    aotx_recall_device d; aotx_check(!d.load(f.wire(false, f.rows.size())).status, "duplicate relationship corpus admission");
    auto q = aotx_ar_queries(n, f.rows.size()); auto rows = d.search(q, n); aotx_status_rows(rows, 0, "ambiguous source group recall");
    for (unsigned i = 0; i < n; ++i) aotx_check(!aotx_context_has(rows[i], aotx_ar_id(i, 6)) &&
        !aotx_context_has(rows[i], aotx_ar_id(i, 7)) && !aotx_context_has(rows[i], aotx_ar_id(i, 20)),
        "two current relationships cannot count the same exposure twice");
    for (unsigned i = 0; i < n; ++i) aotx_id(aotx_query_at(q, i) + 16, 99000 + i);
    rows = d.search(q, n); aotx_status_rows(rows, 0, "foreign private principal recall");
    for (const auto &r : rows) aotx_check(!r.count, "private source and relationship records remain inaccessible");
}
static void aotx_ar_stale(unsigned n) {
    for (unsigned mode = 0; mode < 5; ++mode) {
        auto f = aotx_ar_corpus(n); aotx_recall_device d; aotx_check(!d.load(f.wire(false, f.rows.size())).status, "dependency change setup");
        auto q = aotx_ar_queries(n, f.rows.size()); auto rows = d.search(q, n); aotx_ar_expected(rows, n, false);
        for (unsigned i = 0; i < n; ++i) {
            unsigned part = mode == 0 ? 3 : mode == 1 ? 5 : mode == 2 ? 7 : mode == 3 ? 1 : 6;
            unsigned old = aotx_ar_find(f, part == 1 ? 4000 + i : aotx_ar_id(i, part));
            auto r = f.rows[old]; aotx_put(r.data() + AOTX_CO_VERSION, 2); aotx_put(r.data() + AOTX_CO_UPDATED, f.rows.size() + 1);
            if (mode == 0) aotx_put(r.data() + AOTX_CO_FLAGS, AOTX_COG_TOMBSTONE, 4);
            if (mode == 4) aotx_put(r.data() + AOTX_CO_EVIDENCE, 3, 4);
            f.add(r, mode == 0 ? aotx_bytes{} : f.payloads[old]);
        }
        auto loaded = d.load(f.wire(false, f.rows.size()));
        aotx_check(!loaded.status, "current source dependency changes are admitted");
        if (loaded.status) { fprintf(stderr, "dependency mode=%u status=%u\n", mode, loaded.status); continue; }
        auto checked = aotx_ar_validate(d, rows);
        for (const auto &r : checked) aotx_check(r.status && !r.count && !r.context_bytes, "recorded recall refuses withdrawn or stale exact dependencies");
    }
}
static void aotx_ar_focused(unsigned n) {
    auto f = aotx_ar_corpus(n); aotx_recall_device d;
    aotx_check(!d.load(f.wire(false, f.rows.size())).status, "focused automatic source corpus admission");
    auto before = d.checkpoint();
    for (unsigned group = 0; group < 2; ++group) {
        auto q = aotx_ar_queries(n, f.rows.size());
        for (unsigned i = 0; i < n; ++i) aotx_pin(aotx_query_at(q, i), group, 0, aotx_ar_id(i, 4));
        auto rows = d.search(q, n); aotx_status_rows(rows, 0, "already selected working memory can add appraisal evidence");
        for (unsigned i = 0; i < n; ++i) {
            aotx_check(rows[i].count == 6 && aotx_selected(rows[i], group) == aotx_ar_id(i, 4) &&
                rows[i].reason[group] == (group ? AOTX_RECALL_FOCUS : AOTX_RECALL_REQUIRED),
                "required and focused candidate retain their complete prefix position and reason");
            for (unsigned part = 3; part <= 7; ++part) aotx_check(aotx_context_has(rows[i], aotx_ar_id(i, part)),
                "a selected working source adds its exact complete appraisal group");
        }
        auto replay = aotx_ar_validate(d, rows); aotx_status_rows(replay, 0, "focused source group passes independent recorded validation");
        for (unsigned i = 0; i < n; ++i) aotx_check(aotx_context(rows[i]) == aotx_context(replay[i]),
            "focused source group renders exact recorded context");
        for (unsigned pressure = 0; pressure < 3; ++pressure) {
            auto small = q;
            for (unsigned i = 0; i < n; ++i) {
                auto p = aotx_query_at(small, i);
                if (pressure == 0) aotx_put(p + 132, 5, 4);
                if (pressure == 1) aotx_put(p + 136, 800, 4);
                if (pressure == 2) aotx_put(p + AOTX_RECALL_EXTENSION + 36, 990000, 4);
            }
            auto limited = d.search(small, n); aotx_status_rows(limited, 0, "focused source priority respects unchanged caps and floor");
            for (unsigned i = 0; i < n; ++i) aotx_check(limited[i].count == 2 &&
                aotx_context_has(limited[i], aotx_ar_id(i, 0)) && aotx_context_has(limited[i], aotx_ar_id(i, 4)),
                "mandatory source survives without partial or below-floor appraisal evidence");
        }
    }
    aotx_check(d.checkpoint() == before, "focused recall never adds source exposure");
}
int main(int argc, char **argv) {
    const char *names[] = {"selection0", "selection1", "selection2", "selection3", "selection4", "selection5",
        "selection6", "selection7", "selection8", "selection9", "unknown", "config", "stale", "duplicate", "focused"};
    unsigned only = argc > 1 ? !strcmp(argv[1], "1") ? 1 : !strcmp(argv[1], "64") ? 64 : 0 : 0;
    bool valid = argc <= 3 && (argc == 1 || only);
    if (argc == 3) { bool found = false; for (auto name : names) found |= !strcmp(argv[2], name); valid &= found; }
    if (!valid) { fprintf(stderr, "usage: aotx_appraisal_recall_test [1|64] [selection0..9|unknown|config|stale|duplicate|focused]\n"); return 2; }
    for (unsigned n : {1u, 64u}) {
        if (only && n != only) continue;
        for (unsigned group = 0; group < 15; ++group) {
            if (argc == 3 && strcmp(argv[2], names[group])) continue;
            auto start = std::chrono::steady_clock::now();
            printf("automatic recall N=%u case=%s start\n", n, names[group]); fflush(stdout);
            if (group < 10) {
                unsigned index = group < 6 ? group : group - 6;
                aotx_ar_selection(n, index / 2, index % 2, group >= 6);
            } else if (group == 10) aotx_ar_unknown(n);
            else if (group == 11) aotx_ar_config_checks(n);
            else if (group == 12) aotx_ar_stale(n);
            else if (group == 13) aotx_ar_duplicate(n);
            else aotx_ar_focused(n);
            double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            printf("automatic recall N=%u case=%s checks=%u failed=%u seconds=%.3f\n",
                n, names[group], aotx_checks, aotx_failures, seconds); fflush(stdout);
        }
    }
    printf("automatic recall: %u checks, %u failed\n", aotx_checks, aotx_failures);
    return aotx_failures || !aotx_checks || (argc == 1 && aotx_checks < 5000) ? 1 : 0;
}
