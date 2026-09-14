/* Purpose: Build distinct source groups for automatic appraisal recall checks.
 * Owns: Independent typed bytes, actors, tasks and expected selections.
 * Launch shape: One or 64 separate principals with mixed outcome records.
 * Lifetime: One device test process without model weights. */
#ifndef AOTX_TEST_APPRAISAL_RECALL_FIXTURE_H
#define AOTX_TEST_APPRAISAL_RECALL_FIXTURE_H
#include "context_fixture.h"
#include "appraisal/format.h"

static uint64_t aotx_ar_id(unsigned i, unsigned part) { return 700000 + i * 32 + part; }
static void aotx_ar_source(aotx_row &r, unsigned i, unsigned part) {
    aotx_id(r.data() + AOTX_CO_SOURCE, aotx_ar_id(i, part)); aotx_put(r.data() + AOTX_CO_SOURCE_VERSION, 1);
}
static aotx_bytes aotx_ar_config(unsigned flags = 3) {
    aotx_bytes p(AOTX_APPRAISAL_CONFIG_BYTES, 0); memcpy(p.data(), "AOTXAPC1", 8);
    aotx_put(p.data() + 8, 1, 4); aotx_put(p.data() + 12, flags, 4);
    aotx_put(p.data() + 16, 160, 4); aotx_put(p.data() + 20, 512, 4); aotx_put(p.data() + 24, 4096, 4);
    aotx_put(p.data() + 28, 600000, 4); aotx_put(p.data() + 32, 800000, 4); aotx_put(p.data() + 36, 64, 4);
    const unsigned char processor[32] = AOTX_APPRAISAL_PROCESSOR_BYTES;
    memcpy(p.data() + 40, processor, 32); return p;
}
static std::string aotx_ar_task(unsigned i) { return "packing crate " + std::to_string(i); }
static std::string aotx_ar_text(unsigned i, bool corrected) {
    return std::string(corrected ? "I correct the damage report for " : "I helped and damaged a tool during ") + aotx_ar_task(i) + ".";
}
static void aotx_ar_episode(aotx_fixture &f, unsigned i, bool corrected, unsigned scope) {
    unsigned base = corrected ? 12 : 3;
    std::string text = aotx_ar_text(i, corrected), task = aotx_ar_task(i);
    for (unsigned part = 0; part < 5; ++part) {
        unsigned kind = part == 0 ? AOTX_COG_EVENT : part == 1 ? AOTX_COG_WORKING :
            part == 2 ? AOTX_COG_POLICY : part == 3 ? AOTX_COG_APPRAISAL : AOTX_COG_RELATIONSHIP;
        auto r = aotx_memory_row(i, kind, aotx_ar_id(i, base + part), f.rows.size() + 1, scope);
        aotx_bytes p;
        if (part == 0) p = aotx_memory_text(text);
        else {
            aotx_ar_source(r, i, base); aotx_put(r.data() + AOTX_CO_SOURCE_KIND, AOTX_COG_INFERRED, 4);
            if (part == 1) {
                p = aotx_memory_text(text); aotx_id(r.data() + AOTX_CO_EMBEDDING, aotx_ar_id(i, 2));
                aotx_put(r.data() + AOTX_CO_EMBED_VERSION, 1);
            } else if (part == 2) {
                p.resize(AOTX_APPRAISAL_QUEUE_BYTES, 0); memcpy(p.data(), "AOTXAPQ1", 8);
                aotx_put(p.data() + 8, 1, 4); aotx_put(p.data() + 12, 1, 4);
                aotx_id(p.data() + 16, 690000); aotx_put(p.data() + 32, 1);
                aotx_id(p.data() + 40, 4000 + i); aotx_id(p.data() + 128, 4000 + i); aotx_put(p.data() + 144, 1);
                auto config = aotx_ar_config(); memcpy(p.data() + 64, config.data() + 40, 32); memset(p.data() + 96, 0x5a, 32);
            } else {
                bool relation = part == 4; p.resize(relation ? AOTX_APPRAISAL_RELATION_BYTES : AOTX_APPRAISAL_ASSESS_BYTES, 0);
                const unsigned char processor[32] = AOTX_APPRAISAL_PROCESSOR_BYTES;
                if (relation) {
                    memcpy(p.data(), "AOTXREL1", 8); aotx_put(p.data() + 8, 1, 4); aotx_put(p.data() + 12, 1, 4);
                    aotx_put(p.data() + 16, 700000 + i, 4); aotx_put(p.data() + 20, corrected ? 0 : 300000 + i, 4);
                    aotx_put(p.data() + 24, 600000 + i, 4); aotx_put(p.data() + 28, AOTX_COG_UNKNOWN, 4);
                    aotx_id(p.data() + 32, 4000 + i); aotx_put(p.data() + 52, text.size(), 4);
                    aotx_put(p.data() + 56, text.find(task), 4); aotx_put(p.data() + 60, task.size(), 4);
                    memcpy(p.data() + 72, processor, 32); memset(p.data() + 104, 0x5a, 32);
                    aotx_id(p.data() + 136, aotx_ar_id(i, base + 2)); aotx_put(p.data() + 152, 1);
                } else {
                    aotx_put(p.data(), 2, 4); aotx_put(p.data() + 4, 800000 + i, 4);
                    aotx_put(p.data() + 8, corrected ? 0 : 900000 - i, 4); aotx_put(p.data() + 12, 500000 + i, 4);
                    aotx_put(p.data() + 16, 3, 4); aotx_put(p.data() + 20, AOTX_COG_UNKNOWN, 4); aotx_put(p.data() + 24, 1, 4);
                    memcpy(p.data() + 32, processor, 32); memset(p.data() + 64, 0x5a, 32);
                    aotx_id(p.data() + 96, aotx_ar_id(i, base + 2)); aotx_put(p.data() + 112, 1); aotx_put(p.data() + 124, text.size(), 4);
                }
                if (corrected) {
                    aotx_id(r.data() + AOTX_CO_SUPERSEDES, aotx_ar_id(i, relation ? 7 : 6));
                    aotx_put(r.data() + AOTX_CO_SUPER_VERSION, 1);
                }
            }
        }
        f.add(r, p);
    }
}
static aotx_fixture aotx_ar_corpus(unsigned n, bool corrected = false, unsigned scope = 0) {
    aotx_fixture f; auto config = aotx_memory_row(0, AOTX_COG_POLICY, 690000, 1, 2);
    memset(config.data() + AOTX_CO_SUBJECT, 0, 16); aotx_put(config.data() + AOTX_CO_SOURCE_KIND, AOTX_COG_AUTHORED, 4);
    f.add(config, aotx_ar_config());
    for (unsigned i = 0; i < n; ++i) {
        auto task = aotx_memory_row(i, AOTX_COG_CUE, 4000 + i, f.rows.size() + 1, scope);
        aotx_put(task.data() + AOTX_CO_SOURCE_KIND, AOTX_COG_AUTHORED, 4); memset(task.data() + AOTX_CO_SUBJECT, 0, 16);
        f.add(task, aotx_memory_text(aotx_ar_task(i)));
        f.add(aotx_memory_row(i, AOTX_COG_COMPONENT, aotx_ar_id(i, 2), f.rows.size() + 1, scope),
            aotx_memory_vector(1.0f, 0.6f + i * 0.0002f, 0.01f));
        aotx_ar_episode(f, i, false, scope);
        auto cue = aotx_memory_row(i, AOTX_COG_CUE, aotx_ar_id(i, 0), f.rows.size() + 1, scope);
        memset(cue.data() + AOTX_CO_SUBJECT, 0, 16);
        aotx_id(cue.data() + AOTX_CO_SOURCE, 4000 + i); aotx_put(cue.data() + AOTX_CO_SOURCE_VERSION, 1);
        f.add(cue, aotx_context_text(i, true, "Check the current task requirements " + std::to_string(i)));
    }
    if (corrected) for (unsigned i = 0; i < n; ++i) aotx_ar_episode(f, i, true, scope);
    return f;
}
static aotx_bytes aotx_ar_queries(unsigned n, uint64_t cut, unsigned scope = 0, unsigned flags = 3) {
    auto q = aotx_context_queries(n, cut, scope, flags);
    for (unsigned i = 0; i < n; ++i) aotx_put(aotx_query_at(q, i) + 132, 6, 4);
    return q;
}
static void aotx_ar_expected(const std::vector<aotx_recall_result> &rows, unsigned n, bool corrected) {
    aotx_status_rows(rows, 0, "complete automatic recall admission");
    for (unsigned i = 0; i < n; ++i) {
        unsigned base = corrected ? 12 : 3;
        aotx_check(rows[i].count == 6 && aotx_selected(rows[i], 0) == aotx_ar_id(i, 0), "task requirement precedes the complete source group");
        for (unsigned part = 0; part < 5; ++part)
            aotx_check(aotx_context_has(rows[i], aotx_ar_id(i, base + part)), "exact source, candidate, queue, assessment and relationship are selected");
        auto text = aotx_context(rows[i]);
        aotx_check(text.find(aotx_ar_text(i, corrected)) != std::string::npos, "exact actor report is present");
        aotx_check(text.find("benefit=" + std::to_string(800000 + i) + " harm=" + std::to_string(corrected ? 0 : 900000 - i)) != std::string::npos,
            "benefit and harm remain separate");
        aotx_check(text.find("regard_gain=" + std::to_string(700000 + i)) != std::string::npos &&
            text.find("task_trust_loss=unknown") != std::string::npos && text.find("exposure=1") != std::string::npos,
            "relationship values preserve uncertainty and one exposure");
        aotx_check(text.find("reported commitment candidate: unknown") != std::string::npos && text.find("no authority") != std::string::npos,
            "a reported commitment confers no authority");
        if (corrected) aotx_check(!aotx_context_has(rows[i], aotx_ar_id(i, 6)) && !aotx_context_has(rows[i], aotx_ar_id(i, 7)),
            "superseded assessment and relationship are absent");
    }
}
#endif
