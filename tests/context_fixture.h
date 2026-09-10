/* Purpose: Build distinct task, subject, memory and appraisal fixtures.
 * Owns: Independent portable bytes and expected IDs for both batch sizes.
 * Launch shape: One or 64 principals with separate task cues and subjects.
 * Lifetime: One recall or live-memory test process. */
#ifndef AOTX_TEST_CONTEXT_FIXTURE_H
#define AOTX_TEST_CONTEXT_FIXTURE_H
#include "recall_fixture.h"

static uint64_t aotx_context_id(unsigned i, unsigned part) { return 500000 + i * 64 + part; }
static aotx_bytes aotx_context_text(unsigned i, bool required, const std::string &text) {
    aotx_bytes p(64 + text.size(), 0); memcpy(p.data(), "AOTXMEM2", 8);
    aotx_put(p.data() + 8, 2, 4); aotx_put(p.data() + 12, text.size(), 4);
    aotx_id(p.data() + 16, 4000 + i); aotx_put(p.data() + 32, required, 4);
    memcpy(p.data() + 64, text.data(), text.size()); return p;
}
static aotx_bytes aotx_context_appraisal(unsigned benefit, unsigned harm, unsigned i) {
    aotx_bytes p(32, 0); aotx_put(p.data(), 1, 4);
    aotx_put(p.data() + 4, benefit, 4); aotx_put(p.data() + 8, harm, 4);
    aotx_put(p.data() + 12, 600000 + i, 4); aotx_put(p.data() + 16, 3, 4);
    aotx_put(p.data() + 20, AOTX_COG_UNKNOWN, 4); aotx_put(p.data() + 24, 1, 4); return p;
}
static void aotx_context_source(aotx_row &r, unsigned i, unsigned part) {
    aotx_id(r.data() + AOTX_CO_SOURCE, aotx_context_id(i, part));
    aotx_put(r.data() + AOTX_CO_SOURCE_VERSION, 1);
}
static aotx_fixture aotx_context_corpus(unsigned n, unsigned scope = 0) {
    aotx_fixture f;
    for (unsigned i = 0; i < n; ++i) for (unsigned part = 0; part < 12; ++part) {
        unsigned kind = part < 3 ? AOTX_COG_COMPONENT : part == 3 ? AOTX_COG_EVENT :
            part == 5 ? AOTX_COG_CUE : part < 9 ? AOTX_COG_ASSERTION : AOTX_COG_APPRAISAL;
        auto r = aotx_memory_row(i, kind, aotx_context_id(i, part), f.rows.size() + 1, scope);
        aotx_bytes p;
        if (part < 3) {
            const float x[] = {1.0f, 0.7f, -1.0f}, y[] = {0.1f, 0.7f, 0.0f};
            p = aotx_memory_vector(x[part], y[part] + i * 0.0003f, 0.01f);
        } else if (part == 3) p = aotx_memory_text("reported source " + std::to_string(i));
        else if (part < 9) {
            aotx_context_source(r, i, 3);
            if (part == 5) memset(r.data() + AOTX_CO_SUBJECT, 0, 16);
            if (part >= 6) {
                aotx_id(r.data() + AOTX_CO_EMBEDDING, aotx_context_id(i, part - 6));
                aotx_put(r.data() + AOTX_CO_EMBED_VERSION, 1);
                aotx_put(r.data() + AOTX_CO_EVIDENCE, part == 6 ? 1 : 2, 4);
                aotx_put(r.data() + AOTX_CO_SOURCE_KIND, AOTX_COG_INFERRED, 4);
            }
            const char *name[] = {"constraint ", "ask unknown requirements ", "ordinary episode ", "mixed outcome ", "irrelevant episode "};
            p = aotx_context_text(i, part < 6, std::string(name[part - 4]) + std::to_string(i));
        } else {
            aotx_context_source(r, i, part == 9 ? 7 : part == 10 ? 8 : 6);
            aotx_put(r.data() + AOTX_CO_SOURCE_KIND, AOTX_COG_INFERRED, 4);
            p = aotx_context_appraisal(part == 9 ? 700000 + i : 100000,
                part == 9 ? 900000 - i : part == 10 ? 1000000 : 100000, i);
        }
        f.add(r, p);
    }
    return f;
}
static void aotx_context_extend(unsigned char *q, unsigned i, unsigned flags = 3, bool unknown = false) {
    auto p = q + AOTX_RECALL_EXTENSION; memset(p, 0, AOTX_RECALL_QUERY - AOTX_RECALL_EXTENSION);
    if (!flags) return;
    memcpy(p, "AOTXCTX1", 8); aotx_put(p + 8, 1, 4); aotx_put(p + 12, flags, 4); aotx_put(p + 44, 1, 4);
    if (flags & 1) { aotx_id(p + 16, 4000 + i); aotx_put(p + 32, 1, 4); aotx_id(p + 48, (unknown ? 7000 : 3000) + i); }
    if (flags & 2) { aotx_put(p + 36, 600000, 4); aotx_put(p + 40, 800000, 4); }
}
static aotx_bytes aotx_context_queries(unsigned n, uint64_t cut, unsigned scope = 0, unsigned flags = 3) {
    auto q = aotx_memory_queries(n, cut, scope);
    for (unsigned i = 0; i < n; ++i) {
        auto p = aotx_query_at(q, i); aotx_context_extend(p, i, flags);
        aotx_put(p + 132, 4, 4); aotx_put(p + 136, 4096, 4);
        aotx_float_put(p + 160, 1 + i * 0.001f); aotx_float_put(p + 164, (i % 7) * 0.002f); aotx_float_put(p + 168, 0.01f);
    }
    return q;
}
static bool aotx_context_has(const aotx_recall_result &r, uint64_t id) {
    for (unsigned i = 0; i < r.count; ++i) if (aotx_selected(r, i) == id) return true;
    return false;
}
static void aotx_context_expected(const std::vector<aotx_recall_result> &rows, unsigned n) {
    aotx_status_rows(rows, 0, "contextual recall admission");
    for (unsigned i = 0; i < n; ++i) {
        const unsigned ids[] = {4, 5, 7, 9}, reasons[] = {4, 4, 6, 5};
        aotx_check(rows[i].count == 4, "obligations and exact significant pair fit together");
        for (unsigned j = 0; j < 4; ++j)
            aotx_check(aotx_selected(rows[i], j) == aotx_context_id(i, ids[j]) && rows[i].reason[j] == reasons[j],
                "task and subject bind every selected row and reason");
        auto text = aotx_context(rows[i]);
        aotx_check(text.find("benefit=" + std::to_string(700000 + i) + " harm=" + std::to_string(900000 - i)) != std::string::npos,
            "mixed benefit and harm remain separate without cancellation");
        aotx_check(text.find("confidence=unknown") != std::string::npos && text.find("source=4 evidence=2") != std::string::npos,
            "salience does not promote source kind, evidence or unknown confidence");
        aotx_check(text.find("constraint " + std::to_string(i) + "\n") != std::string::npos &&
            text.find("irrelevant episode") == std::string::npos, "mandatory source survives irrelevant intense memory");
    }
}
#endif
