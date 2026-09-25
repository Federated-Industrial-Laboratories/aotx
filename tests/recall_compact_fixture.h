/* Purpose: Build exact source pairs for bounded rendering checks.
 * Owns: Prepared source, appraisal and query fixture bytes.
 * Launch shape: Distinct N=1 and N=64 inputs.
 * Lifetime: One test process without model weights. */
#ifndef AOTX_RECALL_COMPACT_FIXTURE_H
#define AOTX_RECALL_COMPACT_FIXTURE_H
#include "appraisal_recall_fixture.h"
static aotx_fixture aotx_compact_corpus(unsigned n) {
    auto f = aotx_ar_corpus(2 * n);
    for (unsigned j = 1; j < f.rows.size(); ++j) {
        auto &r = f.rows[j]; unsigned owner = aotx_get(r.data() + AOTX_CO_OWNER) - 1000;
        aotx_id(r.data() + AOTX_CO_OWNER, 1000 + owner / 2);
        unsigned kind = aotx_get(r.data() + AOTX_CO_KIND, 2); auto &p = f.payloads[j];
        std::string text = "Member " + std::to_string(owner) + " secured the crate but broke the glass handle. ";
        text += "Reply with exactly one word: noted. "; text.resize(200, 'a' + owner % 26);
        if (kind == AOTX_COG_EVENT || kind == AOTX_COG_WORKING) p = aotx_memory_text(text);
        if (kind == AOTX_COG_APPRAISAL) aotx_put(p.data() + 124, text.size(), 4);
        if (kind == AOTX_COG_RELATIONSHIP) {
            memset(p.data() + 32, 0, 16); aotx_put(p.data() + 52, text.size(), 4);
            memset(p.data() + 56, 0, 16);
            aotx_put(p.data() + 24, AOTX_COG_UNKNOWN, 4); aotx_put(p.data() + 28, AOTX_COG_UNKNOWN, 4);
        }
        if (kind == AOTX_COG_POLICY) { memset(p.data() + 40, 0, 16); memset(p.data() + 128, 0, 24); }
    }
    return f;
}
static void aotx_compact_version(unsigned char *q, unsigned version) {
    auto c = q + AOTX_RECALL_EXTENSION; memset(c, 0, AOTX_RECALL_QUERY - AOTX_RECALL_EXTENSION);
    memcpy(c, version == 4 ? "AOTXCTX4" : version == 3 ? "AOTXCTX3" : "AOTXCTX2", 8);
    aotx_put(c + 8, version, 4); aotx_put(c + 44, version, 4);
    aotx_put(c + 12, 2, 4); aotx_put(c + 40, 1000000, 4);
}
#endif
