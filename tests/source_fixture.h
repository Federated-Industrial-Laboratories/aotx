/* Purpose: Form complete source events with redundant exact interpretations.
 * Owns: Independent expected source, actor, scope and vector values.
 * Launch shape: Distinct one and 64 owner batches.
 * Lifetime: One source selection test without model weights. */
#ifndef AOTX_TEST_SOURCE_FIXTURE_H
#define AOTX_TEST_SOURCE_FIXTURE_H
#include "recall_fixture.h"
#include "cognitive/intake.h"

static std::string aotx_source_unit(unsigned row) {
    unsigned point = 0x4e00 + row;
    return {char(0xe0 | (point >> 12)), char(0x80 | ((point >> 6) & 63)), char(0x80 | (point & 63))};
}
static std::string aotx_source_hex(uint64_t id) {
    unsigned char p[16]; aotx_id(p, id); const char *digits = "0123456789abcdef";
    std::string out;
    for (unsigned j = 0; j < 16; ++j) { out += digits[p[j] >> 4]; out += digits[p[j] & 15]; }
    return out;
}
static void aotx_source_query(unsigned char *q, uint64_t actor = 0) {
    auto p = q + AOTX_RECALL_EXTENSION;
    memset(p, 0, AOTX_RECALL_QUERY - AOTX_RECALL_EXTENSION);
    memcpy(p, "AOTXCTX2", 8); aotx_put(p + 8, 2, 4); aotx_put(p + 44, 2, 4);
    if (actor) aotx_id(q + AOTX_RECALL_ACTOR, actor);
}
static uint64_t aotx_source_id(unsigned owner, unsigned group) { return 524288 + owner * 4096 + group * 256; }
static std::string aotx_source_text(unsigned owner, unsigned group, unsigned length = 0) {
    std::string text = "Person" + std::to_string(owner) + " uses CUDA. Their peer is Iris.";
    if (!group) for (unsigned j = 0; j < 18; ++j) text += " Item" + std::to_string(j) + ".";
    if (length > text.size()) text.resize(length, 'x');
    return text;
}
static aotx_fixture aotx_source_corpus(unsigned n, unsigned scope = 0, unsigned length = 0) {
    aotx_fixture f;
    for (unsigned i = 0; i < n; ++i) for (unsigned group = 0; group < 3; ++group) {
        uint64_t id = aotx_source_id(i, group); auto text = aotx_source_text(i, group, length);
        auto event = aotx_memory_row(i, AOTX_COG_EVENT, id, f.rows.size() + 1, scope);
        aotx_id(event.data() + AOTX_CO_SUBJECT, 7000 + i * 3 + group);
        f.add(event, aotx_memory_text(text));
        auto vector = aotx_memory_row(i, AOTX_COG_COMPONENT, id + 1, f.rows.size() + 1, scope);
        aotx_id(vector.data() + AOTX_CO_SOURCE, id); aotx_put(vector.data() + AOTX_CO_SOURCE_VERSION, 1);
        f.add(vector, aotx_memory_vector(3 - group, group, 0));
        for (unsigned j = 0; j < (group ? 1u : 18u); ++j) {
            std::string quote = group ? "Their peer is Iris." : "Item" + std::to_string(j) + ".";
            auto r = aotx_memory_row(i, AOTX_COG_ASSERTION, id + 2 + j, f.rows.size() + 1, scope);
            memset(r.data() + AOTX_CO_SUBJECT, 0, 16); aotx_put(r.data() + AOTX_CO_SOURCE_KIND, AOTX_COG_INFERRED, 4);
            aotx_id(r.data() + AOTX_CO_SOURCE, id); aotx_put(r.data() + AOTX_CO_SOURCE_VERSION, 1);
            aotx_id(r.data() + AOTX_CO_EMBEDDING, id + 1); aotx_put(r.data() + AOTX_CO_EMBED_VERSION, 1);
            aotx_bytes p(AOTX_INTAKE_PAYLOAD + quote.size(), 0); memcpy(p.data(), "AOTXMEM3", 8);
            aotx_put(p.data() + 8, 3, 4); aotx_put(p.data() + 12, quote.size(), 4);
            aotx_put(p.data() + 16, 3, 4); aotx_put(p.data() + 20, text.find(quote), 4);
            memset(p.data() + 24, 0x51, 32); memset(p.data() + 56, 0x71, 32);
            memcpy(p.data() + AOTX_INTAKE_PAYLOAD, quote.data(), quote.size()); f.add(r, p);
        }
        auto working = aotx_memory_row(i, AOTX_COG_WORKING, id + 30, f.rows.size() + 1, scope);
        aotx_id(working.data() + AOTX_CO_SOURCE, id); aotx_put(working.data() + AOTX_CO_SOURCE_VERSION, 1);
        aotx_id(working.data() + AOTX_CO_EMBEDDING, id + 1); aotx_put(working.data() + AOTX_CO_EMBED_VERSION, 1);
        aotx_id(working.data() + AOTX_CO_SUBJECT, 90000 + i);
        f.add(working, aotx_memory_text(text));
    }
    return f;
}
static aotx_bytes aotx_source_queries(unsigned n, uint64_t cut, unsigned scope = 0, bool modern = true) {
    auto q = aotx_memory_queries(n, cut, scope);
    for (unsigned i = 0; i < n; ++i) {
        auto p = aotx_query_at(q, i); aotx_put(p + 132, 6, 4);
        aotx_float_put(p + 160, 1); aotx_float_put(p + 164, 0); aotx_float_put(p + 168, 0);
        if (modern) aotx_source_query(p, 8000 + i);
    }
    return q;
}
#endif
