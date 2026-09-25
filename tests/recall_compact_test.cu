/* Purpose: Check exact source references, bounded groups and prior recall formats.
 * Owns: Independent source pairs and complete-selection expectations.
 * Launch shape: Distinct N=1 and N=64 queries on the real device path.
 * Lifetime: One test process without model weights. */
#include "recall_compact_fixture.h"
#include "source_fixture.h"
#include "cognitive/recall_compact.cuh"

__global__ void aotx_compact_identity(aotx_cognitive_store *s, const unsigned char *q,
    const unsigned *indices, unsigned *answers, unsigned n) {
    unsigned i = threadIdx.x; if (i >= n) return;
    unsigned working = indices[i * 3], event = indices[i * 3 + 1], other = indices[i * 3 + 2];
    const unsigned char *query = q + 64 + i * AOTX_RECALL_QUERY;
    unsigned selected[2] = {working, event}; unsigned *out = answers + i * 5;
    out[0] = aotx_recall_body_reference(s, query, working, selected, 2);
    out[1] = aotx_recall_body_reference(s, query, working, selected, 1);
    unsigned char *r = s->objects[working], *p = s->payload + aotx_cog_u64(r + AOTX_CO_OFFSET);
    uint64_t version = aotx_cog_u64(r + AOTX_CO_SOURCE_VERSION);
    aotx_cog_put(r + AOTX_CO_SOURCE_VERSION, version + 1, 8);
    out[2] = aotx_recall_body_reference(s, query, working, selected, 2);
    aotx_cog_put(r + AOTX_CO_SOURCE_VERSION, version, 8);
    p[32] ^= 1; out[3] = aotx_recall_body_reference(s, query, working, selected, 2); p[32] ^= 1;
    unsigned char *ep = s->payload + aotx_cog_u64(s->objects[other] + AOTX_CO_OFFSET);
    unsigned char saved[232];
    for (unsigned j = 0; j < 232; ++j) { saved[j] = ep[j]; ep[j] = p[j]; }
    selected[1] = other; out[4] = aotx_recall_body_reference(s, query, working, selected, 2);
    for (unsigned j = 0; j < 232; ++j) ep[j] = saved[j];
}
static void aotx_compact_case(unsigned n) {
    auto f = aotx_compact_corpus(n); auto image = f.wire(false, f.rows.size());
    aotx_recall_device d; aotx_check(!d.load(image).status, "paired source corpus admission");
    auto q = aotx_memory_queries(n, f.rows.size());
    for (unsigned i = 0; i < n; ++i) {
        aotx_compact_version(aotx_query_at(q, i), 3); aotx_put(aotx_query_at(q, i) + 132, 10, 4);
    }
    std::vector<unsigned> indices, answers(n * 5);
    for (unsigned i = 0; i < n; ++i) for (unsigned part : {4u, 3u, 35u}) {
        unsigned index = 0;
        while (index < f.rows.size() && aotx_get(f.rows[index].data() + AOTX_CO_ID) != aotx_ar_id(2 * i, part)) ++index;
        aotx_check(index < f.rows.size(), "independent identity control index"); indices.push_back(index);
    }
    unsigned *di, *da; AOTX_CUDA(cudaMalloc(&di, indices.size() * sizeof(unsigned)));
    AOTX_CUDA(cudaMalloc(&da, answers.size() * sizeof(unsigned)));
    AOTX_CUDA(cudaMemcpy(di, indices.data(), indices.size() * sizeof(unsigned), cudaMemcpyHostToDevice));
    AOTX_CUDA(cudaMemcpy(d.requests, q.data(), q.size(), cudaMemcpyHostToDevice));
    aotx_compact_identity<<<1,64>>>(d.live, d.requests, di, da, n);
    AOTX_CUDA(cudaMemcpy(answers.data(), da, answers.size() * sizeof(unsigned), cudaMemcpyDeviceToHost));
    for (unsigned i = 0; i < n; ++i) for (unsigned mode = 0; mode < 5; ++mode)
        aotx_check(answers[i * 5 + mode] == (mode == 0), "only the selected exact source version and full bytes permit a body reference");
    aotx_check(d.checkpoint() == image, "identity controls restore all mutated fixture bytes");
    cudaFree(di); cudaFree(da);
    auto compact = d.search(q, n); aotx_status_rows(compact, 0, "compact source pair search");
    for (unsigned i = 0; i < n; ++i) {
        aotx_check(compact[i].count == 10, "two complete source groups fit the original byte and reference limits");
        for (unsigned source = 2 * i; source < 2 * i + 2; ++source)
            for (unsigned part = 3; part < 8; ++part)
                aotx_check(aotx_context_has(compact[i], aotx_ar_id(source, part)), "all exact source and appraisal references remain selected");
        auto text = aotx_context(compact[i]); auto at = text.find("text: see source_ref\n");
        aotx_check(at != std::string::npos && text.find("text: see source_ref\n", at + 1) != std::string::npos,
            "each duplicate working body has an explicit source reference");
        aotx_check(text.find("Reply with exactly one word: noted.") != std::string::npos,
            "historical commands remain verbatim source evidence");
    }
    auto exact = q;
    for (unsigned i = 0; i < n; ++i)
        aotx_put(aotx_query_at(exact, i) + 136, compact[i].context_bytes - 8 - aotx_get(aotx_query_at(q, i) + 148, 4), 4);
    auto bounded = d.search(exact, n); aotx_status_rows(bounded, 0, "exact compact byte bound");
    for (unsigned i = 0; i < n; ++i) {
        aotx_check(aotx_context(bounded[i]) == aotx_context(compact[i]), "capacity uses the final rendered context");
        aotx_put(aotx_query_at(exact, i) + 136, aotx_get(aotx_query_at(exact, i) + 136, 4) - 1, 4);
    }
    auto short_rows = d.search(exact, n); aotx_status_rows(short_rows, 0, "one byte less preserves atomic source groups");
    for (const auto &r : short_rows) aotx_check(r.count == 5, "insufficient space never admits a partial second group");
    for (unsigned version : {2u, 3u}) {
        aotx_check(!d.load(image).status, "restore original source cut");
        for (unsigned i = 0; i < n; ++i) aotx_compact_version(aotx_query_at(q, i), version);
        auto rows = d.search(q, n); aotx_status_rows(rows, 0, "versioned source search");
        for (const auto &r : rows) {
            aotx_check(r.count == (version == 3 ? 10u : 5u), "disabled compaction detects lost source coverage");
            aotx_check((aotx_context(r).find("text: see source_ref\n") != std::string::npos) == (version == 3),
                "old queries keep their original full source rendering");
        }
        aotx_check(!d.record(q, n, true).status, "record exact versioned selection");
        auto saved = d.checkpoint(); aotx_check(!d.load(saved).status, "restore versioned selected state");
        auto replay = d.search(d.saved_queries(n), n, true); aotx_status_rows(replay, 0, "recorded context replay");
        for (unsigned i = 0; i < n; ++i) aotx_check(!replay[i].searches && aotx_context(replay[i]) == aotx_context(rows[i]),
            "both versions restore exact context without a new search");
    }
    aotx_check(!d.load(image).status, "unknown version setup");
    for (unsigned mode = 0; mode < 3; ++mode) {
        auto bad = q;
        for (unsigned i = 0; i < n; ++i) {
            auto c = aotx_query_at(bad, i) + AOTX_RECALL_EXTENSION;
            if (mode == 0) c[7] = '4';
            if (mode == 1) aotx_put(c + 8, 4, 4);
            if (mode == 2) aotx_put(c + 44, 4, 4);
        }
        aotx_status_rows(d.search(bad, n), AOTX_COG_FORMAT, "unknown rendering revisions refuse the entire query");
    }
}
static void aotx_compact_required(unsigned n) {
    auto f = aotx_source_corpus(n, 0, 1900); aotx_recall_device d;
    for (auto &r : f.rows) {
        unsigned owner = aotx_get(r.data() + AOTX_CO_OWNER) - 1000;
        if (aotx_get(r.data() + AOTX_CO_ID) >= aotx_source_id(owner, 1))
            aotx_id(r.data() + AOTX_CO_OWNER, 9000 + owner);
    }
    aotx_check(!d.load(f.wire(false, f.rows.size())).status, "required source and competing fragments load");
    for (unsigned version : {2u, 3u}) {
        auto q = aotx_source_queries(n, f.rows.size());
        for (unsigned i = 0; i < n; ++i) {
            auto p = aotx_query_at(q, i); auto c = p + AOTX_RECALL_EXTENSION;
            memcpy(c, version == 3 ? "AOTXCTX3" : "AOTXCTX2", 8);
            aotx_put(c + 8, version, 4); aotx_put(c + 44, version, 4);
            aotx_put(p + 132, 2, 4); aotx_put(p + 136, 2500, 4);
            aotx_pin(p, 0, 0, aotx_source_id(i, 0));
        }
        auto rows = d.search(q, n); aotx_status_rows(rows, 0, "compact complete-source preference");
        for (unsigned i = 0; i < n; ++i) {
            aotx_check(rows[i].count == 2 && aotx_selected(rows[i], 0) == aotx_source_id(i, 0) &&
                aotx_selected(rows[i], 1) == aotx_source_id(i, 0) + (version == 3 ? 30 : 2),
                "an exact required source permits its fitting complete working record before a fragment");
            aotx_check(rows[i].context_bytes - 8 - aotx_get(aotx_query_at(q, i) + 148, 4) <= 2500,
                "complete-source preference uses the compact byte limit");
        }
    }
}
int main(void) {
    for (unsigned n : {1u, 64u}) { aotx_compact_case(n); aotx_compact_required(n); printf("compact recall N=%u complete\n", n); }
    printf("compact recall: %u checks, %u failures\n", aotx_checks, aotx_failures);
    return aotx_failures ? 1 : 0;
}
