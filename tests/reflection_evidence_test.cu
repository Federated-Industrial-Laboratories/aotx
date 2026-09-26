/* Purpose: Check source-bound task review construction and complete recall.
 * Owns: Distinct evidence, task, scope and defective dependency fixtures.
 * Launch shape: Actual GPU construction and recall at N=1 and N=64.
 * Lifetime: One test process without model weights. */
#include "appraisal_recall_fixture.h"
#include "reflection/build.cuh"
#include "cognitive/recall_context.cuh"

__global__ void aotx_review_test_build(const aotx_cognitive_store *s, const unsigned char *q,
    const uint32_t *indices, uint32_t count, unsigned char *tail, aotx_cognitive_result *result) {
    aotx_review_build_block(s, q + 64, indices, count, tail, result);
}
__global__ void aotx_review_test_validate(const aotx_cognitive_store *s, const unsigned char *q,
    aotx_recall_result *rows, uint32_t count) {
    for (uint32_t i = threadIdx.x; i < count; i += blockDim.x) {
        uint32_t status = aotx_recall_render(s, q + 64 + i * AOTX_RECALL_QUERY, rows + i);
        if (status) aotx_recall_refuse(rows + i, status);
    }
}
static bool aotx_review_construction_only;
static void aotx_review_evidence_case(unsigned count, unsigned scope) {
    auto f = aotx_ar_corpus(count, false, scope);
    auto queries = aotx_ar_queries(count, f.rows.size(), scope);
    std::vector<uint32_t> indices(count);
    for (unsigned i = 0; i < count; ++i) {
        for (unsigned j = 0; j < f.rows.size(); ++j)
            if (aotx_get(f.rows[j].data() + AOTX_CO_ID) == aotx_ar_id(i, 6)) indices[i] = j;
        aotx_put(aotx_query_at(queries, i) + 132, AOTX_RECALL_LIMIT, 4);
    }
    aotx_recall_device d;
    aotx_check(!d.load(f.wire(false, f.rows.size())).status, "distinct source evidence is admitted");
    AOTX_CUDA(cudaMemcpy(d.requests, queries.data(), queries.size(), cudaMemcpyHostToDevice));
    uint32_t *device_indices;
    AOTX_CUDA(cudaMalloc(&device_indices, count * sizeof(uint32_t)));
    AOTX_CUDA(cudaMemcpy(device_indices, indices.data(), count * sizeof(uint32_t), cudaMemcpyHostToDevice));
    auto before = d.checkpoint();
    aotx_review_test_build<<<1,64>>>(d.live, d.requests, device_indices, count, d.image, d.result);
    auto built = d.finish();
    aotx_check(!built.status && built.applied == 2 * count, "the GPU builds each cue and exact dependency selection");
    if (built.status) { cudaFree(device_indices); return; }
    aotx_bytes tail(built.bytes);
    AOTX_CUDA(cudaMemcpy(tail.data(), d.image, tail.size(), cudaMemcpyDeviceToHost));
    aotx_check(d.checkpoint() == before, "construction does not mutate the source store");
    AOTX_CUDA(cudaMemcpy(d.image, tail.data(), tail.size(), cudaMemcpyHostToDevice));
    aotx_cognitive_apply<<<1,64>>>(d.live, d.stage, d.image, tail.size(), d.result);
    auto applied = d.finish();
    aotx_check(!applied.status && applied.applied == 2 * count, "complete reviews publish atomically");
    if (applied.status) { cudaFree(device_indices); return; }
    aotx_put(queries.data() + 32, f.rows.size() + 2 * count);
    auto rows = d.search(queries, count);
    aotx_status_rows(rows, 0, "matching tasks consume complete reviews");
    for (unsigned i = 0; i < count; ++i) {
        auto text = aotx_context(rows[i]);
        aotx_check(text.find(AOTX_REVIEW_TEXT) != std::string::npos, "the later task contains the internal review cue");
        aotx_check(text.find(aotx_ar_text(i, false)) != std::string::npos, "the exact source remains in task context");
        for (unsigned k : {3u, 5u, 6u, 7u})
            aotx_check(aotx_context_has(rows[i], aotx_ar_id(i, k)), "all supporting evidence is selected");
        aotx_check(aotx_context_has(rows[i], 4000 + i), "the registered task remains in context");
    }
    if (aotx_review_construction_only) { cudaFree(device_indices); return; }
    for (unsigned defect = 0; defect < 2; ++defect) {
        auto damaged = rows;
        for (unsigned i = 0; i < count; ++i) {
            auto &r = damaged[i]; unsigned at = 0;
            while (at < r.count && aotx_selected(r, at) != aotx_ar_id(i, defect ? 7 : 3)) ++at;
            aotx_check(at < r.count, "the defect targets a required source dependency");
            if (at < r.count) {
                for (unsigned j = at; j + 1 < r.count; ++j) {
                    r.index[j] = r.index[j + 1]; memcpy(r.selection + 16 + j * 32, r.selection + 48 + j * 32, 32);
                }
                --r.count; memset(r.selection + 16 + r.count * 32, 0, 32); aotx_put(r.selection + 4, r.count, 4);
            }
        }
        AOTX_CUDA(cudaMemcpy(d.rows, damaged.data(), count * sizeof(*d.rows), cudaMemcpyHostToDevice));
        aotx_review_test_validate<<<1,64>>>(d.live, d.requests, d.rows, count);
        AOTX_CUDA(cudaMemcpy(damaged.data(), d.rows, count * sizeof(*d.rows), cudaMemcpyDeviceToHost));
        aotx_status_rows(damaged, AOTX_COG_REFERENCE, "omitted source or relationship cannot render a partial review");
    }
    auto saved = d.checkpoint();
    aotx_check(!d.load(saved).status, "review objects restore with exact dependencies");
    auto recovered = d.search(queries, count);
    for (unsigned i = 0; i < count; ++i)
        aotx_check(aotx_context(rows[i]) == aotx_context(recovered[i]), "restored task context is identical");
    aotx_review_test_build<<<1,64>>>(d.live, d.requests, device_indices, count, d.image, d.result);
    aotx_check(d.finish().status == AOTX_COG_REFERENCE, "completed source reviews cannot be duplicated");
    for (unsigned mode = 0; mode < 3; ++mode) {
        auto changed = queries;
        for (unsigned i = 0; i < count; ++i) {
            auto q = aotx_query_at(changed, i), c = q + AOTX_RECALL_EXTENSION;
            if (mode == 0) aotx_id(c + 16, 123000 + i);
            if (mode == 1) aotx_id(c + 48, 124000 + i);
            if (mode == 2) aotx_put(q + 132, 2, 4);
        }
        auto excluded = d.search(changed, count);
        for (const auto &r : excluded) {
            aotx_check(aotx_context(r).find(AOTX_REVIEW_TEXT) == std::string::npos,
                "wrong task, wrong subject and small context cannot expose a partial review");
            if (mode == 2) aotx_check(r.status == AOTX_COG_CAPACITY, "a required review that cannot fit refuses");
        }
    }
    aotx_check(d.checkpoint() == saved, "review recall and refusal create no evidence or state changes");
    aotx_fixture correction;
    for (unsigned i = 0; i < count; ++i) aotx_ar_episode(correction, i, true, scope);
    uint64_t cut = f.rows.size() + 2 * count;
    for (auto &r : correction.rows) {
        aotx_put(r.data() + AOTX_CO_CREATED, aotx_get(r.data() + AOTX_CO_CREATED) + cut);
        aotx_put(r.data() + AOTX_CO_UPDATED, aotx_get(r.data() + AOTX_CO_UPDATED) + cut);
    }
    auto changed = correction.wire(true, cut + 1, 7);
    AOTX_CUDA(cudaMemcpy(d.image, changed.data(), changed.size(), cudaMemcpyHostToDevice));
    aotx_cognitive_apply<<<1,64>>>(d.live, d.stage, d.image, changed.size(), d.result);
    aotx_check(!d.finish().status, "corrected source evidence is admitted without changing old records");
    cut += correction.rows.size(); aotx_put(queries.data() + 32, cut);
    auto corrected = d.search(queries, count);
    for (const auto &r : corrected) aotx_check(aotx_context(r).find(AOTX_REVIEW_TEXT) == std::string::npos,
        "a correction invalidates the old review through its exact dependency selection");
    for (unsigned i = 0; i < count; ++i) indices[i] = f.rows.size() + 2 * count + 5 * i + 3;
    AOTX_CUDA(cudaMemcpy(device_indices, indices.data(), count * sizeof(uint32_t), cudaMemcpyHostToDevice));
    aotx_review_test_build<<<1,64>>>(d.live, d.requests, device_indices, count, d.image, d.result);
    auto replacement = d.finish();
    aotx_check(!replacement.status, "each corrected source can produce a new supported review");
    if (!replacement.status) {
        aotx_cognitive_apply<<<1,64>>>(d.live, d.stage, d.image, replacement.bytes, d.result);
        aotx_check(!d.finish().status, "replacement reviews publish with corrected exact evidence");
        cut += 2 * count; aotx_put(queries.data() + 32, cut);
        auto fresh = d.search(queries, count);
        for (unsigned i = 0; i < count; ++i) aotx_check(!fresh[i].status &&
            aotx_context(fresh[i]).find(AOTX_REVIEW_TEXT) != std::string::npos &&
            aotx_context(fresh[i]).find(aotx_ar_text(i, true)) != std::string::npos &&
            !aotx_context_has(fresh[i], aotx_ar_id(i, 6)), "later tasks consume the corrected outcome and exclude the superseded appraisal");
    }
    aotx_fixture withdrawal;
    for (unsigned i = 0; i < count; ++i) {
        auto r = correction.rows[i * 5];
        aotx_put(r.data() + AOTX_CO_FLAGS, AOTX_COG_TOMBSTONE, 4);
        aotx_put(r.data() + AOTX_CO_VERSION, 2);
        aotx_put(r.data() + AOTX_CO_UPDATED, cut + i + 1);
        withdrawal.add(r, {});
    }
    auto withdrawn = withdrawal.wire(true, cut + 1, 9);
    AOTX_CUDA(cudaMemcpy(d.image, withdrawn.data(), withdrawn.size(), cudaMemcpyHostToDevice));
    aotx_cognitive_apply<<<1,64>>>(d.live, d.stage, d.image, withdrawn.size(), d.result);
    aotx_check(!d.finish().status, "source withdrawal is an admitted versioned update");
    aotx_put(queries.data() + 32, cut + count);
    auto absent = d.search(queries, count);
    for (const auto &r : absent) aotx_check(aotx_context(r).find(AOTX_REVIEW_TEXT) == std::string::npos,
        "withdrawn evidence cannot support a later task review");
    cudaFree(device_indices);
}
int main(int argc, char **argv) {
    if (argc > 2 || (argc == 2 && strcmp(argv[1], "--construction-only"))) return 2;
    aotx_review_construction_only = argc == 2;
    for (unsigned count : {1u, 64u}) for (unsigned scope : {0u, 1u, 2u})
        if (!aotx_review_construction_only || !scope) aotx_review_evidence_case(count, scope);
    printf("reflection evidence: %u checks, %u failures\n", aotx_checks, aotx_failures);
    return aotx_failures ? 1 : 0;
}
