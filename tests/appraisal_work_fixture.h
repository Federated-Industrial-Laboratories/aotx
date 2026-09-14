/* Purpose: Drive appraisal admission and recorded results with independent source bytes.
 * Owns: Distinct source batches and a controlled decoder output boundary.
 * Launch shape: Real live kernels at N=1 and N=64 without model weights.
 * Lifetime: One maintained admission and recovery test process. */
#ifndef AOTX_TEST_APPRAISAL_WORK_FIXTURE_H
#define AOTX_TEST_APPRAISAL_WORK_FIXTURE_H
#include "retain_fixture.h"
#include "appraisal/appraisal.cuh"
#include "model/load.cuh"
#include "policy/state.cuh"

static const unsigned char aotx_appraisal_test_processor[32] = AOTX_APPRAISAL_PROCESSOR_BYTES;
static void aotx_appraisal_join(aotx_live_records &out, const aotx_live_records &rows) { out.insert(out.end(), rows.begin(), rows.end()); }
__global__ void aotx_appraisal_test_models(void) {
    aotx_model_load.files = 1; aotx_model_load.file[0].role = AOTX_MODEL_LANGUAGE;
    for (unsigned j = 0; j < 32; ++j) aotx_model_load.file[0].digest[j] = 77 + j;
}
struct aotx_appraisal_device : aotx_live_device {
    explicit aotx_appraisal_device(unsigned n) : aotx_live_device(n) {
        AOTX_LIVE_CLEAR(aotx_appraisal); AOTX_LIVE_CLEAR(aotx_intake); AOTX_LIVE_CLEAR(aotx_policy);
        AOTX_LIVE_CLEAR(aotx_model_load); AOTX_LIVE_CLEAR(aotx_decode); AOTX_LIVE_CLEAR(aotx_seqs); AOTX_LIVE_CLEAR(aotx_kv);
        aotx_appraisal_test_models<<<1,1>>>(); AOTX_CUDA(cudaDeviceSynchronize());
    }
    aotx_appraisal_state appraisal() {
        aotx_appraisal_state s; AOTX_CUDA(cudaMemcpyFromSymbol(&s, aotx_appraisal, sizeof(s))); return s;
    }
};
static aotx_bytes aotx_appraisal_config_bytes(unsigned flags = 1) {
    aotx_bytes p(96, 0); memcpy(p.data(), "AOTXAPC1", 8);
    aotx_put(p.data() + 8, 1, 4); aotx_put(p.data() + 12, flags, 4);
    aotx_put(p.data() + 16, 160, 4); aotx_put(p.data() + 20, 512, 4);
    aotx_put(p.data() + 24, 16384, 4); aotx_put(p.data() + 36, 64, 4);
    memcpy(p.data() + 40, aotx_appraisal_test_processor, 32); return p;
}
static aotx_live_records aotx_appraisal_sources(aotx_appraisal_device &d, unsigned n, unsigned scope = 0) {
    aotx_fixture empty;
    auto records = d.send(aotx_live_load_bytes(empty.wire(false, 0)), 1);
    aotx_appraisal_join(records, d.send(aotx_appraisal_config_bytes(), AOTX_APPRAISAL_CONTROL));
    aotx_check(!d.state().status, "appraisal configuration is admitted to typed memory");
    auto bound = aotx_live_binding_bytes(n, 1, scope);
    for (unsigned i = 0; i < n; ++i) aotx_put(bound.data() + 124 + i * 64, 1, 4);
    aotx_appraisal_join(records, d.send(bound, 3));
    auto before = aotx_retain_store();
    auto query = aotx_retain_query(n, 1, 1, false, scope, 3, 80);
    auto parts = d.process(aotx_live_parts(query, 4, d.next_id++), false, false);
    aotx_check(d.state().phase == AOTX_LIVE_WRITE && aotx_retain_store() == before,
        "pending queues and external sources remain unpublished until the whole result is recorded");
    aotx_appraisal_join(parts, d.process({})); aotx_appraisal_join(records, parts);
    aotx_check(!d.state().status, "automatic source and pending queue batch is admitted");
    auto saved = aotx_retain_store(); auto s = (const aotx_cognitive_store *)saved.data();
    aotx_check(s->count == 1 + n * 4 && s->sequence == 1 + n * 4, "one configuration and four objects per external source");
    for (unsigned i = 0; i < n; ++i) {
        const auto q = s->objects[1 + 3 * n + i], event = s->objects[1 + 3 * i];
        const auto p = s->payload + aotx_get(q + AOTX_CO_OFFSET);
        aotx_check(aotx_get(q + AOTX_CO_KIND, 2) == AOTX_COG_POLICY && !memcmp(p, "AOTXAPQ1", 8), "typed pending queue exists");
        aotx_check(aotx_get(q + AOTX_CO_BYTES) == 160 && !aotx_get(p + 12, 4) && !aotx_get(p + 56, 4), "queue starts pending without a result");
        aotx_check(!memcmp(q + AOTX_CO_SOURCE, event + AOTX_CO_ID, 16) && aotx_get(q + AOTX_CO_SOURCE_VERSION) == 1,
            "queue refers to its exact independent source");
        aotx_check(!memcmp(q + AOTX_CO_OWNER, event + AOTX_CO_OWNER, 32) &&
            !memcmp(q + AOTX_CO_SUBJECT, event + AOTX_CO_SUBJECT, 16) && aotx_get(q + AOTX_CO_SCOPE, 4) == scope,
            "queue keeps the admitted actor, owner and scope");
        aotx_check(aotx_get(q + AOTX_CO_RETENTION, 4) == 2 && !memcmp(p + 64, aotx_appraisal_test_processor, 32),
            "pending work retains its exact processor and source dependency");
    }
    d.idle(n); return records;
}
static aotx_bytes aotx_appraisal_work_request(unsigned n) {
    auto saved = aotx_retain_store(); auto s = (const aotx_cognitive_store *)saved.data();
    aotx_bytes p(64 + n * 32, 0); memcpy(p.data(), "AOTXAPR1", 8);
    aotx_put(p.data() + 8, 1, 4); aotx_put(p.data() + 12, n, 4); aotx_put(p.data() + 16, s->sequence);
    memcpy(p.data() + 24, s->objects[0] + AOTX_CO_ID, 16); aotx_put(p.data() + 40, 1);
    for (unsigned i = 0; i < n; ++i) {
        memcpy(p.data() + 64 + i * 32, s->objects[1 + 3 * n + i] + AOTX_CO_ID, 16);
        aotx_put(p.data() + 80 + i * 32, 1); aotx_put(p.data() + 88 + i * 32, i, 4);
    }
    return p;
}
__global__ void aotx_appraisal_test_output(const unsigned char *p, const unsigned *lengths, unsigned status) {
    unsigned i = threadIdx.x;
    if (i < aotx_appraisal.count) {
        auto r = aotx_intake.rows + i; r->bytes = lengths[i]; r->status = status;
        for (unsigned j = 0; j < r->bytes; ++j) r->reply[j] = p[i * 4096 + j];
        for (unsigned j = 0; j < 32; ++j) r->model[j] = 77 + j;
    }
    if (!i) { aotx_live.status = status; aotx_live.phase = AOTX_INTAKE_DONE; }
}
static void aotx_appraisal_output(unsigned n, unsigned status = 0) {
    aotx_bytes all(n * 4096, 0); std::vector<unsigned> sizes(n);
    for (unsigned i = 0; i < n; ++i) {
        std::string text = "{\"benefit\":" + std::to_string(100000 + i) +
            ",\"harm\":" + std::to_string(800000 - i) + ",\"arousal\":4294967295,\"consequence\":3,\"confidence\":500000,"
            "\"regard_gain\":700000,\"regard_loss\":100000,\"trust_gain\":4294967295,\"trust_loss\":4294967295,"
            "\"evidence\":\"retained source " + std::to_string(i) + " \",\"task\":\"\",\"commitment\":\"\",\"correction\":0}";
        sizes[i] = text.size(); memcpy(all.data() + i * 4096, text.data(), text.size());
    }
    unsigned char *p; unsigned *lengths;
    AOTX_CUDA(cudaMalloc(&p, all.size())); AOTX_CUDA(cudaMalloc(&lengths, n * 4));
    AOTX_CUDA(cudaMemcpy(p, all.data(), all.size(), cudaMemcpyHostToDevice));
    AOTX_CUDA(cudaMemcpy(lengths, sizes.data(), n * 4, cudaMemcpyHostToDevice));
    aotx_appraisal_test_output<<<1,64>>>(p, lengths, status); AOTX_CUDA(cudaDeviceSynchronize());
    cudaFree(lengths); cudaFree(p);
}
#endif
