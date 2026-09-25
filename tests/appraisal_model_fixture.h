/* Purpose: Supply distinct source and prior assessment rows for output checks.
 * Owns: Independent text, scope and actor fixtures with bounded device buffers.
 * Launch shape: One and 64 different source rows use the production parser and index.
 * Lifetime: One test process without model weights or persisted fixture state. */
#ifndef AOTX_TEST_APPRAISAL_MODEL_FIXTURE_H
#define AOTX_TEST_APPRAISAL_MODEL_FIXTURE_H
#include "cognitive_fixture.h"
#include "appraisal/parse.cuh"
#include "appraisal/grammar.cuh"
#include "cognitive/intake_index.cuh"
#include "sched/sched.cuh"

static std::string aotx_appraisal_test_source(unsigned i) {
    return "I helped with task " + std::to_string(i) + ". Ren\xc3\xa9 says \"check\". Soup \xf0\x9f\x8d\xb2. repeat repeat. I promise to finish.";
}
static aotx_bytes aotx_appraisal_test_text(const std::string &text) {
    aotx_bytes p(32 + text.size(), 0); memcpy(p.data(), "AOTXMEM1", 8);
    aotx_put(p.data() + 8, 1, 4); aotx_put(p.data() + 12, text.size(), 4);
    memcpy(p.data() + 32, text.data(), text.size()); return p;
}
static std::string aotx_appraisal_response(const std::string &quote, const std::string &values,
    const std::string &task = "", const std::string &commitment = "", const std::string &correction = "0") {
    const char *fields[] = {"benefit", "harm", "arousal", "consequence", "confidence",
        "regard_gain", "regard_loss", "trust_gain", "trust_loss"};
    std::string out = quote.empty() ? "{\"support\":0," : "{\"support\":1,";
    size_t start = 0;
    for (unsigned j = 0; j < 9; ++j) {
        size_t end = values.find(',', start);
        out += (j ? ",\"" : "\"") + std::string(fields[j]) + "\":" + values.substr(start, end - start);
        if (end == std::string::npos) break;
        start = end + 1;
    }
    return out + ",\"evidence\":\"" + quote + "\",\"task\":\"" + task + "\",\"commitment\":\"" + commitment + "\",\"correction\":" + correction + "}";
}
__global__ void aotx_appraisal_model_setup(const unsigned char *wire, unsigned n) {
    unsigned i = threadIdx.x;
    unsigned count = aotx_cog_u32(wire + 20), bytes = (unsigned)aotx_cog_u64(wire + 24);
    for (unsigned j = i; j < count * AOTX_COG_OBJECT; j += blockDim.x)
        ((unsigned char *)aotx_live_store.objects)[j] = wire[AOTX_COG_HEADER + j];
    for (unsigned j = i; j < bytes; j += blockDim.x)
        aotx_live_store.payload[j] = wire[AOTX_COG_HEADER + count * AOTX_COG_OBJECT + j];
    if (!i) {
        aotx_live_store.count = count; aotx_live_store.bytes = bytes; aotx_live_store.sequence = count;
        aotx_live.count = n; aotx_live.status = 0; aotx_live.phase = AOTX_INTAKE_RUN;
        aotx_appraisal.active = 1; aotx_appraisal.count = n;
        aotx_sched.held = 0; aotx_seam.replaying = 0;
    }
    __syncthreads();
    if (i >= n) return;
    aotx_appraisal_row *r = aotx_appraisal.rows + i; *r = {};
    r->source = 2 * n + i; r->slot = i; r->prior_count = 1; r->prior[0] = n + i;
    r->task_source = 3 * n + i;
    for (unsigned j = 0; j < 16; ++j) r->task[j] = aotx_live_store.objects[r->task_source][AOTX_CO_ID + j];
    aotx_intake.rows[i].state = 1; aotx_intake.rows[i].status = 0;
    aotx_intake.rows[i].bytes = 0; aotx_intake.row[i] = i + 1;
    const unsigned char *event = aotx_live_store.objects[r->source];
    const unsigned char *p = aotx_live_store.payload + aotx_cog_u64(event + AOTX_CO_OFFSET);
    unsigned char *q = aotx_live.requests + 64 + i * AOTX_RECALL_QUERY;
    for (unsigned j = 0; j < 4; ++j) q[148 + j] = p[12 + j];
    for (unsigned j = 0; j < aotx_cog_u32(p + 12); ++j) q[4640 + j] = p[32 + j];
}
struct aotx_appraisal_model_device {
    unsigned char *wire = nullptr, *reply = nullptr;
    unsigned *lengths = nullptr, *out = nullptr;
    unsigned n;
    explicit aotx_appraisal_model_device(unsigned count) : n(count) {
        aotx_fixture f;
        for (unsigned i = 0; i < n; ++i)
            f.add(aotx_object(i, AOTX_COG_EVENT, 1 + i, 1 + i), aotx_appraisal_test_text("Earlier contribution " + std::to_string(i) + "."));
        for (unsigned i = 0; i < n; ++i) {
            auto r = aotx_object(i, AOTX_COG_APPRAISAL, 1000 + i, n + i + 1);
            aotx_id(r.data() + AOTX_CO_SOURCE, i + 1); aotx_put(r.data() + AOTX_CO_SOURCE_VERSION, 1);
            aotx_put(r.data() + AOTX_CO_SOURCE_KIND, AOTX_COG_INFERRED, 4);
            aotx_bytes p(AOTX_APPRAISAL_ASSESS_BYTES, 0); aotx_put(p.data(), 2, 4);
            aotx_put(p.data() + 124, std::string("Earlier contribution " + std::to_string(i) + ".").size(), 4);
            f.add(r, p);
        }
        for (unsigned i = 0; i < n; ++i)
            f.add(aotx_object(i, AOTX_COG_EVENT, 2000 + i, 2 * n + i + 1), aotx_appraisal_test_text(aotx_appraisal_test_source(i)));
        for (unsigned i = 0; i < n; ++i) {
            auto r = aotx_object(i, AOTX_COG_CUE, 4000 + i, 3 * n + i + 1);
            aotx_put(r.data() + AOTX_CO_SOURCE_KIND, AOTX_COG_AUTHORED, 4);
            f.add(r, aotx_appraisal_test_text("task " + std::to_string(i)));
        }
        auto bytes = f.wire(false, 4 * n);
        AOTX_CUDA(cudaMalloc(&wire, bytes.size())); AOTX_CUDA(cudaMemcpy(wire, bytes.data(), bytes.size(), cudaMemcpyHostToDevice));
        AOTX_CUDA(cudaMallocManaged(&reply, n * AOTX_INTAKE_REPLY));
        AOTX_CUDA(cudaMallocManaged(&lengths, n * sizeof(unsigned))); AOTX_CUDA(cudaMallocManaged(&out, n * 16 * sizeof(unsigned)));
        reset();
    }
    void reset() {
        aotx_appraisal_model_setup<<<1,64>>>(wire, n); aotx_intake_index<<<n,64>>>(); AOTX_CUDA(cudaDeviceSynchronize());
    }
    void responses(const std::vector<std::string> &responses) {
        for (unsigned i = 0; i < n; ++i) {
            lengths[i] = responses[i].size();
            aotx_check(lengths[i] <= AOTX_INTAKE_REPLY, "response fits the declared output allocation");
            memcpy(reply + i * AOTX_INTAKE_REPLY, responses[i].data(), lengths[i]);
        }
    }
    ~aotx_appraisal_model_device() { cudaFree(wire); cudaFree(reply); cudaFree(lengths); cudaFree(out); }
};
__global__ void aotx_appraisal_model_parse(const unsigned char *bytes, const unsigned *lengths,
    unsigned *out, unsigned n, unsigned split) {
    unsigned i = threadIdx.x; if (i >= n) return;
    aotx_intake_row *r = aotx_intake.rows + i; r->bytes = lengths[i];
    aotx_appraisal.rows[i].prefix = {};
    for (unsigned j = 0; j < r->bytes; ++j) r->reply[j] = bytes[i * AOTX_INTAKE_REPLY + j];
    bool valid = true;
    for (unsigned j = 0; j < r->bytes && valid; j += split)
        valid = aotx_appraisal_advance(i, r->reply + j, min(split, r->bytes - j));
    out[i * 16] = valid && aotx_appraisal.rows[i].prefix.stage == 12;
    out[i * 16 + 1] = aotx_appraisal_parse(i);
    const aotx_appraisal_row *a = aotx_appraisal.rows + i;
    for (unsigned j = 0; j < AOTX_APPRAISAL_VALUES; ++j) out[i * 16 + 2 + j] = a->values[j];
    out[i * 16 + 11] = a->quote_start; out[i * 16 + 12] = a->quote_length;
    out[i * 16 + 13] = a->task_length; out[i * 16 + 14] = a->commitment_length; out[i * 16 + 15] = a->correction;
}
#endif
