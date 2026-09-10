/* Purpose: Supply distinct model responses for semantic admission and replay checks.
 * Owns: Test response bytes and exact source, scope and correction expectations.
 * Launch shape: N=1 and N=64 through the real live memory kernels.
 * Lifetime: One maintained test; actual decoder output has a separate boot check. */
#ifndef AOTX_TEST_INTAKE_FIXTURE_H
#define AOTX_TEST_INTAKE_FIXTURE_H
#include "retain_fixture.h"
#include "cognitive/intake_parse.cuh"
#include "model/decode_state.cuh"
#include "model/load.cuh"

static std::vector<std::array<unsigned char,16>> aotx_intake_targets;
static std::vector<std::string> aotx_intake_outputs;
static unsigned aotx_intake_fixture_n;
__device__ unsigned char aotx_intake_fixture_bytes[AOTX_RECALL_BATCH][AOTX_INTAKE_REPLY];
__global__ void aotx_intake_fixture_setup(void) {
    if (threadIdx.x) return;
    aotx_decode.ready = 1; aotx_decode.role = AOTX_MODEL_LANGUAGE;
    aotx_model_load.resident[AOTX_MODEL_LANGUAGE].active = 1;
    aotx_model_load.resident[AOTX_MODEL_LANGUAGE].slot = AOTX_MODEL_LANGUAGE;
    for (unsigned j = 0; j < 32; ++j) aotx_model_load.resident[AOTX_MODEL_LANGUAGE].body.digest[j] = 0x51 + j;
}
__global__ void aotx_intake_fixture_complete(unsigned n) {
    if (aotx_live.phase != AOTX_INTAKE_RUN || aotx_seam.replaying) return;
    unsigned i = threadIdx.x;
    if (i < n) {
        auto *r = aotx_intake.rows + i;
        unsigned slot = aotx_cog_u32(aotx_live.prefixes[i]);
        if (aotx_live_bindings[slot].auto_retain == 2) {
            r->bytes = 0;
            while (r->bytes < AOTX_INTAKE_REPLY && aotx_intake_fixture_bytes[i][r->bytes]) {
                r->reply[r->bytes] = aotx_intake_fixture_bytes[i][r->bytes]; ++r->bytes;
            }
            r->state = 4;
        }
        aotx_say.slot[slot].wanted = 0; aotx_intake.row[slot] = 0;
    }
    __syncthreads();
    if (!i) aotx_live.phase = AOTX_INTAKE_DONE;
}
static void aotx_intake_fixture_service(bool replay) {
    if (replay) return;
    unsigned phase = 0;
    AOTX_CUDA(cudaMemcpyFromSymbol(&phase, aotx_live, sizeof(phase), offsetof(aotx_live_state, phase)));
    if (phase == AOTX_INTAKE_RUN && !aotx_intake_targets.empty()) {
        std::vector<aotx_recall_result> rows(aotx_intake_fixture_n);
        AOTX_CUDA(cudaMemcpyFromSymbol(rows.data(), aotx_live, rows.size() * sizeof(rows[0]), offsetof(aotx_live_state, results)));
        for (unsigned i = 0; i < rows.size(); ++i) {
            unsigned target = 0;
            for (unsigned j = 0; j < rows[i].count; ++j)
                if (!memcmp(rows[i].selection + 16 + j * 32, aotx_intake_targets[i].data(), 16)) target = j + 1;
            aotx_check(target != 0, "independent correction target is present in prior recall");
            auto response = aotx_intake_outputs[i]; auto at = response.find('@');
            if (at != std::string::npos) response.replace(at, 1, std::to_string(target));
            aotx_bytes bytes(AOTX_INTAKE_REPLY, 0); memcpy(bytes.data(), response.data(), response.size());
            AOTX_CUDA(cudaMemcpyToSymbol(aotx_intake_fixture_bytes, bytes.data(), bytes.size(), i * AOTX_INTAKE_REPLY));
        }
    }
    aotx_intake_fixture_complete<<<1,64>>>(aotx_intake_fixture_n);
}
struct aotx_intake_device : aotx_live_device {
    explicit aotx_intake_device(unsigned n) : aotx_live_device(n) {
        AOTX_LIVE_CLEAR(aotx_intake); AOTX_LIVE_CLEAR(aotx_seqs); AOTX_LIVE_CLEAR(aotx_decode);
        AOTX_LIVE_CLEAR(aotx_kv); AOTX_LIVE_CLEAR(aotx_model_load);
        aotx_intake_fixture_n = n; aotx_intake_targets.clear();
        aotx_intake_fixture_setup<<<1,1>>>(); AOTX_CUDA(cudaDeviceSynchronize());
    }
    aotx_live_records intake(const aotx_bytes &p, const std::vector<std::string> &replies, bool finish = true) {
        aotx_intake_outputs = replies;
        aotx_bytes bytes(64 * AOTX_INTAKE_REPLY, 0);
        for (unsigned i = 0; i < replies.size(); ++i) {
            aotx_check(replies[i].size() < AOTX_INTAKE_REPLY, "fixture response fits");
            memcpy(bytes.data() + i * AOTX_INTAKE_REPLY, replies[i].data(), replies[i].size());
        }
        AOTX_CUDA(cudaMemcpyToSymbol(aotx_intake_fixture_bytes, bytes.data(), bytes.size()));
        return process(aotx_live_parts(p, AOTX_LIVE_QUERY, next_id++), false, finish, aotx_intake_fixture_service);
    }
};
static aotx_bytes aotx_intake_bind(unsigned n, unsigned scope = 0, bool mixed = false) {
    auto p = aotx_live_binding_bytes(n, 0, scope);
    for (unsigned i = 0; i < n; ++i) aotx_put(p.data() + 124 + i * 64, mixed ? i % 3 : 2, 4);
    return p;
}
static aotx_bytes aotx_intake_query(unsigned n, uint64_t cut, unsigned ordinal, unsigned scope = 0) {
    auto p = aotx_retain_query(n, cut, ordinal, false, scope);
    for (unsigned i = 0; i < n; ++i) {
        auto q = p.data() + 128 + i * AOTX_LIVE_QUERY_ROW;
        std::string source = "Iris" + std::to_string(i) + (ordinal == 1 ? " will cook." : " will not cook.");
        memset(q + 4640, 0, AOTX_RECALL_TEXT); memcpy(q + 4640, source.data(), source.size());
        aotx_put(q + 148, source.size(), 4);
    }
    return p;
}
static std::vector<std::string> aotx_intake_initial(unsigned n) {
    std::vector<std::string> replies;
    for (unsigned i = 0; i < n; ++i) {
        std::string person = "Iris" + std::to_string(i);
        replies.push_back("[[1,\"" + person + "\",0],[2,\"will cook\",0],[3,\"" + person + " will cook.\",0]]");
    }
    return replies;
}
#endif
