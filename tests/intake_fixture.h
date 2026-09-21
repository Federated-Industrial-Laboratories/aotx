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
#include "cognitive/intake_capability.cuh"

static std::vector<std::array<unsigned char,16>> aotx_intake_targets;
static std::vector<std::string> aotx_intake_outputs;
static std::vector<std::string> aotx_intake_first_outputs;
static unsigned aotx_intake_fixture_n;
__device__ unsigned char aotx_intake_fixture_bytes[AOTX_RECALL_BATCH][AOTX_INTAKE_REPLY];
__device__ unsigned char aotx_intake_fixture_first[AOTX_RECALL_BATCH][AOTX_INTAKE_REPLY];
__global__ void aotx_intake_fixture_setup(void) {
    if (threadIdx.x) return;
    aotx_decode.ready = 1; aotx_decode.role = AOTX_MODEL_LANGUAGE;
    aotx_model_load.resident[AOTX_MODEL_LANGUAGE].active = 1;
    aotx_model_load.resident[AOTX_MODEL_LANGUAGE].slot = AOTX_MODEL_LANGUAGE;
    for (unsigned j = 0; j < 32; ++j) aotx_model_load.resident[AOTX_MODEL_LANGUAGE].body.digest[j] = 0x51 + j;
}
/* Fixed test responses use explicit synthetic qualification rows. Native checks retain the product table. */
static void aotx_intake_fixture_qualify(void) {
    aotx_intake_capability rows[AOTX_INTAKE_CAPABILITIES] = {};
    for (unsigned j = 0; j < 32; ++j) rows[0].model[j] = 0x51 + j;
    AOTX_CUDA(cudaMemcpyFromSymbol(&rows[0].wrapper, aotx_model_wrap, sizeof(aotx_wrap),
        AOTX_MODEL_LANGUAGE * sizeof(aotx_wrap)));
    rows[1] = rows[0];
    AOTX_CUDA(cudaMemcpyFromSymbol(rows[0].source, aotx_intake_processor, 32));
    AOTX_CUDA(cudaMemcpyFromSymbol(rows[1].source, aotx_intake_source_processor, 32));
    AOTX_CUDA(cudaMemcpyFromSymbol(rows[1].statement, aotx_intake_statement_processor, 32));
    AOTX_CUDA(cudaMemcpyFromSymbol(rows[1].profile, aotx_source_profile_digest, 32));
    AOTX_CUDA(cudaMemcpyToSymbol(aotx_intake_capabilities, rows, sizeof(rows)));
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
            if (r->phase == 2 && r->first_count) r->second_call = 1;
            r->state = 4;
        }
        aotx_say.slot[slot].wanted = 0; aotx_intake.row[slot] = 0;
    }
    __syncthreads();
    if (!i) aotx_live.phase = AOTX_INTAKE_DONE;
}
__global__ void aotx_intake_fixture_statements(unsigned n) {
    unsigned i = threadIdx.x;
    if (i >= n || aotx_live.phase != AOTX_INTAKE_RUN) return;
    auto *r = aotx_intake.rows + i;
    if (r->phase != 1 || !aotx_intake_owns(aotx_cog_u32(aotx_live.prefixes[i]))) return;
    unsigned slot = aotx_cog_u32(aotx_live.prefixes[i]);
    r->bytes = 0;
    while (r->bytes < AOTX_INTAKE_REPLY && aotx_intake_fixture_first[i][r->bytes]) {
        r->reply[r->bytes] = aotx_intake_fixture_first[i][r->bytes]; ++r->bytes;
    }
    r->status = aotx_intake_parse(i); aotx_say.slot[slot].wanted = 0;
    if (!r->status) r->status = aotx_intake_classify(i, slot);
    if (r->status) aotx_live.status = r->status;
}
static std::string aotx_intake_fixture_extract(const std::string &reply) {
    std::string out = "[";
    for (size_t at = 0; at + 3 < reply.size(); ++at) {
        if (reply[at] != '[' || (reply[at + 1] != '3' && reply[at + 1] != '4')) continue;
        size_t from = reply.find('"', at + 2), to = from + 1;
        if (from == std::string::npos) break;
        for (; to < reply.size(); ++to) {
            if (reply[to] == '\\') ++to;
            else if (reply[to] == '"') break;
        }
        if (to == reply.size()) break;
        if (out.size() > 1) out += ',';
        out += "[" + reply.substr(from, to - from + 1) + ",\"statement\"]"; at = to;
    }
    return out + "]";
}
static void aotx_intake_fixture_first_upload(const std::vector<std::string> &replies) {
    aotx_bytes bytes(64 * AOTX_INTAKE_REPLY, 0);
    for (unsigned i = 0; i < replies.size(); ++i) {
        aotx_check(replies[i].size() < AOTX_INTAKE_REPLY, "statement fixture response fits");
        memcpy(bytes.data() + i * AOTX_INTAKE_REPLY, replies[i].data(), replies[i].size());
    }
    AOTX_CUDA(cudaMemcpyToSymbol(aotx_intake_fixture_first, bytes.data(), bytes.size()));
}
static void aotx_intake_fixture_service(bool replay) {
    if (replay) return;
    unsigned phase = 0;
    AOTX_CUDA(cudaMemcpyFromSymbol(&phase, aotx_live, sizeof(phase), offsetof(aotx_live_state, phase)));
    if (phase == AOTX_INTAKE_RUN) {
        aotx_intake_fixture_statements<<<1,64>>>(aotx_intake_fixture_n); AOTX_CUDA(cudaDeviceSynchronize());
    }
    if (phase == AOTX_INTAKE_RUN && !aotx_intake_targets.empty()) {
        std::vector<aotx_recall_result> rows(aotx_intake_fixture_n);
        std::vector<aotx_intake_row> interpreted(aotx_intake_fixture_n);
        AOTX_CUDA(cudaMemcpyFromSymbol(interpreted.data(), aotx_intake, interpreted.size() * sizeof(interpreted[0]), offsetof(aotx_intake_state, rows)));
        AOTX_CUDA(cudaMemcpyFromSymbol(rows.data(), aotx_live, rows.size() * sizeof(rows[0]), offsetof(aotx_live_state, results)));
        for (unsigned i = 0; i < rows.size(); ++i) {
            unsigned target = 0;
            const unsigned char *table = interpreted[i].targets[0] ? interpreted[i].targets : rows[i].selection;
            for (unsigned j = 0; j < aotx_get(table + 4, 4); ++j)
                if (!memcmp(table + 16 + j * 32, aotx_intake_targets[i].data(), 16)) target = j + 1;
            aotx_check(target != 0, "independent correction target is present in the exact target table");
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
        aotx_intake_fixture_n = n; aotx_intake_targets.clear(); aotx_intake_first_outputs.clear();
        aotx_intake_fixture_setup<<<1,1>>>(); AOTX_CUDA(cudaDeviceSynchronize());
        aotx_intake_fixture_qualify();
    }
    aotx_live_records intake(const aotx_bytes &p, const std::vector<std::string> &replies, bool finish = true,
        void (*service)(bool) = aotx_intake_fixture_service) {
        aotx_intake_outputs = replies;
        std::vector<std::string> first = aotx_intake_first_outputs;
        if (first.empty()) for (const auto &reply : replies) first.push_back(aotx_intake_fixture_extract(reply));
        aotx_intake_fixture_first_upload(first);
        aotx_bytes bytes(64 * AOTX_INTAKE_REPLY, 0);
        for (unsigned i = 0; i < replies.size(); ++i) {
            aotx_check(replies[i].size() < AOTX_INTAKE_REPLY, "fixture response fits");
            memcpy(bytes.data() + i * AOTX_INTAKE_REPLY, replies[i].data(), replies[i].size());
        }
        AOTX_CUDA(cudaMemcpyToSymbol(aotx_intake_fixture_bytes, bytes.data(), bytes.size()));
        return process(aotx_live_parts(p, AOTX_LIVE_QUERY, next_id++), false, finish, service);
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
