/* Purpose: Check exact control doses and refusal at the sampler and residual hook.
 * Owns: Distinct rows, requests and expected results at batch sizes one and 64.
 * Launch shape: One block for each residual row and one thread for each request.
 * Lifetime: One test with synthetic values and no model files. */
#include "tests/control_fixture.h"
#include "model/conduct.cuh"
#include "model/sampler.cuh"
#include <stdio.h>
#include <initializer_list>

static __device__ float residual[AOTX_SLOTS * 4], direction[4];
static __device__ unsigned agents[AOTX_SLOTS], offsets[AOTX_SLOTS + 1], results[AOTX_SLOTS];
static __device__ aotx_model_how choices[AOTX_SLOTS];

__global__ void aotx_qualification_fixture(unsigned count, unsigned defect) {
    if (threadIdx.x || blockIdx.x) return;
    const unsigned role = AOTX_MODEL_LANGUAGE;
    aotx_model[role].hidden = 4;
    aotx_conduct = {}; aotx_conduct.vectors = 2; aotx_conduct.voices = 1;
    for (unsigned v = 0; v < 2; ++v) {
        aotx_steer_vector *r = aotx_conduct.vector + v;
        for (unsigned i = 0; i < 32; ++i) r->identity.model[i] = aotx_model_load.resident[role].body.digest[i];
        r->identity.wrap = aotx_model_wrap[role]; r->identity.wrap.usable = 0;
        r->hidden = 4; r->layers = 1; r->layer_count = 1;
        r->value = (unsigned long long)direction;
        r->name[0] = 'a' + v;
        r->permit.status = defect == 1 ? 0 : defect == 8 ? 3 : defect == 12 ? 2 : 1;
        r->permit.count = 2; r->permit.dose[0] = 5000; r->permit.dose[1] = 10000;
    }
    aotx_conduct.voice[0].name[0] = 'v';
    aotx_conduct.voice[0].identity = aotx_conduct.vector[0].identity;
    if (defect == 6) aotx_conduct.vector[0].identity.model[19] ^= 1;
    if (defect == 7) ++aotx_conduct.vector[0].identity.wrap.end_ids[0];
    for (unsigned x = 0; x < 4; ++x) direction[x] = (x + 1) * 0.25f;
    aotx_model_space[role].resid = residual;
    aotx_model_call[role] = {}; aotx_model_call[role].tokens = count;
    aotx_model_call[role].seqs = count; aotx_model_call[role].agent = agents;
    aotx_model_call[role].offset = offsets; aotx_model_call[role].how = choices;
    for (unsigned i = 0; i < count; ++i) {
        agents[i] = i; offsets[i] = i; choices[i] = {};
        for (unsigned j = 0; j < AOTX_MODEL_STEERS; ++j) choices[i].steer[j] = AOTX_MODEL_CONDUCT_NONE;
        choices[i].voice = AOTX_MODEL_CONDUCT_NONE;
        choices[i].steer[0] = 0; choices[i].steer_strength[0] = i % 2 ? 1.0f : 0.5f;
        if (defect == 2) choices[i].steer_strength[0] = 0.75f;
        if (defect == 3) choices[i].steer_strength[0] *= -1;
        if (defect == 4 || defect == 5) {
            choices[i].steer[1] = defect == 4 ? 1 : 0; choices[i].steer_strength[1] = 0.5f;
        }
        if (defect == 9) choices[i].steer_strength[0] = i % 2 ? INFINITY : NAN;
        if (defect == 10) { choices[i].voice = 0; choices[i].voice_scale = 1; }
        if (defect == 11) choices[i].steer_strength[0] = 0;
        if (defect == 12) choices[i].steer_strength[0] = (i + 1) * 0.125f;
        for (unsigned x = 0; x < 4; ++x) residual[i * 4 + x] = i + x * 0.0625f;
    }
    offsets[count] = count;
}
__global__ void aotx_qualification_verify(unsigned count, unsigned defect) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    bool good = true;
    for (unsigned x = 0; x < 4; ++x) {
        float wanted = i + x * 0.0625f;
        if (!defect || defect == 12) wanted += choices[i].steer_strength[0] * direction[x];
        good &= residual[i * 4 + x] == wanted;
    }
    results[i] = good;
}
__global__ void aotx_qualification_requests(unsigned count) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    aotx_sampler_reset(i);
    aotx_model_how *r = &aotx_sampler.row[i];
    r->steer_from = 17 + i;
    bool good = aotx_sampler_set(i, "decode.voice", 12, "absent", 6) == AOTX_SAMPLER_TOOK;
    good &= r->steer_from == 17 + i;
    const char *dose = i % 2 ? "a:1.0" : "a:0.5";
    good &= aotx_sampler_set(i, "decode.steer0", 13, dose, 5) == AOTX_SAMPLER_TOOK;
    good &= aotx_sampler_set(i, "decode.steer0", 13, "a:0.75", 6) == AOTX_SAMPLER_RANGE;
    good &= r->steer_strength[0] == (i % 2 ? 1.0f : 0.5f);
    good &= aotx_sampler_set(i, "decode.steer1", 13, "b:0.5", 5) == AOTX_SAMPLER_RANGE;
    good &= r->steer[1] == AOTX_MODEL_CONDUCT_NONE;
    good &= aotx_sampler_set(i, "decode.voice", 12, "v", 1) == AOTX_SAMPLER_RANGE;
    good &= r->voice == AOTX_MODEL_CONDUCT_NONE;
    good &= aotx_sampler_set(i, "decode.steer0", 13, "absent", 6) == AOTX_SAMPLER_TOOK;
    good &= aotx_sampler_set(i, "decode.voice", 12, "v", 1) == AOTX_SAMPLER_TOOK;
    good &= aotx_sampler_set(i, "decode.steer0", 13, dose, 5) == AOTX_SAMPLER_RANGE;
    results[i] = good;
}
#include "tests/control_selection.h"
int main(void) {
    unsigned cases = 0, failed = 0, out[AOTX_SLOTS];
    aotx_control_test_model(AOTX_MODEL_LANGUAGE);
    for (unsigned count : {1u, AOTX_SLOTS}) {
        for (unsigned defect = 0; defect < 13; ++defect) {
            aotx_qualification_fixture<<<1, 1>>>(count, defect);
            aotx_model_conduct<<<count, 128>>>(AOTX_MODEL_LANGUAGE, 0);
            aotx_qualification_verify<<<1, AOTX_SLOTS>>>(count, defect);
            aotx_check_runtime(cudaMemcpyFromSymbol(out, results, count * sizeof(*out)), "cudaMemcpyFromSymbol");
            for (unsigned i = 0; i < count; ++i) { ++cases; if (!out[i]) { ++failed; printf("residual case %u row %u failed\n", defect, i); } }
        }
        aotx_qualification_fixture<<<1, 1>>>(count, 0);
        aotx_qualification_requests<<<1, AOTX_SLOTS>>>(count);
        aotx_check_runtime(cudaMemcpyFromSymbol(out, results, count * sizeof(*out)), "cudaMemcpyFromSymbol");
        for (unsigned i = 0; i < count; ++i) { ++cases; if (!out[i]) { ++failed; printf("request row %u failed\n", i); } }
    }
    aotx_selection_cases(1, &cases, &failed);
    aotx_selection_cases(AOTX_SLOTS, &cases, &failed);
    printf("qualification device: %u checks, %u failures\n", cases, failed);
    unsigned expected = 14 + aotx_selection_case_count;
    return failed || cases != expected * (1 + AOTX_SLOTS);
}
