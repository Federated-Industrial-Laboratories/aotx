/* Purpose: Check control use and refusal after exact model and turn-format changes.
 * Owns: Distinct residual rows and control doses for every tested agent.
 * Launch shape: N residual blocks for N=1 and N=AOTX_SLOTS.
 * Lifetime: One device test with no model files or weight allocations. */
#include "model/control.cuh"
#include "model/control_position.cuh"
#include "model/conduct.cuh"
#include "model/decode_state.cuh"
#include "boot/check.h"
#ifdef AOTX_AFFECT
#include "affect/affect.cuh"
#endif
#include <stdio.h>
#include <math.h>
#include <initializer_list>

static __device__ float residual[AOTX_SLOTS * 16], direction[4], composite[AOTX_SLOTS * 4], probe[4];
static __device__ unsigned finite_bad, bases[AOTX_SLOTS];
static __device__ unsigned agents[AOTX_SLOTS], offsets[AOTX_SLOTS + 1], results[AOTX_SLOTS * 3];
static __device__ aotx_model_how choices[AOTX_SLOTS];

__global__ void aotx_control_fixture(unsigned count, unsigned defect) {
    if (threadIdx.x || blockIdx.x) return;
    unsigned role = AOTX_MODEL_LANGUAGE;
    aotx_model[role].hidden = 4; aotx_model[role].layers = 1; aotx_model[role].vocab = 16;
    aotx_model_load.resident[role].active = 1;
    for (unsigned i = 0; i < 32; ++i) aotx_model_load.resident[role].body.digest[i] = i + 1;
    aotx_model_wrap[role] = {}; aotx_model_wrap[role].usable = 1;
    aotx_model_wrap[role].end_count = 1; aotx_model_wrap[role].end_ids[0] = 7;
    aotx_control_identity identity = {};
    for (unsigned i = 0; i < 32; ++i) identity.model[i] = i + 1;
    identity.wrap = aotx_model_wrap[role]; identity.wrap.usable = 0;
    identity.wrap.think_open_id = identity.wrap.think_close_id = UINT32_MAX;
    aotx_conduct = {}; aotx_conduct.vectors = 1;
    aotx_conduct.vector[0].identity = identity;
    aotx_conduct.vector[0].permit.status = AOTX_QUALIFICATION_MEASUREMENT;
    aotx_conduct.vector[0].hidden = 4; aotx_conduct.vector[0].layers = 1;
    aotx_conduct.vector[0].layer_count = 1; aotx_conduct.vector[0].value = (unsigned long long)direction;
    for (unsigned x = 0; x < 4; ++x) { direction[x] = 0.25f * (x + 1); probe[x] = x == 0 ? 1.0f : 0.0f; }
    aotx_model_space[role].resid = residual;
    aotx_model_call[role] = {};
    aotx_model_call[role].tokens = count; aotx_model_call[role].seqs = count;
    aotx_model_call[role].agent = agents; aotx_model_call[role].offset = offsets;
    aotx_model_call[role].how = choices;
#ifdef AOTX_AFFECT
    aotx_affect_composite_table = {}; aotx_affect_composite_table.identity = identity;
    aotx_affect_composite_table.hidden = 4; aotx_affect_composite_table.layers = 1;
    aotx_affect_composite_table.layer_count = 1; aotx_affect_composite_table.trusted = 1;
    aotx_affect_steer = composite;
    aotx_affect_rows = {}; aotx_affect_rows.identity = identity;
    aotx_affect_rows.count = 1; aotx_affect_rows.hidden = 4; aotx_affect_rows.layers = 1;
    aotx_affect_rows.row[0].scale = 1; aotx_affect_probe = probe;
#endif
    for (unsigned i = 0; i < count; ++i) {
        agents[i] = i; offsets[i] = i; choices[i] = {};
        for (unsigned k = 0; k < AOTX_MODEL_STEERS; ++k) choices[i].steer[k] = AOTX_MODEL_CONDUCT_NONE;
        choices[i].steer[0] = defect == 3 ? AOTX_MODEL_CONDUCT_NONE : 0;
        choices[i].steer_strength[0] = (i + 1) * (i % 2 ? -0.125f : 0.125f);
        for (unsigned x = 0; x < 4; ++x) {
            residual[i * 4 + x] = 1.0f + i + 0.5f * x;
            composite[i * 4 + x] = (i + 1) * (x + 1) * 0.0625f;
        }
#ifdef AOTX_AFFECT
        choices[i].affect = defect != 3;
        choices[i].steer[AOTX_MODEL_CONDUCT_AFFECT] = defect == 3 ? AOTX_MODEL_CONDUCT_NONE : AOTX_MODEL_CONDUCT_AFFECT;
        aotx_affect_acc[i] = {}; aotx_decode.first[i] = 0; aotx_seqs.slot[i].prompt = 1;
#endif
    }
    offsets[count] = count;
    if (defect == 1) aotx_model_load.resident[role].body.digest[17] ^= 1;
    if (defect == 2) ++aotx_model_wrap[role].end_ids[0];
    if (defect == 4) aotx_model_load.resident[role].active = 0;
    if (defect == 5) aotx_model_wrap[role].bytes[9] ^= 1;
    if (defect == 6) aotx_model_wrap[role].usable = 0;
}
__global__ void aotx_control_verify(unsigned count, unsigned defect) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    float norm = 0, first = 0;
    unsigned good = 1;
    for (unsigned x = 0; x < 4; ++x) {
        float expected = 1.0f + i + 0.5f * x;
        if (!defect) {
            expected += (i + 1) * (i % 2 ? -0.125f : 0.125f) * 0.25f * (x + 1);
#ifdef AOTX_AFFECT
            expected += (i + 1) * (x + 1) * 0.0625f;
#endif
        }
        good &= fabsf(residual[i * 4 + x] - expected) < 0.00001f;
        norm += expected * expected; if (!x) first = expected;
    }
    results[3 * i] = good;
    results[3 * i + 1] = aotx_control_matches(&aotx_conduct.vector[0].identity,
        AOTX_MODEL_LANGUAGE) == (defect == 0 || defect == 3);
#ifdef AOTX_AFFECT
    float expected = defect ? 0.0f : first / sqrtf(norm);
    results[3 * i + 2] = fabsf(aotx_affect_acc[i].prompt[0] - expected) < 0.00001f;
#else
    results[3 * i + 2] = 1;
#endif
}
__global__ void aotx_control_nonfinite(unsigned kind) {
    for (unsigned i = threadIdx.x; i < AOTX_SLOTS * 4; i += blockDim.x) residual[i] = i + 1;
    __syncthreads();
    if (!threadIdx.x) {
        finite_bad = 0;
        if (kind) residual[AOTX_SLOTS * 4 - 1] = kind == 1 ? NAN : INFINITY;
    }
}

__global__ void aotx_control_position_fixture(unsigned count, unsigned mode, unsigned scenario) {
    if (threadIdx.x || blockIdx.x) return;
    unsigned role = AOTX_MODEL_LANGUAGE;
    aotx_model_call[role].tokens = count * 4;
    aotx_model_space[role].base = scenario == 4 ? nullptr : bases;
    if (scenario == 5) aotx_model_call[role].how = nullptr;
    aotx_conduct.vector[0].positions = mode;
    for (unsigned i = 0; i < count; ++i) {
        offsets[i] = i * 4;
        bases[i] = scenario == 2 ? 100 + i : scenario == 3 ? 102 + i : 0;
        choices[i].steer_from = scenario == 1 ? 0 : scenario == 2 || scenario == 3 ? 102 + i : 2 + i % 3;
        if (scenario == 6) choices[i].steer[0] = AOTX_MODEL_CONDUCT_NONE;
#ifdef AOTX_AFFECT
        choices[i].affect = 0;
        choices[i].steer[AOTX_MODEL_CONDUCT_AFFECT] = AOTX_MODEL_CONDUCT_NONE;
#endif
        for (unsigned j = 0; j < 16; ++j) residual[i * 16 + j] = i + j * 0.125f;
    }
    offsets[count] = count * 4;
}
__global__ void aotx_control_position_verify(unsigned count, unsigned mode, unsigned scenario) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    unsigned good = 1;
    for (unsigned row = 0; row < 4; ++row) {
        bool apply = mode == 0 || (mode == 1 && scenario != 1 && scenario != 4 &&
            (scenario == 2 ? row >= 1 : scenario == 3 ? true : row >= 1 + i % 3));
        apply = apply && scenario != 5 && scenario != 6;
        for (unsigned x = 0; x < 4; ++x) {
            float expected = i + (row * 4 + x) * 0.125f;
            if (apply) expected += (i + 1) * (i % 2 ? -0.125f : 0.125f) * 0.25f * (x + 1);
            good &= fabsf(residual[i * 16 + row * 4 + x] - expected) < 0.00001f;
        }
    }
    results[i] = good;
}
__global__ void aotx_control_boundary_verify(unsigned count, unsigned scenario) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    aotx_wrap wrap = {}; wrap.usable = 1;
    wrap.bytes[0] = 'G'; wrap.bytes[1] = '('; wrap.bytes[2] = ')';
    for (unsigned s = 6; s < 9; ++s) { wrap.offset[s] = s - 6; wrap.length[s] = 1; }
    unsigned char clean[] = "abcdG()";
    unsigned start[] = {0, 4, 5, 6}, length[] = {4, 1, 1, 1}, chunks[] = {i + 1, 1, 1, 1};
    unsigned expected = i + 2;
    if (scenario == 1) { start[1] = 3; length[0] = 3; length[1] = 2; expected = i + 3; }
    if (scenario == 2) { clean[6] = '?'; expected = 0; }
    if (scenario == 3) { wrap.length[6] = wrap.length[7] = wrap.length[8] = 0; expected = 0; }
    if (scenario == 4) { wrap.usable = 0; expected = 0; }
    if (scenario == 5) { chunks[3] = 0; expected = 0; }
    if (scenario == 6) { length[3] = 2; expected = 0; }
    if (scenario == 7) { start[0] = 4; expected = 0; }
    results[i] = aotx_control_response(&wrap, clean, 0, 7, start, length, chunks, 4, i + 4) == expected;
}

int main(void) {
    unsigned checks = 0, failures = 0, rows[AOTX_SLOTS * 3];
    for (unsigned n = 1; n <= AOTX_SLOTS; n = n == 1 ? AOTX_SLOTS : AOTX_SLOTS + 1) {
        for (unsigned defect = 0; defect < 7; ++defect) {
            aotx_control_fixture<<<1, 1>>>(n, defect);
            aotx_model_conduct<<<n, AOTX_MODEL_ROW_THREADS>>>(AOTX_MODEL_LANGUAGE, 0);
            aotx_control_verify<<<(n + 63) / 64, 64>>>(n, defect);
            aotx_check_runtime(cudaMemcpyFromSymbol(rows, results, n * 3 * sizeof(unsigned)), "cudaMemcpyFromSymbol");
            for (unsigned i = 0; i < n * 3; ++i) { ++checks; if (!rows[i]) { ++failures; printf("N=%u case=%u check=%u fails\n", n, defect, i); } }
        }
    }
    for (unsigned n : {1u, (unsigned)AOTX_SLOTS}) {
        for (unsigned mode = 0; mode < 3; ++mode) for (unsigned scenario = 0; scenario < 7; ++scenario) {
            aotx_control_fixture<<<1, 1>>>(n, 0);
            aotx_control_position_fixture<<<1, 1>>>(n, mode, scenario);
            aotx_model_conduct<<<n * 4, AOTX_MODEL_ROW_THREADS>>>(AOTX_MODEL_LANGUAGE, 0);
            aotx_control_position_verify<<<(n + 63) / 64, 64>>>(n, mode, scenario);
            aotx_check_runtime(cudaMemcpyFromSymbol(rows, results, n * sizeof(unsigned)), "cudaMemcpyFromSymbol");
            for (unsigned i = 0; i < n; ++i) {
                ++checks; if (!rows[i]) { ++failures; printf("position N=%u mode=%u case=%u slot=%u fails\n", n, mode, scenario, i); }
            }
        }
        for (unsigned scenario = 0; scenario < 8; ++scenario) {
            aotx_control_boundary_verify<<<(n + 63) / 64, 64>>>(n, scenario);
            aotx_check_runtime(cudaMemcpyFromSymbol(rows, results, n * sizeof(unsigned)), "cudaMemcpyFromSymbol");
            for (unsigned i = 0; i < n; ++i) {
                ++checks; if (!rows[i]) { ++failures; printf("boundary N=%u case=%u slot=%u fails\n", n, scenario, i); }
            }
        }
    }
    for (unsigned kind = 0; kind < 3; ++kind) {
        aotx_control_nonfinite<<<1, 64>>>(kind);
        float *values; unsigned *bad;
        aotx_check_runtime(cudaGetSymbolAddress((void **)&values, residual), "cudaGetSymbolAddress");
        aotx_check_runtime(cudaGetSymbolAddress((void **)&bad, finite_bad), "cudaGetSymbolAddress");
        aotx_control_values_check<<<2, 64>>>(values, AOTX_SLOTS * 4, bad);
        unsigned flag; aotx_check_runtime(cudaMemcpyFromSymbol(&flag, finite_bad, sizeof(flag)), "cudaMemcpyFromSymbol");
        ++checks; failures += flag != (kind != 0);
    }
    printf("control device: %u checks, %u failures\n", checks, failures);
    return failures ? 1 : 0;
}
