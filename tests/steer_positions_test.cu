/* Purpose: Check candidate measurement selection and response boundaries.
 * Owns: Distinct text pieces and control choices at N=1 and N=64.
 * Launch shape: One thread for each text in the bounded tokenizer batch.
 * Lifetime: One test with synthetic token pieces and no model files. */
#include "tools/steer_set.h"
#include "model/wrap.cuh"
#include <initializer_list>
static __device__ unsigned char text[64 * 16];
static __device__ unsigned start[64], length[64], pieces[64], tokens[64];
static __device__ unsigned first[64 * 4], size[64 * 4], chunk[64 * 4], bad;
static __device__ aotx_model_how how[64];
static __device__ aotx_steer_text tokenizer;
__global__ void aotx_steer_positions_fixture(unsigned count, unsigned mode, unsigned defect) {
    if (threadIdx.x || blockIdx.x) return;
    aotx_conduct = {}; aotx_conduct.vectors = 16;
    for (unsigned i = 0; i < 16; ++i) {
        aotx_conduct.vector[i].name[0] = i ? 'a' + i : 'v';
        aotx_conduct.vector[i].hidden = 8 + i;
    }
    aotx_conduct.vector[0].positions = AOTX_CONTROL_RESPONSE;
    aotx_wrap wrap = {}; wrap.usable = 1;
    wrap.bytes[0] = 'G'; wrap.bytes[1] = '('; wrap.bytes[2] = ')';
    for (unsigned j = 6; j < 9; ++j) { wrap.offset[j] = j - 6; wrap.length[j] = 1; }
    aotx_model_wrap[AOTX_MODEL_LANGUAGE] = wrap;
    tokenizer = {}; tokenizer.clean = text; tokenizer.clean_start = start; tokenizer.clean_length = length;
    tokenizer.pieces.start = first; tokenizer.pieces.length = size; tokenizer.pieces.count = pieces;
    tokenizer.pieces.stride = 4; tokenizer.tokens.chunk = chunk; tokenizer.tokens.count = tokens; bad = 0;
    for (unsigned i = 0; i < count; ++i) {
        start[i] = 16 * i; length[i] = 7; pieces[i] = 4; tokens[i] = i + 4;
        for (unsigned j = 0; j < 4; ++j) text[16 * i + j] = 'a' + (i + j) % 26;
        text[16 * i + 4] = 'G'; text[16 * i + 5] = '('; text[16 * i + 6] = defect ? '_' : ')';
        for (unsigned j = 0; j < 4; ++j) {
            first[4 * i + j] = 16 * i + (j ? 3 + j : 0);
            size[4 * i + j] = j ? 1 : 4; chunk[4 * i + j] = j ? 1 : i + 1;
        }
        how[i] = {}; how[i].steer_from = 99;
        for (unsigned j = 0; j < AOTX_MODEL_STEERS; ++j) how[i].steer[j] = AOTX_MODEL_CONDUCT_NONE;
        how[i].steer[0] = mode == 0 ? 1 : mode == 1 ? 0 : i % 2;
        how[i].steer_strength[0] = 0.5f;
    }
}
int main(void) {
    unsigned checks = 0, failed = 0;
    for (unsigned count : {1u, 64u}) for (unsigned mode = 0; mode < 3; ++mode) for (unsigned defect = 0; defect < 2; ++defect) {
        aotx_steer_positions_fixture<<<1, 1>>>(count, mode, defect);
        aotx_steer_text t; aotx_model_how *rows; unsigned *errors;
        aotx_check_runtime(cudaMemcpyFromSymbol(&t, tokenizer, sizeof(t)), "cudaMemcpyFromSymbol");
        aotx_check_runtime(cudaGetSymbolAddress((void **)&rows, how), "cudaGetSymbolAddress");
        aotx_check_runtime(cudaGetSymbolAddress((void **)&errors, bad), "cudaGetSymbolAddress");
        aotx_steer_positions<<<1, 64>>>(t, AOTX_MODEL_LANGUAGE, count, rows, errors);
        aotx_model_how out[64]; unsigned got = 0, wanted = 0;
        aotx_check_runtime(cudaMemcpyFromSymbol(out, how, sizeof(out)), "cudaMemcpyFromSymbol");
        aotx_check_runtime(cudaMemcpyFromSymbol(&got, bad, sizeof(got)), "cudaMemcpyFromSymbol");
        for (unsigned i = 0; i < count; ++i) {
            bool response = mode == 1 || (mode == 2 && i % 2 == 0);
            wanted += response && defect; ++checks;
            failed += out[i].steer_from != (response && !defect ? i + 2 : 0);
        }
        ++checks; failed += got != wanted;
        ++checks; failed += aotx_steer_positions_run(t, AOTX_MODEL_LANGUAGE, count, rows) != (wanted != 0);
        aotx_steer_measure candidate[64] = {};
        for (unsigned i = 0; i < count; ++i) {
            unsigned v = i % 18;
            candidate[i].name[0] = v == 16 ? 0 : v == 17 ? '?' : v ? 'a' + v : 'v';
        }
        ++checks; failed += aotx_steer_measure_vectors(candidate, count) != 0;
        for (unsigned i = 0; i < count; ++i) {
            unsigned v = i % 18; ++checks;
            failed += v < 16 ? candidate[i].id != v || candidate[i].vector.hidden != 8 + v ||
                candidate[i].vector.permit.status != AOTX_QUALIFICATION_MEASUREMENT :
                candidate[i].id != AOTX_MODEL_CONDUCT_NONE || candidate[i].vector.permit.status != 0;
        }
        aotx_conduct_table placed;
        aotx_check_runtime(cudaMemcpyFromSymbol(&placed, aotx_conduct, sizeof(placed)), "cudaMemcpyFromSymbol");
        for (unsigned i = 0; i < 16; ++i) {
            ++checks; failed += placed.vector[i].permit.status != (i < count ? AOTX_QUALIFICATION_MEASUREMENT : 0);
        }
    }
    printf("steer positions: %u checks, %u failures\n", checks, failed);
    return failed || checks != 1008;
}
