/* Purpose: Check configured sequence bounds through real device admission.
 * Owns: Distinct token lists, boundary cases and per-slot results.
 * Launch shape: One thread per slot at N=1 and N=64.
 * Lifetime: One test process without model weights. */
#include "live_fixture.h"
#include "model/decode_state.cuh"

__global__ void aotx_capacity_open(unsigned n, unsigned mode, int *ids, unsigned *out) {
    unsigned slot = threadIdx.x;
    if (slot >= n) return;
    unsigned cap = AOTX_SEQ_MAX_TOKENS;
    unsigned count = cap - 1, limit = 1;
    if (mode == 1) ++limit;
    if (mode == 2) count = cap + 1;
    if (mode == 5) count = 1980, limit = 1024;
    if (mode == 3) limit = min(65u, cap - 1) - slot % min(65u, cap - 1), count = cap - limit;
    for (unsigned j = 0; j < cap; ++j) ids[slot * cap + j] = 1000 + slot;
    aotx_model_how sample = {}; sample.seed = 710 + slot;
    sample.repeat_penalty = 1; sample.think_limit = -1;
    out[slot * 3] = aotx_seq_open(slot, AOTX_MODEL_LANGUAGE,
        ids + slot * cap, count, limit,
        mode == 4 ? 1 : AOTX_KV_PAGES_EACH, &sample, 1);
    out[slot * 3 + 1] = aotx_seqs.slot[slot].prompt;
    out[slot * 3 + 2] = !out[slot * 3] ? aotx_seqs.tokens[slot][count - 1] : 0;
}

__global__ void aotx_capacity_shape(unsigned mode) {
    aotx_kvl_make(&aotx_model_space[AOTX_MODEL_LANGUAGE].shape, mode == 4 ? 64 : 1, mode == 4 ? 32 : 1, 128);
}

static void run(unsigned n) {
    aotx_live_device d(n);
    int *ids; AOTX_CUDA(cudaMalloc(&ids, (size_t)n * AOTX_SEQ_MAX_TOKENS * sizeof(int)));
    unsigned *output; AOTX_CUDA(cudaMalloc(&output, n * 3 * sizeof(unsigned)));
    for (unsigned mode = 0; mode < 6; ++mode) {
        AOTX_LIVE_CLEAR(aotx_seqs); AOTX_LIVE_CLEAR(aotx_kv);
        aotx_capacity_shape<<<1,1>>>(mode);
        aotx_capacity_open<<<1,64>>>(n, mode, ids, output);
        AOTX_CUDA(cudaGetLastError()); AOTX_CUDA(cudaDeviceSynchronize());
        std::vector<unsigned> values(n * 3);
        AOTX_CUDA(cudaMemcpy(values.data(), output, values.size() * sizeof(unsigned), cudaMemcpyDeviceToHost));
        for (unsigned i = 0; i < n; ++i) {
            bool accepted = mode == 0 || mode == 3 || (mode == 5 && AOTX_SEQ_MAX_TOKENS >= 3004);
            aotx_check((values[i * 3] == 0) == accepted, "exact context fits; excess tokens and insufficient pages are refused");
            if (accepted) {
                unsigned count = mode == 5 ? 1980 : mode == 3 ? AOTX_SEQ_MAX_TOKENS - (std::min(65u, AOTX_SEQ_MAX_TOKENS - 1) - i % std::min(65u, AOTX_SEQ_MAX_TOKENS - 1)) : AOTX_SEQ_MAX_TOKENS - 1;
                aotx_check(values[i * 3 + 1] == count && values[i * 3 + 2] == 1000 + i,
                    "the complete prompt reaches its own final token in every slot");
            } else aotx_check(!values[i * 3 + 1], "a refused request does not publish a prompt");
        }
    }
    AOTX_CUDA(cudaFree(output)); AOTX_CUDA(cudaFree(ids));
}

int main() {
    aotx_check(!AOTX_SEQUENCE_TOKENS || AOTX_SEQ_MAX_TOKENS == AOTX_SEQUENCE_TOKENS,
        "the configured capacity reaches the device translation unit");
    run(1); run(64);
    printf("sequence capacity %u: %u checks, %u failures\n", AOTX_SEQ_MAX_TOKENS, aotx_checks, aotx_failures);
    return aotx_failures ? 1 : 0;
}
