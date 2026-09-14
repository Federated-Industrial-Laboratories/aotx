/* Purpose: Supply incompatible and malformed native entries for admission tests.
 * Owns: Only the output spans supplied by each test row.
 * Launch shape: Strided batches; the wrong entry deliberately has another ABI.
 * Lifetime: Test-owned driver modules. */
#include "policy/abi.h"
extern "C" __global__ void aotx_policy_wrong(uint32_t value) { (void)value; }
extern "C" __global__ void aotx_policy_malformed(const aotx_policy_input *input,
    const unsigned char *prior, aotx_policy_output *output, unsigned char *next,
    uint32_t count, uint32_t stride) {
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        output[i].action = 99; output[i].reason = input[i].source;
        next[(uint64_t)i * stride] = prior[(uint64_t)i * stride] + 1;
    }
}
extern "C" __global__ void aotx_policy_appraisal_native(const aotx_policy_input *input,
    const unsigned char *prior, aotx_policy_output *output, unsigned char *next,
    uint32_t count, uint32_t stride) {
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        if (!input[i].valid) continue;
        output[i] = {};
        for (uint32_t j = 0; j < stride; ++j) next[(uint64_t)i * stride + j] = prior[(uint64_t)i * stride + j];
        if (stride < 16) { output[i].status = 1; continue; }
        uint64_t calls = 0, revision = (uint64_t)input[i].reserved1[1] | (uint64_t)input[i].reserved1[2] << 32;
        for (unsigned j = 0; j < 8; ++j) calls |= (uint64_t)prior[(uint64_t)i * stride + j] << (8 * j);
        calls += (uint64_t)input[i].reserved1[0] + 1;
        for (unsigned j = 0; j < 8; ++j) {
            next[(uint64_t)i * stride + j] = (unsigned char)(calls >> (8 * j));
            next[(uint64_t)i * stride + 8 + j] = (unsigned char)(revision >> (8 * j));
        }
        if (input[i].reserved0 == AOTX_POLICY_APPRAISAL_ABI && input[i].reserved1[0] > 1 &&
            !input[i].paused && !input[i].foreground) {
            output[i].action = AOTX_POLICY_APPRAISE; output[i].reason = input[i].source ^ revision;
        }
    }
}
extern "C" __global__ void aotx_policy_appraisal_unchecked(const aotx_policy_input *input,
    const unsigned char *prior, aotx_policy_output *output, unsigned char *next,
    uint32_t count, uint32_t stride) {
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        if (!input[i].valid) continue;
        output[i] = {}; output[i].action = AOTX_POLICY_APPRAISE;
        for (uint32_t j = 0; j < stride; ++j) next[(uint64_t)i * stride + j] = prior[(uint64_t)i * stride + j];
        next[(uint64_t)i * stride] += 1;
    }
}
