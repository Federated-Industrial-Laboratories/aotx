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
