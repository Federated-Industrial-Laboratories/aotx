/* Purpose: Select memory maintenance with an authored native pressure policy.
 * Owns: Two portable counters in each row's private state span.
 * Launch shape: Independent rows in a strided thread grid.
 * Lifetime: One finite CUDA entry compiled to PTX or cubin. */
#include "policy/rules.cuh"
extern "C" __global__ void aotx_creator_maintenance(const aotx_policy_input *input,
    const unsigned char *prior, aotx_policy_output *output, unsigned char *next,
    uint32_t count, uint32_t stride) {
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += gridDim.x * blockDim.x)
        aotx_policy_rule_row(input + i, prior + (uint64_t)i * stride, output + i,
            next + (uint64_t)i * stride, stride);
}
