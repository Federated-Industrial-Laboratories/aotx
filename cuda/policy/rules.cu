/* Purpose: Evaluate the supplied maintenance rules over independent policy rows.
 * Owns: Output and candidate state for each valid row.
 * Launch shape: A strided thread grid over the supplied batch count.
 * Lifetime: One finite node between observation and recorded publication. */
#include "policy/state.cuh"
#include "policy/rules.cuh"
__global__ void aotx_policy_rules(const aotx_policy_input *input, const unsigned char *prior,
    aotx_policy_output *output, unsigned char *next, uint32_t count, uint32_t stride) {
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x)
        aotx_policy_rule_row(input + i, prior + (uint64_t)i * stride, output + i,
            next + (uint64_t)i * stride, stride);
}
