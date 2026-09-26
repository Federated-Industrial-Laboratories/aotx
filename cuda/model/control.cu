/* Purpose: Refuse nonfinite fitted residual values before registration.
 * Owns: No persistent state.
 * Launch shape: A bounded grid checks the complete layer and width batch.
 * Lifetime: One asset admission. */
#include "model/control.cuh"
__global__ void aotx_control_values_check(const float *values, unsigned long long count,
    unsigned *bad) {
    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < count; i += (unsigned long long)blockDim.x * gridDim.x)
        if (!isfinite(values[i])) atomicExch(bad, 1u);
}
