/* Purpose: Apply one matrix arithmetic to all recurrent batch partitions.
 * Owns: Nothing; the model workspace holds each input and output.
 * Launch shape: One block per group of weight rows; all live token rows are consumed.
 * Lifetime: One graph launch. */
#include "model/gemv.cuh"
#include "model/hybrid.cuh"

__global__ void aotx_model_hybrid_product(unsigned int role, const void *weight,
                                          unsigned int type, unsigned int n,
                                          unsigned int k, const half *x, float *y)
{
    unsigned int tokens = aotx_model_call[role].tokens;
    aotx_gemv_type(weight, type, n, k, x, tokens, y, blockIdx.x);
}
