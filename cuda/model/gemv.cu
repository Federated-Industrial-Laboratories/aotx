/* Purpose: Multiply a small batch of activation rows by a weight tensor.
 * Owns: Nothing; the caller owns the input and the output.
 * Launch shape: One warp for each run of 2 rows of the weight tensor, 256 threads a block.
 * Lifetime: One launch. */
#include "model/gemv.cuh"

__global__ void aotx_model_gemv(const void *w, unsigned int type, unsigned int n,
                                unsigned int k, const half *x, unsigned int m, float *y)
{
    aotx_gemv_type(w, type, n, k, x, m, y, blockIdx.x);
}
