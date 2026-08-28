/* Purpose: Multiply a batch of activation rows by a weight tensor on the tensor cores.
 * Owns: The two shared tiles of one block.
 * Launch shape: One block for each tile of y, 128 rows by 64 columns, with 256 threads.
 * Lifetime: One launch. */
#include "model/gemm.cuh"

__global__ __launch_bounds__(AOTX_GEMM_THREADS, 4) void aotx_model_gemm(
    const void *w, unsigned int type, unsigned int n, unsigned int k, const half *x,
    unsigned int m, float *y)
{
    __shared__ half sx[AOTX_GEMM_TILE_M * AOTX_GEMM_LINE];
    __shared__ half sw[AOTX_GEMM_TILE_N * AOTX_GEMM_LINE];
    aotx_gemm_type(w, type, n, k, x, m, y, sx, sw);
}
