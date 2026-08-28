/* Purpose: Run the matrix nodes of a captured graph at the batch of the tick.
 * Owns: The two shared tiles of one block of the tensor core node.
 * Launch shape: One block for each tile of the tensor core product. The memory bound
 *   product takes one warp for each run of rows of the weight tensor.
 * Lifetime: Two nodes of every matrix product of the tick.
 *
 * A decode tick holds one row for each sequence. A prefill tick holds up to the whole
 * token budget. The two products have different bounds. Below AOTX_MODEL_TENSOR_MIN rows
 * the read of the weights is the whole cost, and the memory bound product reads each
 * weight one time. At that count and above the tensor cores give more.
 *
 * The batch is not known when the graph is captured. Each node therefore reads the batch
 * through a pointer into the call block and exits when the batch is not its own. The
 * pointer keeps the read of the batch to one load of one word. Two nodes and not
 * one hold the two products. One kernel would hold the registers of both products. The
 * memory bound product would then lose its rows of the batch to the register bound.
 *
 * A module node stands in front of the two where the weight is of the eight bit type. That
 * node takes a batch of one row, and the memory bound node then exits. */
#include "model/forward.cuh"
#include "model/gemm.cuh"
#include "model/gemv.cuh"

__global__ __launch_bounds__(AOTX_GEMM_THREADS, 3) void aotx_model_product(
    const unsigned int *batch, const void *w, unsigned int type, unsigned int n,
    unsigned int k, const half *x, float *y)
{
    __shared__ half sx[AOTX_GEMM_TILE_M * AOTX_GEMM_LINE];
    __shared__ half sw[AOTX_GEMM_TILE_N * AOTX_GEMM_LINE];
    unsigned int m = *batch;
    if (m < AOTX_MODEL_TENSOR_MIN || blockIdx.y * AOTX_GEMM_TILE_M >= m) {
        return;
    }
    aotx_gemm_type(w, type, n, k, x, m, y, sx, sw);
}

__global__ void aotx_model_line(const unsigned int *batch, const void *w, unsigned int type,
                                unsigned int n, unsigned int k, const half *x, float *y,
                                unsigned int module)
{
    unsigned int m = *batch;
    if (m == 0u || m >= AOTX_MODEL_TENSOR_MIN) {
        return;
    }
    if (module != 0u && m == 1u) {
        return;
    }
    aotx_gemv_type(w, type, n, k, x, m, y, blockIdx.x);
}
