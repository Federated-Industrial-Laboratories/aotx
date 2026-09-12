/* Purpose: Apply trained half precision matrices to a batch of image rows.
 * Owns: Shared matrix tiles only; each job owns its output span.
 * Launch shape: Matrix tiles in x and y, independent jobs in z.
 * Lifetime: One product node of an encoder graph. */
#include "vision/vision.cuh"
#include "model/gemm.cuh"

__global__ __launch_bounds__(AOTX_GEMM_THREADS, 3) void aotx_vision_product(
    aotx_vision_job *jobs, unsigned count, const unsigned char *weights,
    const aotx_vision_desc *desc, unsigned operation, unsigned low)
{
    __shared__ half sx[AOTX_GEMM_TILE_M * AOTX_GEMM_LINE];
    __shared__ half sw[AOTX_GEMM_TILE_N * AOTX_GEMM_LINE];
    if (blockIdx.z >= count) return;
    aotx_vision_job &j = jobs[blockIdx.z];
    unsigned phase = operation < 2u ? AOTX_VISION_PATCH : operation == 2u ? AOTX_VISION_PREPARE
        : operation < 6u ? AOTX_VISION_BLOCK : AOTX_VISION_MERGE;
    if (j.phase != phase) return;
    unsigned n = 768u, k = 768u, m = j.patches;
    unsigned long long offset;
    if (operation < 2u) offset = desc->base[AOTX_VISION_PATCH0 + operation];
    else if (operation == 2u) { offset = desc->layer[j.layer][AOTX_VISION_QKV]; n = 2304; }
    else if (operation == 3u) offset = desc->layer[j.layer][AOTX_VISION_OUT];
    else if (operation == 4u) { offset = desc->layer[j.layer][AOTX_VISION_UP]; n = 3072; }
    else if (operation == 5u) { offset = desc->layer[j.layer][AOTX_VISION_DOWN]; k = 3072; }
    else if (operation == 6u) {
        offset = desc->base[AOTX_VISION_MERGE0]; n = k = 3072; m /= 4u;
    } else { offset = desc->base[AOTX_VISION_MERGE1]; n = 1024; k = 3072; m /= 4u; }
    if (blockIdx.y * AOTX_GEMM_TILE_M >= m || blockIdx.x * AOTX_GEMM_TILE_N >= n) return;
    if (low) aotx_gemm_tile<AOTX_WEIGHT_F16, true>(weights + offset, n, k, j.input_low, m, j.product, sx, sw);
    else aotx_gemm_tile<AOTX_WEIGHT_F16>(weights + offset, n, k, j.input, m, j.product, sx, sw);
}
