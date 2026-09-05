/* Purpose: Apply a query and key norm over each full projection row.
 * Owns: Nothing; the model workspace holds the rows.
 * Launch shape: One block per token and projection; threads share the full width.
 * Lifetime: One forward pass. */
#include "model/blocks.cuh"
#include "model/forward.cuh"

__global__ void aotx_model_qk_norm(unsigned int role, unsigned int layer)
{
    __shared__ float sums[32];
    const aotx_model_desc *desc = &aotx_model[role];
    const aotx_model_run *run = &aotx_model_call[role];
    const aotx_model_work *work = &aotx_model_space[role];
    unsigned int width = ((blockIdx.y == 0u) ? desc->heads : desc->kv_heads) * desc->head_dim;
    float *data = (blockIdx.y == 0u) ? work->q : work->k;
    unsigned long long at = (blockIdx.y == 0u) ? desc->layer[layer].attn_q_norm
                                               : desc->layer[layer].attn_k_norm;
    const float *weight = (const float *)aotx_block_tensor(work->weights, at);
    unsigned int lane = threadIdx.x & 31u;
    unsigned int warp = threadIdx.x >> 5;
    for (unsigned int t = blockIdx.x; t < run->tokens; t += gridDim.x) {
        float *row = data + (size_t)t * width;
        float sum = 0.0f;
        for (unsigned int d = threadIdx.x; d < width; d += blockDim.x) {
            sum += row[d] * row[d];
        }
        for (unsigned int step = 16u; step != 0u; step >>= 1) {
            sum += __shfl_down_sync(0xffffffffu, sum, step);
        }
        if (lane == 0u) {
            sums[warp] = sum;
        }
        __syncthreads();
        if (threadIdx.x == 0u) {
            float total = 0.0f;
            for (unsigned int w = 0u; w < blockDim.x / 32u; ++w) {
                total += sums[w];
            }
            sums[0] = rsqrtf(total / (float)width + desc->rms_eps);
        }
        __syncthreads();
        float scale = sums[0];
        for (unsigned int d = threadIdx.x; d < width; d += blockDim.x) {
            row[d] = scale * row[d] * weight[d];
        }
        __syncthreads();
    }
}
