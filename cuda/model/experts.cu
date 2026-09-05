/* Purpose: Select experts from full router probabilities and add their weighted outputs.
 * Owns: Nothing; the forward buffers hold the probabilities, indices and sums.
 * Launch shape: 256 threads per block; blocks step through token rows.
 * Lifetime: One forward graph. */
#include <math_constants.h>

#include "model/blocks.cuh"
#include "model/experts.cuh"

__global__ void aotx_model_expert_route(unsigned int role, unsigned int layer)
{
    __shared__ float probability[AOTX_LAYER_EXPERTS_MAX];
    __shared__ float reduce[AOTX_EXPERT_ROUTE_THREADS];
    const aotx_model_desc *desc = &aotx_model[role];
    const aotx_model_run *run = &aotx_model_call[role];
    const aotx_model_work *work = &aotx_model_space[role];
    const float *router = (const float *)aotx_block_tensor(work->weights,
                                                          desc->layer[layer].offset[AOTX_EXPERT_ROUTER]);
    unsigned int lane = threadIdx.x & 31u;
    unsigned int warp = threadIdx.x >> 5;
    unsigned int experts = desc->expert_count;
    for (unsigned int t = blockIdx.x; t < run->tokens; t += gridDim.x) {
        const half *x = work->x + (size_t)t * desc->hidden;
        for (unsigned int e = warp; e < experts; e += AOTX_EXPERT_ROUTE_THREADS / 32u) {
            const float *row = router + (size_t)e * desc->hidden;
            float dot = 0.0f;
            for (unsigned int d = lane; d < desc->hidden; d += 32u) {
                dot += row[d] * __half2float(x[d]);
            }
            for (unsigned int step = 16u; step != 0u; step >>= 1) {
                dot += __shfl_down_sync(0xFFFFFFFFu, dot, step);
            }
            if (lane == 0u) {
                probability[e] = dot;
            }
        }
        __syncthreads();
        unsigned int e = threadIdx.x;
        reduce[e] = e < experts ? probability[e] : -CUDART_INF_F;
        __syncthreads();
        for (unsigned int step = AOTX_EXPERT_ROUTE_THREADS / 2u; step != 0u; step >>= 1) {
            if (e < step) {
                reduce[e] = fmaxf(reduce[e], reduce[e + step]);
            }
            __syncthreads();
        }
        float value = e < experts ? expf(probability[e] - reduce[0]) : 0.0f;
        __syncthreads();
        reduce[e] = value;
        __syncthreads();
        for (unsigned int step = AOTX_EXPERT_ROUTE_THREADS / 2u; step != 0u; step >>= 1) {
            if (e < step) {
                reduce[e] += reduce[e + step];
            }
            __syncthreads();
        }
        if (e < experts) {
            value /= reduce[0];
            probability[e] = value;
            work->expert_prob[(size_t)t * experts + e] = value;
        }
        __syncthreads();
        if (e < experts) {
            unsigned int rank = 0u;
            for (unsigned int other = 0u; other < experts; ++other) {
                float candidate = probability[other];
                rank += candidate > value || (candidate == value && other < e);
            }
            if (rank < desc->expert_used_count) {
                work->expert_id[(size_t)t * desc->expert_used_count + rank] = e;
            }
        }
        __syncthreads();
    }
}

__global__ void aotx_model_expert_add(unsigned int role, unsigned int rank)
{
    const aotx_model_desc *desc = &aotx_model[role];
    const aotx_model_run *run = &aotx_model_call[role];
    const aotx_model_work *work = &aotx_model_space[role];
    for (unsigned int t = blockIdx.x; t < run->tokens; t += gridDim.x) {
        unsigned int expert = work->expert_id[(size_t)t * desc->expert_used_count + rank];
        float probability = work->expert_prob[(size_t)t * desc->expert_count + expert];
        size_t first = (size_t)t * desc->hidden;
        for (unsigned int d = threadIdx.x; d < desc->hidden; d += blockDim.x) {
            /* Round the product before the sum, as separate graph operations do. */
            float value = __fmul_rn(work->proj[first + d], probability);
            if (rank != 0u) {
                value = __fadd_rn(work->expert_sum[first + d], value);
            }
            work->expert_sum[first + d] = value;
            if (rank + 1u == desc->expert_used_count) {
                work->proj[first + d] = value;
            }
        }
    }
}
