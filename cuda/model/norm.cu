/* Purpose: Apply the root mean square norm, the residual sum and the gated unit.
 * Owns: Nothing; the buffer block holds the rows.
 * Launch shape: One block for a run of rows of the batch; the threads hold the width.
 * Lifetime: One pass of the forward graph. */
#include "model/blocks.cuh"
#include "model/forward.cuh"

/* Add one value of every thread of the block. The first warp adds the warp sums, and the
 * result goes to every thread through shared memory. */
__device__ __forceinline__ static float aotx_model_total(float value, float *share)
{
    unsigned int lane = threadIdx.x & 31u;
    unsigned int warp = threadIdx.x >> 5;
    for (unsigned int step = 16u; step != 0u; step >>= 1) {
        value += __shfl_down_sync(0xFFFFFFFFu, value, step);
    }
    if (lane == 0u) {
        share[warp] = value;
    }
    __syncthreads();
    unsigned int warps = (blockDim.x + 31u) >> 5;
    if (threadIdx.x == 0u) {
        float sum = 0.0f;
        for (unsigned int i = 0u; i < warps; ++i) {
            sum += share[i];
        }
        share[0] = sum;
    }
    __syncthreads();
    return share[0];
}

/* The norm weight of one step. Every norm weight of this model family is a single precision
 * tensor of the hidden width. */
__device__ __forceinline__ static const float *aotx_model_weight(const aotx_model_desc *desc,
                                                                 unsigned long long base,
                                                                 unsigned int layer,
                                                                 unsigned int which)
{
    unsigned long long at = desc->output_norm;
    if (which == AOTX_MODEL_NORM_ATTN) {
        at = desc->layer[layer].attn_norm;
    } else if (which == AOTX_MODEL_NORM_FFN) {
        at = desc->layer[layer].ffn_norm;
    }
    return (const float *)aotx_block_tensor(base, at);
}

__global__ void aotx_model_norm(unsigned int role, unsigned int layer, unsigned int which)
{
    __shared__ float share[32];
    const aotx_model_desc *desc = &aotx_model[role];
    const aotx_model_run *run = &aotx_model_call[role];
    const aotx_model_work *work = &aotx_model_space[role];
    const float *weight = aotx_model_weight(desc, work->weights, layer, which);
    if (weight == 0) {
        return;
    }

    /* One block takes a run of rows. The grid therefore holds the machine and not the
     * batch, and a batch of one row leaves few blocks to start and to stop. */
    for (unsigned int t = blockIdx.x; t < run->tokens; t += gridDim.x) {
        const float *row = work->resid + (unsigned long long)t * desc->hidden;
        float sum = 0.0f;
        for (unsigned int d = threadIdx.x; d < desc->hidden; d += blockDim.x) {
            sum += row[d] * row[d];
        }
        sum = aotx_model_total(sum, share);
        float scale = rsqrtf(sum / (float)desc->hidden + desc->rms_eps);
        half *out = (which == AOTX_MODEL_NORM_OUT) ? work->xnorm : work->x;
        out += (unsigned long long)t * desc->hidden;
        for (unsigned int d = threadIdx.x; d < desc->hidden; d += blockDim.x) {
            out[d] = __float2half(scale * row[d] * weight[d]);
        }
        __syncthreads();
    }
}

__global__ void aotx_model_residual(unsigned int role)
{
    const aotx_model_desc *desc = &aotx_model[role];
    const aotx_model_run *run = &aotx_model_call[role];
    const aotx_model_work *work = &aotx_model_space[role];
    for (unsigned int t = blockIdx.x; t < run->tokens; t += gridDim.x) {
        unsigned long long first = (unsigned long long)t * desc->hidden;
        for (unsigned int d = threadIdx.x; d < desc->hidden; d += blockDim.x) {
            work->resid[first + d] += work->proj[first + d];
        }
    }
}

__global__ void aotx_model_swiglu(unsigned int role)
{
    const aotx_model_desc *desc = &aotx_model[role];
    const aotx_model_run *run = &aotx_model_call[role];
    const aotx_model_work *work = &aotx_model_space[role];
    for (unsigned int t = blockIdx.x; t < run->tokens; t += gridDim.x) {
        unsigned long long first = (unsigned long long)t * desc->ffn;
        for (unsigned int d = threadIdx.x; d < desc->ffn; d += blockDim.x) {
            float gate = work->gate[first + d];
            float unit = gate / (1.0f + expf(-gate));
            work->act[first + d] = __float2half(unit * work->up[first + d]);
        }
    }
}
