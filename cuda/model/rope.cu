/* Purpose: Norm each head of the query and the key, turn them, and fill the cache pages.
 * Owns: Nothing; the buffer block and the cache pages hold the rows.
 * Launch shape: One block for each token and head; the threads hold the pairs of a head.
 * Lifetime: One pass of the forward graph. */
#include "model/blocks.cuh"
#include "model/forward.cuh"

/* Add one value of every thread of the block. The block is small, so one warp adds the
 * warp sums and the result goes back through shared memory. */
__device__ __forceinline__ static float aotx_rope_total(float value, float *share)
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

__global__ void aotx_model_qkv(unsigned int role, unsigned int layer)
{
    __shared__ float share[32];
    const aotx_model_desc *desc = &aotx_model[role];
    const aotx_model_run *run = &aotx_model_call[role];
    const aotx_model_work *work = &aotx_model_space[role];
    unsigned int head = blockIdx.y;
    if (head >= desc->heads + desc->kv_heads) {
        return;
    }
    unsigned int dim = desc->head_dim;
    unsigned int half_dim = dim / 2u;

    /* The query heads come first and the key heads come after them. A key head also writes
     * the value row of the same head, which takes no norm and no turn. */
    unsigned int kv_head = 0u;
    const float *weight;
    if (head < desc->heads) {
        weight = (const float *)aotx_block_tensor(work->weights,
                                                  desc->layer[layer].attn_q_norm);
    } else {
        kv_head = head - desc->heads;
        weight = (const float *)aotx_block_tensor(work->weights,
                                                  desc->layer[layer].attn_k_norm);
    }
    if (weight == 0) {
        return;
    }

    /* The turn takes the pair of element i and element i plus half the head. The angle of
     * the pair falls with the pair number, and the position gives the whole angle. */
    float step = powf(desc->rope_theta, -2.0f / (float)dim);

    /* One block takes a run of rows of one head, so the grid holds the machine and not the
     * batch. Every thread of the block takes the same row, because the norm of a head adds
     * over the block. */
    for (unsigned int t = blockIdx.x; t < run->tokens; t += gridDim.x) {
        unsigned int s = aotx_model_which(run->offset, run->seqs, t);
        unsigned int position = work->base[s] + (t - run->offset[s]);
        unsigned int agent = run->agent[s];
        const float *src = (head < desc->heads)
            ? (work->q + (unsigned long long)(t * desc->heads + head) * dim)
            : (work->k + (unsigned long long)(t * desc->kv_heads + kv_head) * dim);

        /* The norm covers the whole head. Each thread holds the two elements of one pair.
         * The sum of the squares of a head is the sum over the threads of the block. */
        float sum = 0.0f;
        for (unsigned int i = threadIdx.x; i < half_dim; i += blockDim.x) {
            float low = src[i];
            float high = src[i + half_dim];
            sum += low * low + high * high;
        }
        sum = aotx_rope_total(sum, share);
        float scale = rsqrtf(sum / (float)dim + desc->rms_eps);

        half *dst;
        if (head < desc->heads) {
            dst = work->qh + (unsigned long long)(t * desc->heads + head) * dim;
        } else {
            dst = aotx_kvl_key(&work->shape, agent, layer, kv_head, position);
            half *value = aotx_kvl_value(&work->shape, agent, layer, kv_head, position);
            const float *from = work->v
                + (unsigned long long)(t * desc->kv_heads + kv_head) * dim;
            if (value != 0) {
                for (unsigned int d = threadIdx.x; d < dim; d += blockDim.x) {
                    value[d] = __float2half(from[d]);
                }
            }
        }
        if (dst == 0) {
            if (threadIdx.x == 0u) {
                atomicAdd(&aotx_model_faults, 1u);
            }
            __syncthreads();
            continue;
        }
        for (unsigned int i = threadIdx.x; i < half_dim; i += blockDim.x) {
            float low = scale * src[i] * weight[i];
            float high = scale * src[i + half_dim] * weight[i + half_dim];
            float angle = (float)position * powf(step, (float)i);
            float cosine;
            float sine;
            sincosf(angle, &sine, &cosine);
            dst[i] = __float2half(low * cosine - high * sine);
            dst[i + half_dim] = __float2half(low * sine + high * cosine);
        }
        __syncthreads();
    }
}
