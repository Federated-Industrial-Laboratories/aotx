/* Purpose: Turn each head of the query and the key by its position, and fill the cache pages.
 * Kernels apply a head norm, a projection bias, or neither before the turn.
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

/* Find the cache position and the agent of one token of the batch. */
__device__ __forceinline__ static void aotx_rope_where(const aotx_model_run *run,
                                                       const aotx_model_work *work,
                                                       unsigned int t,
                                                       unsigned int *position,
                                                       unsigned int *agent)
{
    unsigned int s = aotx_model_which(run->offset, run->seqs, t);
    *position = work->base[s] + (t - run->offset[s]);
    *agent = run->agent[s];
}

/* The row of one head of one token. The query heads come first and the key heads come
 * after them. */
__device__ __forceinline__ static const float *aotx_rope_source(const aotx_model_desc *desc,
                                                                const aotx_model_work *work,
                                                                unsigned int head,
                                                                unsigned int kv_head,
                                                                unsigned int t)
{
    unsigned int dim = desc->head_dim;
    return (head < desc->heads)
        ? (work->q + (unsigned long long)(t * desc->heads + head) * dim)
        : (work->k + (unsigned long long)(t * desc->kv_heads + kv_head) * dim);
}

/* The row that takes the turned head: the half query buffer for a query head, the key page
 * for a key head. A key head also writes the value row of the same head, which takes no
 * norm and no turn. A null result says that the page of the position is absent. */
__device__ __forceinline__ static half *aotx_rope_target(const aotx_model_desc *desc,
                                                         const aotx_model_work *work,
                                                         unsigned int head,
                                                         unsigned int kv_head,
                                                         unsigned int t,
                                                         unsigned int agent,
                                                         unsigned int layer,
                                                         unsigned int position)
{
    unsigned int dim = desc->head_dim;
    if (head < desc->heads) {
        return work->qh + (unsigned long long)(t * desc->heads + head) * dim;
    }
    half *value = aotx_kvl_value(&work->shape, agent, layer, kv_head, position);
    if (value != 0) {
        const float *from = work->v + (unsigned long long)(t * desc->kv_heads + kv_head) * dim;
        for (unsigned int d = threadIdx.x; d < dim; d += blockDim.x) {
            value[d] = __float2half(from[d]);
        }
    }
    return aotx_kvl_key(&work->shape, agent, layer, kv_head, position);
}

/* The two elements of pair i of a head. The split rule takes element i and element i plus
 * half the head; the adjacent rule takes element 2i and the element after it. */
__device__ __forceinline__ static void aotx_rope_pair(const aotx_model_desc *desc,
                                                      unsigned int i, unsigned int *low,
                                                      unsigned int *high)
{
    if (desc->rope_pairs == AOTX_ROPE_PAIRS_ADJACENT) {
        *low = 2u * i;
        *high = 2u * i + 1u;
    } else {
        *low = i;
        *high = i + desc->head_dim / 2u;
    }
}

/* The angle of pair i at one position. The angle of the pair falls with the pair number,
 * and the position gives the whole angle. A file that holds a row of frequency factors
 * divides the angle of each pair by its factor. */
__device__ __forceinline__ static float aotx_rope_angle(const aotx_model_desc *desc,
                                                        const float *factor, float step,
                                                        unsigned int i, unsigned int position)
{
    float angle = (float)position * powf(step, (float)i);
    return (factor != 0) ? angle / factor[i] : angle;
}

__device__ __forceinline__ static float aotx_rope_step(const aotx_model_desc *desc)
{
    return powf(desc->rope_theta, -2.0f / (float)desc->head_dim);
}

__device__ __forceinline__ static void aotx_rope_turn(half *dst, unsigned int low,
                                                      unsigned int high, float a, float b,
                                                      float angle)
{
    float cosine;
    float sine;
    sincosf(angle, &sine, &cosine);
    dst[low] = __float2half(a * cosine - b * sine);
    dst[high] = __float2half(a * sine + b * cosine);
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
    const float *factor = (const float *)aotx_block_tensor(work->weights, desc->rope_freqs);
    float step = aotx_rope_step(desc);

    /* One block takes a run of rows of one head, so the grid holds the machine and not the
     * batch. Every thread of the block takes the same row, because the norm of a head adds
     * over the block. */
    for (unsigned int t = blockIdx.x; t < run->tokens; t += gridDim.x) {
        unsigned int position;
        unsigned int agent;
        aotx_rope_where(run, work, t, &position, &agent);
        const float *src = aotx_rope_source(desc, work, head, kv_head, t);

        /* The norm covers the whole head. Each thread holds the two elements of one pair.
         * The sum of the squares of a head is the sum over the threads of the block. */
        float sum = 0.0f;
        for (unsigned int i = threadIdx.x; i < half_dim; i += blockDim.x) {
            unsigned int low;
            unsigned int high;
            aotx_rope_pair(desc, i, &low, &high);
            sum += src[low] * src[low] + src[high] * src[high];
        }
        sum = aotx_rope_total(sum, share);
        float scale = rsqrtf(sum / (float)dim + desc->rms_eps);

        half *dst = aotx_rope_target(desc, work, head, kv_head, t, agent, layer, position);
        if (dst == 0) {
            if (threadIdx.x == 0u) {
                atomicAdd(&aotx_model_faults, 1u);
            }
            __syncthreads();
            continue;
        }
        for (unsigned int i = threadIdx.x; i < half_dim; i += blockDim.x) {
            unsigned int low;
            unsigned int high;
            aotx_rope_pair(desc, i, &low, &high);
            aotx_rope_turn(dst, low, high, scale * src[low] * weight[low],
                           scale * src[high] * weight[high],
                           aotx_rope_angle(desc, factor, step, i, position));
        }
        __syncthreads();
    }
}

/* The turn without the head norm. No value crosses the threads, so the kernel holds no
 * shared array and no barrier, and one thread takes one pair. */
__global__ void aotx_model_qkv_turn(unsigned int role, unsigned int layer)
{
    const aotx_model_desc *desc = &aotx_model[role];
    const aotx_model_run *run = &aotx_model_call[role];
    const aotx_model_work *work = &aotx_model_space[role];
    unsigned int head = blockIdx.y;
    if (head >= desc->heads + desc->kv_heads) {
        return;
    }
    unsigned int half_dim = desc->head_dim / 2u;
    unsigned int kv_head = (head < desc->heads) ? 0u : head - desc->heads;
    const float *factor = (const float *)aotx_block_tensor(work->weights, desc->rope_freqs);
    float step = aotx_rope_step(desc);

    for (unsigned int t = blockIdx.x; t < run->tokens; t += gridDim.x) {
        unsigned int position;
        unsigned int agent;
        aotx_rope_where(run, work, t, &position, &agent);
        const float *src = aotx_rope_source(desc, work, head, kv_head, t);
        half *dst = aotx_rope_target(desc, work, head, kv_head, t, agent, layer, position);
        if (dst == 0) {
            if (threadIdx.x == 0u) {
                atomicAdd(&aotx_model_faults, 1u);
            }
            continue;
        }
        for (unsigned int i = threadIdx.x; i < half_dim; i += blockDim.x) {
            unsigned int low;
            unsigned int high;
            aotx_rope_pair(desc, i, &low, &high);
            aotx_rope_turn(dst, low, high, src[low], src[high],
                           aotx_rope_angle(desc, factor, step, i, position));
        }
    }
}

/* Bias spans the full projection. Each head reads its own part before the half conversion. */
__global__ void aotx_model_qkv_bias(unsigned int role, unsigned int layer)
{
    const aotx_model_desc *desc = &aotx_model[role];
    const aotx_model_run *run = &aotx_model_call[role];
    const aotx_model_work *work = &aotx_model_space[role];
    unsigned int head = blockIdx.y;
    if (head >= desc->heads + desc->kv_heads) {
        return;
    }
    unsigned int dim = desc->head_dim;
    unsigned int kv_head = (head < desc->heads) ? 0u : head - desc->heads;
    const unsigned char *weights = (const unsigned char *)work->weights;
    unsigned long long bias_at = (head < desc->heads) ? desc->layer[layer].attn_q_bias
                                                     : desc->layer[layer].attn_k_bias;
    unsigned int bias_head = (head < desc->heads) ? head : kv_head;
    const float *bias = (const float *)(weights + bias_at)
                     + (unsigned long long)bias_head * dim;
    const float *factor = (const float *)aotx_block_tensor(work->weights, desc->rope_freqs);
    float step = aotx_rope_step(desc);

    for (unsigned int t = blockIdx.x; t < run->tokens; t += gridDim.x) {
        unsigned int position;
        unsigned int agent;
        aotx_rope_where(run, work, t, &position, &agent);
        const float *src = aotx_rope_source(desc, work, head, kv_head, t);
        half *dst;
        if (head < desc->heads) {
            dst = work->qh + (unsigned long long)(t * desc->heads + head) * dim;
        } else {
            dst = aotx_kvl_key(&work->shape, agent, layer, kv_head, position);
            half *value = aotx_kvl_value(&work->shape, agent, layer, kv_head, position);
            if (dst == 0 || value == 0) {
                if (threadIdx.x == 0u) {
                    atomicAdd(&aotx_model_faults, 1u);
                }
                continue;
            }
            const float *from = work->v
                              + (unsigned long long)(t * desc->kv_heads + kv_head) * dim;
            const float *v_bias = (const float *)(weights + desc->layer[layer].attn_v_bias)
                                + (unsigned long long)kv_head * dim;
            for (unsigned int d = threadIdx.x; d < dim; d += blockDim.x) {
                value[d] = __float2half(from[d] + v_bias[d]);
            }
        }
        for (unsigned int i = threadIdx.x; i < dim / 2u; i += blockDim.x) {
            unsigned int low;
            unsigned int high;
            aotx_rope_pair(desc, i, &low, &high);
            aotx_rope_turn(dst, low, high, src[low] + bias[low], src[high] + bias[high],
                           aotx_rope_angle(desc, factor, step, i, position));
        }
    }
}
