/* Purpose: Normalize and rotate gated query heads, then attend over compact cache pages.
 * Owns: Nothing; the model buffers and cache pages hold all rows.
 * Launch shape: One warp per query or key head; attention uses four token warps per block.
 * Lifetime: One pass of the forward graph. */
#include "model/blocks.cuh"
#include "model/conduct.cuh"
#include "model/hybrid.cuh"

#define AOTX_GATED_ELEMENTS 8u

__device__ __forceinline__ static float aotx_gated_total(float value)
{
    for (unsigned int step = 16u; step != 0u; step >>= 1)
        value += __shfl_xor_sync(0xFFFFFFFFu, value, step);
    return value;
}

__global__ void aotx_model_gated_qkv(unsigned int role, unsigned int layer)
{
    const aotx_model_desc *desc = &aotx_model[role];
    const aotx_model_run *run = &aotx_model_call[role];
    const aotx_model_work *work = &aotx_model_space[role];
    unsigned int head = blockIdx.y;
    unsigned int lane = threadIdx.x;
    unsigned int dim = desc->head_dim;
    if (head >= desc->heads + desc->kv_heads) return;
    unsigned int query = head < desc->heads;
    unsigned int kh = query ? 0u : head - desc->heads;
    unsigned int slot = query ? AOTX_ATTENTION_Q_NORM : AOTX_ATTENTION_K_NORM;
    const float *weight = (const float *)aotx_block_tensor(
        work->weights, desc->layer[layer].offset[slot]);
    unsigned int pairs = desc->rope_dim / 2u;

    for (unsigned int t = blockIdx.x; t < run->tokens; t += gridDim.x) {
        unsigned int s = aotx_model_which(run->offset, run->seqs, t);
        unsigned int position = work->base[s] + t - run->offset[s];
        const float *src = query
            ? work->qgate + ((unsigned long long)t * desc->heads + head) * 2u * dim
            : work->k + ((unsigned long long)t * desc->kv_heads + kh) * dim;
        float sum = 0.0f;
        for (unsigned int d = lane; d < dim; d += 32u) sum += src[d] * src[d];
        float scale = rsqrtf(aotx_gated_total(sum) / (float)dim + desc->rms_eps);
        half *dst;
        if (query) {
            dst = work->qh + ((unsigned long long)t * desc->heads + head) * dim;
        } else {
            half *block = aotx_kvl_block(&work->shape, run->agent[s], layer, position);
            if (block == 0) {
                if (lane == 0u) atomicAdd(&aotx_model_faults, 1u);
                continue;
            }
            unsigned int at = position % AOTX_KVL_BLOCK;
            dst = block + ((unsigned long long)kh * AOTX_KVL_BLOCK + at) * dim;
            half *value = dst + (unsigned long long)desc->kv_heads * AOTX_KVL_BLOCK * dim;
            const float *from = work->v + ((unsigned long long)t * desc->kv_heads + kh) * dim;
            for (unsigned int d = lane; d < dim; d += 32u) value[d] = __float2half(from[d]);
        }
        /* Text positions coincide in every rotary section. Only the rotary prefix turns. */
        for (unsigned int i = lane; i < pairs; i += 32u) {
            float angle = (float)position * powf(desc->rope_theta, -2.0f * (float)i
                                                                              / desc->rope_dim);
            float sine, cosine;
            sincosf(angle, &sine, &cosine);
            float a = scale * src[i] * weight[i];
            float b = scale * src[i + pairs] * weight[i + pairs];
            dst[i] = __float2half(a * cosine - b * sine);
            dst[i + pairs] = __float2half(a * sine + b * cosine);
        }
        for (unsigned int d = desc->rope_dim + lane; d < dim; d += 32u)
            dst[d] = __float2half(scale * src[d] * weight[d]);
    }
}

__global__ void aotx_model_gated_attend(unsigned int role, unsigned int layer)
{
    const aotx_model_desc *desc = &aotx_model[role];
    const aotx_model_run *run = &aotx_model_call[role];
    const aotx_model_work *work = &aotx_model_space[role];
    unsigned int lane = threadIdx.x & 31u;
    unsigned int warp = threadIdx.x >> 5;
    unsigned int head = blockIdx.y;
    unsigned int dim = desc->head_dim;
    if (head >= desc->heads) return;
    unsigned int kh = head / (desc->heads / desc->kv_heads);
    float scale = rsqrtf((float)dim);
    for (unsigned int t = blockIdx.x * AOTX_MODEL_ATTN_TOKENS + warp;
         t < run->tokens; t += gridDim.x * AOTX_MODEL_ATTN_TOKENS) {
        unsigned int s = aotx_model_which(run->offset, run->seqs, t);
        unsigned int position = work->base[s] + t - run->offset[s];
        unsigned int agent = run->agent[s];
        unsigned long long row = ((unsigned long long)t * desc->heads + head) * dim;
        float query[AOTX_GATED_ELEMENTS], sum[AOTX_GATED_ELEMENTS];
        #pragma unroll
        for (unsigned int u = 0u; u < AOTX_GATED_ELEMENTS; ++u) {
            unsigned int d = lane + u * 32u;
            query[u] = d < dim ? __half2float(work->qh[row + d]) : 0.0f;
            sum[u] = 0.0f;
        }
        float top = -INFINITY;
        float mass = 0.0f;
        int missing = 0;
        for (unsigned int first = 0u; first <= position; first += AOTX_KVL_BLOCK) {
            const half *block = aotx_kvl_block(&work->shape, agent, layer, first);
            if (block == 0) {
                if (lane == 0u) atomicAdd(&aotx_model_faults, 1u);
                missing = 1;
                break;
            }
            const half *keys = block + (unsigned long long)kh * AOTX_KVL_BLOCK * dim;
            const half *values = keys + (unsigned long long)desc->kv_heads * AOTX_KVL_BLOCK * dim;
            unsigned int count = min(AOTX_KVL_BLOCK, position + 1u - first);
            for (unsigned int j = 0u; j < count; ++j) {
                float dot = 0.0f;
                #pragma unroll
                for (unsigned int u = 0u; u < AOTX_GATED_ELEMENTS; ++u) {
                    unsigned int d = lane + u * 32u;
                    if (d < dim) dot += query[u] * __half2float(keys[j * dim + d]);
                }
                dot = aotx_gated_total(dot) * scale;
                float raised = fmaxf(top, dot);
                float shift = expf(top - raised);
                float weight = expf(dot - raised);
                #pragma unroll
                for (unsigned int u = 0u; u < AOTX_GATED_ELEMENTS; ++u) {
                    unsigned int d = lane + u * 32u;
                    if (d < dim)
                        sum[u] = sum[u] * shift + weight * __half2float(values[j * dim + d]);
                }
                mass = mass * shift + weight;
                top = raised;
            }
        }
        if (missing) continue;
        const float *gate = work->qgate + row * 2u + dim;
        #pragma unroll
        for (unsigned int u = 0u; u < AOTX_GATED_ELEMENTS; ++u) {
            unsigned int d = lane + u * 32u;
            if (d < dim) {
                float sigmoid = 1.0f / (1.0f + expf(-gate[d]));
                work->att[row + d] = __float2half((sum[u] / mass) * sigmoid);
            }
        }
        /* Page mass describes softmax weights, before the output gate. */
        if (run->telemetry == 0u) continue;
        for (unsigned int first = 0u; first <= position; first += AOTX_KVL_BLOCK) {
            const half *block = aotx_kvl_block(&work->shape, agent, layer, first);
            const half *keys = block + (unsigned long long)kh * AOTX_KVL_BLOCK * dim;
            unsigned int count = min(AOTX_KVL_BLOCK, position + 1u - first);
            float page_mass = 0.0f;
            for (unsigned int j = 0u; j < count; ++j) {
                float dot = 0.0f;
                #pragma unroll
                for (unsigned int u = 0u; u < AOTX_GATED_ELEMENTS; ++u) {
                    unsigned int d = lane + u * 32u;
                    if (d < dim) dot += query[u] * __half2float(keys[j * dim + d]);
                }
                page_mass += expf(aotx_gated_total(dot) * scale - top) / mass;
            }
            unsigned int page = aotx_kvl_page_of(&work->shape, layer, first);
            if (lane == 0u) atomicAdd(&aotx_page_mass[agent][page], page_mass);
        }
    }
}
