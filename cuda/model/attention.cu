/* Purpose: Attend over the key and value pages of one agent slot, for every token.
 * Owns: Nothing; the pages and the buffer block hold the rows.
 * Launch shape: One block for a run of tokens of one head; one warp for each token.
 * Lifetime: One pass of the forward graph. */
#include "model/blocks.cuh"
#include "model/conduct.cuh"
#include "model/forward.cuh"

/* Elements of a head that one lane holds. A warp of 32 lanes therefore covers a head of up
 * to 128 elements, which is the head width of this model family. */
#define AOTX_ATTN_SLOTS   4u

/* Add one value over the lanes of a warp. Every lane keeps the sum. */
__device__ __forceinline__ static float aotx_attn_total(float value)
{
    for (unsigned int step = 16u; step != 0u; step >>= 1) {
        value += __shfl_xor_sync(0xFFFFFFFFu, value, step);
    }
    return value;
}

__global__ void aotx_model_attend(unsigned int role, unsigned int layer)
{
    const aotx_model_desc *desc = &aotx_model[role];
    const aotx_model_run *run = &aotx_model_call[role];
    const aotx_model_work *work = &aotx_model_space[role];
    unsigned int lane = threadIdx.x & 31u;
    unsigned int warp = threadIdx.x >> 5;
    unsigned int head = blockIdx.y;
    unsigned int dim = desc->head_dim;
    unsigned int slots = dim / 32u;
    if (head >= desc->heads || slots > AOTX_ATTN_SLOTS) {
        return;
    }
    unsigned int kv_head = head / (desc->heads / desc->kv_heads);
    float scale = 1.0f / sqrtf((float)dim);

    /* One block takes a run of tokens of one head, so the grid holds the machine and not
     * the batch. Each warp of the block holds one token of the run. */
    for (unsigned int base = blockIdx.x * AOTX_MODEL_ATTN_TOKENS; base < run->tokens;
         base += gridDim.x * AOTX_MODEL_ATTN_TOKENS) {
        unsigned int t = base + warp;
        if (t >= run->tokens) {
            continue;
        }
        unsigned int s = aotx_model_which(run->offset, run->seqs, t);
        unsigned int position = work->base[s] + (t - run->offset[s]);
        unsigned int agent = run->agent[s];

        /* The lane holds one element of every quarter of the head. A read of a key row or
         * of a value row by the warp is therefore one run of bytes. */
        const half *from = work->qh + (unsigned long long)(t * desc->heads + head) * dim;
        float query[AOTX_ATTN_SLOTS];
        float sum[AOTX_ATTN_SLOTS];
        #pragma unroll
        for (unsigned int u = 0u; u < AOTX_ATTN_SLOTS; ++u) {
            query[u] = (u < slots) ? __half2float(from[lane + u * 32u]) : 0.0f;
            sum[u] = 0.0f;
        }
        float top = -INFINITY;
        float mass = 0.0f;
        int missing = 0;

        /* One layout block holds a run of positions of this layer. The address of the
         * block comes once for the run, so the inner loop holds no division. */
        for (unsigned int first = 0u; first <= position; first += AOTX_KVL_BLOCK) {
            const half *block = aotx_kvl_block(&work->shape, agent, layer, first);
            if (block == 0) {
                if (lane == 0u) {
                    atomicAdd(&aotx_model_faults, 1u);
                }
                missing = 1;
                break;
            }
            const half *keys = block + (unsigned long long)kv_head * AOTX_KVL_BLOCK * dim;
            const half *values = block
                + (unsigned long long)(desc->kv_heads + kv_head) * AOTX_KVL_BLOCK * dim;
            unsigned int last = AOTX_KVL_BLOCK;
            if (first + last > position + 1u) {
                last = position + 1u - first;
            }
            for (unsigned int j = 0u; j < last; ++j) {
                const half *key = keys + (unsigned long long)j * dim;
                float dot = 0.0f;
                #pragma unroll
                for (unsigned int u = 0u; u < AOTX_ATTN_SLOTS; ++u) {
                    if (u < slots) {
                        dot += query[u] * __half2float(key[lane + u * 32u]);
                    }
                }
                dot = aotx_attn_total(dot) * scale;

                /* The running maximum keeps the exponent in range, and the sums that came
                 * before it move to the new maximum by one factor. */
                float raised = fmaxf(top, dot);
                float shift = expf(top - raised);
                float weight = expf(dot - raised);
                const half *value = values + (unsigned long long)j * dim;
                #pragma unroll
                for (unsigned int u = 0u; u < AOTX_ATTN_SLOTS; ++u) {
                    if (u < slots) {
                        sum[u] = sum[u] * shift
                               + weight * __half2float(value[lane + u * 32u]);
                    }
                }
                mass = mass * shift + weight;
                top = raised;
            }
        }
        if (missing != 0) {
            continue;
        }
        half *out = work->att + (unsigned long long)(t * desc->heads + head) * dim;
        float scale_back = (mass > 0.0f) ? (1.0f / mass) : 0.0f;
        #pragma unroll
        for (unsigned int u = 0u; u < AOTX_ATTN_SLOTS; ++u) {
            if (u < slots) {
                out[lane + u * 32u] = __float2half(sum[u] * scale_back);
            }
        }

        /* Decode telemetry makes a second read after the output is complete. It attributes
         * each normalized attention weight to the page that supplied its key. */
        if (run->telemetry != 0u && mass > 0.0f) {
            for (unsigned int first = 0u; first <= position; first += AOTX_KVL_BLOCK) {
                const half *block = aotx_kvl_block(&work->shape, agent, layer, first);
                if (block == 0) {
                    break;
                }
                const half *keys = block
                    + (unsigned long long)kv_head * AOTX_KVL_BLOCK * dim;
                unsigned int last = AOTX_KVL_BLOCK;
                if (first + last > position + 1u) {
                    last = position + 1u - first;
                }
                unsigned int page = aotx_kvl_page_of(&work->shape, layer, first);
                for (unsigned int j = 0u; j < last; ++j) {
                    const half *key = keys + (unsigned long long)j * dim;
                    float dot = 0.0f;
                    #pragma unroll
                    for (unsigned int u = 0u; u < AOTX_ATTN_SLOTS; ++u) {
                        if (u < slots) {
                            dot += query[u] * __half2float(key[lane + u * 32u]);
                        }
                    }
                    dot = aotx_attn_total(dot) * scale;
                    if (lane == 0u && page < AOTX_KV_PAGES_EACH) {
                        atomicAdd(&aotx_page_mass[agent][page], expf(dot - top) / mass);
                    }
                }
            }
        }
    }
}
