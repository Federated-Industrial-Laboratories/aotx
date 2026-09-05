/* Purpose: Multiply each token by the weight slice of its selected expert.
 * Owns: Nothing; the caller holds the weights, input rows and output rows.
 * Launch shape: 256 threads; grid.x takes 16 weight rows and grid.y steps through tokens.
 * Lifetime: One selected rank of one forward layer. */
#include "model/experts.cuh"
#include "model/gemv.cuh"

__global__ void aotx_model_expert_matrix(unsigned int role, unsigned int rank,
                                        const void *w, unsigned int type,
                                        unsigned int n, unsigned int k,
                                        const half *x, float *y)
{
    const aotx_model_desc *desc = &aotx_model[role];
    const aotx_model_run *run = &aotx_model_call[role];
    const aotx_model_work *work = &aotx_model_space[role];
    unsigned long long slice = (unsigned long long)n * aotx_matrix_row_bytes(type, k);
    for (unsigned int t = blockIdx.y; t < run->tokens; t += gridDim.y) {
        unsigned int expert = work->expert_id[(size_t)t * desc->expert_used_count + rank];
        const unsigned char *weight = (const unsigned char *)w + (unsigned long long)expert * slice;
        const half *input = x + (size_t)t * k;
        float *output = y + (size_t)t * n;
        switch (type) {
        case AOTX_WEIGHT_Q8_0:
            aotx_gemv_pass<AOTX_WEIGHT_Q8_0, 1u>(weight, n, k, input, 1u, 0u, output, blockIdx.x);
            break;
        case AOTX_WEIGHT_Q4_0:
            aotx_gemv_pass<AOTX_WEIGHT_Q4_0, 1u>(weight, n, k, input, 1u, 0u, output, blockIdx.x);
            break;
        case AOTX_WEIGHT_Q4_1:
            aotx_gemv_pass<AOTX_WEIGHT_Q4_1, 1u>(weight, n, k, input, 1u, 0u, output, blockIdx.x);
            break;
        case AOTX_WEIGHT_Q5_0:
            aotx_gemv_pass<AOTX_WEIGHT_Q5_0, 1u>(weight, n, k, input, 1u, 0u, output, blockIdx.x);
            break;
        case AOTX_WEIGHT_Q5_1:
            aotx_gemv_pass<AOTX_WEIGHT_Q5_1, 1u>(weight, n, k, input, 1u, 0u, output, blockIdx.x);
            break;
        case AOTX_WEIGHT_Q2_K:
            aotx_gemv_pass<AOTX_WEIGHT_Q2_K, 1u>(weight, n, k, input, 1u, 0u, output, blockIdx.x);
            break;
        case AOTX_WEIGHT_Q3_K:
            aotx_gemv_pass<AOTX_WEIGHT_Q3_K, 1u>(weight, n, k, input, 1u, 0u, output, blockIdx.x);
            break;
        case AOTX_WEIGHT_Q4_K:
            aotx_gemv_pass<AOTX_WEIGHT_Q4_K, 1u>(weight, n, k, input, 1u, 0u, output, blockIdx.x);
            break;
        case AOTX_WEIGHT_Q5_K:
            aotx_gemv_pass<AOTX_WEIGHT_Q5_K, 1u>(weight, n, k, input, 1u, 0u, output, blockIdx.x);
            break;
        case AOTX_WEIGHT_Q6_K:
            aotx_gemv_pass<AOTX_WEIGHT_Q6_K, 1u>(weight, n, k, input, 1u, 0u, output, blockIdx.x);
            break;
        case AOTX_WEIGHT_F16:
            aotx_gemv_pass<AOTX_WEIGHT_F16, 1u>(weight, n, k, input, 1u, 0u, output, blockIdx.x);
            break;
        case AOTX_WEIGHT_F32:
            aotx_gemv_pass<AOTX_WEIGHT_F32, 1u>(weight, n, k, input, 1u, 0u, output, blockIdx.x);
            break;
        default:
            break;
        }
    }
}
