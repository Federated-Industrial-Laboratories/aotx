/* Purpose: Give the probability of the answer yes for each query and document pair.
 * Owns: Nothing; the buffer block holds the rows and the caller holds the scores.
 * Launch shape: One block for each pair; the threads hold the hidden width.
 * Lifetime: One pass of the forward graph. */
#include "model/blocks.cuh"
#include "rerank/rerank.cuh"

__global__ void aotx_rerank_score(unsigned int role)
{
    __shared__ float part[AOTX_MODEL_ROW_THREADS];
    const aotx_model_desc *desc = &aotx_model[role];
    const aotx_model_run *run = &aotx_model_call[role];
    const aotx_model_work *work = &aotx_model_space[role];
    unsigned int r = blockIdx.x;
    if (r >= run->seqs || run->score == 0) {
        return;
    }
    const float *weight = (const float *)aotx_block_tensor(work->weights, desc->output_norm);
    const void *head = aotx_block_tensor(work->weights, desc->cls_output);
    if (weight == 0 || head == 0) {
        return;
    }

    /* The class head is two rows of the hidden width, so the two products run in single
     * precision from the residual stream. A half copy of the row would move the value by
     * more than the bound of this head. */
    unsigned int src = run->offset[r + 1u] - 1u;
    if (src >= run->tokens) {
        return;
    }
    const float *row = work->resid + (unsigned long long)src * desc->hidden;
    float square = 0.0f;
    for (unsigned int d = threadIdx.x; d < desc->hidden; d += blockDim.x) {
        square += row[d] * row[d];
    }
    part[threadIdx.x] = square;
    __syncthreads();
    if (threadIdx.x == 0u) {
        for (unsigned int i = 1u; i < blockDim.x; ++i) {
            part[0] += part[i];
        }
    }
    __syncthreads();
    float scale = rsqrtf(part[0] / (float)desc->hidden + desc->rms_eps);
    unsigned int type = aotx_model_head_type[role][1];

    float logit[AOTX_RERANK_CLASSES];
    for (unsigned int c = 0u; c < AOTX_RERANK_CLASSES; ++c) {
        float sum = 0.0f;
        for (unsigned int d = threadIdx.x; d < desc->hidden; d += blockDim.x) {
            sum += scale * row[d] * weight[d]
                 * aotx_block_at(head, type, (unsigned long long)c * desc->hidden + d);
        }
        part[threadIdx.x] = sum;
        __syncthreads();
        if (threadIdx.x == 0u) {
            for (unsigned int i = 1u; i < blockDim.x; ++i) {
                part[0] += part[i];
            }
        }
        __syncthreads();
        logit[c] = part[0];
        __syncthreads();
    }
    if (threadIdx.x != 0u) {
        return;
    }

    /* The first class output is the answer yes and the second is the answer no. The two
     * come from a softmax over the pair, so the score is between zero and one. */
    float top = fmaxf(logit[0], logit[1]);
    float yes = expf(logit[0] - top);
    float no = expf(logit[1] - top);
    run->score[r] = yes / (yes + no);
}
