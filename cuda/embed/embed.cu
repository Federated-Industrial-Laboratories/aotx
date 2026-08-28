/* Purpose: Give one unit vector for each sequence from its last row.
 * Owns: Nothing; the buffer block holds the rows and the caller holds the vectors.
 * Launch shape: One block for each sequence; the threads hold the hidden width.
 * Lifetime: One pass of the forward graph. */
#include "embed/embed.cuh"
#include "model/blocks.cuh"

__global__ void aotx_embed_pool(unsigned int role)
{
    __shared__ float part[AOTX_MODEL_ROW_THREADS];
    const aotx_model_desc *desc = &aotx_model[role];
    const aotx_model_run *run = &aotx_model_call[role];
    const aotx_model_work *work = &aotx_model_space[role];
    unsigned int r = blockIdx.x;
    if (r >= run->seqs || run->pooled == 0) {
        return;
    }
    const float *weight = (const float *)aotx_block_tensor(work->weights, desc->output_norm);
    if (weight == 0) {
        return;
    }

    /* The last row of a sequence is the row before the first row of the sequence after it.
     * The norm and the unit vector both run in single precision. A half copy of the row
     * loses more than the cosine gate of this head allows. */
    unsigned int src = run->offset[r + 1u] - 1u;
    if (src >= run->tokens) {
        return;
    }
    const float *row = work->resid + (unsigned long long)src * desc->hidden;
    float sum = 0.0f;
    for (unsigned int d = threadIdx.x; d < desc->hidden; d += blockDim.x) {
        sum += row[d] * row[d];
    }
    part[threadIdx.x] = sum;
    __syncthreads();
    if (threadIdx.x == 0u) {
        for (unsigned int i = 1u; i < blockDim.x; ++i) {
            part[0] += part[i];
        }
    }
    __syncthreads();
    float scale = rsqrtf(part[0] / (float)desc->hidden + desc->rms_eps);

    float *out = run->pooled + (unsigned long long)r * desc->hidden;
    float length = 0.0f;
    for (unsigned int d = threadIdx.x; d < desc->hidden; d += blockDim.x) {
        float value = scale * row[d] * weight[d];
        out[d] = value;
        length += value * value;
    }
    part[threadIdx.x] = length;
    __syncthreads();
    if (threadIdx.x == 0u) {
        for (unsigned int i = 1u; i < blockDim.x; ++i) {
            part[0] += part[i];
        }
    }
    __syncthreads();
    float unit = (part[0] > 0.0f) ? rsqrtf(part[0]) : 0.0f;
    for (unsigned int d = threadIdx.x; d < desc->hidden; d += blockDim.x) {
        out[d] *= unit;
    }
}
