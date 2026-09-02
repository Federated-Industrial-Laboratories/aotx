/* Purpose: Score four-choice items from the last-row logits of their queries.
 * Owns: Nothing; the score program owns the logits, the letter list and the item scores.
 * Launch shape: One thread for each sequence of a pass; one block for the mean over items.
 * Lifetime: One score run. */
#include <cuda_runtime.h>

/* The letter with the largest logit at the last row of each sequence, and one when it is
 * the answer of the item. The logits hold one row for each sequence of the pass. The pass
 * starts at the item first; the largest letter and the score of each item go to their
 * place in the set. A tie keeps the earlier letter. */
__global__ void aotx_quality_score_letter(const float *logits, unsigned int vocab,
                                          const unsigned int *letter, unsigned int letters,
                                          const unsigned int *answer, unsigned int first,
                                          unsigned int seqs, unsigned int *largest, float *right)
{
    unsigned int seq = blockIdx.x * blockDim.x + threadIdx.x;
    if (seq >= seqs) return;
    const float *row = logits + (unsigned long long)seq * vocab;
    unsigned int best = 0u;
    float top = row[letter[0]];
    for (unsigned int l = 1u; l < letters; ++l) {
        float value = row[letter[l]];
        if (value > top) { top = value; best = l; }
    }
    largest[first + seq] = best;
    right[first + seq] = (best == answer[first + seq]) ? 1.0f : 0.0f;
}

/* The mean of the item scores over the set. One block; each thread sums a stride of the
 * items and the first thread adds the parts. The mean is a double, so a share of a small
 * set prints as the fraction it is. */
__global__ void aotx_quality_score_mean(const float *right, unsigned int items, double *mean)
{
    __shared__ double part[256];
    double sum = 0.0;
    for (unsigned int i = threadIdx.x; i < items; i += blockDim.x) sum += (double)right[i];
    part[threadIdx.x] = sum;
    __syncthreads();
    if (threadIdx.x == 0u) {
        for (unsigned int i = 1u; i < blockDim.x; ++i) part[0] += part[i];
        *mean = part[0] / (double)items;
    }
}
