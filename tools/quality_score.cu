/* Purpose: Score four-choice items and rubric answers from the last-row logits of their queries.
 * Owns: Nothing; the score program owns the logits, the token lists and the scores.
 * Launch shape: One thread for each sequence of a pass, or for each pair; one block for a tally.
 * Lifetime: One score run. */
#include <cuda_runtime.h>

/* The letter with the largest logit at the last row of each sequence, and one when it is
 * the answer of the item. The logits hold one row for each sequence of the pass. The pass
 * starts at the item first; the largest letter, its logit and the score of each item go
 * to their place in the set. A tie keeps the earlier letter. */
__global__ void aotx_quality_score_letter(const float *logits, unsigned int vocab,
                                          const unsigned int *letter, unsigned int letters,
                                          const unsigned int *answer, unsigned int first,
                                          unsigned int seqs, unsigned int *largest, float *top_of,
                                          float *right)
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
    top_of[first + seq] = top;
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

/* The log-odds of yes at the last row of each sequence of a pass: the logit of yes less
 * the logit of no, in nats. No sigmoid is applied, so two answers that are both near
 * certain still differ. The pass starts at the query first of its set, so the log-odds
 * of each query go to their place. */
__global__ void aotx_quality_score_answer(const float *logits, unsigned int vocab,
                                          unsigned int yes, unsigned int no,
                                          unsigned int first, unsigned int seqs, float *p)
{
    unsigned int seq = blockIdx.x * blockDim.x + threadIdx.x;
    if (seq >= seqs) return;
    const float *row = logits + (unsigned long long)seq * vocab;
    p[first + seq] = row[yes] - row[no];
}

/* The side scores and the result of each pair. The log-odds hold, for each pair, the
 * items of side a and then the items of side b. The side score is the mean over the
 * items, in nats. The result is 1 for a win of b, 0 for a loss and one half for a tie. A
 * tie is a gap inside the margin, in nats. The item marks hold the same figure for each
 * item. */
__global__ void aotx_quality_score_pairs(const float *p, unsigned int pairs,
                                         unsigned int items, float margin, float *side,
                                         float *result, float *mark)
{
    unsigned int pair = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair >= pairs) return;
    const float *a = p + (unsigned long long)pair * 2u * items;
    const float *b = a + items;
    double sum_a = 0.0, sum_b = 0.0;
    for (unsigned int i = 0u; i < items; ++i) {
        float gap = b[i] - a[i];
        sum_a += (double)a[i];
        sum_b += (double)b[i];
        mark[pair * items + i] = (gap > margin) ? 1.0f : ((gap < -margin) ? 0.0f : 0.5f);
    }
    float mean_a = (float)(sum_a / (double)items), mean_b = (float)(sum_b / (double)items);
    float gap = mean_b - mean_a;
    side[pair * 2u] = mean_a;
    side[pair * 2u + 1u] = mean_b;
    result[pair] = (gap > margin) ? 1.0f : ((gap < -margin) ? 0.0f : 0.5f);
}

/* The tallies over the pairs: the wins and the ties of b, the win rate w and its Wilson
 * interval at the normal quantile z. The win rate of each item follows. One block; each
 * thread sums a stride of the pairs and the first thread adds the parts. The figures go
 * out as wins, ties, w, low, high and then one rate for each item. */
__global__ void aotx_quality_score_tally(const float *result, const float *mark,
                                         unsigned int pairs, unsigned int items, float z,
                                         double *figures)
{
    __shared__ double part[256];
    __shared__ double wins_part[256];
    double wins = 0.0, ties = 0.0;
    for (unsigned int i = threadIdx.x; i < pairs; i += blockDim.x) {
        wins += (result[i] == 1.0f) ? 1.0 : 0.0;
        ties += (result[i] == 0.5f) ? 1.0 : 0.0;
    }
    wins_part[threadIdx.x] = wins;
    part[threadIdx.x] = ties;
    __syncthreads();
    if (threadIdx.x == 0u) {
        for (unsigned int i = 1u; i < blockDim.x; ++i) { wins_part[0] += wins_part[i]; part[0] += part[i]; }
        double n = (double)pairs, w = (wins_part[0] + 0.5 * part[0]) / n;
        double zz = (double)z * (double)z, denominator = 1.0 + zz / n;
        double center = (w + zz / (2.0 * n)) / denominator;
        double half = (double)z * sqrt(w * (1.0 - w) / n + zz / (4.0 * n * n)) / denominator;
        figures[0] = wins_part[0];
        figures[1] = part[0];
        figures[2] = w;
        figures[3] = center - half;
        figures[4] = center + half;
    }
    for (unsigned int item = 0u; item < items; ++item) {
        double sum = 0.0;
        __syncthreads();
        for (unsigned int i = threadIdx.x; i < pairs; i += blockDim.x) sum += (double)mark[i * items + item];
        part[threadIdx.x] = sum;
        __syncthreads();
        if (threadIdx.x == 0u) {
            for (unsigned int i = 1u; i < blockDim.x; ++i) part[0] += part[i];
            figures[5u + item] = part[0] / (double)pairs;
        }
    }
}

/* The blind order of each pair: 1 when side b prints first, from a hash of the pair
 * number under the seed. The order is a function of the seed alone, so a key and its
 * transcripts come from one run and agree. */
__global__ void aotx_quality_score_shuffle(unsigned int pairs, unsigned int seed,
                                           unsigned int *swap)
{
    unsigned int pair = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair >= pairs) return;
    unsigned int h = (pair + 1u) * 2654435761u ^ (seed * 40503u + 0x9e3779b9u);
    h ^= h >> 15; h *= 0x85ebca6bu; h ^= h >> 13; h *= 0xc2b2ae35u; h ^= h >> 16;
    swap[pair] = h & 1u;
}
