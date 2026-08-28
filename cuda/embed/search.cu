/* Purpose: Keep the note store and give the nearest notes to a query vector.
 * Owns: The note store of the run.
 * Launch shape: One block for each query; the threads of the block hold the hidden width.
 * Lifetime: The whole run. */
#include "embed/embed.cuh"

__device__ aotx_embed_store aotx_embed_notes;

__device__ unsigned int aotx_embed_keep(const float *vector, unsigned int width,
                                        const char *text, unsigned int length,
                                        unsigned long long seq)
{
    if (vector == 0 || width == 0u || width > AOTX_EMBED_WIDTH) {
        atomicAdd(&aotx_embed_notes.refused, 1u);
        return AOTX_EMBED_NOTES;
    }
    /* The store keeps one width. A model of another width would give a cosine that has no
     * meaning against the notes that are in the store. */
    unsigned int held = atomicCAS(&aotx_embed_notes.width, 0u, width);
    if (held != 0u && held != width) {
        atomicAdd(&aotx_embed_notes.refused, 1u);
        return AOTX_EMBED_NOTES;
    }
    unsigned int at = atomicAdd(&aotx_embed_notes.count, 1u);
    if (at >= AOTX_EMBED_NOTES) {
        atomicSub(&aotx_embed_notes.count, 1u);
        atomicAdd(&aotx_embed_notes.refused, 1u);
        return AOTX_EMBED_NOTES;
    }
    for (unsigned int d = 0u; d < width; ++d) {
        aotx_embed_notes.vector[at][d] = vector[d];
    }
    unsigned int bytes = (length > AOTX_EMBED_TEXT) ? AOTX_EMBED_TEXT : length;
    for (unsigned int i = 0u; i < bytes; ++i) {
        aotx_embed_notes.text[at][i] = (unsigned char)text[i];
    }
    aotx_embed_notes.len[at] = bytes;
    aotx_embed_notes.seq[at] = seq;
    return at;
}

/* The threads of the block hold the width, so one dot product is one reduction over the
 * block. The result of the reduction goes to every thread through shared memory. */
__global__ void aotx_embed_search(aotx_embed_query set)
{
    __shared__ float part[AOTX_MODEL_ROW_THREADS];
    __shared__ float best[AOTX_EMBED_HITS];
    __shared__ unsigned int where[AOTX_EMBED_HITS];

    unsigned int q = blockIdx.x;
    if (set.vector == 0 || set.hit == 0 || set.count == 0) {
        return;
    }
    if (q >= *set.count) {
        return;
    }
    if (set.live != 0 && set.live[q] == 0u) {
        return;
    }
    unsigned int width = set.width;
    unsigned int count = aotx_embed_notes.count;
    if (count > AOTX_EMBED_NOTES) {
        count = AOTX_EMBED_NOTES;
    }
    const float *query = set.vector + (unsigned long long)q * width;

    if (threadIdx.x < AOTX_EMBED_HITS) {
        best[threadIdx.x] = -1.0f;
        where[threadIdx.x] = AOTX_EMBED_NOTES;
    }
    __syncthreads();

    for (unsigned int n = 0u; n < count; ++n) {
        const float *note = aotx_embed_notes.vector[n];
        float sum = 0.0f;
        for (unsigned int d = threadIdx.x; d < width; d += blockDim.x) {
            sum += query[d] * note[d];
        }
        part[threadIdx.x] = sum;
        __syncthreads();
        for (unsigned int step = blockDim.x / 2u; step > 0u; step >>= 1) {
            if (threadIdx.x < step) {
                part[threadIdx.x] += part[threadIdx.x + step];
            }
            __syncthreads();
        }
        /* One thread keeps the list in order, so two notes of the same cosine keep the
         * order they went in the store. The list is four long. */
        if (threadIdx.x == 0u) {
            float value = part[0];
            for (unsigned int h = 0u; h < AOTX_EMBED_HITS; ++h) {
                if (value > best[h]) {
                    for (unsigned int k = AOTX_EMBED_HITS - 1u; k > h; --k) {
                        best[k] = best[k - 1u];
                        where[k] = where[k - 1u];
                    }
                    best[h] = value;
                    where[h] = n;
                    break;
                }
            }
        }
        __syncthreads();
    }

    if (threadIdx.x < AOTX_EMBED_HITS) {
        set.hit[q * AOTX_EMBED_HITS + threadIdx.x] = where[threadIdx.x];
        if (set.score != 0) {
            set.score[q * AOTX_EMBED_HITS + threadIdx.x] = best[threadIdx.x];
        }
    }
}
