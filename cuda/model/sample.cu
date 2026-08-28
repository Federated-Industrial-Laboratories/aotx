/* Purpose: Take one token for each sequence from the logits of its last row.
 * Owns: Nothing; the buffer block holds the logits and the caller holds the token.
 * Launch shape: One block for each sequence; the threads hold the vocabulary.
 * Lifetime: One pass of the forward graph. */
#include "model/forward.cuh"
#include "rng/rng.cuh"

/* Buckets of the count over a band of the scaled logits. The buckets give a cut which keeps
 * at most AOTX_MODEL_PICK_MAX candidates, and one pass over the row makes them. */
#define AOTX_PICK_BUCKETS  256u

/* Passes of the bucket search. The first covers the whole span. The second covers the one
 * bucket that holds more candidates than the array takes. A row of near equal logits
 * therefore gives a full set and not the largest alone. */
#define AOTX_PICK_PASSES   2u

/* Sort the candidates by value, from the largest down. The count is a power of two, so the
 * sort is a bitonic exchange network over the threads of the block. */
__device__ __forceinline__ static void aotx_pick_sort(float *value, unsigned int *index,
                                                      unsigned int count)
{
    for (unsigned int span = 2u; span <= count; span <<= 1) {
        for (unsigned int step = span >> 1; step != 0u; step >>= 1) {
            unsigned int i = threadIdx.x;
            unsigned int other = i ^ step;
            if (other > i && i < count && other < count) {
                int down = ((i & span) == 0u);
                int swap = down ? (value[i] < value[other]) : (value[i] > value[other]);
                if (swap) {
                    float keep = value[i];
                    unsigned int mark = index[i];
                    value[i] = value[other];
                    index[i] = index[other];
                    value[other] = keep;
                    index[other] = mark;
                }
            }
            __syncthreads();
        }
    }
}

__global__ void aotx_model_pick(unsigned int role)
{
    __shared__ float part[AOTX_MODEL_ROW_THREADS];
    __shared__ unsigned int mark[AOTX_MODEL_ROW_THREADS];
    __shared__ unsigned int bucket[AOTX_PICK_BUCKETS];
    __shared__ float value[AOTX_MODEL_PICK_MAX];
    __shared__ unsigned int index[AOTX_MODEL_PICK_MAX];
    __shared__ float shared_top;
    __shared__ float shared_sum;
    __shared__ float shared_cut;
    __shared__ float shared_lo;
    __shared__ float shared_hi;
    __shared__ unsigned int shared_more;
    __shared__ unsigned int shared_count;
    __shared__ unsigned int shared_position;

    const aotx_model_desc *desc = &aotx_model[role];
    const aotx_model_run *run = &aotx_model_call[role];
    const aotx_model_work *work = &aotx_model_space[role];
    unsigned int r = blockIdx.x;
    if (r >= run->rows || run->token == 0) {
        return;
    }
    const float *row = work->head + (unsigned long long)r * desc->vocab;
    unsigned int vocab = desc->vocab;
    unsigned int agent = run->agent[r];
    if (agent >= AOTX_SLOTS) {
        agent = 0u;
    }

    /* A sequence may carry its own sample. A batch that gives no list of them takes the
     * four values of the call block for every sequence of the batch. */
    const aotx_model_how *how = run->how;
    unsigned int want_k = (how != 0) ? how[r].top_k : run->top_k;
    float want_p = (how != 0) ? how[r].top_p : run->top_p;
    float warmth = (how != 0) ? how[r].temperature : run->temperature;
    unsigned long long stream = (how != 0) ? how[r].seed : run->seed;

    /* The largest logit and its place. A temperature of zero gives that place at once. */
    float best = -INFINITY;
    unsigned int at = 0u;
    for (unsigned int i = threadIdx.x; i < vocab; i += blockDim.x) {
        if (row[i] > best) {
            best = row[i];
            at = i;
        }
    }
    part[threadIdx.x] = best;
    mark[threadIdx.x] = at;
    __syncthreads();
    if (threadIdx.x == 0u) {
        for (unsigned int i = 1u; i < blockDim.x; ++i) {
            if (part[i] > part[0]) {
                part[0] = part[i];
                mark[0] = mark[i];
            }
        }
        shared_top = part[0];
    }
    __syncthreads();
    float top = shared_top;

    /* Every draw moves the stream of the slot on by one, so the seed and the position name
     * the token again at a replay. */
    if (threadIdx.x == 0u) {
        shared_position = atomicAdd(&aotx_model_draw[agent], 1u);
        if (run->draw != 0) {
            run->draw[r] = shared_position;
        }
    }
    __syncthreads();
    unsigned int position = shared_position;
    if (warmth <= 0.0f) {
        if (threadIdx.x == 0u) {
            run->token[r] = (int)mark[0];
        }
        return;
    }

    /* The bucket search. The first pass also adds the whole row, which is the divisor of
     * the probabilities that the top of the list cuts by mass. */
    float lo = -AOTX_MODEL_PICK_SPAN;
    float hi = 0.0f;
    for (unsigned int pass = 0u; pass < AOTX_PICK_PASSES; ++pass) {
        for (unsigned int i = threadIdx.x; i < AOTX_PICK_BUCKETS; i += blockDim.x) {
            bucket[i] = 0u;
        }
        __syncthreads();
        float band = hi - lo;
        float total = 0.0f;
        for (unsigned int i = threadIdx.x; i < vocab; i += blockDim.x) {
            float scaled = row[i] - top;
            if (pass == 0u) {
                total += expf(scaled);
            }
            if (scaled >= lo && scaled <= hi) {
                unsigned int b = (unsigned int)((scaled - lo) * (float)AOTX_PICK_BUCKETS
                                                / band);
                if (b >= AOTX_PICK_BUCKETS) {
                    b = AOTX_PICK_BUCKETS - 1u;
                }
                atomicAdd(&bucket[b], 1u);
            }
        }
        part[threadIdx.x] = total;
        __syncthreads();
        if (threadIdx.x == 0u) {
            if (pass == 0u) {
                float sum = 0.0f;
                for (unsigned int i = 0u; i < blockDim.x; ++i) {
                    sum += part[i];
                }
                shared_sum = sum;
            }
            unsigned int held = 0u;
            shared_more = 0u;
            shared_cut = lo;
            for (unsigned int b = AOTX_PICK_BUCKETS; b != 0u; --b) {
                if (held + bucket[b - 1u] > AOTX_MODEL_PICK_MAX) {
                    shared_cut = lo + band * (float)b / (float)AOTX_PICK_BUCKETS;
                    if (held == 0u) {
                        /* The highest bucket that holds anything is over the bound on its
                         * own. The band of that bucket takes a second, finer search. */
                        shared_more = 1u;
                        shared_lo = lo + band * (float)(b - 1u) / (float)AOTX_PICK_BUCKETS;
                        shared_hi = shared_cut;
                    }
                    break;
                }
                held += bucket[b - 1u];
            }
        }
        __syncthreads();
        if (shared_more == 0u) {
            break;
        }
        lo = shared_lo;
        hi = shared_hi;
    }

    /* Collect the candidates above the cut. A row whose logits are equal to the width of
     * the finest band gives an arbitrary set of them, which is the same distribution. */
    float cut = shared_cut;
    if (threadIdx.x == 0u) {
        shared_count = 0u;
    }
    __syncthreads();
    for (unsigned int i = threadIdx.x; i < vocab; i += blockDim.x) {
        float scaled = row[i] - top;
        if (scaled >= cut) {
            unsigned int slot = atomicAdd(&shared_count, 1u);
            if (slot < AOTX_MODEL_PICK_MAX) {
                value[slot] = scaled;
                index[slot] = i;
            }
        }
    }
    __syncthreads();
    unsigned int count = shared_count;
    if (count > AOTX_MODEL_PICK_MAX) {
        count = AOTX_MODEL_PICK_MAX;
    }
    for (unsigned int i = threadIdx.x; i < AOTX_MODEL_PICK_MAX; i += blockDim.x) {
        if (i >= count) {
            value[i] = -INFINITY;
            index[i] = 0u;
        }
    }
    __syncthreads();
    aotx_pick_sort(value, index, AOTX_MODEL_PICK_MAX);

    if (threadIdx.x != 0u) {
        return;
    }

    /* The order of the reference sampler: the count cut, then the mass cut, and the
     * temperature last. The mass cut therefore reads the probabilities of the model and
     * not the probabilities the temperature makes. */
    unsigned int keep = (want_k == 0u || want_k > count) ? count : want_k;
    float limit = (want_p <= 0.0f || want_p > 1.0f) ? 1.0f : want_p;
    float mass = 0.0f;
    unsigned int taken = 0u;
    for (unsigned int i = 0u; i < keep; ++i) {
        mass += expf(value[i]) / shared_sum;
        taken = i + 1u;
        if (mass >= limit) {
            break;
        }
    }

    /* The temperature scales what is left. The largest value takes the exponent to zero,
     * so no term of the sum goes over one. */
    float heat = warmth;
    float total = 0.0f;
    for (unsigned int i = 0u; i < taken; ++i) {
        total += expf((value[i] - value[0]) / heat);
    }
    uint4 word = aotx_rng_lane(stream, agent, 0u, (unsigned long long)position);
    float pick = aotx_rng_unit(word.x) * total;
    float walk = 0.0f;
    unsigned int chosen = index[0];
    for (unsigned int i = 0u; i < taken; ++i) {
        walk += expf((value[i] - value[0]) / heat);
        chosen = index[i];
        if (walk >= pick) {
            break;
        }
    }
    run->token[r] = (int)chosen;
}
