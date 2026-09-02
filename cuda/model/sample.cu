/* Purpose: Hold sampler rows and take one token from each last row of logits.
 * Owns: The per-agent sampler table; the caller owns the selected token.
 * Launch shape: One block for each sequence; the threads hold the vocabulary.
 * Lifetime: One pass of the forward graph. */
#include "agent/agent.cuh"
#include "model/decode_state.cuh"
#include "model/conduct.cuh"
#include "model/sampler.cuh"
#include "rng/rng.cuh"
#include "settings/settings.cuh"
#ifdef AOTX_AFFECT
#include "affect/affect.cuh"
#endif

__device__ aotx_sampler_table aotx_sampler;

__device__ void aotx_sampler_reset(unsigned int agent)
{
    if (agent >= AOTX_SLOTS) {
        return;
    }
    aotx_model_how *row = &aotx_sampler.row[agent];
    row->temperature = aotx_setting_fraction(AOTX_SET_TEMPERATURE);
    row->top_k = aotx_setting_count(AOTX_SET_TOP_K);
    row->top_p = aotx_setting_fraction(AOTX_SET_TOP_P);
    row->min_p = aotx_setting_fraction(AOTX_SET_MIN_P);
    row->repeat_penalty = aotx_setting_fraction(AOTX_SET_REPEAT_PENALTY);
    row->repeat_window = aotx_setting_count(AOTX_SET_REPEAT_WINDOW);
    row->presence_penalty = aotx_setting_fraction(AOTX_SET_PRESENCE_PENALTY);
    row->frequency_penalty = aotx_setting_fraction(AOTX_SET_FREQUENCY_PENALTY);
    row->seed = (unsigned long long)aotx_setting_value(AOTX_SET_SAMPLE_SEED);
    row->think_limit = (int)aotx_setting_value(AOTX_SET_THINK_LIMIT);
    for (unsigned int i = 0u; i < AOTX_MODEL_STEERS; ++i) {
        row->steer[i] = AOTX_MODEL_CONDUCT_NONE;
        row->steer_strength[i] = 0.0f;
        aotx_sampler.name[agent].steer[i][0] = '\0';
    }
    row->voice = AOTX_MODEL_CONDUCT_NONE;
    row->affect = 0u;
    row->voice_scale = 1.0f;
    aotx_sampler.name[agent].voice[0] = '\0';
    aotx_sampler.changed[agent] = 0u;
}

static __device__ __forceinline__ int aotx_sampler_is(const char *key,
                                                       unsigned int length,
                                                       const char *name)
{
    unsigned int i = 0u;
    while (i < length && name[i] != '\0' && key[i] == name[i]) {
        i += 1u;
    }
    return (i == length && name[i] == '\0') ? 1 : 0;
}

/* Parse a signed number with at most four decimal places. */
static __device__ int aotx_sampler_number(const char *text, unsigned int length,
                                          long long *out)
{
    unsigned int at = 0u;
    int minus = 0;
    long long whole = 0;
    long long fraction = 0;
    long long place = 1000;
    unsigned int decimals = 0u;
    if (length != 0u && (text[0] == '-' || text[0] == '+')) {
        minus = (text[0] == '-') ? 1 : 0;
        at = 1u;
    }
    if (at >= length) {
        return 0;
    }
    while (at < length && text[at] >= '0' && text[at] <= '9') {
        if (whole > 2147483647ll) {
            return 0;
        }
        whole = whole * 10ll + (long long)(text[at] - '0');
        at += 1u;
    }
    if (at < length && text[at] == '.') {
        at += 1u;
        while (at < length && text[at] >= '0' && text[at] <= '9') {
            if (decimals >= 4u) {
                return 0;
            }
            fraction += (long long)(text[at] - '0') * place;
            place /= 10ll;
            decimals += 1u;
            at += 1u;
        }
        if (decimals == 0u) {
            return 0;
        }
    }
    if (at != length) {
        return 0;
    }
    *out = (minus != 0) ? -(whole * 10000ll + fraction)
                        : whole * 10000ll + fraction;
    return 1;
}

static __device__ unsigned int aotx_sampler_fixed(float *field, long long value,
                                                   long long least, long long most)
{
    if (value < least || value > most) {
        return AOTX_SAMPLER_RANGE;
    }
    *field = (float)value / 10000.0f;
    return AOTX_SAMPLER_TOOK;
}

static __device__ void aotx_sampler_name(char *out, const char *value, unsigned int length)
{
    unsigned int take = (length < AOTX_SAMPLER_NAME_BYTES - 1u)
                      ? length : AOTX_SAMPLER_NAME_BYTES - 1u;
    for (unsigned int i = 0u; i < take; ++i) {
        out[i] = value[i];
    }
    out[take] = '\0';
}

static __device__ unsigned int aotx_sampler_conduct(unsigned int agent, const char *key,
                                                    unsigned int key_len, const char *value,
                                                    unsigned int value_len)
{
    aotx_model_how *row = &aotx_sampler.row[agent];
    if (aotx_sampler_is(key, key_len, "decode.voice")) {
        if (aotx_sampler_is(value, value_len, "absent")) {
            row->voice = AOTX_MODEL_CONDUCT_NONE;
            aotx_sampler.name[agent].voice[0] = '\0';
            return AOTX_SAMPLER_TOOK;
        }
        unsigned int id = aotx_conduct_voice(value, value_len);
        if (id == AOTX_MODEL_CONDUCT_NONE) {
            return AOTX_SAMPLER_ITEM;
        }
        row->voice = id;
        aotx_sampler_name(aotx_sampler.name[agent].voice, value, value_len);
        return AOTX_SAMPLER_TOOK;
    }
    unsigned int slot = aotx_sampler_is(key, key_len, "decode.steer0") ? 0u
                      : (aotx_sampler_is(key, key_len, "decode.steer1") ? 1u
                                                                         : AOTX_MODEL_STEERS);
    if (slot >= AOTX_MODEL_STEERS) {
        return AOTX_SAMPLER_UNKNOWN;
    }
    if (aotx_sampler_is(value, value_len, "absent")) {
        row->steer[slot] = AOTX_MODEL_CONDUCT_NONE;
        row->steer_strength[slot] = 0.0f;
        aotx_sampler.name[agent].steer[slot][0] = '\0';
        return AOTX_SAMPLER_TOOK;
    }
    unsigned int split = 0u;
    while (split < value_len && value[split] != ':') {
        split += 1u;
    }
    long long strength = 0ll;
    if (split == 0u || split >= value_len
        || aotx_sampler_number(value + split + 1u, value_len - split - 1u, &strength) == 0) {
        return AOTX_SAMPLER_VALUE;
    }
    if (strength < -40000ll || strength > 40000ll) {
        return AOTX_SAMPLER_RANGE;
    }
    unsigned int id = aotx_conduct_vector(value, split);
    if (id == AOTX_MODEL_CONDUCT_NONE) {
        return AOTX_SAMPLER_ITEM;
    }
    row->steer[slot] = id;
    row->steer_strength[slot] = (float)strength / 10000.0f;
    aotx_sampler_name(aotx_sampler.name[agent].steer[slot], value, split);
    return AOTX_SAMPLER_TOOK;
}

__device__ unsigned int aotx_sampler_set(unsigned int agent, const char *key,
                                         unsigned int key_len, const char *value,
                                         unsigned int value_len)
{
    if (agent >= AOTX_SLOTS) {
        return AOTX_SAMPLER_RANGE;
    }
    if (aotx_sampler_is(key, key_len, "decode.voice")
        || aotx_sampler_is(key, key_len, "decode.steer0")
        || aotx_sampler_is(key, key_len, "decode.steer1")) {
        unsigned int conduct = aotx_sampler_conduct(agent, key, key_len, value, value_len);
        if (conduct == AOTX_SAMPLER_TOOK) {
            aotx_sampler.changed[agent] += 1u;
        } else {
            aotx_sampler.refused += 1u;
        }
        return conduct;
    }
    long long number = 0;
    if (aotx_sampler_is(key, key_len, "decode.think_limit")
        && aotx_sampler_is(value, value_len, "absent")) {
        aotx_sampler.row[agent].think_limit = -1;
        aotx_sampler.changed[agent] += 1u;
        return AOTX_SAMPLER_TOOK;
    }
    if (aotx_sampler_number(value, value_len, &number) == 0) {
        aotx_sampler.refused += 1u;
        return AOTX_SAMPLER_VALUE;
    }
    aotx_model_how *row = &aotx_sampler.row[agent];
    unsigned int result = AOTX_SAMPLER_UNKNOWN;
    if (aotx_sampler_is(key, key_len, "decode.temperature")) {
        result = aotx_sampler_fixed(&row->temperature, number, 0ll, 20000ll);
    } else if (aotx_sampler_is(key, key_len, "decode.top_p")) {
        result = aotx_sampler_fixed(&row->top_p, number, 1ll, 10000ll);
    } else if (aotx_sampler_is(key, key_len, "decode.min_p")) {
        result = aotx_sampler_fixed(&row->min_p, number, 0ll, 10000ll);
    } else if (aotx_sampler_is(key, key_len, "decode.repeat_penalty")) {
        result = aotx_sampler_fixed(&row->repeat_penalty, number, 1ll, 20000ll);
    } else if (aotx_sampler_is(key, key_len, "decode.presence_penalty")) {
        result = aotx_sampler_fixed(&row->presence_penalty, number, -20000ll, 20000ll);
    } else if (aotx_sampler_is(key, key_len, "decode.frequency_penalty")) {
        result = aotx_sampler_fixed(&row->frequency_penalty, number, -20000ll, 20000ll);
    } else if (aotx_sampler_is(key, key_len, "decode.top_k")) {
        long long count = number / 10000ll;
        if (number % 10000ll != 0ll || count < 0ll || count > 256ll) {
            result = AOTX_SAMPLER_RANGE;
        } else {
            row->top_k = (unsigned int)count;
            result = AOTX_SAMPLER_TOOK;
        }
    } else if (aotx_sampler_is(key, key_len, "decode.repeat_window")) {
        long long count = number / 10000ll;
        if (number % 10000ll != 0ll || count < 0ll || count > 8191ll) {
            result = AOTX_SAMPLER_RANGE;
        } else {
            row->repeat_window = (unsigned int)count;
            result = AOTX_SAMPLER_TOOK;
        }
    } else if (aotx_sampler_is(key, key_len, "decode.seed")) {
        long long count = number / 10000ll;
        if (number % 10000ll != 0ll || count < 0ll || count > 2147483647ll) {
            result = AOTX_SAMPLER_RANGE;
        } else {
            row->seed = (unsigned long long)count;
            result = AOTX_SAMPLER_TOOK;
        }
    } else if (aotx_sampler_is(key, key_len, "decode.think_limit")) {
        long long count = number / 10000ll;
        if (number % 10000ll != 0ll || count < 0ll || count > 8191ll) {
            result = AOTX_SAMPLER_RANGE;
        } else {
            row->think_limit = (int)count;
            result = AOTX_SAMPLER_TOOK;
        }
    }
    if (result == AOTX_SAMPLER_TOOK) {
        aotx_sampler.changed[agent] += 1u;
    } else {
        aotx_sampler.refused += 1u;
    }
    return result;
}

/* Buckets of the count over a band of the scaled logits. The buckets give a cut which keeps
 * at most AOTX_MODEL_PICK_MAX candidates, and one pass over the row makes them. */
#define AOTX_PICK_BUCKETS  256u

/* Passes of the bucket search. The first covers the whole span. The second covers the one
 * bucket that holds more candidates than the array takes. A row of near equal logits
 * therefore gives a full set and not the largest alone. */
#define AOTX_PICK_PASSES   2u

/* Give one adjusted logit. Neutral values return the model logit unchanged. */
static __device__ __forceinline__ float aotx_pick_value(const float *row,
                                                        unsigned int token,
                                                        unsigned int agent,
                                                        const aotx_model_how *how)
{
    float value = row[token];
    if (how == 0 || agent >= AOTX_SLOTS) {
        return value;
    }
    const aotx_seq *seq = &aotx_seqs.slot[agent];
    value += aotx_conduct_bias(how->voice, token);
    if (seq->sampled == 0u && how->think_limit == 0
        && token == AOTX_DECODE_THINK_OPEN) {
        return -INFINITY;
    }
    if (seq->thinking != 0u && how->think_limit >= 0
        && seq->think_tokens >= (unsigned int)how->think_limit) {
        return (token == AOTX_DECODE_THINK_CLOSE) ? value : -INFINITY;
    }
    unsigned int list = seq->prompt + seq->sampled;
    unsigned int frequency = 0u;
    if (how->presence_penalty != 0.0f || how->frequency_penalty != 0.0f) {
        for (unsigned int i = 0u; i < list; ++i) {
            frequency += ((unsigned int)aotx_seqs.tokens[agent][i] == token) ? 1u : 0u;
        }
        if (frequency != 0u) {
            value -= how->presence_penalty + how->frequency_penalty * (float)frequency;
        }
    }
    if (how->repeat_window != 0u && how->repeat_penalty != 1.0f) {
        unsigned int first = (list > how->repeat_window) ? list - how->repeat_window : 0u;
        for (unsigned int i = first; i < list; ++i) {
            if ((unsigned int)aotx_seqs.tokens[agent][i] == token) {
                value = (value > 0.0f) ? value / how->repeat_penalty
                                       : value * how->repeat_penalty;
                break;
            }
        }
    }
    return value;
}

/* Write the instrument of one emitted token. The affect sums of the agent take the same
 * two figures while the how row of the sequence carries the affect mark. */
static __device__ __forceinline__ void aotx_pick_stats(const aotx_model_run *run,
                                                       unsigned int agent,
                                                       const aotx_model_how *choice,
                                                       unsigned int token,
                                                       float logprob, float entropy)
{
    if (run->telemetry == 0u || agent >= AOTX_SLOTS) {
        return;
    }
#ifdef AOTX_AFFECT
    aotx_affect_pick(agent, choice, logprob, entropy);
    aotx_quality_pick(agent, choice, token);
#else
    (void)choice;
#endif
    const aotx_seq *seq = &aotx_seqs.slot[agent];
    aotx_token_stats_body body;
    body.agent = agent;
    body.turn = aotx_agents.agent[agent].turn;
    body.index = seq->sampled;
    body.flags = (seq->thinking != 0u || token == AOTX_DECODE_THINK_OPEN)
               ? AOTX_TOKEN_STATS_THINK : 0u;
    body.logprob = logprob;
    body.entropy = entropy;
    body.token = token;
    body.reserved = 0u;
    aotx_seam_write(AOTX_WRITER_AGENT_BASE + agent, AOTX_CLASS_B,
                    AOTX_REC_TOKEN_STATS, 0u, &body, (unsigned int)sizeof body);
}

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
    __shared__ float weighted[AOTX_MODEL_ROW_THREADS];
    __shared__ unsigned int mark[AOTX_MODEL_ROW_THREADS];
    __shared__ unsigned int bucket[AOTX_PICK_BUCKETS];
    __shared__ float value[AOTX_MODEL_PICK_MAX];
    __shared__ unsigned int index[AOTX_MODEL_PICK_MAX];
    __shared__ float shared_top;
    __shared__ float shared_sum;
    __shared__ float shared_weighted;
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
    const aotx_model_how *choice = (how != 0) ? &how[r] : 0;
    unsigned int want_k = (choice != 0) ? choice->top_k : run->top_k;
    float want_p = (choice != 0) ? choice->top_p : run->top_p;
    float warmth = (choice != 0) ? choice->temperature : run->temperature;
    unsigned long long stream = (choice != 0) ? choice->seed : run->seed;

    /* The largest logit and its place. A temperature of zero gives that place at once. */
    float best = -INFINITY;
    unsigned int at = 0u;
    for (unsigned int i = threadIdx.x; i < vocab; i += blockDim.x) {
        float one = aotx_pick_value(row, i, agent, choice);
        if (one > best) {
            best = one;
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

    /* The token figures use the complete adjusted distribution. The reduction order is
     * fixed by thread number, so equal inputs give equal figures. */
    if (run->telemetry != 0u) {
        float total = 0.0f;
        float moment = 0.0f;
        for (unsigned int i = threadIdx.x; i < vocab; i += blockDim.x) {
            float one = aotx_pick_value(row, i, agent, choice);
            float chance = expf(one - top);
            total += chance;
            if (chance > 0.0f) {
                moment += chance * one;
            }
        }
        part[threadIdx.x] = total;
        weighted[threadIdx.x] = moment;
        __syncthreads();
        if (threadIdx.x == 0u) {
            float sum = 0.0f;
            float mean = 0.0f;
            for (unsigned int i = 0u; i < blockDim.x; ++i) {
                sum += part[i];
                mean += weighted[i];
            }
            shared_sum = sum;
            shared_weighted = mean;
        }
    } else if (threadIdx.x == 0u) {
        shared_sum = 1.0f;
        shared_weighted = top;
    }
    __syncthreads();

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
            float logprob = -logf(shared_sum);
            float entropy = logf(shared_sum) + top - shared_weighted / shared_sum;
            aotx_pick_stats(run, agent, choice, mark[0], logprob, entropy);
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
        float row_total = 0.0f;
        for (unsigned int i = threadIdx.x; i < vocab; i += blockDim.x) {
            float scaled = aotx_pick_value(row, i, agent, choice) - top;
            if (pass == 0u) row_total += expf(scaled);
            if (scaled >= lo && scaled <= hi) {
                unsigned int b = (unsigned int)((scaled - lo) * (float)AOTX_PICK_BUCKETS
                                                / band);
                if (b >= AOTX_PICK_BUCKETS) {
                    b = AOTX_PICK_BUCKETS - 1u;
                }
                atomicAdd(&bucket[b], 1u);
            }
        }
        part[threadIdx.x] = row_total;
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
        float scaled = aotx_pick_value(row, i, agent, choice) - top;
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
    float min_p = (choice != 0) ? choice->min_p : 0.0f;
    if (min_p > 0.0f && min_p <= 1.0f) {
        unsigned int min_keep = 1u;
        for (unsigned int i = 1u; i < keep; ++i) {
            if (expf(value[i] - value[0]) < min_p) {
                break;
            }
            min_keep = i + 1u;
        }
        keep = min_keep;
    }
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
    float heat_total = 0.0f;
    for (unsigned int i = 0u; i < taken; ++i) {
        heat_total += expf((value[i] - value[0]) / heat);
    }
    uint4 word = aotx_rng_lane(stream, agent, 0u, (unsigned long long)position);
    float pick = aotx_rng_unit(word.x) * heat_total;
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
    float chosen_value = aotx_pick_value(row, chosen, agent, choice);
    float logprob = chosen_value - top - logf(shared_sum);
    float entropy = logf(shared_sum) + top - shared_weighted / shared_sum;
    aotx_pick_stats(run, agent, choice, chosen, logprob, entropy);
}
