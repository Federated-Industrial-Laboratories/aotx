/* Purpose: Check the sample kernel against the distribution it is asked for.
 * Owns: The logits rows, the draw counts and the counts of the cases.
 * Launch shape: The sample kernel alone, one block for each sequence.
 * Lifetime: The program. */
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "boot/check.h"
#include "agent/agent.cuh"
#include "model/decode_state.cuh"
#include "model/forward.cuh"
#include "model/conduct.cuh"
#include "seam/seam.cuh"

#define AOTX_PICK_ROLE     AOTX_MODEL_LANGUAGE
#define AOTX_PICK_VOCAB    2048u
#define AOTX_PICK_SEQS     AOTX_SLOTS
#define AOTX_PICK_THINK_OPEN  151667u
#define AOTX_PICK_THINK_CLOSE 151668u
/* The draws of a case are the same count on every profile. The shape of the chi square
 * therefore does not change with the slot count. The rounds take what the sequences
 * leave. */
#define AOTX_PICK_DRAWS    12800u
#define AOTX_PICK_ROUNDS   (AOTX_PICK_DRAWS / AOTX_PICK_SEQS)
#define AOTX_PICK_SEED     0xA0A1A2A3A4A5A6A7ull

/* Cells of the test hold at least this many draws, which is the count the normal shape of
 * the chi square asks for. */
#define AOTX_PICK_CELL     5.0

static unsigned int aotx_pick_cases = 0u;
static unsigned int aotx_pick_bad = 0u;

static void aotx_pick_note(const char *name, int good, const char *how, double value,
                           double bound)
{
    aotx_pick_cases += 1u;
    if (!good) {
        aotx_pick_bad += 1u;
    }
    printf("%-40s %-4s %s %.3f of %.3f\n", name, good ? "ok" : "BAD", how, value, bound);
}

/* The buffers the kernel reads and writes. No model and no pass take part: the logits are
 * written straight into the head buffer, so the distribution is exactly known. */
typedef struct aotx_pick_gear {
    float *head;
    unsigned int *agent;
    int *token;
    unsigned int *draw;
    aotx_model_how *how;
    int *out;
} aotx_pick_gear;

static void *aotx_pick_take(unsigned long long bytes)
{
    void *block = 0;
    aotx_check_runtime(cudaMalloc(&block, (size_t)bytes), "cudaMalloc");
    aotx_check_runtime(cudaMemset(block, 0, (size_t)bytes), "cudaMemset");
    return block;
}

static void aotx_pick_open(aotx_pick_gear *gear)
{
    unsigned int slots[AOTX_PICK_SEQS];
    memset(gear, 0, sizeof *gear);
    gear->head = (float *)aotx_pick_take((unsigned long long)AOTX_PICK_SEQS
                                         * AOTX_PICK_VOCAB * sizeof(float));
    gear->agent = (unsigned int *)aotx_pick_take(AOTX_PICK_SEQS * sizeof(unsigned int));
    gear->token = (int *)aotx_pick_take(AOTX_PICK_SEQS * sizeof(int));
    gear->draw = (unsigned int *)aotx_pick_take(AOTX_PICK_SEQS * sizeof(unsigned int));
    gear->how = (aotx_model_how *)aotx_pick_take(AOTX_PICK_SEQS * sizeof(aotx_model_how));
    gear->out = (int *)malloc((size_t)AOTX_PICK_DRAWS * sizeof(int));
    for (unsigned int i = 0u; i < AOTX_PICK_SEQS; ++i) {
        slots[i] = i;
    }
    aotx_check_runtime(cudaMemcpy(gear->agent, slots, sizeof slots, cudaMemcpyHostToDevice),
                       "cudaMemcpy");

    /* The descriptor needs one field for this kernel, and the buffer block one. */
    aotx_model_desc desc;
    aotx_model_work work;
    memset(&desc, 0, sizeof desc);
    memset(&work, 0, sizeof work);
    desc.role = AOTX_PICK_ROLE;
    desc.layers = 1u;
    desc.vocab = AOTX_PICK_VOCAB;
    work.head = gear->head;
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_model, &desc, sizeof desc,
                                          (size_t)AOTX_PICK_ROLE * sizeof desc),
                       "cudaMemcpyToSymbol");
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_model_space, &work, sizeof work,
                                          (size_t)AOTX_PICK_ROLE * sizeof work),
                       "cudaMemcpyToSymbol");
}

/* Put one row of logits in every sequence of the head buffer. */
static void aotx_pick_row(aotx_pick_gear *gear, const float *row)
{
    for (unsigned int r = 0u; r < AOTX_PICK_SEQS; ++r) {
        aotx_check_runtime(cudaMemcpy(gear->head + (size_t)r * AOTX_PICK_VOCAB, row,
                                      AOTX_PICK_VOCAB * sizeof(float),
                                      cudaMemcpyHostToDevice), "cudaMemcpy");
    }
}

/* Take the draws. Each pass gives one draw for each slot. The stream of a slot moves on by
 * one at each draw, so a later pass is not the same draw again. */
static void aotx_pick_run_count(aotx_pick_gear *gear, const aotx_model_how *how,
                                unsigned int passes, unsigned int count,
                                unsigned int telemetry, unsigned int role = AOTX_PICK_ROLE)
{
    aotx_model_run set;
    aotx_model_how rows[AOTX_PICK_SEQS];
    memset(&set, 0, sizeof set);
    for (unsigned int i = 0u; i < count; ++i) {
        rows[i] = *how;
    }
    aotx_check_runtime(cudaMemcpy(gear->how, rows, count * sizeof *rows,
                                  cudaMemcpyHostToDevice), "cudaMemcpy");
    set.agent = gear->agent;
    set.token = gear->token;
    set.draw = gear->draw;
    set.how = gear->how;
    set.seqs = count;
    set.rows = count;
    set.seed = how->seed;
    set.top_k = how->top_k;
    set.top_p = how->top_p;
    set.temperature = how->temperature;
    set.telemetry = telemetry;
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_model_call, &set, sizeof set,
                                          (size_t)role * sizeof set),
                       "cudaMemcpyToSymbol");
    aotx_model_restream();
    for (unsigned int p = 0u; p < passes; ++p) {
        aotx_model_pick<<<AOTX_PICK_SEQS, AOTX_MODEL_ROW_THREADS>>>(role);
        aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        aotx_check_runtime(cudaMemcpy(gear->out + (size_t)p * count, gear->token,
                                      count * sizeof(int), cudaMemcpyDeviceToHost),
                           "cudaMemcpy");
    }
}

static void aotx_pick_run(aotx_pick_gear *gear, const aotx_model_how *how,
                          unsigned int passes)
{
    aotx_pick_run_count(gear, how, passes, AOTX_PICK_SEQS, 0u);
}

/* One candidate of the reference: its identity and its value below the largest logit. */
typedef struct aotx_pick_one {
    float value;
    unsigned int id;
} aotx_pick_one;

static int aotx_pick_order(const void *left, const void *right)
{
    const aotx_pick_one *a = (const aotx_pick_one *)left;
    const aotx_pick_one *b = (const aotx_pick_one *)right;
    if (a->value > b->value) {
        return -1;
    }
    if (a->value < b->value) {
        return 1;
    }
    return (a->id < b->id) ? -1 : 1;
}

/* The set the sampler is allowed to draw from, and the chance of each of its members. The
 * count cut comes first, then the mass cut, and the temperature last. */
static unsigned int aotx_pick_allow(const float *row, const aotx_model_how *how,
                                    unsigned int *id, double *chance)
{
    aotx_pick_one *all = (aotx_pick_one *)malloc(AOTX_PICK_VOCAB * sizeof *all);
    double top = row[0];
    for (unsigned int i = 1u; i < AOTX_PICK_VOCAB; ++i) {
        top = (row[i] > top) ? row[i] : top;
    }
    for (unsigned int i = 0u; i < AOTX_PICK_VOCAB; ++i) {
        all[i].value = (float)((double)row[i] - top);
        all[i].id = i;
    }
    qsort(all, AOTX_PICK_VOCAB, sizeof *all, aotx_pick_order);
    double sum = 0.0;
    for (unsigned int i = 0u; i < AOTX_PICK_VOCAB; ++i) {
        sum += exp((double)all[i].value);
    }
    unsigned int keep = (how->top_k == 0u || how->top_k > AOTX_PICK_VOCAB)
        ? AOTX_PICK_VOCAB : how->top_k;
    if (how->min_p > 0.0f && how->min_p <= 1.0f) {
        unsigned int min_keep = 1u;
        for (unsigned int i = 1u; i < keep; ++i) {
            if (exp((double)all[i].value - all[0].value) < (double)how->min_p) {
                break;
            }
            min_keep = i + 1u;
        }
        keep = min_keep;
    }
    double limit = (how->top_p <= 0.0f || how->top_p > 1.0f) ? 1.0 : (double)how->top_p;
    double mass = 0.0;
    unsigned int taken = 0u;
    for (unsigned int i = 0u; i < keep; ++i) {
        mass += exp((double)all[i].value) / sum;
        taken = i + 1u;
        if (mass >= limit) {
            break;
        }
    }
    double total = 0.0;
    for (unsigned int i = 0u; i < taken; ++i) {
        total += exp(((double)all[i].value - all[0].value) / how->temperature);
    }
    for (unsigned int i = 0u; i < taken; ++i) {
        id[i] = all[i].id;
        chance[i] = exp(((double)all[i].value - all[0].value) / how->temperature) / total;
    }
    free(all);
    return taken;
}

/* The chi square of the counts against the chances, over the cells that hold enough draws.
 * The bound is the degrees of freedom plus four times its own spread, which a correct
 * sampler passes and a wrong distribution does not. */
static double aotx_pick_chi(const unsigned int *count, const double *chance,
                            unsigned int cells, unsigned int draws, double *bound)
{
    double chi = 0.0;
    double rest_seen = 0.0;
    double rest_want = 0.0;
    unsigned int used = 0u;
    for (unsigned int i = 0u; i < cells; ++i) {
        double want = chance[i] * (double)draws;
        if (want >= AOTX_PICK_CELL) {
            double gap = (double)count[i] - want;
            chi += gap * gap / want;
            used += 1u;
        } else {
            rest_seen += (double)count[i];
            rest_want += want;
        }
    }
    if (rest_want >= AOTX_PICK_CELL) {
        double gap = rest_seen - rest_want;
        chi += gap * gap / rest_want;
        used += 1u;
    }
    double freedom = (used > 1u) ? (double)(used - 1u) : 1.0;
    *bound = freedom + 4.0 * sqrt(2.0 * freedom) + 10.0;
    return chi;
}

/* Count the draws that fell on each member of the allowed set, and the draws that fell
 * outside it. The identity of every draw must be a member. */
static unsigned int aotx_pick_tally(const int *out, unsigned int draws,
                                    const unsigned int *id, unsigned int members,
                                    unsigned int *count, unsigned int *distinct)
{
    unsigned int *place = (unsigned int *)malloc(AOTX_PICK_VOCAB * sizeof *place);
    unsigned int outside = 0u;
    for (unsigned int i = 0u; i < AOTX_PICK_VOCAB; ++i) {
        place[i] = ~0u;
    }
    for (unsigned int i = 0u; i < members; ++i) {
        place[id[i]] = i;
        count[i] = 0u;
    }
    unsigned int *seen = (unsigned int *)calloc(AOTX_PICK_VOCAB, sizeof *seen);
    for (unsigned int i = 0u; i < draws; ++i) {
        unsigned int got = (unsigned int)out[i];
        if (got >= AOTX_PICK_VOCAB || place[got] == ~0u) {
            outside += 1u;
            continue;
        }
        count[place[got]] += 1u;
        seen[got] = 1u;
    }
    unsigned int held = 0u;
    for (unsigned int i = 0u; i < AOTX_PICK_VOCAB; ++i) {
        held += seen[i];
    }
    *distinct = held;
    free(place);
    free(seen);
    return outside;
}

/* One arm: draw, check that every draw is a member, and check the shape of the counts. */
static void aotx_pick_arm(aotx_pick_gear *gear, const float *row, const aotx_model_how *how,
                          const char *label)
{
    unsigned int *id = (unsigned int *)malloc(AOTX_PICK_VOCAB * sizeof *id);
    double *chance = (double *)malloc(AOTX_PICK_VOCAB * sizeof *chance);
    unsigned int *count = (unsigned int *)malloc(AOTX_PICK_VOCAB * sizeof *count);
    unsigned int distinct = 0u;
    unsigned int members = aotx_pick_allow(row, how, id, chance);
    char name[80];

    aotx_pick_run(gear, how, AOTX_PICK_ROUNDS);
    unsigned int outside = aotx_pick_tally(gear->out, AOTX_PICK_DRAWS, id, members, count,
                                           &distinct);
    snprintf(name, sizeof name, "%s draws inside the set", label);
    aotx_pick_note(name, outside == 0u, "outside", (double)outside, 0.0);
    if (members > 1u) {
        double bound = 0.0;
        double chi = aotx_pick_chi(count, chance, members, AOTX_PICK_DRAWS, &bound);
        snprintf(name, sizeof name, "%s shape of the counts", label);
        aotx_pick_note(name, chi <= bound, "chi", chi, bound);
    }
    free(id);
    free(chance);
    free(count);
}

/* A row of logits that falls away from the largest, with no two values the same. */
static void aotx_pick_slope(float *row, unsigned long long seed)
{
    unsigned long long state = seed;
    for (unsigned int i = 0u; i < AOTX_PICK_VOCAB; ++i) {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        unsigned int word = (unsigned int)((state * 2685821657736338717ull) >> 40);
        row[i] = 8.0f - (float)i * 0.01f + (float)(word & 1023u) * 0.001f;
    }
}

/* A row where a thousand logits sit inside one bucket of the first search, each distinct
 * from the next by one part in ten thousand. The search must still give a full set of
 * candidates and not the largest alone. */
static void aotx_pick_flat(float *row)
{
    for (unsigned int i = 0u; i < AOTX_PICK_VOCAB; ++i) {
        row[i] = (i < 1000u) ? (12.0f - (float)i * 1.0e-4f) : -20.0f;
    }
}

/* Give the sample kernel the sequence state used by penalties and thinking limits. */
/* Every second agent takes the prompt token one above, so a wrong agent index gives a
 * different pick. A case that names no step gives every agent the same token. */
__global__ void aotx_pick_fixture(unsigned int count, unsigned int prompt_token,
                                  unsigned int sampled, unsigned int thinking,
                                  unsigned int think_tokens, unsigned int prompt_step)
{
    unsigned int agent = blockIdx.x * blockDim.x + threadIdx.x;
    if (agent >= count) {
        return;
    }
    prompt_token += (agent & 1u) * prompt_step;
    aotx_seqs.slot[agent] = {};
    aotx_seqs.slot[agent].state = AOTX_SEQ_STATE_DECODE;
    aotx_seqs.slot[agent].role = AOTX_MODEL_LANGUAGE_Q4;
    aotx_seqs.slot[agent].prompt = 1u;
    aotx_seqs.slot[agent].sampled = sampled;
    aotx_seqs.slot[agent].thinking = thinking;
    aotx_seqs.slot[agent].think_tokens = think_tokens;
    aotx_seqs.tokens[agent][0] = (int)prompt_token;
    aotx_agents.agent[agent].turn = agent + 3u;
}

static void aotx_pick_one_pass(aotx_pick_gear *gear, const float *row,
                               const aotx_model_how *how, unsigned int count,
                               unsigned int telemetry)
{
    aotx_pick_row(gear, row);
    aotx_pick_run_count(gear, how, 1u, count, telemetry);
}

static void aotx_pick_penalties(aotx_pick_gear *gear, float *row)
{
    aotx_model_how how = {};
    static const unsigned int counts[2] = { 1u, AOTX_PICK_SEQS };
    for (unsigned int i = 0u; i < AOTX_PICK_VOCAB; ++i) row[i] = -100.0f;
    row[10] = 10.0f;
    row[11] = 9.0f;
    how.top_p = 1.0f;
    how.repeat_penalty = 2.0f;
    how.repeat_window = 1u;
    how.think_limit = -1;
    for (unsigned int c = 0u; c < 2u; ++c) {
        unsigned int count = counts[c];
        how.repeat_penalty = 2.0f;
        how.repeat_window = 1u;
        how.presence_penalty = 0.0f;
        how.frequency_penalty = 0.0f;
        /* An even agent holds token 10 in its prompt, so the penalty moves its argmax
         * to 11. An odd agent holds 11, so its argmax stays at 10. */
        aotx_pick_fixture<<<1, count>>>(count, 10u, 0u, 0u, 0u, 1u);
        aotx_pick_one_pass(gear, row, &how, count, 0u);
        unsigned int same = 0u;
        for (unsigned int i = 0u; i < count; ++i) same += (gear->out[i] == ((i & 1u) ? 10 : 11)) ? 1u : 0u;
        aotx_pick_note("repeat penalty changes the argmax", same == count, "rows",
                       (double)same, (double)count);

        how.repeat_penalty = 1.0f;
        how.repeat_window = 0u;
        how.presence_penalty = 2.0f;
        aotx_pick_one_pass(gear, row, &how, count, 0u);
        same = 0u;
        for (unsigned int i = 0u; i < count; ++i) same += (gear->out[i] == ((i & 1u) ? 10 : 11)) ? 1u : 0u;
        aotx_pick_note("presence penalty changes the argmax", same == count, "rows",
                       (double)same, (double)count);

        how.presence_penalty = 0.0f;
        how.frequency_penalty = 2.0f;
        aotx_pick_one_pass(gear, row, &how, count, 0u);
        same = 0u;
        for (unsigned int i = 0u; i < count; ++i) same += (gear->out[i] == ((i & 1u) ? 10 : 11)) ? 1u : 0u;
        aotx_pick_note("frequency penalty changes the argmax", same == count, "rows",
                       (double)same, (double)count);
        how.frequency_penalty = 0.0f;
    }
}

static int aotx_pick_stats_same(const aotx_token_stats_body *got,
                                const aotx_token_stats_body *want)
{
    return got->agent == want->agent && got->turn == want->turn
        && got->index == want->index && got->flags == want->flags
        && got->token == want->token && got->reserved == 0u
        && fabs((double)got->logprob - (double)want->logprob) < 1.0e-5
        && fabs((double)got->entropy - (double)want->entropy) < 1.0e-5;
}

/* The sampler uses the launch role, not the unrelated role left in the sequence slot. */
static void aotx_pick_wrap(unsigned int present, unsigned int opening)
{
    aotx_wrap wrap = {};
    if (present) {
        memcpy(wrap.bytes, "<think></think>", 15u);
        wrap.length[AOTX_WRAP_THINK_OPEN] = 7u;
        wrap.offset[AOTX_WRAP_THINK_CLOSE] = 7u;
        wrap.length[AOTX_WRAP_THINK_CLOSE] = 8u;
        wrap.think_open_id = opening;
        wrap.think_close_id = AOTX_PICK_THINK_CLOSE;
    }
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_model_wrap, &wrap, sizeof wrap,
                        (size_t)AOTX_PICK_ROLE * sizeof wrap), "cudaMemcpyToSymbol");
}

static void aotx_pick_telemetry(aotx_pick_gear *gear, float *row)
{
    static const unsigned int counts[2] = { 1u, AOTX_PICK_SEQS };
    aotx_model_how how = {};
    how.top_p = 1.0f;
    how.repeat_penalty = 1.0f;
    how.think_limit = -1;
    for (unsigned int i = 0u; i < AOTX_PICK_VOCAB; ++i) row[i] = -100.0f;
    row[0] = logf(3.0f);
    row[1] = 0.0f;
    for (unsigned int c = 0u; c < 6u; ++c) {
        unsigned int count = counts[c % 2u], kind = c / 2u;
        aotx_pick_wrap(kind != 1u, kind == 2u ? 0u : AOTX_PICK_THINK_OPEN);
        unsigned char *device = NULL;
        unsigned char *records = (unsigned char *)calloc(count, AOTX_SLOT_BYTES);
        aotx_seam_state seam = {};
        unsigned long long tick = 41ull;
        aotx_check_runtime(cudaMalloc(&device, (size_t)count * AOTX_SLOT_BYTES), "cudaMalloc");
        aotx_check_runtime(cudaMemset(device, 0, (size_t)count * AOTX_SLOT_BYTES), "cudaMemset");
        seam.dev.base = device;
        seam.dev.slot_count = count;
        seam.dev.mask = count - 1u;
        seam.boot_id = 0x00a0250000000001ull + count;
        aotx_check_runtime(cudaMemcpyToSymbol(aotx_seam, &seam, sizeof seam),
                           "cudaMemcpyToSymbol");
        aotx_check_runtime(cudaMemcpyToSymbol(aotx_time_tick, &tick, sizeof tick),
                           "cudaMemcpyToSymbol");
        aotx_pick_fixture<<<1, count>>>(count, 7u, 5u, kind == 0u, 2u, 0u);
        aotx_pick_one_pass(gear, row, &how, count, 1u);
        aotx_check_runtime(cudaMemcpy(records, device, (size_t)count * AOTX_SLOT_BYTES,
                                      cudaMemcpyDeviceToHost), "cudaMemcpy");
        unsigned int exact = 0u;
        unsigned int mutation = 0u;
        for (unsigned int i = 0u; i < count; ++i) {
            const aotx_record_header *header = (const aotx_record_header *)
                                               (records + (size_t)i * AOTX_SLOT_BYTES);
            const aotx_token_stats_body *body = (const aotx_token_stats_body *)
                                                ((const unsigned char *)header
                                                 + AOTX_HEADER_BYTES);
            aotx_token_stats_body want = {};
            want.agent = body->agent;
            want.turn = body->agent + 3u;
            want.index = 5u;
            want.flags = kind == 1u ? 0u : AOTX_TOKEN_STATS_THINK;
            want.token = (uint32_t)gear->out[i];
            want.logprob = logf(0.75f);
            want.entropy = -0.75f * logf(0.75f) - 0.25f * logf(0.25f);
            exact += (header->seq == i + 1u && header->tick == tick
                      && header->cls == AOTX_CLASS_B && header->type == AOTX_REC_TOKEN_STATS
                      && header->body_len == sizeof *body && body->agent < count
                      && aotx_pick_stats_same(body, &want)) ? 1u : 0u;
            want.index += 1u;
            want.token += 1u;
            mutation += (aotx_pick_stats_same(body, &want) == 0) ? 1u : 0u;
        }
        aotx_pick_note("token records carry exact figures", exact == count, "records",
                       (double)exact, (double)count);
        aotx_pick_note("token record mutation is refused", mutation == count, "records",
                       (double)mutation, (double)count);
        cudaFree(device);
        free(records);
    }
    aotx_seam_state clear = {};
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_seam, &clear, sizeof clear),
                       "cudaMemcpyToSymbol");
}

__global__ void aotx_pick_voice_one(unsigned int *bits)
{
    float bias = aotx_conduct_bias(0u, 19u);
    bits[0] = __float_as_uint(bias);
    bits[1] = __float_as_uint(1.0f * bias);
}

/* A positive profile bias must move the frequency of its token. The token records above
 * carry the selected identity, so this same count is available for each reply. */
static void aotx_pick_voice(aotx_pick_gear *gear, float *row)
{
    static const unsigned int counts[2] = { 1u, AOTX_PICK_SEQS };
    unsigned int token = 19u;
    float bias = 4.0f;
    aotx_model_how how = {};
    for (unsigned int i = 0u; i < AOTX_PICK_VOCAB; ++i) row[i] = -100.0f;
    row[19] = 0.0f;
    row[20] = 0.0f;
    how.top_p = 1.0f;
    how.temperature = 1.0f;
    how.repeat_penalty = 1.0f;
    how.think_limit = -1;
    how.voice_scale = 1.0f;
    how.voice = AOTX_MODEL_CONDUCT_NONE;
    if (aotx_conduct_register_voice("plain", &token, &bias, 1u) != 0) {
        aotx_pick_note("voice profile registers", 0, "state", 1.0, 0.0);
        return;
    }
    unsigned int bits[2];
    aotx_pick_voice_one<<<1, 1>>>(gear->draw);
    aotx_check_runtime(cudaMemcpy(bits, gear->draw, sizeof bits, cudaMemcpyDeviceToHost),
                       "cudaMemcpy");
    aotx_pick_note("voice scale one keeps the bias bits", bits[0] == bits[1], "bits",
                   bits[1], bits[0]);
    for (unsigned int c = 0u; c < 2u; ++c) {
        unsigned int count = counts[c];
        unsigned int passes = 4096u / count;
        aotx_pick_fixture<<<1, count>>>(count, 7u, 0u, 0u, 0u, 0u);
        aotx_pick_row(gear, row);
        aotx_pick_run_count(gear, &how, passes, count, 0u);
        unsigned int plain = 0u;
        for (unsigned int p = 0u; p < passes; ++p)
            for (unsigned int i = 0u; i < count; ++i)
                plain += (gear->out[(size_t)p * count + i] == 19) ? 1u : 0u;
        how.voice = 0u;
        aotx_pick_run_count(gear, &how, passes, count, 0u);
        unsigned int voiced = 0u;
        for (unsigned int p = 0u; p < passes; ++p)
            for (unsigned int i = 0u; i < count; ++i)
                voiced += (gear->out[(size_t)p * count + i] == 19) ? 1u : 0u;
        aotx_pick_note("voice bias shifts token frequency",
                       plain > 1600u && plain < 2500u && voiced > 3900u,
                       "voiced", (double)voiced, 3900.0);
        aotx_model_desc owner, specialist = {}, saved;
        aotx_model_work work = {};
        aotx_check_runtime(cudaMemcpyFromSymbol(&owner, aotx_model, sizeof owner,
            AOTX_PICK_ROLE * sizeof owner), "cudaMemcpyFromSymbol");
        aotx_check_runtime(cudaMemcpyFromSymbol(&saved, aotx_model, sizeof saved,
            AOTX_MODEL_LANGUAGE_AUDIO * sizeof saved), "cudaMemcpyFromSymbol");
        specialist = owner; specialist.role = AOTX_MODEL_LANGUAGE_AUDIO; work.head = gear->head;
        aotx_check_runtime(cudaMemcpyToSymbol(aotx_model, &specialist, sizeof specialist,
            AOTX_MODEL_LANGUAGE_AUDIO * sizeof specialist), "cudaMemcpyToSymbol");
        aotx_check_runtime(cudaMemcpyToSymbol(aotx_model_space, &work, sizeof work,
            AOTX_MODEL_LANGUAGE_AUDIO * sizeof work), "cudaMemcpyToSymbol");
        for (unsigned mode = 0u; mode < 2u; ++mode) {
            if (mode) {
                aotx_model_desc absent = {};
                aotx_check_runtime(cudaMemcpyToSymbol(aotx_model, &absent, sizeof absent,
                    AOTX_PICK_ROLE * sizeof absent), "cudaMemcpyToSymbol");
            }
            aotx_pick_run_count(gear, &how, passes, count, 0u, AOTX_MODEL_LANGUAGE_AUDIO);
            unsigned biased = 0u;
            for (unsigned i = 0u; i < passes * count; ++i) biased += gear->out[i] == 19;
            aotx_pick_note(mode ? "audio default applies its voice bias" : "audio specialist excludes default voice bias",
                mode ? biased > 3900u : biased > 1600u && biased < 2500u,
                "draws", biased, mode ? 3900.0 : 2048.0);
        }
        aotx_check_runtime(cudaMemcpyToSymbol(aotx_model, &owner, sizeof owner,
            AOTX_PICK_ROLE * sizeof owner), "cudaMemcpyToSymbol");
        aotx_check_runtime(cudaMemcpyToSymbol(aotx_model, &saved, sizeof saved,
            AOTX_MODEL_LANGUAGE_AUDIO * sizeof saved), "cudaMemcpyToSymbol");
        how.voice = AOTX_MODEL_CONDUCT_NONE;
    }
}

static void aotx_pick_catalog_refusal(void)
{
    char dir[] = "/tmp/aotx-conduct-XXXXXX";
    char path[128];
    int good = mkdtemp(dir) != 0;
    snprintf(path, sizeof path, "%s/steer.jsonl", dir);
    FILE *out = good ? fopen(path, "w") : 0;
    if (out != 0) {
        fputs("{\"name\":\"missing\",\"file\":\"missing.aotxvec\"}\n", out);
        good = fclose(out) == 0 && aotx_conduct_load_store(dir) != 0;
    } else good = 0;
    aotx_pick_note("vector without potency is refused", good, "state", good, 1.0);
    unlink(path);
    rmdir(dir);
}

static void aotx_pick_thinking_case(unsigned int count, unsigned int sampled,
                                    unsigned int thinking, unsigned int tokens,
                                    int limit, unsigned int largest, unsigned int want, unsigned int spans,
                                    const char *label)
{
    const unsigned int vocab = AOTX_PICK_THINK_OPEN + 2u;
    size_t cells = (size_t)count * vocab;
    float *host = (float *)malloc(cells * sizeof *host);
    float *head = (float *)aotx_pick_take((unsigned long long)cells * sizeof(float));
    unsigned int *agent = (unsigned int *)aotx_pick_take(count * sizeof *agent);
    int *token = (int *)aotx_pick_take(count * sizeof *token);
    unsigned int *draw = (unsigned int *)aotx_pick_take(count * sizeof *draw);
    aotx_model_how *device_how = (aotx_model_how *)aotx_pick_take(count * sizeof *device_how);
    unsigned int slots[AOTX_PICK_SEQS];
    aotx_model_how rows[AOTX_PICK_SEQS];
    aotx_model_how how = {};
    aotx_model_desc desc = {};
    aotx_model_work work = {};
    aotx_model_run run = {};
    for (unsigned int i = 0u; i < vocab; ++i) host[i] = -100.0f;
    host[largest] = 10.0f;
    host[want] = (largest == want) ? 10.0f : 9.0f;
    for (unsigned int i = 1u; i < count; ++i) {
        memcpy(host + (size_t)i * vocab, host, vocab * sizeof *host);
    }
    how.top_p = 1.0f;
    how.repeat_penalty = 1.0f;
    how.think_limit = limit;
    /* An odd agent takes no limit, so its pick is the largest logit; a wrong row index
     * gives the even agents that pick too. */
    for (unsigned int i = 0u; i < count; ++i) {
        slots[i] = i;
        rows[i] = how;
        rows[i].think_limit = (i & 1u) ? -1 : limit;
    }
    desc.role = AOTX_PICK_ROLE;
    desc.vocab = vocab;
    aotx_pick_wrap(spans, AOTX_PICK_THINK_OPEN);
    work.head = head;
    run.agent = agent;
    run.token = token;
    run.draw = draw;
    run.how = device_how;
    run.seqs = count;
    run.rows = count;
    aotx_check_runtime(cudaMemcpy(head, host, cells * sizeof *host, cudaMemcpyHostToDevice),
                       "cudaMemcpy");
    aotx_check_runtime(cudaMemcpy(agent, slots, count * sizeof *slots, cudaMemcpyHostToDevice),
                       "cudaMemcpy");
    aotx_check_runtime(cudaMemcpy(device_how, rows, count * sizeof *rows,
                                  cudaMemcpyHostToDevice), "cudaMemcpy");
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_model, &desc, sizeof desc,
                                          (size_t)AOTX_PICK_ROLE * sizeof desc),
                       "cudaMemcpyToSymbol");
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_model_space, &work, sizeof work,
                                          (size_t)AOTX_PICK_ROLE * sizeof work),
                       "cudaMemcpyToSymbol");
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_model_call, &run, sizeof run,
                                          (size_t)AOTX_PICK_ROLE * sizeof run),
                       "cudaMemcpyToSymbol");
    aotx_pick_fixture<<<1, count>>>(count, 7u, sampled, thinking, tokens, 0u);
    aotx_model_pick<<<count, AOTX_MODEL_ROW_THREADS>>>(AOTX_PICK_ROLE);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    int out[AOTX_PICK_SEQS];
    aotx_check_runtime(cudaMemcpy(out, token, count * sizeof *out, cudaMemcpyDeviceToHost),
                       "cudaMemcpy");
    unsigned int same = 0u;
    for (unsigned int i = 0u; i < count; ++i) {
        unsigned int expect = (i & 1u) ? largest : want;
        same += ((unsigned int)out[i] == expect) ? 1u : 0u;
    }
    aotx_pick_note(label, same == count, "rows", (double)same, (double)count);
    cudaFree(head);
    cudaFree(agent);
    cudaFree(token);
    cudaFree(draw);
    cudaFree(device_how);
    free(host);
}

static void aotx_pick_thinking(void)
{
    static const unsigned int counts[2] = { 1u, AOTX_PICK_SEQS };
    for (unsigned int c = 0u; c < 2u; ++c) {
        unsigned int count = counts[c];
        aotx_pick_thinking_case(count, 0u, 0u, 0u, 0, AOTX_PICK_THINK_OPEN, 7u, 1u,
                                "launch wrap masks the opening token");
        aotx_pick_thinking_case(count, 4u, 1u, 2u, 2, 7u, AOTX_PICK_THINK_CLOSE, 1u,
                                "launch wrap permits only the closing token");
        aotx_pick_thinking_case(count, 4u, 1u, 2u, -1, 7u, 7u, 1u,
                                "absent think limit leaves the span open");
        aotx_pick_thinking_case(count, 0u, 0u, 0u, 0, 0u, 0u, 0u,
                                "an empty wrap does not mask token zero");
        aotx_pick_thinking_case(count, 4u, 1u, 2u, 2, 7u, 7u, 0u,
                                "an empty wrap does not force a closing token");
    }
}

int main(void)
{
    aotx_check_runtime(cudaFree(0), "cudaFree");
    aotx_pick_gear gear;
    aotx_pick_open(&gear);
    float *row = (float *)malloc(AOTX_PICK_VOCAB * sizeof *row);
    aotx_pick_slope(row, 0x5A3D1177ull);
    aotx_pick_row(&gear, row);

    /* The count cut, the mass cut and the temperature, one at a time and together. */
    aotx_model_how how;
    memset(&how, 0, sizeof how);
    how.seed = AOTX_PICK_SEED;
    how.top_p = 1.0f;
    how.temperature = 1.0f;
    how.repeat_penalty = 1.0f;
    how.think_limit = -1;
    how.top_k = 1u;
    aotx_pick_arm(&gear, row, &how, "one candidate");
    how.top_k = 10u;
    aotx_pick_arm(&gear, row, &how, "ten candidates");
    how.top_k = 40u;
    how.temperature = 2.0f;
    aotx_pick_arm(&gear, row, &how, "40 candidates at two");
    how.top_k = 0u;
    how.temperature = 1.0f;
    how.top_p = 0.5f;
    aotx_pick_arm(&gear, row, &how, "half the mass");
    how.top_p = 0.9f;
    how.temperature = 2.0f;
    aotx_pick_arm(&gear, row, &how, "nine tenths at two");
    how.top_k = 40u;
    how.top_p = 0.9f;
    how.temperature = 1.0f;
    aotx_pick_arm(&gear, row, &how, "40 and nine tenths");
    how.top_k = 40u;
    how.top_p = 1.0f;
    how.min_p = 0.8f;
    aotx_pick_arm(&gear, row, &how, "eight tenths of the largest");
    how.min_p = 0.0f;

    /* The bucket search: a thousand near equal logits must give many tokens and not one. */
    aotx_pick_flat(row);
    aotx_pick_row(&gear, row);
    how.top_k = 40u;
    how.top_p = 1.0f;
    how.temperature = 1.0f;
    aotx_pick_run(&gear, &how, AOTX_PICK_ROUNDS);
    unsigned int *count = (unsigned int *)malloc(AOTX_PICK_VOCAB * sizeof *count);
    unsigned int *id = (unsigned int *)malloc(AOTX_PICK_VOCAB * sizeof *id);
    unsigned int distinct = 0u;
    for (unsigned int i = 0u; i < 1000u; ++i) {
        id[i] = i;
    }
    unsigned int outside = aotx_pick_tally(gear.out, AOTX_PICK_DRAWS, id, 1000u, count,
                                           &distinct);
    aotx_pick_note("near equal logits stay in the set", outside == 0u, "outside",
                   (double)outside, 0.0);
    aotx_pick_note("near equal logits give many tokens", distinct >= 20u, "tokens",
                   (double)distinct, 20.0);

    /* The two arms must be able to fail. A set which is too small sees draws outside it,
     * and counts of one shape do not pass the test of another. */
    aotx_pick_slope(row, 0x5A3D1177ull);
    aotx_pick_row(&gear, row);
    how.top_k = 40u;
    how.top_p = 1.0f;
    how.temperature = 1.0f;
    aotx_pick_run(&gear, &how, AOTX_PICK_ROUNDS);
    double *chance = (double *)malloc(AOTX_PICK_VOCAB * sizeof *chance);
    aotx_model_how small = how;
    small.top_k = 10u;
    unsigned int members = aotx_pick_allow(row, &small, id, chance);
    outside = aotx_pick_tally(gear.out, AOTX_PICK_DRAWS, id, members, count, &distinct);
    aotx_pick_note("a set that is too small is refused", outside > 0u, "outside",
                   (double)outside, 0.0);
    members = aotx_pick_allow(row, &how, id, chance);
    aotx_pick_tally(gear.out, AOTX_PICK_DRAWS, id, members, count, &distinct);
    for (unsigned int i = 0u; i < members; ++i) {
        chance[i] = 1.0 / (double)members;
    }
    double bound = 0.0;
    double chi = aotx_pick_chi(count, chance, members, AOTX_PICK_DRAWS, &bound);
    aotx_pick_note("a flat shape is refused", chi > bound, "chi", chi, bound);

    aotx_pick_penalties(&gear, row);
    aotx_pick_telemetry(&gear, row);
    aotx_pick_voice(&gear, row);
    aotx_pick_catalog_refusal();
    aotx_pick_thinking();

    free(row);
    free(count);
    free(id);
    free(chance);
    printf("sample: %u cases, %u bad, 0 skipped, %u draws for each arm\n", aotx_pick_cases,
           aotx_pick_bad, AOTX_PICK_DRAWS);
    return (aotx_pick_bad == 0u) ? 0 : 1;
}
