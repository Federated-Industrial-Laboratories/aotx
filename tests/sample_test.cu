/* Purpose: Check the sample kernel against the distribution it is asked for.
 * Owns: The logits rows, the draw counts and the counts of the cases.
 * Launch shape: The sample kernel alone, one block for each sequence.
 * Lifetime: The program. */
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "boot/check.h"
#include "model/forward.cuh"

#define AOTX_PICK_ROLE     AOTX_MODEL_LANGUAGE
#define AOTX_PICK_VOCAB    2048u
#define AOTX_PICK_SEQS     AOTX_SLOTS
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
static void aotx_pick_run(aotx_pick_gear *gear, const aotx_model_how *how,
                          unsigned int passes)
{
    aotx_model_run set;
    memset(&set, 0, sizeof set);
    set.agent = gear->agent;
    set.token = gear->token;
    set.draw = gear->draw;
    set.seqs = AOTX_PICK_SEQS;
    set.rows = AOTX_PICK_SEQS;
    set.seed = how->seed;
    set.top_k = how->top_k;
    set.top_p = how->top_p;
    set.temperature = how->temperature;
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_model_call, &set, sizeof set,
                                          (size_t)AOTX_PICK_ROLE * sizeof set),
                       "cudaMemcpyToSymbol");
    aotx_model_restream();
    for (unsigned int p = 0u; p < passes; ++p) {
        aotx_model_pick<<<AOTX_PICK_SEQS, AOTX_MODEL_ROW_THREADS>>>(AOTX_PICK_ROLE);
        aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        aotx_check_runtime(cudaMemcpy(gear->out + (size_t)p * AOTX_PICK_SEQS, gear->token,
                                      AOTX_PICK_SEQS * sizeof(int), cudaMemcpyDeviceToHost),
                           "cudaMemcpy");
    }
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
    how.seed = AOTX_PICK_SEED;
    how.top_p = 1.0f;
    how.temperature = 1.0f;
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

    free(row);
    free(count);
    free(id);
    free(chance);
    printf("sample: %u cases, %u bad, 0 skipped, %u draws for each arm\n", aotx_pick_cases,
           aotx_pick_bad, AOTX_PICK_DRAWS);
    return (aotx_pick_bad == 0u) ? 0 : 1;
}
