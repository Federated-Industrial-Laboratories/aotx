/* Purpose: Check the decode of the tick graph: its records, its states and its rate.
 * Owns: The test consumer of the host ring and the counts of the cases.
 * Launch shape: One thread for each sequence slot; the consumer is a host thread.
 * Lifetime: One run of the test program. */
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "boot/boot.cuh"
#include "boot/check.h"
#include "mem/mem.cuh"
#include "model/decode_state.cuh"
#include "model/graph_host.h"
#include "model/roles.h"
#include "sched/sched.cuh"
#include "seam/seam.cuh"

#include "decode_drain.h"

/* Ticks one run may take before the check gives up on it. */
#define AOTX_DECODE_TEST_TICKS   4000u

/* Bytes of one take of the reply text. */
#define AOTX_DECODE_TEST_TEXT    512u

/* The reply of a check run, and the reply of a rate run. */
#define AOTX_DECODE_TEST_REPLY   12u
#define AOTX_DECODE_TEST_RATE    16u

/* The budget of a decode tick at 16 sequences, in nanoseconds. One tick reads every weight
 * of the model one time. At 16 rows the tensor core product pays for the whole 128 rows
 * of its tile. That is about 58 ms for the 4.02 billion weights of this file at the 17.8
 * TFLOPS the product gives. The budget is twice that figure. It covers the plan, the
 * commit, the flush, the page service and the launch of every node of the graph. */
#define AOTX_DECODE_TEST_BUDGET  120000000ull

/* The sample of a rate run, from the model card of the language file. */
#define AOTX_DECODE_TEST_TOP_K   20u
#define AOTX_DECODE_TEST_TOP_P   0.95f
#define AOTX_DECODE_TEST_HEAT    0.7f

/* Open one sequence for each prompt, from one thread, as the command layer does. */
__global__ void aotx_decode_test_open(const int *ids, const unsigned int *start,
                                      const unsigned int *count, unsigned int first,
                                      unsigned int seqs, unsigned int role,
                                      unsigned int limit, unsigned long long seed,
                                      unsigned int top_k, float top_p, float heat,
                                      unsigned int *bad)
{
    if (blockIdx.x != 0u || threadIdx.x != 0u) {
        return;
    }
    for (unsigned int s = first; s < first + seqs; ++s) {
        if (aotx_seq_open(s, role, ids + start[s], count[s], limit, seed + s, top_k,
                          top_p, heat, aotx_time_tick) != 0) {
            *bad += 1u;
        }
    }
}

/* Mark a run of sequences to end at the next commit. */
__global__ void aotx_decode_test_stop(unsigned int seqs)
{
    unsigned int slot = threadIdx.x;
    if (slot < seqs) {
        aotx_seq_stop(slot);
    }
}

/* Apply a run of token records, in order, as the serial apply of a restore does. */
__global__ void aotx_decode_test_apply(const aotx_token_body *body, unsigned int count,
                                       unsigned int *bad)
{
    if (blockIdx.x != 0u || threadIdx.x != 0u) {
        return;
    }
    for (unsigned int i = 0u; i < count; ++i) {
        if (aotx_seq_apply(&body[i]) != 0) {
            *bad += 1u;
        }
    }
}

/* Put every slot back in the free state and ask for its pages back. */
__global__ void aotx_decode_test_clear(void)
{
    unsigned int slot = threadIdx.x;
    if (slot >= AOTX_SEQ_SLOTS) {
        return;
    }
    aotx_seqs.slot[slot].state = AOTX_SEQ_STATE_FREE;
    aotx_seqs.slot[slot].flags = 0u;
    aotx_seqs.slot[slot].prompt = 0u;
    aotx_seqs.slot[slot].sampled = 0u;
    aotx_seqs.slot[slot].held = 0u;
    aotx_seq_kept[slot] = 0u;
    aotx_seq_shown[slot] = 0u;
    aotx_model_seen[slot] = 0u;
    aotx_model_draw[slot] = 0u;
    aotx_decode.rows[slot] = 0u;
    aotx_kv_release(slot);
    if (slot == 0u) {
        aotx_seqs.live = 0u;
        aotx_seqs.refused = 0u;
    }
}

/* Take the reply bytes of one slot, as the console does. */
__global__ void aotx_decode_test_text(unsigned int slot, unsigned char *out,
                                      unsigned int max, unsigned int *length)
{
    if (blockIdx.x == 0u && threadIdx.x == 0u) {
        *length = aotx_seq_take_text(slot, out, max);
    }
}

typedef struct aotx_decode_test_gear {
    int *ids;
    unsigned int *start;
    unsigned int *count;
    unsigned int *bad;
    aotx_token_body *body;
    unsigned char *text;
    unsigned int *length;
    aotx_seq slot[AOTX_SEQ_SLOTS];
} aotx_decode_test_gear;

static void *aotx_decode_test_take(unsigned long long bytes)
{
    void *at = 0;
    aotx_check_runtime(cudaMalloc(&at, (size_t)bytes), "cudaMalloc");
    aotx_check_runtime(cudaMemset(at, 0, (size_t)bytes), "cudaMemset");
    return at;
}

/* Read the sequence table of the device. The token lists stay on the device. */
static void aotx_decode_test_read(aotx_decode_test_gear *gear)
{
    aotx_check_runtime(cudaMemcpyFromSymbol(gear->slot, aotx_seqs, sizeof gear->slot, 0,
                                            cudaMemcpyDeviceToHost),
                       "cudaMemcpyFromSymbol");
}

/* Slots that are not free. */
static unsigned int aotx_decode_test_live(aotx_decode_test_gear *gear)
{
    unsigned int live = 0u;
    aotx_decode_test_read(gear);
    for (unsigned int s = 0u; s < AOTX_SEQ_SLOTS; ++s) {
        if (gear->slot[s].state != AOTX_SEQ_STATE_FREE) {
            live += 1u;
        }
    }
    return live;
}

/* Run ticks until every slot is free again, or until the tick bound. */
static unsigned int aotx_decode_test_drive(aotx_pump *pump, aotx_decode_test_gear *gear,
                                           unsigned int bound)
{
    unsigned int ticks = 0u;
    while (ticks < bound) {
        aotx_pump_tick(pump);
        ticks += 1u;
        if (aotx_decode_test_live(gear) == 0u) {
            break;
        }
    }
    return ticks;
}

/* Open one sequence for each of the first seqs prompts and give the tick count back. */
static void aotx_decode_test_ask(aotx_decode_test_gear *gear, unsigned int seqs,
                                 unsigned int role, unsigned int limit,
                                 unsigned int top_k, float top_p, float heat)
{
    aotx_decode_test_open<<<1, 1>>>(gear->ids, gear->start, gear->count, 0u, seqs, role,
                                    limit, 0x51EEDull, top_k, top_p, heat, gear->bad);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
}

static void aotx_decode_test_reset(aotx_pump *pump)
{
    aotx_decode_test_clear<<<1, AOTX_SEQ_SLOTS>>>();
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_kv_serve(&pump->kv, 0);
}

/* One run of a set of sequences to the end, with the records and the pages checked. */
static void aotx_decode_test_case_run(aotx_pump *pump, aotx_decode_test_gear *gear,
                                      aotx_decode_test_drain *drain, unsigned int seqs,
                                      unsigned int role, unsigned int *applied,
                                      unsigned int *failed)
{
    unsigned int mark = drain->taken;
    unsigned int events = drain->released;
    aotx_decode_test_reset(pump);
    aotx_decode_test_ask(gear, seqs, role, AOTX_DECODE_TEST_REPLY,
                         AOTX_DECODE_TEST_TOP_K, AOTX_DECODE_TEST_TOP_P,
                         AOTX_DECODE_TEST_HEAT);
    unsigned int ticks = aotx_decode_test_drive(pump, gear, AOTX_DECODE_TEST_TICKS);
    usleep(50000);

    unsigned int bad = 0u;
    unsigned int made = 0u;
    aotx_decode_test_read(gear);
    for (unsigned int s = 0u; s < seqs; ++s) {
        const aotx_seq *seq = &gear->slot[s];
        if (seq->state != AOTX_SEQ_STATE_FREE || seq->sampled == 0u
            || seq->sampled > AOTX_DECODE_TEST_REPLY) {
            bad += 1u;
            continue;
        }
        made += seq->sampled;
        aotx_decode_test_records(drain, mark, s, seq->prompt, seq->sampled, &bad);
    }
    *applied += 1u;
    if (bad != 0u) {
        printf("decode: %u sequences of %u did not end with a whole token list\n", bad, seqs);
        *failed += 1u;
    }

    /* Every slot gives its pages back after the tick that ends it. */
    aotx_kv_table table;
    aotx_check_runtime(cudaMemcpyFromSymbol(&table, aotx_kv, sizeof table),
                       "cudaMemcpyFromSymbol");
    unsigned int held = 0u;
    for (unsigned int s = 0u; s < seqs; ++s) {
        held += table.count[s];
    }
    *applied += 1u;
    if (held != 0u || drain->released - events != seqs) {
        printf("decode: %u pages stay and %u slots reported a release of %u\n", held,
               drain->released - events, seqs);
        *failed += 1u;
    }
    printf("decode: %u sequences, %u reply tokens, %u ticks, %u pages held after the run\n",
           seqs, made, ticks, held);
}

/* The stop command ends a reply at the commit of the tick that follows it. */
static void aotx_decode_test_case_stop(aotx_pump *pump, aotx_decode_test_gear *gear,
                                       unsigned int role, unsigned int seqs,
                                       unsigned int *applied, unsigned int *failed)
{
    aotx_decode_test_reset(pump);
    aotx_decode_test_ask(gear, seqs, role, AOTX_SEQ_REPLY_DEFAULT, AOTX_DECODE_TEST_TOP_K,
                         AOTX_DECODE_TEST_TOP_P, AOTX_DECODE_TEST_HEAT);
    for (unsigned int i = 0u; i < 5u; ++i) {
        aotx_pump_tick(pump);
    }
    aotx_decode_test_read(gear);
    unsigned int running = 0u;
    unsigned int before = gear->slot[0].sampled;
    for (unsigned int s = 0u; s < seqs; ++s) {
        running += (gear->slot[s].state == AOTX_SEQ_STATE_DECODE) ? 1u : 0u;
    }
    aotx_decode_test_stop<<<1, AOTX_SEQ_SLOTS>>>(seqs);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_pump_tick(pump);
    aotx_decode_test_read(gear);
    unsigned int ended = 0u;
    for (unsigned int s = 0u; s < seqs; ++s) {
        ended += (gear->slot[s].state == AOTX_SEQ_STATE_DONE) ? 1u : 0u;
    }
    aotx_pump_tick(pump);
    aotx_decode_test_read(gear);
    unsigned int freed = 0u;
    for (unsigned int s = 0u; s < seqs; ++s) {
        freed += (gear->slot[s].state == AOTX_SEQ_STATE_FREE) ? 1u : 0u;
    }
    *applied += 1u;
    if (running == 0u || ended != seqs || freed != seqs) {
        printf("decode: a stop of %u sequences found %u running, ended %u and freed %u\n",
               seqs, running, ended, freed);
        *failed += 1u;
    } else {
        printf("decode: a stop of %u sequences, %u of them in reply after %u tokens, ended "
               "every one of them in one tick\n", seqs, running, before);
    }
    aotx_decode_test_reset(pump);
}

/* The bytes of the reply that the console takes. A second take gives nothing more. */
static void aotx_decode_test_case_text(aotx_pump *pump, aotx_decode_test_gear *gear,
                                       unsigned int role, unsigned int seqs,
                                       unsigned int *applied, unsigned int *failed)
{
    unsigned char text[AOTX_DECODE_TEST_TEXT];
    unsigned int length = 0u;
    unsigned int again = 0u;
    unsigned int gave = 0u;
    unsigned int more = 0u;
    aotx_decode_test_reset(pump);
    aotx_decode_test_ask(gear, seqs, role, AOTX_DECODE_TEST_REPLY, AOTX_DECODE_TEST_TOP_K,
                         AOTX_DECODE_TEST_TOP_P, AOTX_DECODE_TEST_HEAT);
    for (unsigned int i = 0u; i < 8u; ++i) {
        aotx_pump_tick(pump);
    }
    aotx_decode_test_read(gear);
    for (unsigned int s = 0u; s < seqs; ++s) {
        aotx_decode_test_text<<<1, 1>>>(s, gear->text, sizeof text, gear->length);
        aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        aotx_check_runtime(cudaMemcpy(&length, gear->length, sizeof length,
                                      cudaMemcpyDeviceToHost), "cudaMemcpy");
        if (s == 0u) {
            aotx_check_runtime(cudaMemcpy(text, gear->text, sizeof text,
                                          cudaMemcpyDeviceToHost), "cudaMemcpy");
        }
        aotx_decode_test_text<<<1, 1>>>(s, gear->text, sizeof text, gear->length);
        aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        aotx_check_runtime(cudaMemcpy(&again, gear->length, sizeof again,
                                      cudaMemcpyDeviceToHost), "cudaMemcpy");
        gave += (length != 0u && gear->slot[s].sampled != 0u) ? 1u : 0u;
        more += (again != 0u) ? 1u : 0u;
    }
    *applied += 1u;
    if (gave != seqs || more != 0u) {
        printf("decode: %u of %u takes gave bytes and %u second takes gave more\n", gave,
               seqs, more);
        *failed += 1u;
    } else {
        unsigned int show = (length < 40u) ? length : 40u;
        text[show] = '\0';
        printf("decode: %u replies gave bytes; the first is %s\n", seqs, text);
    }
    aotx_decode_test_reset(pump);
}

/* A prompt that is longer than the token budget of a tick goes in over several ticks. A
 * sequence that opens while that one decodes joins the same batch. */
static void aotx_decode_test_case_long(aotx_pump *pump, aotx_decode_test_gear *gear,
                                       aotx_decode_test_prompt *prompt, unsigned int role,
                                       unsigned int *applied, unsigned int *failed)
{
    unsigned int whole = prompt->start[AOTX_SEQ_SLOTS - 1u]
                       + prompt->count[AOTX_SEQ_SLOTS - 1u];
    unsigned int long_count = (whole > 1024u) ? 1024u : whole;
    unsigned int keep = prompt->count[0];
    aotx_decode_test_reset(pump);
    aotx_check_runtime(cudaMemcpy(gear->count, &long_count, sizeof long_count,
                                  cudaMemcpyHostToDevice), "cudaMemcpy");
    aotx_decode_test_ask(gear, 1u, role, AOTX_DECODE_TEST_REPLY, AOTX_DECODE_TEST_TOP_K,
                         AOTX_DECODE_TEST_TOP_P, AOTX_DECODE_TEST_HEAT);
    aotx_check_runtime(cudaMemcpy(gear->count, &keep, sizeof keep, cudaMemcpyHostToDevice),
                       "cudaMemcpy");
    unsigned int chunks = 0u;
    unsigned int held = 0u;
    while (chunks < AOTX_DECODE_TEST_TICKS) {
        aotx_pump_tick(pump);
        chunks += 1u;
        aotx_decode_test_read(gear);
        held = gear->slot[0].held;
        if (gear->slot[0].state != AOTX_SEQ_STATE_PREFILL) {
            break;
        }
    }

    /* Eight more sequences open while the long one decodes, so one batch holds a decode
     * row and a prompt piece together. */
    aotx_decode_test_open<<<1, 1>>>(gear->ids, gear->start, gear->count, 1u, 8u, role,
                                    AOTX_DECODE_TEST_REPLY, 0x51EEDull,
                                    AOTX_DECODE_TEST_TOP_K, AOTX_DECODE_TEST_TOP_P,
                                    AOTX_DECODE_TEST_HEAT, gear->bad);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_decode_test_drive(pump, gear, AOTX_DECODE_TEST_TICKS);
    aotx_decode_test_read(gear);
    unsigned int made = 0u;
    for (unsigned int s = 0u; s < 9u; ++s) {
        made += (gear->slot[s].sampled == AOTX_DECODE_TEST_REPLY) ? 1u : 0u;
    }
    *applied += 2u;
    if (chunks < 2u || held < long_count) {
        printf("decode: a prompt of %u tokens took %u ticks and holds %u\n", long_count,
               chunks, held);
        *failed += 1u;
    }
    if (made != 9u) {
        printf("decode: %u of 9 sequences of the mixed batch made a whole reply\n", made);
        *failed += 1u;
    }
    printf("decode: a prompt of %u tokens went in over %u ticks, and 8 sequences that "
           "opened after it joined the same batch; %u of 9 gave a whole reply\n",
           long_count, chunks, made);
    aotx_decode_test_reset(pump);
}

/* The cost of a tick that holds no live sequence. Every kernel of the pass exits at once,
 * so the figure is the launch of the nodes of the graph and nothing else. */
static void aotx_decode_test_case_idle(aotx_pump *pump, aotx_decode_test_drain *drain,
                                       unsigned int *applied, unsigned int *failed)
{
    aotx_decode_test_reset(pump);
    for (unsigned int i = 0u; i < 4u; ++i) {
        aotx_pump_tick(pump);
    }
    usleep(50000);
    unsigned int marks = drain->ticks;
    unsigned long long sum = drain->tick_ns_sum;
    for (unsigned int i = 0u; i < 40u; ++i) {
        aotx_pump_tick(pump);
    }
    usleep(50000);
    unsigned int steps = drain->ticks - marks;
    unsigned long long mean = (steps > 0u) ? ((drain->tick_ns_sum - sum) / steps) : 0ull;
    *applied += 1u;
    if (steps == 0u || mean == 0ull) {
        printf("decode: the idle ticks gave no statistics record\n");
        *failed += 1u;
    }
    printf("decode: the pass holds %u nodes; an idle tick costs %llu us over %u ticks\n",
           aotx_decode_nodes(), mean / 1000ull, steps);
}

/* The shape of the tick graph over a run that opens and closes sequences. */
static void aotx_decode_test_case_graph(aotx_pump *pump, aotx_decode_test_gear *gear,
                                        unsigned int role, unsigned int ticks,
                                        unsigned int *applied, unsigned int *failed)
{
    size_t before_nodes = 0;
    size_t before_edges = 0;
    size_t after_nodes = 0;
    size_t after_edges = 0;
    aotx_check_runtime(cudaGraphGetNodes(pump->graph, 0, &before_nodes),
                       "cudaGraphGetNodes");
    aotx_check_runtime(cudaGraphGetEdges(pump->graph, 0, 0, 0, &before_edges),
                       "cudaGraphGetEdges");
    aotx_decode_test_reset(pump);
    unsigned int opens = 0u;
    for (unsigned int t = 0u; t < ticks; ++t) {
        if (aotx_decode_test_live(gear) == 0u) {
            aotx_decode_test_ask(gear, 4u, role, 4u, 1u, 1.0f, 0.0f);
            opens += 4u;
        }
        aotx_pump_tick(pump);
    }
    aotx_check_runtime(cudaGraphGetNodes(pump->graph, 0, &after_nodes),
                       "cudaGraphGetNodes");
    aotx_check_runtime(cudaGraphGetEdges(pump->graph, 0, 0, 0, &after_edges),
                       "cudaGraphGetEdges");
    *applied += 1u;
    if (before_nodes != after_nodes || before_edges != after_edges) {
        printf("decode: the tick graph went from %zu nodes and %zu edges to %zu and %zu\n",
               before_nodes, before_edges, after_nodes, after_edges);
        *failed += 1u;
    } else {
        printf("decode: the tick graph holds %zu nodes and %zu edges over %u ticks with "
               "%u sequences opened and closed\n", after_nodes, after_edges, ticks, opens);
    }
    aotx_decode_test_reset(pump);
}

/* A replay of the token records rebuilds the token list and the pages. The check runs the
 * sequences, replays every token but the last two of each, and lets the decode make them
 * again. The logits of the last row stand beside the list at one sequence. The batch of
 * the last pass holds that sequence and nothing else there. */
static void aotx_decode_test_case_replay(aotx_pump *pump, aotx_decode_test_gear *gear,
                                         aotx_decode_test_drain *drain, unsigned int role,
                                         unsigned int vocab, unsigned int seqs,
                                         unsigned int *applied, unsigned int *failed)
{
    unsigned int mark = drain->taken;
    unsigned int reply = 8u;
    unsigned int list[AOTX_SEQ_SLOTS];
    aotx_decode_test_reset(pump);
    aotx_decode_test_ask(gear, seqs, role, reply, 1u, 1.0f, 0.0f);
    aotx_decode_test_drive(pump, gear, AOTX_DECODE_TEST_TICKS);
    usleep(50000);
    float *first = (seqs == 1u) ? aotx_decode_test_logits(role, vocab) : 0;
    aotx_decode_test_read(gear);
    unsigned int whole = 0u;
    for (unsigned int s = 0u; s < seqs; ++s) {
        list[s] = gear->slot[s].prompt + gear->slot[s].sampled;
        whole += list[s];
    }
    aotx_token_body *body = (aotx_token_body *)calloc(whole, sizeof *body);
    unsigned int count = aotx_decode_test_bodies(drain, mark, seqs, role, 2u, list, body);
    int *before = (int *)malloc(AOTX_DECODE_TEST_LIST_BYTES);
    aotx_decode_test_tokens(before);
    aotx_decode_test_reset(pump);

    aotx_token_body *at = (aotx_token_body *)aotx_decode_test_take(
        (unsigned long long)whole * sizeof *body);
    aotx_check_runtime(cudaMemcpy(at, body, (size_t)count * sizeof *body,
                                  cudaMemcpyHostToDevice), "cudaMemcpy");
    aotx_decode_test_apply<<<1, 1>>>(at, count, gear->bad);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    /* The replay stops at the length of the first run. The last pass of the two runs
     * therefore reads the same pages and gives the logits of the same position. */
    unsigned int steps = 0u;
    while (steps < AOTX_DECODE_TEST_TICKS) {
        aotx_pump_tick(pump);
        steps += 1u;
        aotx_decode_test_read(gear);
        unsigned int done = 0u;
        for (unsigned int s = 0u; s < seqs; ++s) {
            done += (gear->slot[s].prompt + gear->slot[s].sampled >= list[s]
                     || gear->slot[s].state == AOTX_SEQ_STATE_FREE) ? 1u : 0u;
        }
        if (done == seqs) {
            break;
        }
    }
    usleep(50000);
    float *second = (seqs == 1u) ? aotx_decode_test_logits(role, vocab) : 0;
    int *after = (int *)malloc(AOTX_DECODE_TEST_LIST_BYTES);
    aotx_decode_test_tokens(after);
    aotx_decode_test_read(gear);
    unsigned int same = 0u;
    unsigned int made = 0u;
    unsigned int again = 0u;
    for (unsigned int s = 0u; s < seqs; ++s) {
        unsigned int held = list[s] - 2u;
        unsigned int hold = 0u;
        for (unsigned int i = 0u; i < list[s]; ++i) {
            hold += (before[s * AOTX_SEQ_MAX_TOKENS + i]
                     == after[s * AOTX_SEQ_MAX_TOKENS + i]) ? 1u : 0u;
        }
        same += (hold >= held) ? 1u : 0u;
        again += (hold == list[s]) ? 1u : 0u;
        made += (gear->slot[s].prompt + gear->slot[s].sampled >= list[s]) ? 1u : 0u;
    }

    /* The records that were replayed must come back to the letter. The two tokens the
     * decode makes again are a new draw over a batch of another shape. Such a draw is not
     * bitwise reproducible, so the count of them is a figure and not an arm. A draw which
     * gives the stop token ends the sequence one token early, so the length is a figure
     * as well. */
    *applied += 1u;
    if (same != seqs) {
        printf("decode: a replay of %u records gave %u of %u lists the same\n", count,
               same, seqs);
        *failed += 1u;
    }
    printf("decode: a replay of %u records rebuilt %u of %u token lists and their pages; "
           "%u made the two tokens after them again and %u reached the length of the "
           "first run\n", count, same, seqs, again, made);
    if (first != 0 && second != 0) {
        float worst = 0.0f;
        float span = 0.0f;
        for (unsigned int i = 0u; i < vocab; ++i) {
            float step = fabsf(first[i] - second[i]);
            float size = fabsf(first[i]);
            worst = (step > worst) ? step : worst;
            span = (size > span) ? size : span;
        }
        unsigned int pick_one = aotx_decode_test_argmax(first, vocab);
        unsigned int pick_two = aotx_decode_test_argmax(second, vocab);
        *applied += 1u;
        if (pick_one != pick_two) {
            printf("decode: the largest logit moved from %u to %u after the replay\n",
                   pick_one, pick_two);
            *failed += 1u;
        }
        printf("decode: the logits of the last row differ by %.4f at the most of a largest "
               "value of %.2f\n", (double)worst, (double)span);
        free(first);
        free(second);
    }
    free(body);
    free(before);
    free(after);
    cudaFree(at);
    aotx_decode_test_reset(pump);
}

/* A replay that crosses the seam. The records go in the inbound ring in three runs and the
 * device applies them. The state hash and the token lists must come back. One slot holds
 * two sequences one after the other, so the apply must close the first and open the second
 * on the same slot. */
static void aotx_decode_test_case_feed(aotx_pump *pump, aotx_decode_test_gear *gear,
                                       aotx_decode_test_drain *drain,
                                       aotx_decode_test_prompt *prompt,
                                       aotx_seam_rings *rings, unsigned long long boot_id,
                                       unsigned int role, unsigned int *applied,
                                       unsigned int *failed)
{
    unsigned int mark = drain->taken;
    unsigned long long began = 0ull;
    unsigned long long ended = 0ull;
    aotx_decode_test_reset(pump);
    aotx_decode_test_hash(&began, 0);

    /* Two sequences that run together, and then one more on the slot of the first. */
    aotx_decode_test_ask(gear, 2u, role, 6u, 1u, 1.0f, 0.0f);
    aotx_decode_test_drive(pump, gear, AOTX_DECODE_TEST_TICKS);

    /* The second sequence of slot 0 takes another prompt. Its tokens are therefore not
     * the tokens of the first, and a slot that keeps only one of them is seen. */
    aotx_check_runtime(cudaMemcpy(gear->start, prompt->start + 2u, sizeof(unsigned int),
                                  cudaMemcpyHostToDevice), "cudaMemcpy");
    aotx_check_runtime(cudaMemcpy(gear->count, prompt->count + 2u, sizeof(unsigned int),
                                  cudaMemcpyHostToDevice), "cudaMemcpy");
    aotx_decode_test_ask(gear, 1u, role, 4u, 1u, 1.0f, 0.0f);
    aotx_decode_test_drive(pump, gear, AOTX_DECODE_TEST_TICKS);
    aotx_check_runtime(cudaMemcpy(gear->start, prompt->start, sizeof(unsigned int),
                                  cudaMemcpyHostToDevice), "cudaMemcpy");
    aotx_check_runtime(cudaMemcpy(gear->count, prompt->count, sizeof(unsigned int),
                                  cudaMemcpyHostToDevice), "cudaMemcpy");
    usleep(50000);
    aotx_decode_test_hash(&ended, 0);
    int *before = (int *)malloc(AOTX_DECODE_TEST_LIST_BYTES);
    aotx_decode_test_tokens(before);

    /* The records of the run, in the order the ring holds them. */
    unsigned int whole = drain->taken - mark;
    aotx_token_body *body = (aotx_token_body *)calloc(whole, sizeof *body);
    unsigned int count = 0u;
    for (unsigned int i = mark; i < drain->taken; ++i) {
        const aotx_decode_test_token *one = &drain->token[i];
        body[count].slot = one->slot;
        body[count].token = one->token;
        body[count].position = one->position;
        body[count].flags = one->flags;
        body[count].seed = one->seed;
        body[count].draw = one->draw;
        body[count].role = role;
        body[count].reserved = 0u;
        count += 1u;
    }
    aotx_decode_test_reset(pump);
    aotx_decode_test_hash(&began, 1);
    aotx_seam_set_replaying(1);
    unsigned int piece = (count + 2u) / 3u;
    unsigned int runs = 0u;
    for (unsigned int at = 0u; at < count; at += piece) {
        unsigned int take = (count - at < piece) ? (count - at) : piece;
        aotx_decode_test_feed(rings, body + at, take, boot_id);
        aotx_pump_tick(pump);
        aotx_pump_tick(pump);
        runs += 1u;
    }
    aotx_seam_set_replaying(0);
    usleep(50000);
    unsigned long long again = 0ull;
    aotx_decode_test_hash(&again, 0);
    int *after = (int *)malloc(AOTX_DECODE_TEST_LIST_BYTES);
    aotx_decode_test_tokens(after);
    aotx_decode_test_read(gear);
    unsigned int refused = 0u;
    aotx_check_runtime(cudaMemcpyFromSymbol(&refused, aotx_seqs, sizeof refused,
                                            offsetof(aotx_seq_table, refused)),
                       "cudaMemcpyFromSymbol");
    unsigned int same = 0u;
    for (unsigned int s = 0u; s < 2u; ++s) {
        unsigned int hold = 0u;
        for (unsigned int i = 0u; i < AOTX_SEQ_MAX_TOKENS; ++i) {
            hold += (before[s * AOTX_SEQ_MAX_TOKENS + i]
                     == after[s * AOTX_SEQ_MAX_TOKENS + i]) ? 1u : 0u;
        }
        same += (hold == AOTX_SEQ_MAX_TOKENS) ? 1u : 0u;
    }
    *applied += 3u;
    if (runs < 3u || count == 0u) {
        printf("decode: the seam replay took %u runs of %u records\n", runs, count);
        *failed += 1u;
    }
    if (again != ended) {
        printf("decode: the state hash is %llx after the seam replay and was %llx\n",
               again, ended);
        *failed += 1u;
    }
    if (same != 2u || refused != 0u) {
        printf("decode: the seam replay rebuilt %u of 2 token lists and refused %u\n",
               same, refused);
        *failed += 1u;
    }
    printf("decode: %u token records went through the inbound ring in %u runs; the hash "
           "%llx and %u of 2 token lists came back, %u refused\n", count, runs, again,
           same, refused);

    /* The same records again through the serial apply, with no tick between them. The
     * second sequence of slot 0 then reaches a slot that still holds the first. That is
     * the path a restore takes when both sequences fall in one apply. */
    aotx_decode_test_reset(pump);
    aotx_token_body *at = (aotx_token_body *)aotx_decode_test_take(
        (unsigned long long)count * sizeof *body);
    aotx_check_runtime(cudaMemcpy(at, body, (size_t)count * sizeof *body,
                                  cudaMemcpyHostToDevice), "cudaMemcpy");
    aotx_decode_test_apply<<<1, 1>>>(at, count, gear->bad);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    int *once = (int *)malloc(AOTX_DECODE_TEST_LIST_BYTES);
    aotx_decode_test_tokens(once);
    unsigned int held = 0u;
    for (unsigned int i = 0u; i < AOTX_SEQ_MAX_TOKENS; ++i) {
        held += (before[i] == once[i]) ? 1u : 0u;
    }
    *applied += 1u;
    if (held != AOTX_SEQ_MAX_TOKENS) {
        printf("decode: one apply of both sequences of slot 0 gave %u of %u tokens\n",
               held, AOTX_SEQ_MAX_TOKENS);
        *failed += 1u;
    } else {
        printf("decode: one apply of both sequences of slot 0 gave the same token list\n");
    }
    free(once);
    cudaFree(at);
    free(body);
    free(before);
    free(after);
    aotx_decode_test_reset(pump);
}

/* The rate of the decode: reply tokens a second, with the display running. The figure
 * counts the ticks from the end of the prefill to the end of the last reply. The tick
 * times are of that window as well, so no prefill tick is in them. */
static double aotx_decode_test_rate(aotx_pump *pump, aotx_decode_test_gear *gear,
                                    aotx_decode_test_drain *drain, unsigned int seqs,
                                    unsigned int role, unsigned int *ticks,
                                    unsigned long long *mean, unsigned long long *worst)
{
    aotx_decode_test_reset(pump);
    aotx_decode_test_ask(gear, seqs, role, AOTX_DECODE_TEST_RATE, AOTX_DECODE_TEST_TOP_K,
                         AOTX_DECODE_TEST_TOP_P, AOTX_DECODE_TEST_HEAT);

    /* The prefill is over when every live slot holds its whole prompt. */
    unsigned int step = 0u;
    unsigned int made = 0u;
    while (step < AOTX_DECODE_TEST_TICKS) {
        aotx_pump_tick(pump);
        step += 1u;
        aotx_decode_test_read(gear);
        unsigned int ready = 0u;
        made = 0u;
        for (unsigned int s = 0u; s < seqs; ++s) {
            made += gear->slot[s].sampled;
            if (gear->slot[s].state == AOTX_SEQ_STATE_DECODE) {
                ready += 1u;
            }
        }
        if (ready == seqs) {
            break;
        }
    }
    usleep(50000);
    unsigned int first = made;
    unsigned int marks = drain->ticks;
    unsigned long long sum = drain->tick_ns_sum;
    drain->tick_ns_worst = 0ull;
    double began = aotx_decode_test_now();
    unsigned int run = aotx_decode_test_drive(pump, gear, AOTX_DECODE_TEST_TICKS);
    double spent = aotx_decode_test_now() - began;
    usleep(50000);
    unsigned int total = 0u;
    aotx_decode_test_read(gear);
    for (unsigned int s = 0u; s < seqs; ++s) {
        total += gear->slot[s].sampled;
    }
    unsigned int steps = drain->ticks - marks;
    *ticks = run;
    *worst = drain->tick_ns_worst;
    *mean = (steps > 0u) ? ((drain->tick_ns_sum - sum) / steps) : 0ull;
    aotx_decode_test_reset(pump);
    if (spent <= 0.0 || total <= first) {
        return 0.0;
    }
    return (double)(total - first) / spent;
}

/* The rate table of one role: reply tokens a second at four batch counts. */
static void aotx_decode_test_table(aotx_pump *pump, aotx_decode_test_gear *gear,
                                   aotx_decode_test_drain *drain, unsigned int role,
                                   const char *name, unsigned int *applied,
                                   unsigned int *failed)
{
    const unsigned int counts[4] = { 1u, 8u, 16u, 64u };
    for (unsigned int i = 0u; i < 4u; ++i) {
        unsigned int ticks = 0u;
        unsigned long long mean = 0ull;
        unsigned long long worst = 0ull;
        double rate = aotx_decode_test_rate(pump, gear, drain, counts[i], role, &ticks,
                                            &mean, &worst);
        printf("decode rate: %s at %2u sequences: %7.2f reply tokens a second, %u ticks, "
               "tick mean %llu us worst %llu us\n", name, counts[i], rate, ticks,
               mean / 1000ull, worst / 1000ull);
        *applied += 1u;
        if (rate <= 0.0) {
            printf("decode: the rate run at %u sequences made no reply token\n", counts[i]);
            *failed += 1u;
        }
        if (counts[i] == 16u) {
            *applied += 1u;
            if (worst > AOTX_DECODE_TEST_BUDGET) {
                printf("decode: a tick at 16 sequences took %llu us against a budget of "
                       "%llu us\n", worst / 1000ull, AOTX_DECODE_TEST_BUDGET / 1000ull);
                *failed += 1u;
            }
        }
    }
}

int main(int argc, char **argv)
{
    const char *models = (argc > 1) ? argv[1] : "models";
    const char *fixtures = (argc > 2) ? argv[2] : "tests/fixtures/tokenizer";
    char path[1024];
    unsigned int applied = 0u;
    unsigned int failed = 0u;
    snprintf(path, sizeof path, "%s/manifest.jsonl", models);
    if (access(path, R_OK) != 0) {
        printf("decode: 0 cases, 0 bad, 16 skipped, no model files in %s\n", models);
        return 0;
    }
    aotx_decode_test_prompt *prompt =
        (aotx_decode_test_prompt *)calloc(1, sizeof *prompt);
    snprintf(path, sizeof path, "%s/golden-language.ids", fixtures);
    if (aotx_decode_test_prompts(path, prompt) != 0) {
        printf("decode: 0 cases, 0 bad, 16 skipped, no prompt list in %s\n", fixtures);
        return 0;
    }

    /* The four bit file joins the run when the model record names it. */
    snprintf(path, sizeof path, "%s/manifest.jsonl", models);
    FILE *record = fopen(path, "rb");
    char text[65536];
    size_t bytes = (record != NULL) ? fread(text, 1u, sizeof text - 1u, record) : 0u;
    text[bytes] = '\0';
    if (record != NULL) {
        fclose(record);
    }
    int four_bit = (strstr(text, "\"name\":\"language-q4\"") != NULL);
    const char *roles = four_bit ? "language,language-q4" : "language";

    CUdevice device;
    CUcontext context;
    aotx_mem_map map;
    aotx_seam_rings rings;
    aotx_pump pump;
    aotx_check_driver(cuInit(0), "cuInit");
    aotx_check_driver(cuDeviceGet(&device, 0), "cuDeviceGet");
    aotx_check_driver(cuDevicePrimaryCtxRetain(&context, device), "cuDevicePrimaryCtxRetain");
    aotx_check_driver(cuCtxSetCurrent(context), "cuCtxSetCurrent");
    unsigned long long boot_id = 0x0DEC0DEull;
    if (aotx_mem_reserve(&map) != 0 || aotx_seam_open(&rings, boot_id) != 0) {
        printf("decode: the map or the rings did not open\n");
        return 1;
    }
    aotx_seam_bind(&rings, map.ring, map.ring_bytes, boot_id);
    aotx_seam_bind_bulk(&rings, map.scratch, AOTX_BULK_STAGE_BYTES);
    aotx_seam_note_boot<<<1, 1>>>(0ull, 0ull);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    double began = aotx_decode_test_now();
    if (aotx_boot_models(models, roles, 0) != 0) {
        printf("decode: the model files did not load\n");
        return 1;
    }
    printf("decode: roles %s loaded in %.1f s\n", roles, aotx_decode_test_now() - began);

    aotx_decode_test_drain *drain = (aotx_decode_test_drain *)calloc(1, sizeof *drain);
    drain->token = (aotx_decode_test_token *)calloc(AOTX_DECODE_TEST_RECORDS,
                                                    sizeof *drain->token);
    drain->map = rings.host_map;
    drain->data = rings.host_map + sizeof(aotx_host_ring_preamble);
    drain->data_bytes = AOTX_HOST_RING_DATA_BYTES;
    drain->mask = AOTX_HOST_RING_DATA_BYTES - 1ull;
    drain->boot_id = boot_id;
    pthread_t thread;
    pthread_create(&thread, NULL, aotx_decode_test_reader, drain);

    aotx_decode_test_gear *gear = (aotx_decode_test_gear *)calloc(1, sizeof *gear);
    gear->ids = (int *)aotx_decode_test_take(sizeof prompt->ids);
    gear->start = (unsigned int *)aotx_decode_test_take(sizeof prompt->start);
    gear->count = (unsigned int *)aotx_decode_test_take(sizeof prompt->count);
    gear->bad = (unsigned int *)aotx_decode_test_take(sizeof(unsigned int));
    gear->text = (unsigned char *)aotx_decode_test_take(1024ull);
    gear->length = (unsigned int *)aotx_decode_test_take(sizeof(unsigned int));
    aotx_check_runtime(cudaMemcpy(gear->ids, prompt->ids, sizeof prompt->ids,
                                  cudaMemcpyHostToDevice), "cudaMemcpy");
    aotx_check_runtime(cudaMemcpy(gear->start, prompt->start, sizeof prompt->start,
                                  cudaMemcpyHostToDevice), "cudaMemcpy");
    aotx_check_runtime(cudaMemcpy(gear->count, prompt->count, sizeof prompt->count,
                                  cudaMemcpyHostToDevice), "cudaMemcpy");
    printf("decode: %u prompts of %u to %u tokens\n", prompt->rows, prompt->count[0],
           prompt->count[prompt->rows - 1u]);

    const unsigned int order[2] = { AOTX_MODEL_LANGUAGE, AOTX_MODEL_LANGUAGE_Q4 };
    unsigned int roles_run = four_bit ? 2u : 1u;
    for (unsigned int r = 0u; r < roles_run; ++r) {
        unsigned int role = order[r];
        if (r > 0u) {
            aotx_model_shut(order[r - 1u]);
            if (aotx_model_open(role, AOTX_MODEL_MAX_TOKENS) != 0) {
                printf("decode: the graph of role %u did not capture\n", role);
                return 1;
            }
        }
        if (aotx_pump_build(&pump, 0ull, 1u) != 0 || pump.decode == 0u) {
            printf("decode: the tick graph did not take the decode\n");
            return 1;
        }
        unsigned int vocab = 0u;
        aotx_check_runtime(cudaMemcpyFromSymbol(&vocab, aotx_model, sizeof vocab,
                                                (size_t)role * sizeof(aotx_model_desc)
                                                + offsetof(aotx_model_desc, vocab)),
                           "cudaMemcpyFromSymbol");
        printf("decode: role %s, %u nodes in the tick graph, %u from the decode\n",
               aotx_role_name[role], pump.nodes, pump.decode_nodes);
        aotx_decode_test_case_idle(&pump, drain, &applied, &failed);
        aotx_decode_test_case_run(&pump, gear, drain, 1u, role, &applied, &failed);
        aotx_decode_test_case_run(&pump, gear, drain, AOTX_SEQ_SLOTS, role, &applied,
                                  &failed);
        aotx_decode_test_case_stop(&pump, gear, role, 1u, &applied, &failed);
        aotx_decode_test_case_stop(&pump, gear, role, AOTX_SEQ_SLOTS, &applied, &failed);
        aotx_decode_test_case_text(&pump, gear, role, 1u, &applied, &failed);
        aotx_decode_test_case_text(&pump, gear, role, AOTX_SEQ_SLOTS, &applied, &failed);
        aotx_decode_test_case_long(&pump, gear, prompt, role, &applied, &failed);
        aotx_decode_test_case_replay(&pump, gear, drain, role, vocab, 1u, &applied,
                                     &failed);
        aotx_decode_test_case_replay(&pump, gear, drain, role, vocab, AOTX_SEQ_SLOTS,
                                     &applied, &failed);
        aotx_decode_test_case_feed(&pump, gear, drain, prompt, &rings, boot_id, role,
                                   &applied, &failed);
        aotx_decode_test_table(&pump, gear, drain, role, aotx_role_name[role], &applied,
                               &failed);
        if (r == 0u) {
            aotx_decode_test_case_graph(&pump, gear, role, 1000u, &applied, &failed);
        }
        aotx_pump_close(&pump);
    }

    drain->stop = 1;
    pthread_join(thread, NULL);
    unsigned int refused = 0u;
    aotx_check_runtime(cudaMemcpy(&refused, gear->bad, sizeof refused,
                                  cudaMemcpyDeviceToHost), "cudaMemcpy");
    applied += 1u;
    if (refused != 0u || drain->lost != 0u || drain->bad != 0ull) {
        printf("decode: %u calls refused, %u records lost, %llu blocks bad\n", refused,
               drain->lost, drain->bad);
        failed += 1u;
    }
    printf("decode: %u token records and %u sequence events over %llu blocks\n",
           drain->taken, drain->events, drain->blocks);
    printf("decode: %u cases applied, %u failed\n", applied, failed);
    aotx_seam_finish(&rings);
    aotx_seam_close(&rings);
    return (failed == 0u) ? 0 : 1;
}
