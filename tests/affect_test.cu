/* Purpose: Check the probe loader, the read branch of the conduct kernel and the turn node.
 * Owns: The fixture store, the synthetic residual batch, the device ring and the counts.
 * Launch shape: The conduct kernel at one block for each row; the turn node and the
 *   scripts at one thread for each agent.
 * Lifetime: One run of the test program. */
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "affect/affect.cuh"
#include "agent/agent_state.cuh"
#include "boot/check.h"
#include "model/conduct.cuh"
#include "model/decode_state.cuh"
#include "seam/seam.cuh"

#define AOTX_AFFECT_TEST_ROLE        AOTX_MODEL_LANGUAGE
#define AOTX_AFFECT_TEST_HIDDEN      256u
#define AOTX_AFFECT_TEST_LAYERS      36u
#define AOTX_AFFECT_TEST_LAYER       12u
#define AOTX_AFFECT_TEST_GUARD_LAYER 16u
#define AOTX_AFFECT_TEST_ROWS        5u     /* rows each sequence gives the batch */
#define AOTX_AFFECT_TEST_PIECE       3u     /* rows before the last prompt row */
#define AOTX_AFFECT_TEST_SLOTS       128u   /* records the device ring holds */
#define AOTX_AFFECT_TEST_PATH        1024u
#define AOTX_AFFECT_TEST_FILES       8u
#define AOTX_AFFECT_TEST_TICK        41ull
#define AOTX_AFFECT_TEST_NEAR        1.0e-3

/* The bits the agent step marks, which the script marks in their place. */
#define AOTX_AFFECT_TEST_MARKS       0x4bf8u

static unsigned int aotx_affect_cases = 0u;
static unsigned int aotx_affect_bad = 0u;

static void aotx_affect_note(const char *name, int good, const char *how, double value,
                             double bound)
{
    aotx_affect_cases += 1u;
    if (!good) {
        aotx_affect_bad += 1u;
    }
    printf("%-56s %-4s %s %.4f of %.4f\n", name, good ? "ok" : "BAD", how, value, bound);
}

static void *aotx_affect_take(unsigned long long bytes)
{
    void *block = 0;
    aotx_check_runtime(cudaMalloc(&block, (size_t)bytes), "cudaMalloc");
    aotx_check_runtime(cudaMemset(block, 0, (size_t)bytes), "cudaMemset");
    return block;
}

/* The multiple of the direction of one axis in one row of one sequence. */
__host__ __device__ static float aotx_affect_test_multiple(unsigned int seq, unsigned int at,
                                                           unsigned int axis)
{
    return (float)(seq % 7u + 1u) * (float)(at + 1u) * 0.25f * (float)(axis + 1u) - 1.0f;
}

__host__ __device__ static unsigned int aotx_affect_test_prompt(unsigned int seq)
{
    return 8u + seq % 3u;
}

__host__ __device__ static unsigned int aotx_affect_test_agent(unsigned int seq,
                                                               unsigned int count)
{
    return count - 1u - seq;
}

#include "affect_store.h"

/* Clear the sums of every agent. */
static void aotx_affect_test_clear(void)
{
    static aotx_affect_sums none[AOTX_SLOTS];
    memset(none, 0, sizeof none);
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_affect_acc, none, sizeof none),
                       "cudaMemcpyToSymbol");
}

static void aotx_affect_test_sums(aotx_affect_sums *acc)
{
    aotx_check_runtime(cudaMemcpyFromSymbol(acc, aotx_affect_acc,
                                            AOTX_SLOTS * sizeof *acc),
                       "cudaMemcpyFromSymbol");
}

/* Put the prompt length and the first row position of every sequence in the tables the
 * read branch reads. One thread for each sequence. */
__global__ void aotx_affect_test_batch(unsigned int count)
{
    unsigned int seq = threadIdx.x;
    if (seq >= count) {
        return;
    }
    unsigned int agent = aotx_affect_test_agent(seq, count);
    aotx_seqs.slot[agent].prompt = aotx_affect_test_prompt(seq);
    aotx_decode.first[agent] = aotx_affect_test_prompt(seq) - AOTX_AFFECT_TEST_PIECE;
}

static int aotx_affect_test_near(float got, float want)
{
    return fabs((double)got - (double)want) < AOTX_AFFECT_TEST_NEAR;
}

/* The read branch on a batch of count sequences of five rows each. Rows are sums of the
 * four directions with known multiples. The middle row is the last prompt row; two rows
 * come before it and two reply rows after it. */
static void aotx_affect_test_readout(unsigned int count)
{
    unsigned int hidden = AOTX_AFFECT_TEST_HIDDEN;
    unsigned int rows = count * AOTX_AFFECT_TEST_ROWS;
    float direction[4][AOTX_AFFECT_TEST_HIDDEN];
    float *host = (float *)calloc((size_t)rows * hidden, sizeof(float));
    unsigned int offset[AOTX_SLOTS + 1u];
    unsigned int agent[AOTX_SLOTS];
    aotx_model_how how[AOTX_SLOTS];
    aotx_model_work work;
    aotx_model_run run;
    aotx_affect_sums acc[AOTX_SLOTS];
    aotx_affect_sums again[AOTX_SLOTS];
    char label[96];
    for (unsigned int k = 0u; k < 4u; ++k) {
        aotx_affect_test_direction(aotx_affect_test_axis[k], hidden, direction[k]);
    }
    memset(how, 0, sizeof how);
    for (unsigned int s = 0u; s < count; ++s) {
        offset[s] = s * AOTX_AFFECT_TEST_ROWS;
        agent[s] = aotx_affect_test_agent(s, count);
        how[s].affect = (count == 1u || s % 4u != 3u) ? 1u : 0u;
        how[s].steer[0] = AOTX_MODEL_CONDUCT_NONE;
        how[s].steer[1] = AOTX_MODEL_CONDUCT_NONE;
        how[s].voice = AOTX_MODEL_CONDUCT_NONE;
        for (unsigned int r = 0u; r < AOTX_AFFECT_TEST_ROWS; ++r) {
            float *row = host + (size_t)(offset[s] + r) * hidden;
            for (unsigned int k = 0u; k < 4u; ++k) {
                float multiple = aotx_affect_test_multiple(s, r, aotx_affect_test_axis[k]);
                for (unsigned int i = 0u; i < hidden; ++i) {
                    row[i] += multiple * direction[k][i];
                }
            }
        }
    }
    offset[count] = rows;
    float *resid = (float *)aotx_affect_take((unsigned long long)rows * hidden
                                             * sizeof(float));
    unsigned int *device_offset =
        (unsigned int *)aotx_affect_take((count + 1u) * sizeof(unsigned int));
    unsigned int *device_agent = (unsigned int *)aotx_affect_take(count * sizeof(unsigned int));
    aotx_model_how *device_how = (aotx_model_how *)aotx_affect_take(count * sizeof *how);
    aotx_check_runtime(cudaMemcpy(resid, host, (size_t)rows * hidden * sizeof(float),
                                  cudaMemcpyHostToDevice), "cudaMemcpy");
    aotx_check_runtime(cudaMemcpy(device_offset, offset, (count + 1u) * sizeof *offset,
                                  cudaMemcpyHostToDevice), "cudaMemcpy");
    aotx_check_runtime(cudaMemcpy(device_agent, agent, count * sizeof *agent,
                                  cudaMemcpyHostToDevice), "cudaMemcpy");
    aotx_check_runtime(cudaMemcpy(device_how, how, count * sizeof *how,
                                  cudaMemcpyHostToDevice), "cudaMemcpy");
    memset(&work, 0, sizeof work);
    memset(&run, 0, sizeof run);
    work.resid = resid;
    run.offset = device_offset;
    run.agent = device_agent;
    run.how = device_how;
    run.seqs = count;
    run.tokens = rows;
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_model_space, &work, sizeof work,
                                          (size_t)AOTX_AFFECT_TEST_ROLE * sizeof work),
                       "cudaMemcpyToSymbol");
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_model_call, &run, sizeof run,
                                          (size_t)AOTX_AFFECT_TEST_ROLE * sizeof run),
                       "cudaMemcpyToSymbol");
    aotx_affect_test_clear();
    aotx_affect_test_batch<<<1, AOTX_SLOTS>>>(count);
    aotx_model_conduct<<<AOTX_DECODE_WAVE, AOTX_MODEL_ROW_THREADS>>>(AOTX_AFFECT_TEST_ROLE,
                                                                     AOTX_AFFECT_TEST_LAYER);
    aotx_model_conduct<<<AOTX_DECODE_WAVE, AOTX_MODEL_ROW_THREADS>>>(
        AOTX_AFFECT_TEST_ROLE, AOTX_AFFECT_TEST_GUARD_LAYER);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_affect_test_sums(acc);

    /* A layer with no probe row and a pass with no how rows leave the sums as they are. */
    aotx_model_conduct<<<AOTX_DECODE_WAVE, AOTX_MODEL_ROW_THREADS>>>(AOTX_AFFECT_TEST_ROLE,
                                                                     3u);
    run.how = 0;
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_model_call, &run, sizeof run,
                                          (size_t)AOTX_AFFECT_TEST_ROLE * sizeof run),
                       "cudaMemcpyToSymbol");
    aotx_model_conduct<<<AOTX_DECODE_WAVE, AOTX_MODEL_ROW_THREADS>>>(AOTX_AFFECT_TEST_ROLE,
                                                                     AOTX_AFFECT_TEST_LAYER);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_affect_test_sums(again);

    unsigned int right = 0u;
    unsigned int quiet = 0u;
    unsigned int wanted_quiet = 0u;
    for (unsigned int s = 0u; s < count; ++s) {
        const aotx_affect_sums *one = &acc[agent[s]];
        if (how[s].affect == 0u) {
            aotx_affect_sums none;
            memset(&none, 0, sizeof none);
            wanted_quiet += 1u;
            quiet += (memcmp(one, &none, sizeof none) == 0) ? 1u : 0u;
            continue;
        }
        unsigned int good = 1u;
        for (unsigned int k = 0u; k < 4u; ++k) {
            unsigned int axis = aotx_affect_test_axis[k];
            float mean = aotx_affect_test_mean(axis);
            float scale = aotx_affect_test_scale(axis);
            float prompt = (aotx_affect_test_multiple(s, AOTX_AFFECT_TEST_PIECE - 1u, axis)
                            - mean) / scale;
            float reply = 0.0f;
            for (unsigned int r = AOTX_AFFECT_TEST_PIECE; r < AOTX_AFFECT_TEST_ROWS; ++r) {
                reply += (aotx_affect_test_multiple(s, r, axis) - mean) / scale;
            }
            good &= aotx_affect_test_near(one->prompt[axis], prompt) ? 1u : 0u;
            good &= aotx_affect_test_near(one->reply_sum[axis], reply) ? 1u : 0u;
        }
        good &= (one->prompt[2] == 0.0f && one->prompt[3] == 0.0f
                 && one->reply_sum[2] == 0.0f && one->reply_sum[3] == 0.0f) ? 1u : 0u;
        good &= (one->reply_rows == AOTX_AFFECT_TEST_ROWS - AOTX_AFFECT_TEST_PIECE) ? 1u : 0u;
        good &= (one->sampled == 0u && one->logprob_sum == 0.0f) ? 1u : 0u;
        right += good;
    }
    snprintf(label, sizeof label, "readouts and row counts of %u sequences", count);
    aotx_affect_note(label, right == count - wanted_quiet, "agents", (double)right,
                     (double)(count - wanted_quiet));
    if (wanted_quiet != 0u) {
        snprintf(label, sizeof label, "a sequence with no affect mark reads nothing at %u",
                 count);
        aotx_affect_note(label, quiet == wanted_quiet, "agents", (double)quiet,
                         (double)wanted_quiet);
    }
    snprintf(label, sizeof label, "a layer with no row and a pass with no how change nothing at %u",
             count);
    aotx_affect_note(label, memcmp(acc, again, sizeof acc) == 0, "tables",
                     (memcmp(acc, again, sizeof acc) == 0) ? 1.0 : 0.0, 1.0);
    cudaFree(resid);
    cudaFree(device_offset);
    cudaFree(device_agent);
    cudaFree(device_how);
    free(host);
}

/* The script of one agent for the turn node: its gear, its record, its sequence and its
 * sums. One formula gives it; the device writes it and the host reads it. */
typedef struct aotx_affect_test_plan {
    unsigned int flag;
    unsigned int last_token;
    unsigned int limit_end;
    unsigned int stopped;
    unsigned int budget_left;
    unsigned int out_tokens;
    unsigned int think_tokens;
    unsigned int turn;
    unsigned int marks;
    unsigned int reply_rows;
    unsigned int sampled;
    float prompt[AOTX_AFFECT_AXES];
    float reply_sum[AOTX_AFFECT_AXES];
    float logprob_sum;
    float entropy_sum;
} aotx_affect_test_plan;

__host__ __device__ static void aotx_affect_test_script_of(unsigned int a, unsigned int count,
                                                           aotx_affect_test_plan *p)
{
    p->flag = (count == 1u || a % 3u != 2u) ? 1u : 0u;
    p->last_token = (a % 3u == 0u) ? 1u : 0u;
    p->limit_end = (a % 3u == 1u) ? 1u : 0u;
    p->stopped = (a % 5u == 4u) ? 1u : 0u;
    p->budget_left = (a % 4u == 0u) ? 0u : 3u;
    p->out_tokens = (a == 5u) ? 0u : 10u + a;
    p->think_tokens = ((a & 1u) != 0u) ? 7u + a : 2u;
    p->turn = a + 3u;
    p->marks = 0u;
    for (unsigned int b = 0u; b <= AOTX_AFFECT_EVENT_VERDICT_REFUTE; ++b) {
        if (((AOTX_AFFECT_TEST_MARKS >> b) & 1u) != 0u && (((a >> (b % 5u)) ^ b) & 1u) == 0u) {
            p->marks |= 1u << b;
        }
    }
    p->reply_rows = 3u + (a % 2u);
    p->sampled = 4u + (a % 3u);
    for (unsigned int k = 0u; k < AOTX_AFFECT_AXES; ++k) {
        p->prompt[k] = 0.25f * (float)(a + 1u) * (float)(k + 1u);
        p->reply_sum[k] = 1.5f * (float)(a + 1u) * (float)(k + 1u);
    }
    p->logprob_sum = (((a & 1u) != 0u) ? -1.0f : -2.0f) * (float)p->sampled;
    p->entropy_sum = (a == 7u) ? -1.0e-6f : 0.5f * (float)(a + 1u) * (float)p->sampled;
}

/* Write the script of every agent under the count into the device state. */
__global__ void aotx_affect_test_script(unsigned int count, unsigned int ended)
{
    unsigned int a = threadIdx.x;
    if (a >= count) {
        return;
    }
    aotx_affect_test_plan p;
    aotx_affect_test_script_of(a, count, &p);
    aotx_agent_gear[a].last_token = p.last_token;
    aotx_agent_gear[a].limit_end = p.limit_end;
    aotx_agent_gear[a].stopped = p.stopped;
    aotx_agent_gear[a].out_tokens = p.out_tokens;
    aotx_agents.agent[a].budget_left = p.budget_left;
    aotx_agents.agent[a].turn = p.turn;
    aotx_seqs.slot[a].think_tokens = p.think_tokens;
    aotx_affect_sums *acc = &aotx_affect_acc[a];
    for (unsigned int k = 0u; k < AOTX_AFFECT_AXES; ++k) {
        acc->prompt[k] = p.prompt[k];
        acc->reply_sum[k] = p.reply_sum[k];
    }
    acc->reply_rows = p.reply_rows;
    acc->sampled = p.sampled;
    acc->logprob_sum = p.logprob_sum;
    acc->entropy_sum = p.entropy_sum;
    acc->flag = p.flag;
    acc->events = 0u;
    for (unsigned int b = 0u; b <= AOTX_AFFECT_EVENT_VERDICT_REFUTE; ++b) {
        if (((p.marks >> b) & 1u) != 0u) {
            aotx_affect_mark(a, b);
        }
    }
    if (ended != 0u) {
        aotx_affect_end(a);
    }
}

/* The reason mask the node must give for one script. */
static unsigned int aotx_affect_test_reason(const aotx_affect_test_plan *p)
{
    unsigned int think = (p->out_tokens != 0u) ? p->think_tokens : 0u;
    unsigned int mask = p->marks;
    mask |= (p->last_token != 0u) ? (1u << AOTX_AFFECT_EVENT_STOP) : 0u;
    mask |= (p->limit_end != 0u) ? (1u << AOTX_AFFECT_EVENT_LIMIT) : 0u;
    mask |= (p->stopped != 0u) ? (1u << AOTX_AFFECT_EVENT_OPERATOR_STOP) : 0u;
    mask |= (p->budget_left == 0u) ? (1u << AOTX_AFFECT_EVENT_BUDGET) : 0u;
    mask |= (think * 2u > p->out_tokens) ? (1u << AOTX_AFFECT_EVENT_THINK_RATIO) : 0u;
    mask |= (p->logprob_sum / (float)p->sampled < AOTX_AFFECT_LOGPROB_BOUND)
          ? (1u << AOTX_AFFECT_EVENT_LOW_LOGPROB) : 0u;
    return mask;
}

/* The device ring the node writes into, and the records read back from it. */
typedef struct aotx_affect_test_ring {
    unsigned char *device;
    unsigned char *records;
} aotx_affect_test_ring;

static void aotx_affect_test_ring_open(aotx_affect_test_ring *ring, unsigned int replaying)
{
    aotx_seam_state seam;
    unsigned long long tick = AOTX_AFFECT_TEST_TICK;
    size_t bytes = (size_t)AOTX_AFFECT_TEST_SLOTS * AOTX_SLOT_BYTES;
    memset(&seam, 0, sizeof seam);
    if (ring->device == 0) {
        ring->device = (unsigned char *)aotx_affect_take(bytes);
        ring->records = (unsigned char *)calloc(1, bytes);
    }
    aotx_check_runtime(cudaMemset(ring->device, 0, bytes), "cudaMemset");
    seam.dev.base = ring->device;
    seam.dev.slot_count = AOTX_AFFECT_TEST_SLOTS;
    seam.dev.mask = AOTX_AFFECT_TEST_SLOTS - 1u;
    seam.boot_id = 0x00a0260000000002ull;
    seam.replaying = replaying;
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_seam, &seam, sizeof seam),
                       "cudaMemcpyToSymbol");
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_time_tick, &tick, sizeof tick),
                       "cudaMemcpyToSymbol");
}

static void aotx_affect_test_ring_read(aotx_affect_test_ring *ring)
{
    aotx_check_runtime(cudaMemcpy(ring->records, ring->device,
                                  (size_t)AOTX_AFFECT_TEST_SLOTS * AOTX_SLOT_BYTES,
                                  cudaMemcpyDeviceToHost), "cudaMemcpy");
}

static void aotx_affect_test_ring_shut(aotx_affect_test_ring *ring)
{
    aotx_seam_state clear;
    memset(&clear, 0, sizeof clear);
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_seam, &clear, sizeof clear),
                       "cudaMemcpyToSymbol");
    cudaFree(ring->device);
    free(ring->records);
    memset(ring, 0, sizeof *ring);
}

/* One trace record against its script: the header, the body and the refusal rules. */
static int aotx_affect_test_record(const aotx_record_header *header,
                                   const aotx_affect_trace_body *body, unsigned int count,
                                   unsigned int flags)
{
    aotx_affect_test_plan p;
    if (body->agent >= count) {
        return 0;
    }
    aotx_affect_test_script_of(body->agent, count, &p);
    int head = header->cls == AOTX_CLASS_B && header->type == AOTX_REC_AFFECT_TRACE
            && header->body_len == sizeof *body && header->tick == AOTX_AFFECT_TEST_TICK
            && header->writer == AOTX_WRITER_AGENT_BASE + body->agent;
    int good = head && body->turn == p.turn && body->rows == p.reply_rows
            && body->think == ((p.out_tokens != 0u) ? p.think_tokens : 0u)
            && body->reason == aotx_affect_test_reason(&p) && body->flags == flags;
    for (unsigned int k = 0u; k < AOTX_AFFECT_TRACE_AXES; ++k) {
        good = good && aotx_affect_test_near(body->prompt[k], p.prompt[k])
            && aotx_affect_test_near(body->reply[k], p.reply_sum[k] / (float)p.reply_rows)
            && body->effective[k] == 0;
    }
    for (unsigned int k = 0u; k < AOTX_AFFECT_AXES - AOTX_AFFECT_GUARD_AXIS; ++k) {
        good = good && aotx_affect_test_near(body->guard[k],
                                             p.reply_sum[AOTX_AFFECT_GUARD_AXIS + k]
                                             / (float)p.reply_rows);
    }
    float entropy = p.entropy_sum / (float)p.sampled;
    good = good && aotx_affect_test_near(body->logprob, p.logprob_sum / (float)p.sampled)
        && aotx_affect_test_near(body->entropy, (entropy < 0.0f) ? 0.0f : entropy);
    /* The rules of the drain: the agent, the mask, the flags, the entropy, every float. */
    int taken = body->agent < 64u && (body->reason & ~AOTX_AFFECT_EVENT_MASK) == 0u
             && (body->flags & ~0xfu) == 0u && body->entropy >= 0.0f
             && isfinite(body->logprob) && isfinite(body->entropy);
    for (unsigned int k = 0u; k < AOTX_AFFECT_TRACE_AXES; ++k) {
        taken = taken && isfinite(body->prompt[k]) && isfinite(body->reply[k]);
    }
    taken = taken && isfinite(body->guard[0]) && isfinite(body->guard[1]);
    return good && taken;
}

/* The turn node over count scripted agents, with or without probe rows loaded. */
static void aotx_affect_test_turn(aotx_affect_test_ring *ring, unsigned int count,
                                  unsigned int probes)
{
    aotx_affect_sums acc[AOTX_SLOTS];
    aotx_affect_sums none;
    char label[96];
    unsigned int flags = (probes != 0u) ? AOTX_AFFECT_FLAG_PROBES : 0u;
    unsigned int wanted = 0u;
    unsigned int every = 0u;
    for (unsigned int a = 0u; a < count; ++a) {
        aotx_affect_test_plan p;
        aotx_affect_test_script_of(a, count, &p);
        wanted += p.flag;
        every |= (p.flag != 0u) ? aotx_affect_test_reason(&p) : 0u;
    }
    memset(&none, 0, sizeof none);
    aotx_affect_test_ring_open(ring, 0u);
    aotx_affect_test_clear();
    aotx_affect_test_script<<<1, AOTX_SLOTS>>>(count, 1u);
    aotx_affect_turn<<<1, AOTX_SLOTS>>>();
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_affect_test_ring_read(ring);
    aotx_affect_test_sums(acc);
    unsigned int records = 0u;
    unsigned int right = 0u;
    unsigned int seen[AOTX_SLOTS];
    memset(seen, 0, sizeof seen);
    for (unsigned int i = 0u; i < AOTX_AFFECT_TEST_SLOTS; ++i) {
        const aotx_record_header *header =
            (const aotx_record_header *)(ring->records + (size_t)i * AOTX_SLOT_BYTES);
        const aotx_affect_trace_body *body =
            (const aotx_affect_trace_body *)((const unsigned char *)header
                                             + AOTX_HEADER_BYTES);
        if (header->magic != AOTX_WIRE_MAGIC || header->type != AOTX_REC_AFFECT_TRACE) {
            continue;
        }
        records += 1u;
        if (body->agent < AOTX_SLOTS && seen[body->agent] == 0u) {
            seen[body->agent] = 1u;
            right += aotx_affect_test_record(header, body, count, flags) ? 1u : 0u;
        }
    }
    unsigned int silent = 0u;
    unsigned int cleared = 0u;
    for (unsigned int a = 0u; a < count; ++a) {
        aotx_affect_test_plan p;
        aotx_affect_test_script_of(a, count, &p);
        silent += (p.flag == 0u && seen[a] == 0u) ? 1u : 0u;
        cleared += (memcmp(&acc[a], &none, sizeof none) == 0) ? 1u : 0u;
    }
    snprintf(label, sizeof label, "one trace for each flagged agent of %u, probes %u", count,
             probes);
    aotx_affect_note(label, records == wanted && right == wanted, "records",
                     (double)right, (double)wanted);
    snprintf(label, sizeof label, "no trace for an agent with the flag 0 of %u", count);
    aotx_affect_note(label, silent == count - wanted, "agents", (double)silent,
                     (double)(count - wanted));
    snprintf(label, sizeof label, "the sums of %u agents are cleared after the node", count);
    aotx_affect_note(label, cleared == count, "agents", (double)cleared, (double)count);
    if (count == AOTX_SLOTS) {
        aotx_affect_note("every event bit is exercised over the batch",
                         every == AOTX_AFFECT_EVENT_MASK, "mask", (double)every,
                         (double)AOTX_AFFECT_EVENT_MASK);
    }

    /* A second launch with no turn ended writes nothing. */
    aotx_affect_test_ring_open(ring, 0u);
    aotx_affect_turn<<<1, AOTX_SLOTS>>>();
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_affect_test_ring_read(ring);
    records = 0u;
    for (unsigned int i = 0u; i < AOTX_AFFECT_TEST_SLOTS; ++i) {
        const aotx_record_header *header =
            (const aotx_record_header *)(ring->records + (size_t)i * AOTX_SLOT_BYTES);
        records += (header->magic == AOTX_WIRE_MAGIC) ? 1u : 0u;
    }
    snprintf(label, sizeof label, "a tick with no turn ended writes no record at %u", count);
    aotx_affect_note(label, records == 0u, "records", (double)records, 0.0);

    /* A replay writes no trace and takes the turn. */
    aotx_affect_test_ring_open(ring, 1u);
    aotx_affect_test_clear();
    aotx_affect_test_script<<<1, AOTX_SLOTS>>>(count, 1u);
    aotx_affect_turn<<<1, AOTX_SLOTS>>>();
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_affect_test_ring_read(ring);
    aotx_affect_test_sums(acc);
    records = 0u;
    cleared = 0u;
    for (unsigned int i = 0u; i < AOTX_AFFECT_TEST_SLOTS; ++i) {
        const aotx_record_header *header =
            (const aotx_record_header *)(ring->records + (size_t)i * AOTX_SLOT_BYTES);
        records += (header->magic == AOTX_WIRE_MAGIC) ? 1u : 0u;
    }
    for (unsigned int a = 0u; a < count; ++a) {
        cleared += (memcmp(&acc[a], &none, sizeof none) == 0) ? 1u : 0u;
    }
    snprintf(label, sizeof label, "a replay writes no trace and clears %u agents", count);
    aotx_affect_note(label, records == 0u && cleared == count, "agents", (double)cleared,
                     (double)count);
}

int main(void)
{
    static const unsigned int counts[2] = { 1u, AOTX_SLOTS };
    aotx_affect_test_store store;
    aotx_affect_test_ring ring;
    aotx_check_runtime(cudaFree(0), "cudaFree");
    memset(&ring, 0, sizeof ring);
    if (aotx_affect_test_open_store(&store) != 0) {
        printf("affect: the fixture store did not open\n");
        return 1;
    }
    aotx_affect_test_model();
    aotx_affect_test_loader(&store);
    for (unsigned int c = 0u; c < 2u; ++c) {
        aotx_affect_test_readout(counts[c]);
    }
    for (unsigned int c = 0u; c < 2u; ++c) {
        aotx_affect_test_turn(&ring, counts[c], 1u);
    }
    aotx_affect_release();
    for (unsigned int c = 0u; c < 2u; ++c) {
        aotx_affect_test_turn(&ring, counts[c], 0u);
    }
    aotx_affect_test_ring_shut(&ring);
    aotx_affect_test_shut_store(&store);
    printf("affect: %u cases, %u bad, 0 skipped, at 1 and %u\n", aotx_affect_cases,
           aotx_affect_bad, AOTX_SLOTS);
    return (aotx_affect_bad == 0u) ? 0 : 1;
}
