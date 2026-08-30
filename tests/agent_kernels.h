/* Purpose: Give the agent check its kernels, its text batches and its waits.
 * Owns: The device memory of one case and the text lines it holds.
 * Threading: One host thread drives the cases; the kernels take one thread for each agent.
 * Lifetime: One run of the test program. */
#ifndef AOTX_TESTS_AGENT_KERNELS_H
#define AOTX_TESTS_AGENT_KERNELS_H

#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "agent/agent_state.cuh"
#include "agent/transcript.cuh"
#include "boot/check.h"
#include "catalog/catalog.cuh"
#include "cli/prompt.cuh"
#include "model/decode_state.cuh"
#include "sched/sched.cuh"
#include "seam_feed.h"
#include "tool/tool_state.cuh"

static double aotx_agent_test_now(void)
{
    struct timespec at;
    clock_gettime(CLOCK_MONOTONIC, &at);
    return (double)at.tv_sec + 1e-9 * (double)at.tv_nsec;
}

/* Spawn a run of agents from one thread, as the command layer does. */
__global__ void aotx_agent_test_spawn(unsigned int role, unsigned int count,
                                      unsigned int *out, unsigned long long tick)
{
    if (blockIdx.x != 0u || threadIdx.x != 0u) {
        return;
    }
    for (unsigned int i = 0u; i < count; ++i) {
        out[i] = aotx_agent_spawn(role, 0u, tick);
    }
}

/* Give a message to a run of agents, and open a task for a run of agents. */
__global__ void aotx_agent_test_message(const unsigned char *text,
                                        const unsigned int *start,
                                        const unsigned int *length, unsigned int first,
                                        unsigned int count, int *out,
                                        unsigned long long tick)
{
    if (blockIdx.x != 0u || threadIdx.x != 0u) {
        return;
    }
    for (unsigned int i = 0u; i < count; ++i) {
        out[i] = aotx_agent_message(first + i, text + start[i], length[i], tick);
    }
}

__global__ void aotx_agent_test_task(const unsigned char *text, const unsigned int *start,
                                     const unsigned int *length, unsigned int agent,
                                     unsigned int role, unsigned int count,
                                     unsigned int verify, unsigned int *out,
                                     unsigned long long tick)
{
    if (blockIdx.x != 0u || threadIdx.x != 0u) {
        return;
    }
    for (unsigned int i = 0u; i < count; ++i) {
        unsigned int who = (agent == ~0u) ? ~0u : (agent + i);
        out[i] = aotx_task_open(who, role, text + start[i], length[i], verify, tick);
    }
}

/* Put a reply in the hand of an agent, as the run state does when its sequence ends. The
 * arm that gives a fixed reply takes this path, so the loop runs with no model. */
__global__ void aotx_agent_test_force(unsigned int agent, const unsigned char *text,
                                      unsigned int length)
{
    if (blockIdx.x != 0u || threadIdx.x != 0u || agent >= AOTX_SLOTS) {
        return;
    }
    aotx_agent_work *gear = &aotx_agent_gear[agent];
    unsigned int bytes = (length > AOTX_AGENT_REPLY_BYTES) ? AOTX_AGENT_REPLY_BYTES
                                                           : length;
    for (unsigned int i = 0u; i < bytes; ++i) {
        gear->reply[i] = text[i];
    }
    gear->reply_len = bytes;
    aotx_tool_parse(gear->reply, gear->reply_len, &gear->call);

    /* The prompt of the turn is dropped and the sequence of the slot is ended. The
     * language model therefore gives no reply of its own to this turn. */
    aotx_say.slot[agent].wanted = 0u;
    aotx_say.slot[agent].live = 0u;
    if (aotx_seqs.slot[agent].state != AOTX_SEQ_STATE_FREE) {
        aotx_seqs.slot[agent].state = AOTX_SEQ_STATE_FREE;
        aotx_seqs.slot[agent].sampled = 0u;
        aotx_seqs.slot[agent].prompt = 0u;
        aotx_seq_kept[agent] = 0u;
        aotx_seq_shown[agent] = 0u;
        aotx_model_seen[agent] = 0u;
        if (aotx_seqs.live > 0u) {
            aotx_seqs.live -= 1u;
        }
    }
    aotx_agents.agent[agent].state = AOTX_AGENT_STATE_POST;
}

/* Put a reply in the hand of a run of agents, one line for each. One thread takes one
 * agent, so a wide case gives every agent of the batch its own reply in one launch. */
__global__ void aotx_agent_test_force_many(const unsigned char *text,
                                           const unsigned int *start,
                                           const unsigned int *length, unsigned int first,
                                           unsigned int count)
{
    unsigned int at = blockIdx.x * blockDim.x + threadIdx.x;
    if (at >= count || first + at >= AOTX_SLOTS) {
        return;
    }
    unsigned int agent = first + at;
    aotx_agent_work *gear = &aotx_agent_gear[agent];
    unsigned int bytes = (length[at] > AOTX_AGENT_REPLY_BYTES) ? AOTX_AGENT_REPLY_BYTES
                                                               : length[at];
    for (unsigned int i = 0u; i < bytes; ++i) {
        gear->reply[i] = text[start[at] + i];
    }
    gear->reply_len = bytes;
    aotx_tool_parse(gear->reply, gear->reply_len, &gear->call);
    aotx_say.slot[agent].wanted = 0u;
    aotx_say.slot[agent].live = 0u;
    if (aotx_seqs.slot[agent].state != AOTX_SEQ_STATE_FREE) {
        aotx_seqs.slot[agent].state = AOTX_SEQ_STATE_FREE;
        aotx_seqs.slot[agent].sampled = 0u;
        aotx_seqs.slot[agent].prompt = 0u;
        aotx_seq_kept[agent] = 0u;
        aotx_seq_shown[agent] = 0u;
        aotx_model_seen[agent] = 0u;
        atomicSub(&aotx_seqs.live, 1u);
    }
    aotx_agents.agent[agent].state = AOTX_AGENT_STATE_POST;
}

/* Put a reply in the hand of every agent of a run that has started a turn. An agent that
 * has not started one is left alone. An agent that took its reply is no longer in a turn.
 * A launch of each tick therefore gives every agent one reply and no more. */
__global__ void aotx_agent_test_force_ready(const unsigned char *text,
                                            const unsigned int *start,
                                            const unsigned int *length, unsigned int first,
                                            unsigned int count)
{
    unsigned int at = blockIdx.x * blockDim.x + threadIdx.x;
    if (at >= count || first + at >= AOTX_SLOTS) {
        return;
    }
    unsigned int agent = first + at;
    unsigned int state = aotx_agents.agent[agent].state;
    if (state != AOTX_AGENT_STATE_PROMPT && state != AOTX_AGENT_STATE_RUN) {
        return;
    }
    aotx_agent_work *gear = &aotx_agent_gear[agent];
    unsigned int bytes = (length[at] > AOTX_AGENT_REPLY_BYTES) ? AOTX_AGENT_REPLY_BYTES
                                                               : length[at];
    for (unsigned int i = 0u; i < bytes; ++i) {
        gear->reply[i] = text[start[at] + i];
    }
    gear->reply_len = bytes;
    aotx_tool_parse(gear->reply, gear->reply_len, &gear->call);
    aotx_say.slot[agent].wanted = 0u;
    aotx_say.slot[agent].live = 0u;
    if (aotx_seqs.slot[agent].state != AOTX_SEQ_STATE_FREE) {
        aotx_seqs.slot[agent].state = AOTX_SEQ_STATE_FREE;
        aotx_seqs.slot[agent].sampled = 0u;
        aotx_seqs.slot[agent].prompt = 0u;
        aotx_seq_kept[agent] = 0u;
        aotx_seq_shown[agent] = 0u;
        aotx_model_seen[agent] = 0u;
        atomicSub(&aotx_seqs.live, 1u);
    }
    aotx_agents.agent[agent].state = AOTX_AGENT_STATE_POST;
}

/* Open a run of sequences from one thread, as the say path does when its nodes have the
 * tokens of a prompt. The count of the opens that the table refused comes back. */
__global__ void aotx_agent_test_open_many(const int *ids, unsigned int stride,
                                          unsigned int slots, unsigned int count,
                                          unsigned int role, unsigned int limit,
                                          unsigned int *bad)
{
    if (blockIdx.x != 0u || threadIdx.x != 0u) {
        return;
    }
    unsigned int wrong = 0u;
    for (unsigned int s = 0u; s < slots; ++s) {
        if (aotx_seq_open(s, role, ids + (unsigned long long)s * stride, count, limit,
                          AOTX_KV_PAGES_EACH,
                          0x5EEDu + s, aotx_setting_count(AOTX_SET_TOP_K),
                          aotx_setting_fraction(AOTX_SET_TOP_P),
                          aotx_setting_fraction(AOTX_SET_TEMPERATURE),
                          aotx_time_tick) != 0) {
            wrong += 1u;
        }
    }
    *bad = wrong;
}

/* Answer a run of pending authorizations. The call belongs to the serial thread of the
 * command layer, so one thread answers them in order. */
__global__ void aotx_agent_test_auth_many(const unsigned int *id, unsigned int count,
                                          unsigned int granted, unsigned long long tick)
{
    if (blockIdx.x != 0u || threadIdx.x != 0u) {
        return;
    }
    for (unsigned int at = 0u; at < count; ++at) {
        aotx_agent_authorize(id[at], granted, tick);
    }
}

/* Put the deadline of a run of requests in the past. */
__global__ void aotx_agent_test_expire_many(unsigned int first, unsigned int count)
{
    unsigned int at = blockIdx.x * blockDim.x + threadIdx.x;
    if (at < count && first + at < AOTX_SLOTS) {
        aotx_requests.slot[first + at].deadline = 0ull;
    }
}

/* Put the deadline of one request in the past. */
__global__ void aotx_agent_test_expire(unsigned int slot)
{
    if (blockIdx.x == 0u && threadIdx.x == 0u && slot < AOTX_SLOTS) {
        aotx_requests.slot[slot].deadline = 0ull;
    }
}

/* Lower the turn budget of a role, so the budget arm does not need eight replies. */
__global__ void aotx_agent_test_budget(unsigned int role, unsigned int budget)
{
    if (blockIdx.x == 0u && threadIdx.x == 0u && role < AOTX_MODULE_SLOTS) {
        aotx_catalog.entry[role].role.budget = budget;
    }
}

/* Answer one pending authorization, as the console does. */
__global__ void aotx_agent_test_auth(unsigned int request, unsigned int granted, int *out,
                                     unsigned long long tick)
{
    if (blockIdx.x == 0u && threadIdx.x == 0u) {
        *out = aotx_agent_authorize(request, granted, tick);
    }
}

/* Apply a run of reply records, as the apply of the tick does. */
__global__ void aotx_agent_test_apply(const aotx_tool_reply_body *body, unsigned int count,
                                      int *out)
{
    if (blockIdx.x != 0u || threadIdx.x != 0u) {
        return;
    }
    for (unsigned int i = 0u; i < count; ++i) {
        out[i] = aotx_tool_reply_apply(&body[i]);
    }
}

/* Put every agent, task and request slot back in the free state. */
__global__ void aotx_agent_test_clear(void)
{
    unsigned int at = threadIdx.x;
    if (at >= AOTX_SLOTS) {
        return;
    }
    aotx_agents.agent[at].state = AOTX_AGENT_STATE_FREE;
    aotx_agents.agent[at].task = ~0u;
    aotx_agents.agent[at].request = 0u;
    aotx_agents.agent[at].turn = 0u;
    aotx_agent_gear[at].has_message = 0u;
    aotx_agent_gear[at].wrote = 0u;
    aotx_agent_gear[at].reply_len = 0u;
    aotx_requests.slot[at].request = 0u;
    aotx_requests.slot[at].auth = AOTX_AUTH_NONE;
    aotx_requests.slot[at].result_len = 0u;
    aotx_requests.slot[at].parts = 0u;
    aotx_requests.slot[at].parts_in = 0u;
    aotx_requests.slot[at].call_seq = 0ull;
    aotx_requests.slot[at].answer_seq = 0ull;
    aotx_requests.slot[at].result_seq = 0ull;
    aotx_tool_done[at] = 0u;
    aotx_tool_embed.state[at] = AOTX_TOOL_EMBED_NONE;
    aotx_say.slot[at].wanted = 0u;
    aotx_say.slot[at].live = 0u;
    aotx_say.slot[at].ready = 0u;
    aotx_say.slot[at].reply_first = 0ull;
    aotx_say.slot[at].reply_records = 0u;
    unsigned char *transcript = (unsigned char *)&aotx_transcript[at];
    for (unsigned int i = 0u; i < (unsigned int)sizeof(aotx_transcript_agent); ++i) {
        transcript[i] = 0u;
    }
    aotx_seqs.slot[at].state = AOTX_SEQ_STATE_FREE;
    aotx_seqs.slot[at].sampled = 0u;
    aotx_seqs.slot[at].prompt = 0u;
    aotx_seq_kept[at] = 0u;
    aotx_seq_shown[at] = 0u;
    aotx_model_seen[at] = 0u;
    aotx_kv_release(at);
    for (unsigned int t = at; t < AOTX_TASK_SLOTS; t += AOTX_SLOTS) {
        aotx_task_used[t] = 0u;
        aotx_agents.task[t].state = AOTX_TASK_PENDING;
        aotx_agents.task[t].agent = ~0u;
        aotx_agents.task[t].verifier = ~0u;
    }
    if (at == 0u) {
        aotx_agents.live = 0u;
        aotx_agents.tasks = 0u;
        aotx_seqs.live = 0u;
        aotx_requests.pending_auth = 0u;
        aotx_transcript_count.searches = 0ull;
        aotx_transcript_count.replay_selections = 0ull;
        aotx_transcript_count.embedded = 0ull;
        aotx_transcript_count.compacted = 0ull;
        aotx_transcript_count.text_refused = 0ull;
    }
}

static void *aotx_agent_test_take(size_t bytes)
{
    void *at = 0;
    aotx_check_runtime(cudaMalloc(&at, bytes), "cudaMalloc");
    aotx_check_runtime(cudaMemset(at, 0, bytes), "cudaMemset");
    return at;
}

/* The text batch of a case: one line for each agent, in one run of bytes. */
typedef struct aotx_agent_test_text {
    unsigned char *bytes;
    unsigned int *start;
    unsigned int *length;
} aotx_agent_test_text;

#define AOTX_AGENT_TEST_LINE  160u

static aotx_agent_test_text aotx_agent_test_lines(const char *pattern, unsigned int count)
{
    aotx_agent_test_text on;
    unsigned char *bytes =
        (unsigned char *)calloc(AOTX_SLOTS, AOTX_AGENT_TEST_LINE);
    unsigned int *start = (unsigned int *)calloc(AOTX_SLOTS, sizeof(unsigned int));
    unsigned int *length = (unsigned int *)calloc(AOTX_SLOTS, sizeof(unsigned int));
    for (unsigned int i = 0u; i < count; ++i) {
        start[i] = i * AOTX_AGENT_TEST_LINE;
        unsigned int made = (unsigned int)snprintf((char *)bytes + start[i],
                                                   AOTX_AGENT_TEST_LINE, pattern, i);
        length[i] = (made < AOTX_AGENT_TEST_LINE) ? made : (AOTX_AGENT_TEST_LINE - 1u);
    }
    on.bytes = (unsigned char *)aotx_agent_test_take(AOTX_SLOTS
                                                     * AOTX_AGENT_TEST_LINE);
    on.start = (unsigned int *)aotx_agent_test_take(AOTX_SLOTS * sizeof(unsigned int));
    on.length = (unsigned int *)aotx_agent_test_take(AOTX_SLOTS
                                                     * sizeof(unsigned int));
    aotx_check_runtime(cudaMemcpy(on.bytes, bytes, AOTX_SLOTS * AOTX_AGENT_TEST_LINE,
                                  cudaMemcpyHostToDevice), "cudaMemcpy");
    aotx_check_runtime(cudaMemcpy(on.start, start, AOTX_SLOTS * sizeof(unsigned int),
                                  cudaMemcpyHostToDevice), "cudaMemcpy");
    aotx_check_runtime(cudaMemcpy(on.length, length,
                                  AOTX_SLOTS * sizeof(unsigned int),
                                  cudaMemcpyHostToDevice), "cudaMemcpy");
    free(bytes);
    free(start);
    free(length);
    return on;
}

static void aotx_agent_test_free(aotx_agent_test_text *on)
{
    cudaFree(on->bytes);
    cudaFree(on->start);
    cudaFree(on->length);
}

/* Put one text on the device for a forced reply. */
static unsigned char *aotx_agent_test_bytes(const char *text, unsigned int *length)
{
    unsigned int span = (unsigned int)strlen(text);
    unsigned char *at = (unsigned char *)aotx_agent_test_take(span + 1u);
    aotx_check_runtime(cudaMemcpy(at, text, span, cudaMemcpyHostToDevice), "cudaMemcpy");
    *length = span;
    return at;
}

static aotx_agent_table *aotx_agent_test_read(aotx_agent_table *table)
{
    aotx_check_runtime(cudaMemcpyFromSymbol(table, aotx_agents, sizeof *table),
                       "cudaMemcpyFromSymbol");
    return table;
}

/* Run ticks until an agent reaches a state, or the allowance runs out. */
static unsigned int aotx_agent_test_until(aotx_pump *pump, unsigned int agent,
                                          unsigned int state, unsigned int ticks)
{
    unsigned int held = AOTX_AGENT_STATE_FREE;
    for (unsigned int t = 0u; t < ticks; ++t) {
        aotx_pump_tick(pump);
        aotx_check_runtime(cudaMemcpyFromSymbol(&held, aotx_agents, sizeof held,
                                                offsetof(aotx_agent_table, agent)
                                                + (size_t)agent * sizeof(aotx_agent)
                                                + offsetof(aotx_agent, state)),
                           "cudaMemcpyFromSymbol");
        if (held == state) {
            return t + 1u;
        }
    }
    return 0u;
}

/* Run ticks until an agent starts a turn, which is the prompt state or the run state. A
 * forced reply then takes the place of the reply the model would give. */
static unsigned int aotx_agent_test_turn(aotx_pump *pump, unsigned int agent,
                                         unsigned int ticks)
{
    unsigned int held = AOTX_AGENT_STATE_FREE;
    for (unsigned int t = 0u; t < ticks; ++t) {
        aotx_pump_tick(pump);
        aotx_check_runtime(cudaMemcpyFromSymbol(&held, aotx_agents, sizeof held,
                                                offsetof(aotx_agent_table, agent)
                                                + (size_t)agent * sizeof(aotx_agent)
                                                + offsetof(aotx_agent, state)),
                           "cudaMemcpyFromSymbol");
        if (held == AOTX_AGENT_STATE_PROMPT || held == AOTX_AGENT_STATE_RUN) {
            return t + 1u;
        }
    }
    return 0u;
}

/* Run ticks until every agent of a run has started a turn, and give the count that did. */
static unsigned int aotx_agent_test_turn_many(aotx_pump *pump, unsigned int first,
                                              unsigned int count, unsigned int ticks)
{
    aotx_agent_table *table = (aotx_agent_table *)calloc(1, sizeof *table);
    unsigned int ready = 0u;
    for (unsigned int t = 0u; t < ticks; ++t) {
        aotx_pump_tick(pump);
        aotx_check_runtime(cudaMemcpyFromSymbol(table, aotx_agents, sizeof *table),
                           "cudaMemcpyFromSymbol");
        ready = 0u;
        for (unsigned int i = 0u; i < count; ++i) {
            unsigned int state = table->agent[first + i].state;
            ready += (state == AOTX_AGENT_STATE_PROMPT || state == AOTX_AGENT_STATE_RUN)
                   ? 1u : 0u;
        }
        if (ready >= count) {
            break;
        }
    }
    free(table);
    return ready;
}

/* Run ticks until every agent of a run reaches a state, and give the count that did. */
static unsigned int aotx_agent_test_state_many(aotx_pump *pump, unsigned int first,
                                               unsigned int count, unsigned int state,
                                               unsigned int ticks)
{
    aotx_agent_table *table = (aotx_agent_table *)calloc(1, sizeof *table);
    unsigned int ready = 0u;
    for (unsigned int t = 0u; t < ticks; ++t) {
        aotx_pump_tick(pump);
        aotx_check_runtime(cudaMemcpyFromSymbol(table, aotx_agents, sizeof *table),
                           "cudaMemcpyFromSymbol");
        ready = 0u;
        for (unsigned int i = 0u; i < count; ++i) {
            ready += (table->agent[first + i].state == state) ? 1u : 0u;
        }
        if (ready >= count) {
            break;
        }
    }
    free(table);
    return ready;
}

/* Give every agent of a run its reply as soon as it starts a turn. The ticks run until
 * every one of them reaches a state. The language model never finishes a reply of its own
 * this way, so the arm runs the same in every run. */
static unsigned int aotx_agent_test_drive(aotx_pump *pump, unsigned int first,
                                          unsigned int count,
                                          const aotx_agent_test_text *reply,
                                          unsigned int state, unsigned int ticks)
{
    aotx_agent_table *table = (aotx_agent_table *)calloc(1, sizeof *table);
    unsigned int ready = 0u;
    for (unsigned int t = 0u; t < ticks; ++t) {
        aotx_agent_test_force_ready<<<1, AOTX_SLOTS>>>(reply->bytes, reply->start,
                                                             reply->length, first, count);
        aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        aotx_pump_tick(pump);
        aotx_check_runtime(cudaMemcpyFromSymbol(table, aotx_agents, sizeof *table),
                           "cudaMemcpyFromSymbol");
        ready = 0u;
        for (unsigned int i = 0u; i < count; ++i) {
            ready += (table->agent[first + i].state == state) ? 1u : 0u;
        }
        if (ready >= count) {
            break;
        }
    }
    free(table);
    return ready;
}

#endif
