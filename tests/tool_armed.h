/* Purpose: Check the arm of a call over a run of agents, through the agent step alone.
 * Owns: The spawned agents of the case and the slots they took.
 * Launch shape: One thread for each spawned agent; the agent step at one block of one
 *   thread for each agent slot.
 * Lifetime: One case of the tool check. */
#ifndef AOTX_TESTS_TOOL_ARMED_H
#define AOTX_TESTS_TOOL_ARMED_H

#include "cli/prompt.cuh"
#include "tool/policy.cuh"

/* Spawn a run of agents of a role; the slot of each one goes out. */
__global__ void aotx_tool_test_spawn(unsigned int role, unsigned int count, unsigned int *out,
                                     unsigned long long tick)
{
    if (blockIdx.x != 0u || threadIdx.x != 0u) {
        return;
    }
    for (unsigned int i = 0u; i < count; ++i) {
        out[i] = aotx_agent_spawn(role, 0u, tick);
    }
}

/* Give the spawned agents back, so the next case spawns them again. */
__global__ void aotx_tool_test_release(const unsigned int *slots, unsigned int count)
{
    unsigned int i = threadIdx.x;
    if (i >= count || slots[i] >= AOTX_SLOTS) {
        return;
    }
    aotx_agents.agent[slots[i]].state = AOTX_AGENT_STATE_FREE;
    aotx_agents.agent[slots[i]].task = ~0u;
    aotx_say.slot[slots[i]].wanted = 0u;
    if (aotx_agents.live > 0u) {
        atomicSub(&aotx_agents.live, 1u);
    }
}

/* Put the spawned agents at the end of a turn, with the arm of a call on each. The reply
 * of the turn holds a call to memory_recall, or plain text. The agent step then ends the
 * turn as the model wrote it. */
__global__ void aotx_tool_test_armed_turn(const unsigned int *slots, unsigned int count,
                                          unsigned int with_call)
{
    unsigned int i = threadIdx.x;
    if (i >= count || slots[i] >= AOTX_SLOTS) {
        return;
    }
    unsigned int agent = slots[i];
    aotx_agent *me = &aotx_agents.agent[agent];
    aotx_agent_work *gear = &aotx_agent_gear[agent];
    const char *message = "note the plan";
    const char *reply = (with_call != 0u)
        ? "<tool_call>\n{\"name\": \"memory_recall\", \"arguments\": {\"text\": \"the plan\"}}\n</tool_call>"
        : "The plan stands.";
    me->state = AOTX_AGENT_STATE_POST;
    me->task = ~0u;
    aotx_tool_policy_capture(agent, me->role);
    gear->kind = AOTX_AGENT_TURN_MESSAGE;
    gear->source_seq = 0ull;
    gear->stopped = 0u;
    gear->limit_end = 0u;
    gear->last_token = 1u;
    gear->out_tokens = 4u;
    gear->message_len = 0u;
    while (message[gear->message_len] != '\0') {
        gear->message[gear->message_len] = (unsigned char)message[gear->message_len];
        gear->message_len += 1u;
    }
    gear->reply_len = 0u;
    while (reply[gear->reply_len] != '\0') {
        gear->reply[gear->reply_len] = (unsigned char)reply[gear->reply_len];
        gear->reply_len += 1u;
    }
    gear->call.entry = AOTX_CATALOG_NO_ENTRY;
    gear->call.tool = AOTX_TOOL_NONE;
    gear->call.provenance = 0u;
    gear->call.arg_len = 0u;
    if (with_call != 0u && aotx_tool_parse(gear->reply, gear->reply_len, &gear->call) == 0) {
        gear->call.entry = AOTX_CATALOG_NO_ENTRY;
        gear->call.tool = AOTX_TOOL_NONE;
    }
    aotx_requests.slot[agent].request = 0u;
    aotx_requests.slot[agent].result_len = 0u;
    aotx_requests.slot[agent].status = AOTX_TOOL_OK;
    /* The prompt of the next turn needs the say slot free; an earlier case may have
     * left a prompt on it. */
    aotx_say.slot[agent].wanted = 0u;
    aotx_tool_done[agent] = 0u;
    aotx_tool_embed.state[agent] = AOTX_TOOL_EMBED_NONE;
#ifdef AOTX_AFFECT
    aotx_affect_acc[agent].events = 0u;
#endif
    aotx_tool_outcome_arm(agent, AOTX_TOOL_CALL_RESULT);
}

/* The arm of a call over a run of agents. A turn whose reply holds a call ends with the
 * ok result on the slot. The next step takes it as a real result with the tool ok event.
 * A turn with no call ends with no result, no event and the arm clear. The agent step
 * runs on its own, so the events stand for the read. */
static void aotx_tool_test_case_armed(unsigned int count, unsigned int *applied,
                                      unsigned int *failed)
{
    unsigned int worker = aotx_test_catalog_entry("worker", AOTX_MODULE_ROLE);
    unsigned int *slots = (unsigned int *)aotx_tool_test_take(count * sizeof(unsigned int));
    unsigned int host_slots[AOTX_SLOTS];
    unsigned long long tick = 0ull;
    aotx_request_table *requests = (aotx_request_table *)calloc(1, sizeof *requests);
    aotx_agent_table *agents = (aotx_agent_table *)calloc(1, sizeof *agents);
    unsigned int outcome[AOTX_SLOTS], done[AOTX_SLOTS];
    aotx_check_runtime(cudaMemcpyFromSymbol(&tick, aotx_time_tick, sizeof tick),
                       "cudaMemcpyFromSymbol");
    aotx_tool_test_spawn<<<1, 1>>>(worker, count, slots, tick);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_check_runtime(cudaMemcpy(host_slots, slots, count * sizeof(unsigned int),
                                  cudaMemcpyDeviceToHost), "cudaMemcpy");
    unsigned int live = 0u;
    for (unsigned int i = 0u; i < count; ++i) {
        live += (host_slots[i] < AOTX_SLOTS) ? 1u : 0u;
    }

    /* The turn with a call: the step ends it with the completed request, and the next
     * step takes the result. */
    aotx_tool_test_armed_turn<<<1, AOTX_SLOTS>>>(slots, count, 1u);
    aotx_agent_step<<<1, AOTX_SLOTS>>>(tick + 1ull);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_check_runtime(cudaMemcpyFromSymbol(requests, aotx_requests, sizeof *requests),
                       "cudaMemcpyFromSymbol");
    aotx_check_runtime(cudaMemcpyFromSymbol(agents, aotx_agents, sizeof *agents),
                       "cudaMemcpyFromSymbol");
    aotx_check_runtime(cudaMemcpyFromSymbol(outcome, aotx_tool_embed, sizeof outcome,
                                            offsetof(aotx_tool_embed_batch, outcome)),
                       "cudaMemcpyFromSymbol");
    aotx_check_runtime(cudaMemcpyFromSymbol(done, aotx_tool_done, sizeof done),
                       "cudaMemcpyFromSymbol");
    unsigned int completed = 0u;
    for (unsigned int i = 0u; i < count; ++i) {
        unsigned int a = host_slots[i];
        if (a >= AOTX_SLOTS) {
            continue;
        }
        const aotx_request *slot = &requests->slot[a];
        completed += (agents->agent[a].state == AOTX_AGENT_STATE_TOOL
                      && slot->request != 0u && slot->status == AOTX_TOOL_OK
                      && done[a] != 0u && outcome[a] == 0u
                      && slot->result_len == 20u
                      && strncmp(slot->result, "the fixture tool ran", 20u) == 0) ? 1u : 0u;
    }
    aotx_agent_step<<<1, AOTX_SLOTS>>>(tick + 2ull);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_check_runtime(cudaMemcpyFromSymbol(requests, aotx_requests, sizeof *requests),
                       "cudaMemcpyFromSymbol");
    aotx_check_runtime(cudaMemcpyFromSymbol(agents, aotx_agents, sizeof *agents),
                       "cudaMemcpyFromSymbol");
    unsigned int taken = 0u;
    unsigned int marked = 0u;
#ifdef AOTX_AFFECT
    aotx_affect_sums *sums = (aotx_affect_sums *)calloc(AOTX_SLOTS, sizeof *sums);
    aotx_check_runtime(cudaMemcpyFromSymbol(sums, aotx_affect_acc, AOTX_SLOTS * sizeof *sums),
                       "cudaMemcpyFromSymbol");
#endif
    for (unsigned int i = 0u; i < count; ++i) {
        unsigned int a = host_slots[i];
        if (a >= AOTX_SLOTS) {
            continue;
        }
        taken += (requests->slot[a].request == 0u
                  && agents->agent[a].state != AOTX_AGENT_STATE_TOOL) ? 1u : 0u;
#ifdef AOTX_AFFECT
        marked += ((sums[a].events & (1u << AOTX_AFFECT_EVENT_TOOL_OK)) != 0u) ? 1u : 0u;
#else
        marked += 1u;
#endif
    }
    /* The console agent holds one slot, so the wide case spawns one agent fewer. */
    *applied += 1u;
    if (live + 1u < count || completed != live || taken != live || marked != live) {
        printf("tool: the arm of a call at %u agents: %u live, %u completed the call, %u took "
               "the result, %u carry the tool ok event\n", count, live, completed, taken,
               marked);
        *failed += 1u;
    }

    /* The turn with no call: the step ends it with no result and no event, and the arm is
     * clear after it. */
    aotx_tool_test_armed_turn<<<1, AOTX_SLOTS>>>(slots, count, 0u);
    aotx_agent_step<<<1, AOTX_SLOTS>>>(tick + 3ull);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_check_runtime(cudaMemcpyFromSymbol(requests, aotx_requests, sizeof *requests),
                       "cudaMemcpyFromSymbol");
    aotx_check_runtime(cudaMemcpyFromSymbol(agents, aotx_agents, sizeof *agents),
                       "cudaMemcpyFromSymbol");
    aotx_check_runtime(cudaMemcpyFromSymbol(outcome, aotx_tool_embed, sizeof outcome,
                                            offsetof(aotx_tool_embed_batch, outcome)),
                       "cudaMemcpyFromSymbol");
#ifdef AOTX_AFFECT
    aotx_check_runtime(cudaMemcpyFromSymbol(sums, aotx_affect_acc, AOTX_SLOTS * sizeof *sums),
                       "cudaMemcpyFromSymbol");
#endif
    unsigned int quiet = 0u;
    for (unsigned int i = 0u; i < count; ++i) {
        unsigned int a = host_slots[i];
        if (a >= AOTX_SLOTS) {
            continue;
        }
        unsigned int good = (agents->agent[a].state == AOTX_AGENT_STATE_IDLE
                             && requests->slot[a].request == 0u
                             && requests->slot[a].result_len == 0u && outcome[a] == 0u) ? 1u : 0u;
#ifdef AOTX_AFFECT
        good &= ((sums[a].events & (1u << AOTX_AFFECT_EVENT_TOOL_OK)) == 0u) ? 1u : 0u;
#endif
        quiet += good;
    }
    *applied += 1u;
    if (quiet != live) {
        printf("tool: the arm of a call with no call at %u agents: %u of %u ended with no "
               "result, no event and the arm clear\n", count, quiet, live);
        *failed += 1u;
    }
    printf("tool: the arm of a call at %u agents: %u spawned, %u calls completed as ok with "
           "the tool ok event, %u turns with no call ended with no result and no event\n",
           count, live, taken, quiet);
    aotx_tool_test_release<<<1, AOTX_SLOTS>>>(slots, count);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
#ifdef AOTX_AFFECT
    free(sums);
#endif
    free(requests);
    free(agents);
    cudaFree(slots);
}

#endif
