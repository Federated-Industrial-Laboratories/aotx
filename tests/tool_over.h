/* Purpose: Check a call whose values do not fit the argument line, through the agent step.
 * Owns: The spawned agents of the case and the slots they took.
 * Launch shape: One thread for each spawned agent; the agent step at one block of one
 *   thread for each agent slot.
 * Lifetime: One case of the tool check. */
#ifndef AOTX_TESTS_TOOL_OVER_H
#define AOTX_TESTS_TOOL_OVER_H

#include "tool/policy.cuh"

/* The words the error result of such a call carries. The device side writes them. */
#define AOTX_TOOL_OVER_REASON \
    "the arguments of the call do not fit the tool line; make them shorter"

/* Put the spawned agents at the end of a turn. The reply of the turn holds one
 * memory_recall call whose text value is over the room of the argument line. The parse of
 * each turn goes out, so the check reads what the parser gave. An arm goes on the slot
 * when the case asks for one. */
__global__ void aotx_tool_test_over_turn(const unsigned int *slots, unsigned int count,
                                         unsigned int arm, int *parsed)
{
    unsigned int i = threadIdx.x;
    if (i >= count || slots[i] >= AOTX_SLOTS) {
        return;
    }
    unsigned int agent = slots[i];
    aotx_agent *me = &aotx_agents.agent[agent];
    aotx_agent_work *gear = &aotx_agent_gear[agent];
    const char *message = "note the plan";
    const char *head = "<tool_call>\n{\"name\": \"memory_recall\", \"arguments\": "
                       "{\"text\": \"";
    const char *body = "the long list of the notes of the table and the rows it holds ";
    const char *tail = "\"}}\n</tool_call>";
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
    unsigned int at = 0u;
    for (unsigned int b = 0u; head[b] != '\0'; ++b) {
        gear->reply[at++] = (unsigned char)head[b];
    }
    /* Six runs of the words give a value over the room, whatever the profile gives the
     * argument line. */
    for (unsigned int run = 0u; run < 6u; ++run) {
        for (unsigned int b = 0u; body[b] != '\0'; ++b) {
            gear->reply[at++] = (unsigned char)body[b];
        }
    }
    for (unsigned int b = 0u; tail[b] != '\0'; ++b) {
        gear->reply[at++] = (unsigned char)tail[b];
    }
    gear->reply_len = at;
    parsed[i] = aotx_tool_parse(gear->reply, gear->reply_len, &gear->call);
    aotx_requests.slot[agent].request = 0u;
    aotx_requests.slot[agent].result_len = 0u;
    aotx_requests.slot[agent].status = AOTX_TOOL_OK;
    aotx_say.slot[agent].wanted = 0u;
    aotx_tool_done[agent] = 0u;
    aotx_tool_embed.state[agent] = AOTX_TOOL_EMBED_NONE;
#ifdef AOTX_AFFECT
    aotx_affect_acc[agent].events = 0u;
#endif
    aotx_tool_embed.outcome[agent] = 0u;
    if (arm != 0u) {
        aotx_tool_outcome_arm(agent, AOTX_TOOL_CALL_RESULT);
    }
}

/* A call over the room of the argument line, over a run of agents. The turn ends with the
 * tool finish and an error result which names the cause. The next step takes that result
 * with the tool error event, as after any tool error. With an arm, the armed result stands
 * in for the call as it does for any call. */
static void aotx_tool_test_case_over(unsigned int count, unsigned int *applied,
                                     unsigned int *failed)
{
    unsigned int worker = aotx_test_catalog_entry("worker", AOTX_MODULE_ROLE);
    unsigned int *slots = (unsigned int *)aotx_tool_test_take(count * sizeof(unsigned int));
    int *parsed = (int *)aotx_tool_test_take(count * sizeof(int));
    unsigned int host_slots[AOTX_SLOTS];
    int host_parsed[AOTX_SLOTS];
    unsigned long long tick = 0ull;
    aotx_request_table *requests = (aotx_request_table *)calloc(1, sizeof *requests);
    aotx_agent_table *agents = (aotx_agent_table *)calloc(1, sizeof *agents);
    unsigned int outcome[AOTX_SLOTS], done[AOTX_SLOTS];
    unsigned int reason = (unsigned int)strlen(AOTX_TOOL_OVER_REASON);
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

    /* The turn with no arm: the step ends it with the completed error result. */
    aotx_tool_test_over_turn<<<1, AOTX_SLOTS>>>(slots, count, 0u, parsed);
    aotx_agent_step<<<1, AOTX_SLOTS>>>(tick + 1ull);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_check_runtime(cudaMemcpy(host_parsed, parsed, count * sizeof(int),
                                  cudaMemcpyDeviceToHost), "cudaMemcpy");
    aotx_check_runtime(cudaMemcpyFromSymbol(requests, aotx_requests, sizeof *requests),
                       "cudaMemcpyFromSymbol");
    aotx_check_runtime(cudaMemcpyFromSymbol(agents, aotx_agents, sizeof *agents),
                       "cudaMemcpyFromSymbol");
    aotx_check_runtime(cudaMemcpyFromSymbol(done, aotx_tool_done, sizeof done),
                       "cudaMemcpyFromSymbol");
    unsigned int marked_call = 0u;
    unsigned int completed = 0u;
    for (unsigned int i = 0u; i < count; ++i) {
        unsigned int a = host_slots[i];
        if (a >= AOTX_SLOTS) {
            continue;
        }
        marked_call += (host_parsed[i] == 2) ? 1u : 0u;
        const aotx_request *slot = &requests->slot[a];
        completed += (agents->agent[a].state == AOTX_AGENT_STATE_TOOL
                      && slot->request != 0u && slot->status == AOTX_TOOL_ERROR
                      && done[a] != 0u && slot->arg_len == 0u
                      && slot->result_len == reason
                      && strncmp(slot->result, AOTX_TOOL_OVER_REASON, reason) == 0)
                     ? 1u : 0u;
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
        marked += ((sums[a].events & (1u << AOTX_AFFECT_EVENT_TOOL_ERROR)) != 0u) ? 1u : 0u;
#else
        marked += 1u;
#endif
    }
    /* The console agent holds one slot, so the wide case spawns one agent fewer. */
    *applied += 1u;
    if (live + 1u < count || marked_call != live || completed != live || taken != live
        || marked != live) {
        printf("tool: a call over the argument line at %u agents: %u live, %u parsed as a "
               "call over the line, %u ended with the error result, %u took it, %u carry "
               "the tool error event\n", count, live, marked_call, completed, taken, marked);
        *failed += 1u;
    }

    /* The same turn with the arm of a call: the armed result stands in for the call. */
    aotx_tool_test_over_turn<<<1, AOTX_SLOTS>>>(slots, count, 1u, parsed);
    aotx_agent_step<<<1, AOTX_SLOTS>>>(tick + 3ull);
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
    unsigned int stood = 0u;
    for (unsigned int i = 0u; i < count; ++i) {
        unsigned int a = host_slots[i];
        if (a >= AOTX_SLOTS) {
            continue;
        }
        const aotx_request *slot = &requests->slot[a];
        stood += (agents->agent[a].state == AOTX_AGENT_STATE_TOOL
                  && slot->request != 0u && slot->status == AOTX_TOOL_OK
                  && done[a] != 0u && outcome[a] == 0u && slot->result_len == 20u
                  && strncmp(slot->result, "the fixture tool ran", 20u) == 0) ? 1u : 0u;
    }
    *applied += 1u;
    if (stood != live) {
        printf("tool: the arm over a call above the argument line at %u agents: %u of %u "
               "took the armed result\n", count, stood, live);
        *failed += 1u;
    }
    printf("tool: a call over the argument line at %u agents: %u spawned, %u ended with the "
           "error result and the tool error event, %u took the armed result instead\n",
           count, live, taken, stood);
    aotx_agent_step<<<1, AOTX_SLOTS>>>(tick + 4ull);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_tool_test_release<<<1, AOTX_SLOTS>>>(slots, count);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
#ifdef AOTX_AFFECT
    free(sums);
#endif
    free(requests);
    free(agents);
    cudaFree(slots);
    cudaFree(parsed);
}

#endif
