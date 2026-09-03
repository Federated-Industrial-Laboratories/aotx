/* Purpose: Put an armed tool result on the request slot of an agent.
 * Owns: Nothing; the request table and the arm of each slot hold the state.
 * Launch shape: Device functions; one call for each armed turn.
 * Lifetime: The whole run.
 *
 * A scripted run needs the events of a tool result in a known order, whatever the reply
 * of the model holds. The console arms one result for the next turn of an agent. When
 * that turn calls a tool, no request opens and no tool runs. The armed result stands on
 * the request slot as a request that completed at once. The turn that follows carries it
 * as a real result, on the path a real result takes. When the turn makes no call, the
 * armed result is the result of the turn itself, and its event lands on that turn.
 *
 * An arm of no result takes a call off with no event and no result. The turn then ends
 * with the reply the model wrote. When that reply was only the call, the next turn
 * generates with no result in its context. An arm of a call completes a call the model
 * makes as the ok arm does. A turn with no call then gets no result and no event. The
 * tools of the catalog do not change, and a turn with no arm takes its path as before. */
#include "agent/agent_state.cuh"
#include "catalog/catalog.cuh"
#include "tool/tool_state.cuh"

__device__ void aotx_tool_outcome_arm(unsigned int agent, unsigned int status)
{
    if (agent >= AOTX_SLOTS || status > AOTX_TOOL_CALL_RESULT) {
        return;
    }
    aotx_tool_embed.outcome[agent] = status + 1u;
}

__device__ int aotx_tool_outcome_armed(unsigned int agent)
{
    return (agent < AOTX_SLOTS && aotx_tool_embed.outcome[agent] != 0u) ? 1 : 0;
}

/* The words of each armed result. A refused result carries the words of the refusal of
 * the operator, because that is the result the agent reads in a live run. */
__device__ __forceinline__ static const char *aotx_tool_outcome_text(unsigned int status)
{
    switch (status) {
    case AOTX_TOOL_OK:      return "the fixture tool ran";
    case AOTX_TOOL_ERROR:   return "the fixture tool failed";
    default:                return "the operator refused this tool";
    }
}

/* Take the arm off and give its status, or AOTX_TOOL_NO_RESULT for no arm. */
__device__ __forceinline__ static unsigned int aotx_tool_outcome_off(unsigned int agent)
{
    unsigned int armed = aotx_tool_embed.outcome[agent];
    aotx_tool_embed.outcome[agent] = 0u;
    return (armed == 0u) ? AOTX_TOOL_NO_RESULT : armed - 1u;
}

__device__ int aotx_tool_outcome_take(unsigned int agent)
{
    if (agent >= AOTX_SLOTS) {
        return 0;
    }
    unsigned int status = aotx_tool_outcome_off(agent);
    /* A turn with no call takes nothing from the arm of no result or the arm of a call. */
    if (status == AOTX_TOOL_NO_RESULT || status == AOTX_TOOL_CALL_RESULT) {
        return 0;
    }
    aotx_request *slot = &aotx_requests.slot[agent];
    slot->status = status;
    slot->result_len = aotx_tool_put(slot->result, 0u, aotx_tool_outcome_text(status));
    return 1;
}

__device__ unsigned int aotx_tool_outcome_request(unsigned int agent,
                                                  const aotx_tool_call *call,
                                                  unsigned long long tick)
{
    if (agent >= AOTX_SLOTS || call == 0) {
        return 0u;
    }
    unsigned int status = aotx_tool_outcome_off(agent);
    aotx_request *slot = &aotx_requests.slot[agent];
    if (status == AOTX_TOOL_NO_RESULT || slot->request != 0u
        || aotx_catalog_is(call->entry, AOTX_MODULE_TOOL) == 0) {
        return 0u;
    }
    /* The arm of a call completes the call as ok. */
    if (status == AOTX_TOOL_CALL_RESULT) {
        status = AOTX_TOOL_OK;
    }
    /* The request takes the number a real request of this slot takes, so the record of
     * the turn and a replay agree. No record names the request: the arm is a console
     * line, and a replay makes the result again from that line. */
    unsigned int made = aotx_tool_embed.made[agent];
    unsigned int id = made * AOTX_SLOTS + agent + 1u;
    aotx_tool_embed.made[agent] = made + 1u;
    atomicAdd(&aotx_agents.next_request, 1u);
    slot->agent = agent;
    slot->entry = call->entry;
    slot->tool = aotx_catalog_tool_number(call->entry);
    slot->auth = AOTX_AUTH_NONE;
    slot->status = status;
    slot->parts_in = 1u;
    slot->parts = 1u;
    slot->call_seq = 0ull;
    slot->answer_seq = 0ull;
    slot->result_seq = 0ull;
    slot->deadline = tick + aotx_setting_deadline();
    slot->arg_len = aotx_tool_arguments(call, slot->arg, AOTX_TOOL_ARG_BYTES);
    slot->result_len = aotx_tool_put(slot->result, 0u, aotx_tool_outcome_text(status));
    aotx_tool_embed.prov[agent] = call->provenance;
    aotx_tool_embed.state[agent] = AOTX_TOOL_EMBED_NONE;
    /* The result is in hand before the slot opens. The tool step of the next tick passes
     * the request over, and the agent step takes the result. */
    aotx_tool_done[agent] = 1u;
    slot->request = id;
    return id;
}
