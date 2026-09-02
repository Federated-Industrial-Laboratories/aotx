/* Purpose: Put an armed tool result on the request slot of an agent.
 * Owns: Nothing; the request table and the arm of each slot hold the state.
 * Launch shape: Device functions; one call for each armed turn.
 * Lifetime: The whole run.
 *
 * A scripted run needs the events of a tool result in a known order, whatever the reply
 * of the model holds. The console arms one result for the next turn of an agent. That
 * turn makes no call: no request opens and no tool runs. The agent step takes the armed
 * result as the result of the turn, through the same read of the status that a real
 * result takes. An arm of no result takes the call off with no event. The tools of the
 * catalog do not change, and a turn with no arm takes its path as before. */
#include "tool/tool_state.cuh"

__device__ void aotx_tool_outcome_arm(unsigned int agent, unsigned int status)
{
    if (agent >= AOTX_SLOTS || status > AOTX_TOOL_NO_RESULT) {
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

__device__ int aotx_tool_outcome_take(unsigned int agent)
{
    if (agent >= AOTX_SLOTS || aotx_tool_embed.outcome[agent] == 0u) {
        return 0;
    }
    unsigned int status = aotx_tool_embed.outcome[agent] - 1u;
    aotx_request *slot = &aotx_requests.slot[agent];
    aotx_tool_embed.outcome[agent] = 0u;
    if (status == AOTX_TOOL_NO_RESULT) {
        return 0;
    }
    slot->status = status;
    slot->result_len = aotx_tool_put(slot->result, 0u, aotx_tool_outcome_text(status));
    return 1;
}
