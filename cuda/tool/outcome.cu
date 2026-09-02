/* Purpose: Put an armed tool result on the request slot of an agent.
 * Owns: Nothing; the request table and the arm of each slot hold the state.
 * Launch shape: Device functions; one call for each armed turn.
 * Lifetime: The whole run.
 *
 * A scripted run needs the events of a tool result in a known order, whatever the reply
 * of the model holds. The console arms one result for the next turn of an agent. The
 * agent step takes it as the result of that turn, through the same read of the status
 * that a real result takes. The tools of the catalog do not change. */
#include "tool/tool_state.cuh"

__device__ void aotx_tool_outcome_arm(unsigned int agent, unsigned int status)
{
    if (agent >= AOTX_SLOTS || status > AOTX_TOOL_REFUSED) {
        return;
    }
    aotx_tool_embed.outcome[agent] = status + 1u;
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
    slot->status = status;
    slot->result_len = aotx_tool_put(slot->result, 0u, aotx_tool_outcome_text(status));
    return 1;
}
