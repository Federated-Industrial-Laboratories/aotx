/* Purpose: Give the console the names of the agent table and the answer of a request.
 * Owns: The focus of the keyboard between the console and the agents panel.
 * Launch shape: One thread; the apply step calls these in slot order.
 * Lifetime: The whole run. */
#ifndef AOTX_CLI_AGENTS_CUH
#define AOTX_CLI_AGENTS_CUH

#include "agent/agent.cuh"
#include "agent/agent_state.cuh"
#include "catalog/catalog.cuh"
#include "cli/cli.cuh"
#include "tool/tool.cuh"

/* Where the keyboard writes. The tab key moves the focus from one to the other. The editor
 * takes no key while the focus is on the panel, and the panel takes y and n. */
#define AOTX_CLI_FOCUS_CONSOLE 0u
#define AOTX_CLI_FOCUS_AGENTS  1u

/* The two answers a keystroke of the panel gives. */
#define AOTX_CLI_GRANT         1u
#define AOTX_CLI_REFUSE        0u

extern __device__ unsigned int aotx_cli_focus;

/* Give the name of a role, or a dash when the entry holds no role. */
__device__ __forceinline__ const char *aotx_cli_role_name(unsigned int role)
{
    return (aotx_catalog_is(role, AOTX_MODULE_ROLE) != 0) ? aotx_catalog.entry[role].name
                                                          : "-";
}

/* Give the name of an agent state, or a dash when the value is not one of the six. */
__device__ __forceinline__ const char *aotx_cli_agent_state_name(unsigned int state)
{
    switch (state) {
    case AOTX_AGENT_STATE_FREE:   return "free";
    case AOTX_AGENT_STATE_IDLE:   return "idle";
    case AOTX_AGENT_STATE_PROMPT: return "prompt";
    case AOTX_AGENT_STATE_RUN:    return "run";
    case AOTX_AGENT_STATE_TOOL:   return "tool";
    case AOTX_AGENT_STATE_POST:   return "post";
    default:                      return "-";
    }
}

/* Give the name of a tool, or a dash when no tool call waits. */
__device__ __forceinline__ const char *aotx_cli_tool_name(unsigned int tool)
{
    return (aotx_catalog_is(tool, AOTX_MODULE_TOOL) != 0) ? aotx_catalog.entry[tool].name
                                                          : "-";
}

/* Give the request slot of a rank in the pending list, with the lowest number first. The
 * return is the slot count when fewer requests wait than the rank asks for. The panel and
 * the keystroke of the panel take the same order from this function. */
__device__ __forceinline__ unsigned int aotx_cli_pending_at(unsigned int rank)
{
    unsigned int at = AOTX_SLOTS;
    unsigned int below = 0u;
    for (unsigned int i = 0u; i < AOTX_SLOTS; ++i) {
        const aotx_request *slot = &aotx_requests.slot[i];
        if (slot->request == 0u || slot->auth != AOTX_AUTH_PENDING) {
            continue;
        }
        below = 0u;
        for (unsigned int j = 0u; j < AOTX_SLOTS; ++j) {
            const aotx_request *other = &aotx_requests.slot[j];
            if (other->request != 0u && other->auth == AOTX_AUTH_PENDING
                && other->request < slot->request) {
                below += 1u;
            }
        }
        if (below == rank) {
            at = i;
        }
    }
    return at;
}

/* Give the number of the first pending request, or zero when none waits. */
__device__ __forceinline__ unsigned int aotx_cli_first_request(void)
{
    unsigned int at = aotx_cli_pending_at(0u);
    return (at < AOTX_SLOTS) ? aotx_requests.slot[at].request : 0u;
}

/* Count the requests that wait for the operator. */
__device__ __forceinline__ unsigned int aotx_cli_pending_count(void)
{
    unsigned int count = 0u;
    for (unsigned int i = 0u; i < AOTX_SLOTS; ++i) {
        const aotx_request *slot = &aotx_requests.slot[i];
        if (slot->request != 0u && slot->auth == AOTX_AUTH_PENDING) {
            count += 1u;
        }
    }
    return count;
}

/* Add a value to a line, or a dash when the value is the one that stands for nothing. */
__device__ __forceinline__ void aotx_cli_or_dash(aotx_cli_out *out,
                                                 unsigned long long value,
                                                 unsigned long long none)
{
    if (value == none) {
        aotx_cli_say(out, "-");
        return;
    }
    aotx_cli_num(out, value);
}

/* Put one agent on one line. The line holds the identity, the role and the state. It then
 * holds the task in hand, the tool of a request that waits and the number of that request.
 * It ends with the turns taken and the reply tokens. An agent owns the slot of its
 * identity, so the reply tokens come from that sequence slot. */
__device__ __forceinline__ void aotx_cli_agent_line(aotx_cli_out *out, unsigned int id)
{
    const aotx_agent *agent = &aotx_agents.agent[id];
    aotx_cli_num(out, (unsigned long long)id);
    aotx_cli_say(out, " ");
    aotx_cli_say(out, aotx_cli_role_name(agent->role));
    aotx_cli_say(out, " ");
    aotx_cli_say(out, aotx_cli_agent_state_name(agent->state));
    aotx_cli_say(out, " ");
    aotx_cli_or_dash(out, (unsigned long long)agent->task, 0xffffffffull);
    aotx_cli_say(out, " ");
    aotx_cli_say(out, aotx_cli_tool_name(agent->tool));
    aotx_cli_say(out, " ");
    aotx_cli_or_dash(out, (unsigned long long)agent->request, 0ull);
    aotx_cli_say(out, " ");
    aotx_cli_num(out, (unsigned long long)agent->turn);
    aotx_cli_say(out, " ");
    aotx_cli_num(out, (unsigned long long)aotx_seqs.slot[id].sampled);
}

/* Show one row for each agent that is not free. The columns are the columns of the agents
 * panel. */
static __device__ __noinline__ void aotx_cli_show_agents(aotx_cli_out *out)
{
    aotx_cli_say(out, "agents: id role state task tool request turn tokens");
    aotx_cli_console(out);
    unsigned int live = 0u;
    for (unsigned int id = 0u; id < AOTX_SLOTS; ++id) {
        if (aotx_agents.agent[id].state == AOTX_AGENT_STATE_FREE) {
            continue;
        }
        live += 1u;
        aotx_cli_say(out, "  ");
        aotx_cli_agent_line(out, id);
        aotx_cli_console(out);
    }
    if (live == 0u) {
        aotx_cli_say(out, "  no agents");
        aotx_cli_console(out);
    }
}

/* Answer one pending authorization and state the answer on the console. The console line
 * names the request and the answer, whether a command line or a keystroke of the panel
 * gave it. The return is 0 when the request took the answer, and 1 when none waits with
 * that number. The caller is the serial thread of the apply step. */
__device__ __forceinline__ int aotx_cli_answer(aotx_cli_out *out, unsigned int request,
                                               unsigned int granted,
                                               unsigned long long tick)
{
    const char *name = (granted != 0u) ? "authorize" : "refuse";
    if (request == 0u || aotx_agent_authorize(request, granted, tick) != 0) {
        aotx_cli_say(out, name);
        aotx_cli_say(out, ": no request of that number waits");
        aotx_cli_console(out);
        aotx_cli_count.refused += 1u;
        return 1;
    }
    aotx_cli_say(out, name);
    aotx_cli_say(out, ": request ");
    aotx_cli_num(out, (unsigned long long)request);
    aotx_cli_say(out, (granted != 0u) ? " is granted" : " is refused");
    aotx_cli_console(out);
    return 0;
}

#endif
