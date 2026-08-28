/* Purpose: Hold the agent table and take the calls of the command layer.
 * Owns: The agent table, the task table and the role table.
 * Launch shape: Device functions; the command layer calls them from its serial thread.
 * Lifetime: The whole run. */
#include "agent/overlays.cuh"
#include "agent/records.cuh"
#include "tool/tool_state.cuh"

__device__ aotx_agent_table aotx_agents;
__device__ aotx_agent_work aotx_agent_gear[AOTX_AGENT_SLOTS];
__device__ unsigned int aotx_task_role[AOTX_TASK_SLOTS];
__device__ unsigned int aotx_task_used[AOTX_TASK_SLOTS];
__device__ aotx_agent_counts aotx_agent_count;
__device__ unsigned int aotx_agent_refusal;

__global__ void aotx_agent_boot(void)
{
    /* The conductor of the run. It has no parent, so the spawn makes it a root. A second
     * call takes no slot, because slot 0 belongs to the conductor and holds one. */
    if (blockIdx.x == 0u && threadIdx.x == 0u) {
        aotx_agent_spawn(AOTX_ROLE_CONDUCTOR, ~0u, aotx_time_tick);
    }
}

__device__ unsigned int aotx_agent_spawn(unsigned int role, unsigned int parent,
                                         unsigned long long tick)
{
    aotx_agent_roles_set();
    if (role >= AOTX_ROLE_COUNT) {
        return ~0u;
    }
    /* Agent 0 is the conductor and no other role takes that slot. A conductor which is
     * already there is not made a second time. */
    unsigned int first = (role == AOTX_ROLE_CONDUCTOR) ? 0u : 1u;
    unsigned int last = (role == AOTX_ROLE_CONDUCTOR) ? 1u : AOTX_AGENT_SLOTS;
    unsigned int slot = AOTX_AGENT_SLOTS;
    for (unsigned int i = first; i < last; ++i) {
        if (aotx_agents.agent[i].state == AOTX_AGENT_STATE_FREE) {
            slot = i;
            break;
        }
    }
    if (slot >= AOTX_AGENT_SLOTS) {
        return ~0u;
    }
    aotx_agent *me = &aotx_agents.agent[slot];
    me->state = AOTX_AGENT_STATE_IDLE;
    me->role = role;
    me->parent = (parent < AOTX_AGENT_SLOTS) ? parent : slot;
    me->turn = 0u;
    me->task = ~0u;
    me->request = 0u;
    me->tool = AOTX_TOOL_NONE;
    me->budget_left = aotx_agents.role[role].budget;
    me->deadline = 0ull;
    me->spawned = tick;
    me->mailbox = 0ull;
    me->verdict = AOTX_VERDICT_NONE;
    me->reserved = 0u;

    aotx_agent_work *gear = &aotx_agent_gear[slot];
    gear->reply_len = 0u;
    gear->prompt_len = 0u;
    gear->input_hash = 0ull;
    gear->wrote = 0u;
    gear->console = (slot == 0u) ? 1u : 0u;
    gear->kind = 0u;
    gear->result = 0u;
    gear->refused = 0u;
    gear->opens = 0u;
    gear->message_len = 0u;
    gear->has_message = 0u;
    gear->call.tool = AOTX_TOOL_NONE;
    gear->call.provenance = 0u;
    gear->call.arg_len = 0u;

    aotx_agents.live += 1u;
    aotx_agent_count.spawned += 1u;
    aotx_agent_note(slot, AOTX_AGENT_SPAWNED, tick);
    return slot;
}

__device__ int aotx_agent_message(unsigned int agent, const unsigned char *text,
                                  unsigned int length, unsigned long long tick)
{
    (void)tick;
    if (agent >= AOTX_AGENT_SLOTS || text == 0 || length == 0u) {
        return 1;
    }
    aotx_agent *me = &aotx_agents.agent[agent];
    if (me->state != AOTX_AGENT_STATE_IDLE || me->task != ~0u) {
        aotx_agent_refusal = AOTX_AGENT_REFUSE_BUSY;
        aotx_agents.refused += 1u;
        return 1;
    }
    /* A text that does not fit is refused and not cut. A message the agent answers must
     * be the message the operator gave. */
    if (length > AOTX_TASK_TEXT_BYTES) {
        aotx_agent_refusal = AOTX_AGENT_REFUSE_LONG;
        aotx_agents.refused += 1u;
        return 2;
    }
    aotx_agent_work *gear = &aotx_agent_gear[agent];
    unsigned int bytes = length;
    for (unsigned int i = 0u; i < bytes; ++i) {
        gear->message[i] = text[i];
    }
    gear->message_len = bytes;
    aotx_agent_refusal = AOTX_AGENT_REFUSE_NONE;
    gear->has_message = 1u;
    gear->kind = AOTX_AGENT_TURN_MESSAGE;
    return 0;
}

__device__ unsigned int aotx_task_open(unsigned int agent, unsigned int role,
                                       const unsigned char *text, unsigned int length,
                                       unsigned int verify, unsigned long long tick)
{
    aotx_agent_roles_set();
    if (text == 0 || length == 0u || (agent >= AOTX_AGENT_SLOTS && agent != ~0u)
        || (agent == ~0u && role >= AOTX_ROLE_COUNT)) {
        aotx_agent_refusal = AOTX_AGENT_REFUSE_ROLE;
        aotx_agents.refused += 1u;
        return ~0u;
    }
    /* A text that does not fit is refused and not cut, so the task the agent reads is the
     * task the operator gave. */
    if (length > AOTX_TASK_TEXT_BYTES) {
        aotx_agent_refusal = AOTX_AGENT_REFUSE_LONG;
        aotx_agents.refused += 1u;
        return ~0u;
    }
    unsigned int at = AOTX_TASK_SLOTS;
    for (unsigned int i = 0u; i < AOTX_TASK_SLOTS; ++i) {
        if (aotx_task_used[i] == 0u) {
            at = i;
            break;
        }
    }
    if (at >= AOTX_TASK_SLOTS) {
        aotx_agent_refusal = AOTX_AGENT_REFUSE_FULL;
        aotx_agents.refused += 1u;
        return ~0u;
    }
    aotx_task *hold = &aotx_agents.task[at];
    hold->state = AOTX_TASK_PENDING;
    hold->agent = agent;
    hold->giver = ~0u;
    hold->verify = (verify != 0u) ? AOTX_VERIFY_SIBLING : AOTX_VERIFY_NONE;
    hold->attempts = 0u;
    hold->verifier = ~0u;
    hold->result_len = 0u;
    hold->opened = tick;
    for (unsigned int i = 0u; i < length; ++i) {
        hold->text[i] = (char)text[i];
    }
    hold->text_len = length;
    aotx_agent_refusal = AOTX_AGENT_REFUSE_NONE;
    aotx_task_role[at] = (agent < AOTX_AGENT_SLOTS) ? aotx_agents.agent[agent].role : role;
    aotx_task_used[at] = 1u;
    aotx_agents.tasks += 1u;
    aotx_task_note(at, AOTX_WRITER_SYSTEM, hold->text, hold->text_len, tick);
    return at;
}

__device__ int aotx_agent_authorize(unsigned int request, unsigned int granted,
                                    unsigned long long tick)
{
    (void)tick;
    if (request == 0u) {
        return 1;
    }
    for (unsigned int i = 0u; i < AOTX_REQUEST_SLOTS; ++i) {
        aotx_request *slot = &aotx_requests.slot[i];
        if (slot->request != request || slot->auth != AOTX_AUTH_PENDING) {
            continue;
        }
        slot->auth = (granted != 0u) ? AOTX_AUTH_GRANTED : AOTX_AUTH_REFUSED;
        if (aotx_requests.pending_auth > 0u) {
            aotx_requests.pending_auth -= 1u;
        }
        /* The answer is a second record for the same request. The feeder runs the tool
         * when it reads a record whose authorization is granted. */
        aotx_tool_note_request(slot, aotx_agents.agent[slot->agent].turn);
        return 0;
    }
    return 1;
}
