/* Purpose: Hold the agent table and take the calls of the command layer.
 * Owns: The agent table, the task table and the role table.
 * Launch shape: Device functions; the command layer calls them from its serial thread.
 * Lifetime: The whole run. */
#include "agent/overlays.cuh"
#include "cognitive/live.cuh"
#include "agent/records.cuh"
#include "agent/transcript.cuh"
#include "tool/policy.cuh"
#include "service/service.cuh"
#include "shared/state.cuh"
#include "cli/cli.cuh"
#include "model/sampler.cuh"
#include "tool/tool_state.cuh"

__device__ aotx_agent_table aotx_agents;
__device__ aotx_agent_work aotx_agent_gear[AOTX_SLOTS];
__device__ unsigned int aotx_task_role[AOTX_TASK_SLOTS];
__device__ unsigned int aotx_task_used[AOTX_TASK_SLOTS];
__device__ aotx_agent_counts aotx_agent_count;
__device__ unsigned int aotx_agent_refusal;

__device__ unsigned int aotx_agent_boot_mark;

/* The line the boot spawn writes. One thread makes the spawn, so one line is enough and
 * no frame of the kernel holds it. */
static __device__ aotx_cli_out aotx_agent_boot_line;

__device__ void aotx_agent_boot_spawn(void)
{
    /* The agent of the console. It has no parent, so the spawn makes it a root. A second
     * call takes no slot, because slot 0 belongs to that role and holds one agent. The
     * role is a module, so the call waits until the import of the role lands. */
    if (aotx_agents.agent[0].state != AOTX_AGENT_STATE_FREE) {
        return;
    }
    unsigned int role = aotx_catalog.conductor;
    if (aotx_catalog_is(role, AOTX_MODULE_ROLE) == 0) {
        return;
    }
    if (aotx_agent_spawn(role, ~0u, aotx_time_tick) != 0u) {
        return;
    }
    aotx_agent_boot_mark = 1u;
    /* One console line names the tick the agent of the console took its slot. The
     * operator then sees that the run answers a say line from that tick. */
    aotx_cli_out *out = &aotx_agent_boot_line;
    aotx_cli_clear(out);
    aotx_cli_say(out, "spawn: the role ");
    aotx_cli_say(out, aotx_catalog_name(role));
    aotx_cli_say(out, " is installed and its agent holds slot 0");
    aotx_console_write(out->text, out->at);
    aotx_cli_clear(out);
}

__device__ unsigned int aotx_agent_spawn(unsigned int role, unsigned int parent,
                                         unsigned long long tick)
{
    if (aotx_catalog_is(role, AOTX_MODULE_ROLE) == 0) {
        return ~0u;
    }
    /* Agent 0 belongs to the role the console speaks to, and no other role takes that
     * slot. An agent of that role which is already there is not made a second time. */
    unsigned int console = (role == aotx_catalog.conductor) ? 1u : 0u;
    unsigned int first = (console != 0u) ? 0u : 1u;
    unsigned int last = (console != 0u) ? 1u : AOTX_SLOTS;
    unsigned int slot = AOTX_SLOTS;
    for (unsigned int i = first; i < last; ++i) {
        if (aotx_agents.agent[i].state == AOTX_AGENT_STATE_FREE && !aotx_service_owns(i) && !aotx_shared_owns(i)) {
            slot = i;
            break;
        }
    }
    if (slot >= AOTX_SLOTS) {
        return ~0u;
    }
    aotx_agent *me = &aotx_agents.agent[slot];
    me->state = AOTX_AGENT_STATE_IDLE;
    me->role = role;
    me->parent = (parent < AOTX_SLOTS) ? parent : slot;
    me->turn = 0u;
    me->task = ~0u;
    me->request = 0u;
    me->tool = AOTX_CATALOG_NO_ENTRY;
    me->budget_left = aotx_agent_budget_of(role);
    me->deadline = 0ull;
    me->spawned = tick;
    me->mailbox = 0ull;
    me->verdict = AOTX_VERDICT_NONE;
    me->reserved = 0u;

    aotx_agent_work *gear = &aotx_agent_gear[slot];
    gear->reply_len = 0u;
    gear->prompt_len = 0u;
    gear->system_bytes = 0u;
    gear->prompt_refused = 0u;
    aotx_tool_policy_reset(slot);
    gear->input_hash = 0ull;
    gear->wrote = 0u;
    gear->console = (slot == 0u) ? 1u : 0u;
    gear->kind = 0u;
    gear->result = 0u;
    gear->refused = 0u;
    gear->opens = 0u;
    gear->message_len = 0u;
    gear->has_message = 0u;
    gear->limit_end = 0u;
    gear->stopped = 0u;
    gear->continuable = 0u;
    gear->automatic_message = 0u;
    gear->stop_requested = 0u;
    gear->source_seq = 0ull;
    gear->call.entry = AOTX_CATALOG_NO_ENTRY;
    gear->call.tool = AOTX_TOOL_NONE;
    gear->call.provenance = 0u;
    gear->call.arg_len = 0u;

    aotx_sampler_reset(slot);

    aotx_agents.live += 1u;
    aotx_agent_count.spawned += 1u;
    aotx_agent_note(slot, AOTX_AGENT_SPAWNED, tick);
    return slot;
}

__device__ int aotx_agent_message(unsigned int agent, const unsigned char *text,
                                  unsigned int length, unsigned long long tick)
{
    (void)tick;
    if (aotx_live_bound(agent)) {
        aotx_agent_refusal = AOTX_AGENT_REFUSE_BUSY;
        ++aotx_agents.refused;
        const char *reason = "memory: use a typed request for this conversation";
        aotx_console_write(reason, aotx_cli_length(reason));
        return 1;
    }
    return aotx_agent_queue_message(agent, text, length, aotx_transcript_source_seq);
}

__device__ unsigned int aotx_task_open(unsigned int agent, unsigned int role,
                                       const unsigned char *text, unsigned int length,
                                       unsigned int verify, unsigned long long tick)
{
    if (aotx_live_bound(agent) || text == 0 || length == 0u || (agent >= AOTX_SLOTS && agent != ~0u)
        || (agent == ~0u && aotx_catalog_is(role, AOTX_MODULE_ROLE) == 0)) {
        aotx_agent_refusal = AOTX_AGENT_REFUSE_ROLE;
        aotx_agents.refused += 1u;
        return ~0u;
    }
    /* A text that does not fit is refused and not cut, so the task the agent reads is the
     * task the operator gave. */
    if (length > AOTX_SAY_BYTES) {
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
    hold->source_seq = aotx_transcript_source_seq;
    for (unsigned int i = 0u; i < length; ++i) {
        hold->text[i] = (char)text[i];
    }
    hold->text_len = length;
    aotx_agent_refusal = AOTX_AGENT_REFUSE_NONE;
    aotx_task_role[at] = (agent < AOTX_SLOTS) ? aotx_agents.agent[agent].role : role;
    aotx_task_used[at] = 1u;
    aotx_agents.tasks += 1u;
    aotx_task_note(at, AOTX_WRITER_SYSTEM, hold->text, hold->text_len, tick);
    return at;
}

__device__ int aotx_agent_authorize(unsigned int request, unsigned int granted,
                                    unsigned long long tick)
{
    if (request == 0u) {
        return 1;
    }
    for (unsigned int i = 0u; i < AOTX_SLOTS; ++i) {
        aotx_request *slot = &aotx_requests.slot[i];
        if (slot->request != request || slot->auth != AOTX_AUTH_PENDING) {
            continue;
        }
        slot->auth = (granted != 0u) ? AOTX_AUTH_GRANTED : AOTX_AUTH_REFUSED;
        if (aotx_requests.pending_auth > 0u) {
            aotx_requests.pending_auth -= 1u;
        }
        /* The request had no deadline while it waited for the operator. The deadline of a
         * granted request starts at this tick, so the tool gets its full time. A refused
         * request ends in the tool step of this tick and needs no deadline. */
        if (granted != 0u) {
            slot->deadline = tick + aotx_setting_deadline();
        }
        slot->answer_seq = aotx_transcript_source_seq;
        /* The answer is a second record for the same request. The feeder runs the tool
         * when it reads a record whose authorization is granted. */
        aotx_tool_note_request(slot, aotx_agents.agent[slot->agent].turn);
        return 0;
    }
    return 1;
}
