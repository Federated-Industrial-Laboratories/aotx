/* Purpose: Step every agent through its states, and give pending tasks to idle agents.
 * Owns: Nothing; the agent table and the request table hold the state.
 * Launch shape: One block of one thread for each agent slot.
 * Lifetime: One node of every tick.
 *
 * The node runs after the commit of the decode and after the tool step. An agent sees the
 * reply of its sequence and the result of its tool in the tick they arrive. Thread 0 gives
 * the pending tasks to idle agents first. That assignment must be the same in every run,
 * and a race between 64 threads is not. The threads then step their agents. */
#include "agent/prompt.cuh"
#include "agent/records.cuh"
#include "agent/transcript.cuh"
#include "bus/bus.cuh"
#include "tool/tool_state.cuh"
#ifdef AOTX_AFFECT
#include "affect/affect.cuh"
#endif

/* The instruction is fixed, so two compaction turns use the same task text. */
static __device__ const unsigned char aotx_agent_compact_instruction[] =
    "Write one concise summary of the memory turns. Keep facts, decisions and open work.";

/* Replies outside the console slot still cross the seam as joined console parts. The
 * transcript derivation then has their text and their agent writer. */
static __device__ __forceinline__ void aotx_agent_reply_records(unsigned int agent,
                                                                const unsigned char *text,
                                                                unsigned int length)
{
    if (agent == 0u) {
        return;
    }
    unsigned int at = 0u;
    unsigned int part = 0u;
    while (at < length) {
        unsigned int bytes = length - at;
        if (bytes > AOTX_BODY_BYTES) {
            bytes = AOTX_BODY_BYTES;
        }
        unsigned long long seq = aotx_seam_write(
            AOTX_WRITER_AGENT_BASE + agent, AOTX_CLASS_B, AOTX_REC_CONSOLE,
            (part == 0u) ? 0u : AOTX_FLAG_FRAGMENT, text + at, bytes);
        if (part == 0u) {
            aotx_say.slot[agent].reply_first = seq;
        }
        aotx_say.slot[agent].reply_records += 1u;
        at += bytes;
        part += 1u;
    }
}

/* State one parsed call without putting its markup on the console. */
__device__ void aotx_agent_call_line(unsigned int agent)
{
    if (agent != AOTX_SAY_SLOT) {
        return;
    }
    aotx_agent_work *gear = &aotx_agent_gear[agent];
    unsigned int at = 0u;
    const char *head = "calls ";
    for (unsigned int i = 0u; head[i] != '\0' && at < AOTX_BUS_TEXT_BYTES; ++i) {
        gear->line[at++] = head[i];
    }
    const char *name = aotx_cli_tool_name(gear->call.entry);
    for (unsigned int i = 0u; name[i] != '\0' && at < AOTX_BUS_TEXT_BYTES; ++i) {
        gear->line[at++] = name[i];
    }
    if (gear->call.arg_len != 0u && at < AOTX_BUS_TEXT_BYTES) {
        gear->line[at++] = ' ';
    }
    for (unsigned int i = 0u; i < gear->call.arg_len && at < AOTX_BUS_TEXT_BYTES; ++i) {
        gear->line[at++] = gear->call.arg[i];
    }
    aotx_say_slot *state = &aotx_say.slot[agent];
    if (state->column != 0u && state->at != 0ull) {
        aotx_seam_write(AOTX_WRITER_CONSOLE, AOTX_CLASS_B, AOTX_REC_CONSOLE,
                        AOTX_FLAG_FRAGMENT, gear->line, at);
        aotx_console_grow(state->at, (const unsigned char *)gear->line, at);
    } else {
        aotx_console_write(gear->line, at);
    }
}

/* State one refusal with the tool as subject and the reason as its answer. */
static __device__ __noinline__ void aotx_agent_tool_line(unsigned int agent,
                                                         const char *reason,
                                                         unsigned int length)
{
    aotx_agent_work *gear = &aotx_agent_gear[agent];
    const char *name = aotx_cli_tool_name(aotx_requests.slot[agent].entry);
    unsigned int at = 0u;
    for (unsigned int i = 0u; name[i] != '\0' && at < AOTX_BUS_TEXT_BYTES; ++i) {
        gear->line[at++] = name[i];
    }
    if (at < AOTX_BUS_TEXT_BYTES) {
        gear->line[at++] = ':';
    }
    if (at < AOTX_BUS_TEXT_BYTES) {
        gear->line[at++] = ' ';
    }
    for (unsigned int i = 0u; i < length && at < AOTX_BUS_TEXT_BYTES; ++i) {
        gear->line[at++] = reason[i];
    }
    aotx_console_write(gear->line, at);
}

/* State the end of an input without adding words to the generated reply. */
static __device__ __noinline__ void aotx_agent_budget_line(unsigned int agent)
{
    aotx_agent_work *gear = &aotx_agent_gear[agent];
    unsigned int at = 0u;
    const char *head = "agent ";
    const char *tail = ": the turn budget is exhausted; give new input to resume";
    for (unsigned int i = 0u; head[i] != '\0'; ++i) {
        gear->line[at++] = head[i];
    }
    at += aotx_text_utoa(agent, gear->line + at, AOTX_BUS_TEXT_BYTES - at);
    for (unsigned int i = 0u; tail[i] != '\0' && at < AOTX_BUS_TEXT_BYTES; ++i) {
        gear->line[at++] = tail[i];
    }
    aotx_console_write(gear->line, at);
}

#ifdef AOTX_AFFECT
/* Mark the event of one tool result on the turn of an agent. */
static __device__ __forceinline__ void aotx_agent_tool_mark(unsigned int agent,
                                                            unsigned int status)
{
    aotx_affect_mark(agent, (status == AOTX_TOOL_OK) ? AOTX_AFFECT_EVENT_TOOL_OK
                            : ((status == AOTX_TOOL_REFUSED) ? AOTX_AFFECT_EVENT_TOOL_REFUSED
                               : AOTX_AFFECT_EVENT_TOOL_ERROR));
}
#endif

/* Give one task to one agent. The agent takes its turn in the same tick. */
__device__ __forceinline__ static void aotx_agent_assign(unsigned int task,
                                                         unsigned int agent,
                                                         unsigned long long tick)
{
    aotx_task *hold = &aotx_agents.task[task];
    hold->state = AOTX_TASK_ASSIGNED;
    hold->agent = agent;
    hold->attempts += 1u;
    aotx_agents.agent[agent].task = task;
    aotx_agents.agent[agent].budget_left =
        aotx_agent_budget_of(aotx_agents.agent[agent].role);
    aotx_agent_gear[agent].kind = AOTX_AGENT_TURN_TASK;
    aotx_agent_gear[agent].stop_requested = 0u;
    aotx_agent_gear[agent].source_seq = hold->source_seq;
    aotx_task_note(task, AOTX_WRITER_AGENT_BASE + agent, hold->text, hold->text_len, tick);
}

/* Find the first idle agent of a role that holds no task. The return is the agent, or the
 * slot count when every agent of the role is busy. */
__device__ __forceinline__ static unsigned int aotx_agent_idle_of(unsigned int role)
{
    for (unsigned int a = 0u; a < AOTX_SLOTS; ++a) {
        const aotx_agent *me = &aotx_agents.agent[a];
        if (me->state == AOTX_AGENT_STATE_IDLE && me->role == role && me->task == ~0u
            && aotx_agent_gear[a].has_message == 0u) {
            return a;
        }
    }
    return AOTX_SLOTS;
}

/* The engine of the agenda. One thread walks the task table in order, so two runs of the
 * same inputs give the same assignment. */
__device__ __forceinline__ static void aotx_agent_agenda(unsigned long long tick)
{
    for (unsigned int t = 0u; t < AOTX_TASK_SLOTS; ++t) {
        if (aotx_task_used[t] == 0u) {
            continue;
        }
        aotx_task *hold = &aotx_agents.task[t];
        if (hold->state == AOTX_TASK_PENDING) {
            unsigned int who = hold->agent;
            if (who >= AOTX_SLOTS) {
                who = aotx_agent_idle_of(aotx_task_role[t]);
            } else if (aotx_agents.agent[who].state != AOTX_AGENT_STATE_IDLE
                       || aotx_agents.agent[who].task != ~0u) {
                who = AOTX_SLOTS;
            }
            if (who < AOTX_SLOTS) {
                aotx_agent_assign(t, who, tick);
            }
        } else if (hold->state == AOTX_TASK_VERIFYING && hold->verifier >= AOTX_SLOTS) {
            /* The role that judges a result is an entry of the catalog. The catalog keeps
             * that entry, because this path of the engine names it. */
            unsigned int who = aotx_agent_idle_of(aotx_catalog.verifier);
            if (who < AOTX_SLOTS) {
                hold->verifier = who;
                aotx_agents.agent[who].task = t;
                aotx_agents.agent[who].verdict = AOTX_VERDICT_NONE;
                aotx_agents.agent[who].budget_left =
                    aotx_agent_budget_of(aotx_catalog.verifier);
                aotx_agent_gear[who].kind = AOTX_AGENT_TURN_VERIFY;
                aotx_agent_gear[who].stop_requested = 0u;
                aotx_agent_gear[who].source_seq = hold->source_seq;
                aotx_task_note(t, AOTX_WRITER_AGENT_BASE + who, hold->text, hold->text_len,
                               tick);
            }
        }
    }
}

/* Put one bus message on the bus from an agent. A handoff carries the name of the task and
 * the first bytes of the result, with a line feed between them. */
__device__ __forceinline__ static void aotx_agent_handoff(unsigned int agent,
                                                          unsigned int task,
                                                          const char *word,
                                                          const char *text,
                                                          unsigned int length,
                                                          unsigned long long tick)
{
    aotx_agent_work *gear = &aotx_agent_gear[agent];
    unsigned int at = 0u;
    for (unsigned int i = 0u; word[i] != '\0' && at < AOTX_BUS_TEXT_BYTES; ++i) {
        gear->line[at++] = word[i];
    }
    at += aotx_text_utoa(task, gear->line + at, AOTX_BUS_TEXT_BYTES - at);
    if (at < AOTX_BUS_TEXT_BYTES) {
        gear->line[at++] = '\n';
    }
    for (unsigned int i = 0u; i < length && at < AOTX_BUS_TEXT_BYTES; ++i) {
        gear->line[at++] = text[i];
    }
    aotx_bus_append(AOTX_WRITER_AGENT_BASE + agent, AOTX_BUS_HANDOFF, 0u, gear->line, at,
                    0ull, 0ull, 0.0f, tick);
}

/* Start one turn of an agent. The prompt goes in the table of the say path, which opens
 * the sequence of the slot in the tick that follows. */
__device__ __forceinline__ static void aotx_agent_begin(unsigned int agent,
                                                        const char *head,
                                                        const unsigned char *first,
                                                        unsigned int first_len,
                                                        const char *middle,
                                                        const unsigned char *second,
                                                        unsigned int second_len,
                                                        const char *result,
                                                        unsigned int result_len,
                                                        unsigned long long tick)
{
    aotx_agent *me = &aotx_agents.agent[agent];
    if (aotx_agent_prompt(agent, head, first, first_len, middle, second, second_len,
                          result, result_len) == 0u) {
        return;
    }
    me->turn += 1u;
    if (me->budget_left > 0u) {
        me->budget_left -= 1u;
    }
    /* The end of the sequence of the turn stands empty until the reply is taken. A turn
     * whose sequence does not open therefore states no token and no stop. */
    aotx_agent_gear[agent].out_tokens = 0u;
    aotx_agent_gear[agent].last_token = 0u;
    aotx_agent_gear[agent].limit_end = 0u;
    aotx_agent_gear[agent].stopped = 0u;
    me->state = AOTX_AGENT_STATE_PROMPT;
    aotx_agent_note(agent, AOTX_AGENT_TURN, tick);
}

/* End the task of an agent. The result is the reply of the turn. A task that asks for a
 * verifier waits for one; every other task ends here. */
__device__ __forceinline__ static void aotx_agent_finish(unsigned int agent,
                                                         unsigned int state,
                                                         unsigned long long tick)
{
    aotx_agent *me = &aotx_agents.agent[agent];
    aotx_agent_work *gear = &aotx_agent_gear[agent];
    unsigned int task = me->task;
    if (task >= AOTX_TASK_SLOTS) {
        me->state = AOTX_AGENT_STATE_IDLE;
        return;
    }
    aotx_task *hold = &aotx_agents.task[task];
    unsigned int bytes = (gear->reply_len > AOTX_TASK_TEXT_BYTES) ? AOTX_TASK_TEXT_BYTES
                                                                 : gear->reply_len;
    for (unsigned int i = 0u; i < bytes; ++i) {
        hold->result[i] = (char)gear->reply[i];
    }
    hold->result_len = bytes;
    if (state == AOTX_TASK_DONE && hold->verify == AOTX_VERIFY_SIBLING) {
        hold->state = AOTX_TASK_VERIFYING;
        hold->verifier = ~0u;
    } else {
        hold->state = state;
        if (state == AOTX_TASK_DONE) {
            atomicAdd(&aotx_agent_count.done, 1u);
        } else {
            atomicAdd(&aotx_agent_count.failed, 1u);
        }
#ifdef AOTX_AFFECT
        aotx_affect_mark(agent, (state == AOTX_TASK_DONE) ? AOTX_AFFECT_EVENT_TASK_DONE
                                                          : AOTX_AFFECT_EVENT_TASK_FAILED);
#endif
    }
    aotx_task_note(task, AOTX_WRITER_AGENT_BASE + agent, hold->result, hold->result_len,
                   tick);
    aotx_agent_handoff(agent, task, "task ", hold->result, hold->result_len, tick);
    me->task = ~0u;
    me->state = AOTX_AGENT_STATE_IDLE;
}

/* Judge the result of a task from the one word of a verifier. */
__device__ __forceinline__ static void aotx_agent_judge(unsigned int agent,
                                                        unsigned long long tick)
{
    aotx_agent *me = &aotx_agents.agent[agent];
    aotx_agent_work *gear = &aotx_agent_gear[agent];
    unsigned int task = me->task;
    if (task >= AOTX_TASK_SLOTS) {
        me->state = AOTX_AGENT_STATE_IDLE;
        return;
    }
    aotx_task *hold = &aotx_agents.task[task];
    unsigned int verdict = aotx_agent_verdict_of(gear->reply, gear->reply_len);
    me->verdict = verdict;
#ifdef AOTX_AFFECT
    if (verdict == AOTX_VERDICT_REFUTE) {
        aotx_affect_mark(agent, AOTX_AFFECT_EVENT_VERDICT_REFUTE);
    }
    /* The task of the assignee ends here, so its done or failed mark goes to the assignee
     * and rides into the assignee's next turn. */
    aotx_affect_mark(hold->agent, (verdict == AOTX_VERDICT_REFUTE)
                                  ? AOTX_AFFECT_EVENT_TASK_FAILED
                                  : AOTX_AFFECT_EVENT_TASK_DONE);
#endif
    hold->state = (verdict == AOTX_VERDICT_REFUTE) ? AOTX_TASK_FAILED : AOTX_TASK_DONE;
    if (hold->state == AOTX_TASK_DONE) {
        atomicAdd(&aotx_agent_count.done, 1u);
    } else {
        atomicAdd(&aotx_agent_count.failed, 1u);
    }
    atomicAdd(&aotx_agent_count.verdicts, 1u);
    const char *word = (verdict == AOTX_VERDICT_UPHOLD) ? "uphold"
                     : ((verdict == AOTX_VERDICT_REFUTE) ? "refute"
                        : ((verdict == AOTX_VERDICT_UNCERTAIN) ? "uncertain" : "no word"));
    aotx_task_note(task, AOTX_WRITER_AGENT_BASE + agent, hold->result, hold->result_len,
                   tick);
    aotx_agent_handoff(agent, task, "verdict of task ", word, aotx_cli_length(word), tick);
    me->task = ~0u;
    me->state = AOTX_AGENT_STATE_IDLE;
}

/* Decide what follows the turn that ended. The reply holds a tool call, or it holds the
 * answer. A call to a tool the role does not hold fails to no tool, after the source
 * design, and the reply is then the answer. */
__device__ __forceinline__ static void aotx_agent_post(unsigned int agent,
                                                       unsigned long long tick)
{
    aotx_agent *me = &aotx_agents.agent[agent];
    aotx_agent_work *gear = &aotx_agent_gear[agent];
    unsigned int entry = gear->call.entry;
    if (gear->stop_requested != 0u) {
        entry = AOTX_CATALOG_NO_ENTRY;
        gear->stopped = 1u;
    }
    unsigned int request = 0u;
    if (entry < AOTX_MODULE_SLOTS && aotx_tool_outcome_armed(agent) != 0) {
        /* An armed result stands in for the tool the turn called. No request opens and no
         * tool runs. The result stands on the request slot as complete. The turn ends with
         * the tool finish, so the next turn carries the result as a real one. An arm of
         * no result takes the call off, and the turn ends with the reply as written. */
        request = aotx_tool_outcome_request(agent, &gear->call, tick);
        if (request == 0u) {
            entry = AOTX_CATALOG_NO_ENTRY;
        }
    } else if (entry < AOTX_MODULE_SLOTS) {
        if (aotx_agent_may_call(me->role, entry) == 0) {
            gear->refused += 1u;
            atomicAdd(&aotx_agent_count.bad_calls, 1u);
#ifdef AOTX_AFFECT
            aotx_affect_mark(agent, AOTX_AFFECT_EVENT_ROLE_REFUSED);
#endif
            entry = AOTX_CATALOG_NO_ENTRY;
        } else if (gear->call.over != 0u) {
            /* The values of the call do not fit the argument line, so the tool cannot
             * run. The call ends at once with an error result which names the cause. */
            request = aotx_tool_over_request(agent, &gear->call, tick);
            if (request == 0u) {
                entry = AOTX_CATALOG_NO_ENTRY;
            }
        } else if (gear->call.error != 0u) {
            request = aotx_tool_error_request(agent, &gear->call, tick);
            if (request == 0u) {
                entry = AOTX_CATALOG_NO_ENTRY;
            }
        } else {
            request = aotx_tool_request(agent, &gear->call,
                                        aotx_agent_needs_auth(me->role, entry), tick);
            if (request == 0u) {
                entry = AOTX_CATALOG_NO_ENTRY;
            }
        }
    }
    if (entry < AOTX_MODULE_SLOTS) {
        aotx_agent_call_line(agent);
    }
    unsigned int finish = (gear->stopped != 0u) ? AOTX_TURN_STOPPED
                        : ((entry < AOTX_MODULE_SLOTS) ? AOTX_TURN_TOOL
                           : ((gear->limit_end != 0u) ? AOTX_TURN_LIMIT : AOTX_TURN_STOP));
    /* The record of the turn carries the number of a built-in tool, which the seam
     * names. A tool that came in as a module gives zero. The console line and the bus
     * note of that module name it. */
    aotx_agent_reply_records(agent, gear->reply, gear->reply_len);
    unsigned long long manifest = aotx_agent_manifest(
        agent, finish, (entry < AOTX_MODULE_SLOTS) ? gear->call.tool : 0u, request);
    atomicAdd(&aotx_agent_count.turns, 1u);

    if (gear->kind == AOTX_AGENT_TURN_COMPACT) {
        aotx_transcript_summary(agent, gear->reply, gear->reply_len, tick);
        me->state = AOTX_AGENT_STATE_IDLE;
        return;
    }
    const unsigned char *turn_text = gear->message;
    unsigned int turn_len = gear->message_len;
    if (me->task < AOTX_TASK_SLOTS) {
        turn_text = (const unsigned char *)aotx_agents.task[me->task].text;
        turn_len = aotx_agents.task[me->task].text_len;
    }
    unsigned int turn_tokens = aotx_say.slot[agent].turn_tokens + gear->out_tokens;
    aotx_transcript_finish(agent, turn_text, turn_len, gear->reply, gear->reply_len,
                           turn_tokens, manifest);

    if (gear->limit_end != 0u && gear->stopped == 0u
        && entry >= AOTX_MODULE_SLOTS
        && gear->kind == AOTX_AGENT_TURN_MESSAGE && me->task >= AOTX_TASK_SLOTS) {
        char *line = gear->line;
        unsigned int at = 0u;
        const char *head = "reply: the limit of ";
        const char *tail = " tokens ended the reply; give continue to resume";
        while (head[at] != '\0') {
            line[at] = head[at];
            at += 1u;
        }
        at += aotx_text_utoa((unsigned long long)gear->out_tokens, line + at,
                             AOTX_BUS_TEXT_BYTES - at);
        for (unsigned int i = 0u; tail[i] != '\0' && at < AOTX_BUS_TEXT_BYTES; ++i) {
            line[at++] = tail[i];
        }
        if (agent == AOTX_SAY_SLOT) {
            aotx_console_write(line, at);
        }
        gear->continuable = 1u;
    } else {
        gear->continuable = 0u;
    }

    if (entry < AOTX_MODULE_SLOTS) {
        me->request = request;
        me->tool = entry;
        /* The deadline of the agent is the deadline of its request. A request that waits
         * for the operator has none, and the agent then waits with it. */
        me->deadline = aotx_requests.slot[agent].deadline;
        me->state = AOTX_AGENT_STATE_TOOL;
        atomicAdd(&aotx_agent_count.calls, 1u);
        return;
    }
    atomicAdd(&aotx_agent_count.no_calls, 1u);
    /* The armed result of the operator, when one stands, is the result of the turn. It
     * lands on the request slot and its event on this turn. No tool ran, so no turn
     * carries a result. */
    if (aotx_tool_outcome_take(agent) != 0) {
#ifdef AOTX_AFFECT
        aotx_agent_tool_mark(agent, aotx_requests.slot[agent].status);
#endif
    }
    if (gear->kind == AOTX_AGENT_TURN_VERIFY) {
        aotx_agent_judge(agent, tick);
    } else if (me->task < AOTX_TASK_SLOTS) {
        /* A turn that made no reply gives the task no result. That task did not succeed.
         * A sequence which did not open takes this path. */
        aotx_agent_finish(agent, (gear->reply_len != 0u) ? AOTX_TASK_DONE
                                                         : AOTX_TASK_FAILED, tick);
    } else {
        me->state = AOTX_AGENT_STATE_IDLE;
        if (gear->continuable != 0u
            && aotx_setting_count(AOTX_SET_AUTO_CONTINUE) != 0u) {
            if (me->budget_left == 0u) {
                aotx_agent_budget_line(agent);
                return;
            }
            unsigned int length = (unsigned int)sizeof(aotx_agent_continue_text) - 1u;
            if (aotx_agent_queue_message(agent, aotx_agent_continue_text, length,
                                         gear->source_seq) == 0) {
                gear->continuable = 0u;
                gear->automatic_message = 1u;
            }
        }
    }
}

/* Put a text in the hand of an agent as the reply of its turn. */
__device__ __forceinline__ static void aotx_agent_word(unsigned int agent, const char *text)
{
    aotx_agent_work *gear = &aotx_agent_gear[agent];
    unsigned int at = 0u;
    while (text[at] != '\0' && at < AOTX_AGENT_REPLY_BYTES) {
        gear->reply[at] = (unsigned char)text[at];
        at += 1u;
    }
    gear->reply_len = at;
}

/* Take the result of a tool and start the turn that carries it. The caller gives the
 * result, because an agent whose request is gone takes no result at all. */
__device__ __forceinline__ static void aotx_agent_resume(unsigned int agent,
                                                         const char *result,
                                                         unsigned int result_len,
                                                         unsigned long long tick)
{
    aotx_agent *me = &aotx_agents.agent[agent];
    aotx_agent_work *gear = &aotx_agent_gear[agent];
    aotx_request *slot = &aotx_requests.slot[agent];
    int fits = result == 0 || aotx_agent_cut_result(slot, aotx_agent_result_room(me->role, agent));
    if (result != 0) result_len = slot->result_len;
    aotx_transcript_result(agent, &aotx_requests.slot[agent]);
    me->request = 0u;
    me->tool = AOTX_CATALOG_NO_ENTRY;
    me->deadline = 0ull;
    aotx_tool_done[agent] = 0u;
    if (gear->stop_requested != 0u) {
        gear->continuable = 0u;
        me->state = AOTX_AGENT_STATE_IDLE;
#ifdef AOTX_AFFECT
        aotx_affect_end(agent);
        aotx_quality_end(agent);
#endif
        return;
    }
    if (me->budget_left == 0u) {
        /* A verifier that runs out of turns gives no word. The task keeps the result it
         * was given, and the verdict of the record is uncertain. */
        if (gear->kind == AOTX_AGENT_TURN_VERIFY) {
            aotx_agent_word(agent, "uncertain");
            aotx_agent_judge(agent, tick);
        } else if (me->task < AOTX_TASK_SLOTS) {
            aotx_agent_word(agent, "the turns of this task ran out");
            aotx_agent_finish(agent, AOTX_TASK_FAILED, tick);
        } else {
            if (result_len != 0u && aotx_requests.slot[agent].status == AOTX_TOOL_OK) {
                aotx_agent_tool_line(agent, result, result_len);
            }
            aotx_agent_budget_line(agent);
            gear->continuable = 0u;
            me->state = AOTX_AGENT_STATE_IDLE;
        }
#ifdef AOTX_AFFECT
        /* No turn carries the events of this result. The end mark takes them off, and
         * the turn node writes no trace, because no sequence opened. */
        aotx_affect_end(agent);
        aotx_quality_end(agent);
#endif
        return;
    }
    if (fits) {
        if (me->task < AOTX_TASK_SLOTS) {
            const aotx_task *hold = &aotx_agents.task[me->task];
            aotx_agent_begin(agent, 0, (const unsigned char *)hold->text, hold->text_len, 0, 0,
                             0u, result, result_len, tick);
        } else {
            aotx_agent_begin(agent, 0, gear->message, gear->message_len, 0, 0, 0u,
                             result, result_len, tick);
        }
        if (me->state == AOTX_AGENT_STATE_PROMPT) return;
    }
    /* A completed request cannot be retried after its identity is cleared. */
    const char *reason = "agent: the tool continuation prompt was refused; give new input to resume";
    aotx_console_write(reason, aotx_cli_length(reason));
    atomicAdd(&aotx_agent_count.opens_refused, 1u);
    gear->continuable = 0u;
    gear->has_message = 0u;
    gear->wrote = 0u;
    if (me->task < AOTX_TASK_SLOTS) {
        aotx_agent_word(agent, "the tool continuation prompt was refused");
        aotx_agent_finish(agent, AOTX_TASK_FAILED, tick);
    } else {
        me->state = AOTX_AGENT_STATE_IDLE;
    }
#ifdef AOTX_AFFECT
    aotx_affect_end(agent);
    aotx_quality_end(agent);
#endif
}

__global__ void aotx_agent_step(unsigned long long parameter)
{
    /* The node of the tick graph carries the parameter of its capture. The step therefore
     * takes the tick from the device clock, as the commit of the decode does. */
    unsigned long long tick = (parameter != 0ull) ? parameter : aotx_time_tick;
    if (aotx_seam.replaying != 0ull && aotx_seam_replay_clock != 0ull) {
        tick = aotx_seam_replay_clock;
    }
    unsigned int agent = threadIdx.x;
    if (threadIdx.x == 0u) {
        /* The agent of the console stands as soon as the import of its role lands. The
         * role is a module, so no agent of it can stand before the first tick. */
        aotx_agent_boot_spawn();
        aotx_agent_agenda(tick);
    }
    __syncthreads();

    /* The step runs while a replay of the journal runs, exactly as it runs live. The
     * records of this module are derived from the operator lines and the tokens, so a
     * replay must derive them again. A step that stood still would also leave the message
     * of a replayed line in hand. That message would then open a second reply when the
     * replay ended.
     *
     * Three things differ in a replay. No draw is taken, a deadline does not pass, and a
     * request that waits takes a new deadline at the end. Each one is marked where it
     * stands. */
    if (agent >= AOTX_SLOTS) {
        return;
    }
    aotx_agent *me = &aotx_agents.agent[agent];
    aotx_agent_work *gear = &aotx_agent_gear[agent];
    if (me->state == AOTX_AGENT_STATE_FREE) {
        return;
    }

    /* The say path marks every sequence it opens for the console. Only the conductor
     * streams its reply. Every other agent takes that mark off in the same tick, and it
     * does so before the reply node of the console runs. */
    if (gear->wrote != 0u && gear->console == 0u) {
        aotx_say.slot[agent].live = 0u;
    }

    if (me->state == AOTX_AGENT_STATE_IDLE) {
        if (aotx_transcript[agent].force_compact != 0u
            && aotx_transcript_maintain(agent) != 0
            && aotx_transcript[agent].warm >= 2u) {
            gear->kind = AOTX_AGENT_TURN_COMPACT;
            gear->source_seq = 0ull;
            aotx_agent_begin(agent, 0, aotx_agent_compact_instruction,
                             (unsigned int)sizeof(aotx_agent_compact_instruction) - 1u,
                             0, 0, 0u, 0, 0u, tick);
        } else if (me->task < AOTX_TASK_SLOTS) {
            aotx_task *hold = &aotx_agents.task[me->task];
            if (hold->state == AOTX_TASK_ASSIGNED) {
                aotx_agent_begin(agent, 0, (const unsigned char *)hold->text,
                                 hold->text_len, 0, 0, 0u, 0, 0u, tick);
                /* A prompt table that is busy takes no prompt. The task then stays
                 * assigned and the agent starts its turn in a later tick. */
                if (me->state == AOTX_AGENT_STATE_PROMPT) {
                    hold->state = AOTX_TASK_RUNNING;
                }
            } else if (hold->state == AOTX_TASK_VERIFYING && hold->verifier == agent) {
                aotx_agent_begin(agent, "The task:\n",
                                 (const unsigned char *)hold->text, hold->text_len,
                                 "\nThe result:\n", (const unsigned char *)hold->result,
                                 hold->result_len, 0, 0u, tick);
            }
        } else if (gear->has_message != 0u) {
            if (gear->automatic_message == 0u) {
                me->budget_left = aotx_agent_budget_of(me->role);
            } else if (me->budget_left == 0u || gear->stop_requested != 0u) {
                gear->has_message = 0u;
                return;
            }
            aotx_agent_begin(agent, 0, gear->message, gear->message_len, 0, 0, 0u, 0, 0u,
                             tick);
            if (me->state == AOTX_AGENT_STATE_PROMPT) {
                gear->has_message = 0u;
            }
        } else if (aotx_transcript_maintain(agent) != 0
                   && aotx_transcript_needs_compact(agent) != 0u
                   && aotx_transcript[agent].warm >= 2u) {
            gear->kind = AOTX_AGENT_TURN_COMPACT;
            gear->source_seq = 0ull;
            aotx_agent_begin(agent, 0, aotx_agent_compact_instruction,
                             (unsigned int)sizeof(aotx_agent_compact_instruction) - 1u,
                             0, 0, 0u, 0, 0u, tick);
        }
        return;
    }

    if (me->state == AOTX_AGENT_STATE_PROMPT) {
        if (aotx_say.slot[agent].wanted != 0u) {
            return;
        }
        unsigned int state = aotx_seqs.slot[agent].state;
        /* A replay puts the records of many ticks in one tick. The sequence of the turn
         * may therefore be done when the open takes it over. The turn runs from that state
         * as well, and the reply of the journal is the reply of the turn. */
        if (aotx_say.slot[agent].ready != 0u
            && (state == AOTX_SEQ_STATE_PREFILL || state == AOTX_SEQ_STATE_DECODE
                || state == AOTX_SEQ_STATE_DONE)) {
            me->state = AOTX_AGENT_STATE_RUN;
        } else {
            /* The sequence did not open. The turn ends with no reply. */
            atomicAdd(&aotx_agent_count.opens_refused, 1u);
            gear->reply_len = 0u;
            gear->call.tool = AOTX_TOOL_NONE;
            gear->call.entry = AOTX_CATALOG_NO_ENTRY;
            me->state = AOTX_AGENT_STATE_POST;
        }
        return;
    }

    if (me->state == AOTX_AGENT_STATE_RUN) {
        unsigned int state = aotx_seqs.slot[agent].state;
        if (state == AOTX_SEQ_STATE_PREFILL || state == AOTX_SEQ_STATE_DECODE) {
            return;
        }
        gear->reply_len = aotx_agent_take_reply(agent, gear->reply, AOTX_AGENT_REPLY_BYTES);
        /* The reply and the end of the sequence come from one read. The record of the turn
         * states both, and a later read would give the sequence of the turn that follows. */
        gear->out_tokens = aotx_seqs.slot[agent].sampled;
        const aotx_seq *ended = &aotx_seqs.slot[agent];
        gear->last_token = (ended->last == ended->stop
                            || aotx_wrap_end(ended->role, ended->last)) ? 1u : 0u;
        gear->limit_end = (gear->last_token == 0u && ended->sampled >= ended->limit) ? 1u : 0u;
        gear->stopped = ((ended->flags & AOTX_DECODE_MARK_STOP) != 0u) ? 1u : 0u;
        if (gear->stopped != 0u) {
            gear->call.tool = AOTX_TOOL_NONE;
            gear->call.entry = AOTX_CATALOG_NO_ENTRY;
        } else if (aotx_tool_parse(gear->reply, gear->reply_len, &gear->call) != 0) {
            atomicAdd(&aotx_tool_count.parsed, 1u);
            if (gear->call.over != 0u) {
                atomicAdd(&aotx_tool_count.over, 1u);
            }
        } else {
            atomicAdd(&aotx_tool_count.rejected, 1u);
            gear->call.tool = AOTX_TOOL_NONE;
            gear->call.entry = AOTX_CATALOG_NO_ENTRY;
        }
        me->state = AOTX_AGENT_STATE_POST;
        return;
    }

    if (me->state == AOTX_AGENT_STATE_POST) {
        aotx_agent_post(agent, tick);
#ifdef AOTX_AFFECT
        aotx_affect_end(agent);
        aotx_quality_end(agent);
#endif
        return;
    }

    if (me->state == AOTX_AGENT_STATE_TOOL) {
        aotx_request *slot = &aotx_requests.slot[agent];
        if (aotx_tool_done[agent] != 0u && slot->request == me->request) {
#ifdef AOTX_AFFECT
            /* The quality rows of the turn that made the call use the cache of the agent
             * while the agent waits. A result that came at once would take the cache
             * before they ran, so the resume waits a fixed count of ticks. A replay
             * waits the same count, so it opens the next turn in the tick the live run
             * did. */
            if (aotx_quality_wait(agent) != 0) {
                return;
            }
#endif
            slot->request = 0u;
            if (slot->status != AOTX_TOOL_OK) {
                aotx_agent_tool_line(agent, slot->result, slot->result_len);
            }
#ifdef AOTX_AFFECT
            /* The events of the result belong to the turn that carries it. */
            aotx_agent_tool_mark(agent, slot->status);
#endif
            aotx_agent_resume(agent, slot->result, slot->result_len, tick);
            return;
        }
        /* The deadline of the agent follows the deadline of its request. A replay that
         * gives the request a new deadline gives the agent one as well. */
        if (slot->request == me->request && me->request != 0u) {
            me->deadline = slot->deadline;
            return;
        }
        /* The request of this agent is gone from the table. No result can reach it, so
         * the turn ends at the deadline the agent holds. */
        if (tick > me->deadline) {
            static const char reason[] = "the deadline passed";
            aotx_agent_tool_line(agent, reason, (unsigned int)sizeof reason - 1u);
#ifdef AOTX_AFFECT
            aotx_affect_mark(agent, AOTX_AFFECT_EVENT_DEADLINE);
#endif
            aotx_agent_resume(agent, 0, 0u, tick);
        }
    }
}
