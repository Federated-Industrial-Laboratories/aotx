/* Purpose: Run, stop, continue, and report a console conversation.
 * Owns: Nothing; the agent and say tables hold the conversation.
 * Launch shape: Device functions called by the command parser.
 * Lifetime: One command. */
#ifndef AOTX_CLI_CONVERSATION_CUH
#define AOTX_CLI_CONVERSATION_CUH

static __device__ __noinline__ void aotx_cli_say_text(aotx_cli_out *out,
                                                      const unsigned char *text,
                                                      unsigned int length,
                                                      unsigned long long tick)
{
    if (aotx_model[AOTX_MODEL_LANGUAGE].layers == 0u
        && aotx_model[AOTX_MODEL_LANGUAGE_Q4].layers == 0u) {
        aotx_cli_say(out, "say: no language model is loaded");
        aotx_cli_console(out);
        aotx_cli_count.refused += 1u;
        aotx_say.refused += 1u;
        return;
    }
    if (aotx_agents.agent[AOTX_SAY_SLOT].state == AOTX_AGENT_STATE_FREE) {
        aotx_cli_say(out, "say: no conductor agent runs; give the spawn command");
        aotx_cli_console(out);
        aotx_cli_count.refused += 1u;
        aotx_say.refused += 1u;
        return;
    }
    const aotx_say_slot *state = &aotx_say.slot[AOTX_SAY_SLOT];
    if (aotx_agent_gear[AOTX_SAY_SLOT].has_message != 0u || state->wanted != 0u
        || state->live != 0u
        || aotx_agents.agent[AOTX_SAY_SLOT].state != AOTX_AGENT_STATE_IDLE) {
        aotx_cli_say(out, "say: a reply runs; give the stop command to end it");
        aotx_cli_console(out);
        aotx_cli_count.refused += 1u;
        aotx_say.refused += 1u;
        return;
    }
    if (length > AOTX_SAY_BYTES) {
        aotx_cli_say(out, "say: the text is too long; give ");
        aotx_cli_num(out, (unsigned long long)AOTX_SAY_BYTES);
        aotx_cli_say(out, " bytes at most");
        aotx_cli_console(out);
        aotx_cli_count.refused += 1u;
        aotx_say.refused += 1u;
        return;
    }
    if (aotx_agent_message(AOTX_SAY_SLOT, text, length, tick) != 0) {
        aotx_cli_say(out, "say: a reply runs; give the stop command to end it");
        aotx_cli_console(out);
        aotx_cli_count.refused += 1u;
        aotx_say.refused += 1u;
        return;
    }
    aotx_agent_gear[AOTX_SAY_SLOT].continuable = 0u;
    aotx_say.said += 1u;
    if (!aotx_cli_allow()) {
        return;
    }
    aotx_cli_say(out, "conductor: ");
    aotx_say.slot[AOTX_SAY_SLOT].at = aotx_console_start(out->text, out->at);
    aotx_say.slot[AOTX_SAY_SLOT].column = 1u;
    aotx_cli_clear(out);
}

/* Arm the result of the next turn of the console agent. That turn makes no call and takes
 * the armed result. A scripted run gives the line before a say line, so the events of the
 * result come in the order of the script. The status is a tool status, the arm of no
 * result, or the late status for a word the parser did not know. */
static __device__ __noinline__ void aotx_cli_outcome(aotx_cli_out *out, unsigned int status,
                                                     const char *word, unsigned int length)
{
    if (status != AOTX_TOOL_NO_RESULT && status > AOTX_TOOL_REFUSED) {
        aotx_cli_say(out, "outcome: give ok, error, refused or none");
        aotx_cli_console(out);
        aotx_cli_count.refused += 1u;
        return;
    }
    aotx_tool_outcome_arm(AOTX_SAY_SLOT, status);
    aotx_cli_say(out, "outcome: the next turn of the console agent makes no call and ends with ");
    aotx_cli_say(out, (status == AOTX_TOOL_NO_RESULT) ? "no tool result" : "the tool result ");
    if (status != AOTX_TOOL_NO_RESULT) {
        aotx_cli_add(out, word, length);
    }
    aotx_cli_console(out);
}

static __device__ __noinline__ void aotx_cli_stop(aotx_cli_out *out)
{
    aotx_say_slot *state = &aotx_say.slot[AOTX_SAY_SLOT];
    unsigned int seq = aotx_seqs.slot[AOTX_SAY_SLOT].state;
    if (state->wanted == 0u && (seq == AOTX_SEQ_STATE_FREE || state->live == 0u)) {
        aotx_cli_say(out, "stop: no reply runs");
        aotx_cli_console(out);
        aotx_cli_count.refused += 1u;
        return;
    }
    if (state->wanted != 0u) {
        state->wanted = 0u;
    } else {
        aotx_seq_stop(AOTX_SAY_SLOT);
    }
    aotx_say.stopped += 1u;
    aotx_cli_say(out, "stop: the reply ends");
    aotx_cli_console(out);
}

/* Resume the bounded reply of one agent: the conductor for the plain command, a worker
 * for the agent form. */
static __device__ __noinline__ void aotx_cli_continue(aotx_cli_out *out,
                                                       unsigned long long tick,
                                                       unsigned int agent)
{
    aotx_agent_work *gear = &aotx_agent_gear[agent];
    if (gear->continuable == 0u
        || aotx_agents.agent[agent].state != AOTX_AGENT_STATE_IDLE) {
        aotx_cli_say(out, "continue: no reply is available to resume");
        aotx_cli_console(out);
        aotx_cli_count.refused += 1u;
        return;
    }
    unsigned int length = (unsigned int)sizeof(aotx_agent_continue_text) - 1u;
    if (aotx_agent_message(agent, aotx_agent_continue_text, length, tick) != 0) {
        aotx_cli_say(out, "continue: the reply does not resume");
        aotx_cli_console(out);
        aotx_cli_count.refused += 1u;
        return;
    }
    gear->continuable = 0u;
    aotx_cli_say(out, "continue: the reply resumes");
    aotx_cli_console(out);
}

static __device__ __noinline__ void aotx_cli_show_stats(aotx_cli_out *out,
                                                        unsigned long long tick)
{
    unsigned long long seq = aotx_cli_last(AOTX_REC_STATS);
    aotx_cli_say(out, "stats: tick ");
    aotx_cli_num(out, tick);
    if (seq == 0ull) {
        aotx_cli_say(out, " no statistics record");
        aotx_cli_console(out);
        return;
    }
    const volatile aotx_record_header *header = aotx_cli_slot(seq);
    const volatile aotx_stats_body *body =
        (const volatile aotx_stats_body *)((const volatile unsigned char *)header
                                           + AOTX_HEADER_BYTES);
    unsigned long long tick_ns = body->tick_ns;
    unsigned long long records = body->records;
    unsigned long long inbound = body->inbound;
    if (!aotx_cli_holds(header, seq, AOTX_REC_STATS)) {
        aotx_cli_say(out, " no statistics record");
        aotx_cli_console(out);
        return;
    }
    aotx_cli_say(out, " time ");
    aotx_cli_num(out, tick_ns);
    aotx_cli_say(out, " ns records ");
    aotx_cli_num(out, records);
    aotx_cli_say(out, " inbound ");
    aotx_cli_num(out, inbound);
    aotx_cli_console(out);
}

#endif
