/* Purpose: Build the prompt of one turn in the prompt table that the tokenizer nodes read.
 * Owns: Nothing; the prompt table of the say path holds the bytes.
 * Launch shape: Device functions; one call for each agent that starts a turn.
 * Lifetime: The whole run.
 *
 * The nodes of the say path take the whole batch of 64 slots every tick. An agent that
 * starts a turn writes its bytes in its own slot of that table and asks for the tokenize.
 * The nodes then open the sequence of the slot in the tick that follows. */
#ifndef AOTX_AGENT_PROMPT_CUH
#define AOTX_AGENT_PROMPT_CUH

#include "agent/agent_state.cuh"
#include "agent/overlays.cuh"
#include "agent/transcript.cuh"
#include "agent/call.cuh"
#include "catalog/catalog.cuh"
#include "cli/prompt.cuh"
#include "tool/tool_state.cuh"

/* Add a text that ends with a zero byte to the prompt of a slot. */
__device__ __forceinline__ unsigned int aotx_agent_put(unsigned char *out, unsigned int at,
                                                       const char *text)
{
    unsigned int length = 0u;
    while (text[length] != '\0') {
        length += 1u;
    }
    if (at > AOTX_SAY_BYTES || length > AOTX_SAY_BYTES - at) {
        return AOTX_SAY_BYTES + 1u;
    }
    for (unsigned int i = 0u; i < length; ++i) {
        out[at] = (unsigned char)text[i];
        at += 1u;
    }
    return at;
}

/* Add a run of bytes to the prompt of a slot. */
__device__ __forceinline__ unsigned int aotx_agent_put_run(unsigned char *out,
                                                           unsigned int at,
                                                           const unsigned char *text,
                                                           unsigned int length)
{
    if (at > AOTX_SAY_BYTES || length > AOTX_SAY_BYTES - at) {
        return AOTX_SAY_BYTES + 1u;
    }
    for (unsigned int i = 0u; i < length; ++i) {
        out[at] = text[i];
        at += 1u;
    }
    return at;
}


/* Write all arguments of the prior call in the selected model form. */
__device__ __forceinline__ unsigned int aotx_agent_put_call(unsigned char *out,
                                                            unsigned int at,
                                                            const aotx_tool_call *call,
                                                            const unsigned char *reply,
                                                            unsigned int reply_len)
{
    const aotx_wrap *wrap = aotx_wrap_active();
    if (call->prefix_len > reply_len) return AOTX_SAY_BYTES + 1u;
    at = aotx_wrap_put(out, at, AOTX_SAY_BYTES, wrap, AOTX_WRAP_ASSISTANT_HEAD);
    at = aotx_agent_put_run(out, at, reply, call->prefix_len);
    if (call->prefix_len != 0u && aotx_call_format_active()->kind == AOTX_CALL_LLAMA_JSON) {
        at = aotx_wrap_put(out, at, AOTX_SAY_BYTES, wrap, AOTX_WRAP_ASSISTANT_TAIL);
        at = aotx_wrap_put(out, at, AOTX_SAY_BYTES, wrap, AOTX_WRAP_ASSISTANT_HEAD);
    }
    at = aotx_call_render(out, at, call->entry, (const unsigned char *)call->pack,
                          0u, AOTX_TOOL_ARG_BYTES, call->at, call->length);
    at = aotx_wrap_put(out, at, AOTX_SAY_BYTES, wrap, AOTX_WRAP_ASSISTANT_TAIL);
    return at;
}

/* Build the prompt of one turn and ask the tokenize nodes for it. The prompt is the
 * overlay of the role, then the text of the turn. A turn that follows a tool then carries
 * the result of that tool. The header of the answer, with thinking off, comes last. The
 * return is the byte count, or zero when the slot is busy.
 *
 * The bytes of the turn come from the task, the message or the verify text. The caller
 * gives them as one run and a second run. A verify turn therefore gives the task and the
 * result with no buffer of its own. */
__device__ __forceinline__ unsigned int aotx_agent_prompt(unsigned int agent,
                                                          const char *head,
                                                          const unsigned char *first,
                                                          unsigned int first_len,
                                                          const char *middle,
                                                          const unsigned char *second,
                                                          unsigned int second_len,
                                                          const char *result,
                                                          unsigned int result_len)
{
    if (agent >= AOTX_SLOTS) {
        return 0u;
    }
    aotx_say_slot *state = &aotx_say.slot[agent];
    if (state->wanted != 0u) {
        return 0u;
    }
    const aotx_wrap *wrap = aotx_wrap_active();
    if (wrap->usable == 0u) return 0u;
    const aotx_call_format *format = aotx_call_format_active();
    unsigned char *out = aotx_say.prompt[agent];
    aotx_agent_work *gear = &aotx_agent_gear[agent];
    unsigned int role = aotx_agents.agent[agent].role;
    unsigned int next_turn = aotx_agents.agent[agent].turn + 1u;
    unsigned int reserve = first_len + second_len + result_len
                         + AOTX_AGENT_REPLY_BYTES + AOTX_TOOL_RESULT_BYTES + 256u;
    if (aotx_transcript_prepare(agent, first, first_len, next_turn, reserve) == 0) {
        return 0u;
    }
    unsigned int at = 0u;
    for (;;) {
        /* The selected form places the tool list before or after the role text. */
        at = aotx_wrap_put(out, 0u, AOTX_SAY_BYTES, wrap, AOTX_WRAP_SYSTEM_HEAD);
        at = aotx_call_format_put(out, at, AOTX_SAY_BYTES, format, AOTX_CALL_SYSTEM_PREFIX);
        int tools_first = format->kind == AOTX_CALL_QWEN_XML || format->kind == AOTX_CALL_LLAMA_JSON;
        if (tools_first) {
            at = aotx_catalog_tool_list(out, at, role);
            at = aotx_agent_put(out, at, format->kind == AOTX_CALL_QWEN_XML ? "\n\n" : "\n");
        }
        if (role < AOTX_MODULE_SLOTS) {
            const aotx_catalog_run overlay = aotx_catalog.entry[role].role.overlay;
            at = aotx_agent_put_run(out, at, aotx_catalog_arena + overlay.at,
                                    overlay.length);
        }
        at = aotx_catalog_skill_bodies(out, at, role);
        if (!tools_first) at = aotx_catalog_tool_list(out, at, role);
        at = aotx_wrap_put(out, at, AOTX_SAY_BYTES, wrap, AOTX_WRAP_SYSTEM_TAIL);
        aotx_catalog_system_seen(role, (at <= AOTX_SAY_BYTES) ? at : AOTX_SAY_BYTES);
        at = aotx_transcript_prompt(agent, out, at);
        state->turn_at = at;
        const aotx_transcript_agent *history = &aotx_transcript[agent];
        unsigned int last = (history->first + history->count + AOTX_MEMORY_TURNS - 1u)
                          % AOTX_MEMORY_TURNS;
        const aotx_transcript_turn *prior = &history->turn[last];
        int recorded = result != 0 && history->count != 0u && prior->text_live != 0u
                     && prior->tier == AOTX_MEMORY_HOT && prior->result_present != 0u
                     && prior->number == aotx_agents.agent[agent].turn;
        if (!recorded) {
            at = aotx_wrap_put(out, at, AOTX_SAY_BYTES, wrap, AOTX_WRAP_USER_HEAD);
            if (head != 0) at = aotx_agent_put(out, at, head);
            at = aotx_agent_put_run(out, at, first, first_len);
            if (middle != 0) at = aotx_agent_put(out, at, middle);
            at = aotx_agent_put_run(out, at, second, second_len);
            at = aotx_wrap_put(out, at, AOTX_SAY_BYTES, wrap, AOTX_WRAP_USER_TAIL);
            if (result != 0 && result_len != 0u) {
                if (gear->call.entry < AOTX_MODULE_SLOTS) {
                    if (gear->call.over != 0u || format->kind == AOTX_CALL_NONE) {
                        at = aotx_wrap_put(out, at, AOTX_SAY_BYTES, wrap, AOTX_WRAP_ASSISTANT_HEAD);
                        at = aotx_agent_put_run(out, at, gear->reply, gear->reply_len);
                        at = aotx_wrap_put(out, at, AOTX_SAY_BYTES, wrap, AOTX_WRAP_ASSISTANT_TAIL);
                    } else {
                        at = aotx_agent_put_call(out, at, &gear->call, gear->reply, gear->reply_len);
                    }
                }
                at = aotx_call_result(out, at, (const unsigned char *)result, 0u,
                                      result_len, AOTX_TOOL_RESULT_BYTES);
            }
        }
        at = aotx_wrap_generation(out, at, AOTX_SAY_BYTES, wrap);
        if (at <= AOTX_SAY_BYTES) {
            break;
        }
        if (gear->kind == AOTX_AGENT_TURN_COMPACT
            && aotx_transcript_compact_less(agent) != 0) {
            continue;
        }
        if (aotx_transcript_give_hot(agent) != 0) {
            continue;
        }
        aotx_transcript[agent].choice_pending = 0u;
        atomicAdd(&aotx_transcript_count.prompt_refused, 1ull);
        aotx_console_write("agent: the prompt does not fit",
                           aotx_cli_length("agent: the prompt does not fit"));
        return 0u;
    }

    /* The console line of the conductor belongs to the command layer, so the line number
     * and the column of the slot are not touched here. */
    state->length = at;
    state->prompt = 0u;
    state->tokens = 0u;
    state->page_limit = aotx_transcript[agent].limit;
    state->turn_tokens = 0u;
    state->token_deadline = 0ull;
    if (agent != AOTX_SAY_SLOT) {
        aotx_say_count[agent] = 0u;
        state->token_deadline = aotx_time_tick + AOTX_SAY_TOKEN_WAIT_TICKS;
    }
    state->reply_first = 0ull;
    state->reply_records = 0u;
    state->console_mode = 0u;
    state->console_prefix = 0u;
    state->wanted = 1u;

    gear->prompt_len = at;
    gear->input_hash = aotx_agent_hash(out, at);
    gear->wrote = 1u;
    gear->reply_len = 0u;
    gear->call.entry = AOTX_CATALOG_NO_ENTRY;
    gear->call.tool = AOTX_TOOL_NONE;
    gear->call.provenance = 0u;
    gear->call.arg_len = 0u;
    gear->opens += 1u;
    return at;
}

/* Encoded result room excludes the role, turn text, prior call, and selected framing.
 * Current call measurement uses the renderer's count mode without writing prompt bytes.
 * A role-only query uses the task text bound and has no current call to measure. */
__device__ __forceinline__ unsigned int aotx_agent_result_room(unsigned int role,
                                                               unsigned int agent = AOTX_SLOTS)
{
    const aotx_wrap *wrap = aotx_wrap_active();
    const aotx_call_format *format = aotx_call_format_active();
    unsigned int block = aotx_catalog_system_bytes(role);
    if (block == 0u) {
        block = AOTX_CATALOG_OVERLAY_BYTES + AOTX_CATALOG_LIST_BYTES
              + wrap->length[AOTX_WRAP_SYSTEM_HEAD] + wrap->length[AOTX_WRAP_SYSTEM_TAIL]
              + format->length[AOTX_CALL_SYSTEM_PREFIX] + 2u;
    }
    unsigned int framing = wrap->length[AOTX_WRAP_USER_HEAD] + wrap->length[AOTX_WRAP_USER_TAIL]
        + wrap->length[AOTX_WRAP_GENERATION_HEAD] + wrap->length[AOTX_WRAP_THINK_OPEN]
        + wrap->length[AOTX_WRAP_THINK_CLOSE];
    if (format->kind == AOTX_CALL_NONE) {
        framing += wrap->length[AOTX_WRAP_USER_HEAD] + wrap->length[AOTX_WRAP_USER_TAIL]
                 + (unsigned int)sizeof("[tool result]\n") - 1u;
    } else {
        framing += format->length[AOTX_CALL_RESULT_HEAD] + format->length[AOTX_CALL_RESULT_TAIL]
                 + (format->result_json != 0u ? 2u : 0u);
    }
    unsigned int text = 2u * AOTX_TASK_TEXT_BYTES;
    unsigned int call = 0u;
    if (agent < AOTX_SLOTS) {
        if (aotx_say.slot[agent].wanted != 0u) return 0u;
        const aotx_agent_work *gear = &aotx_agent_gear[agent];
        unsigned int task = aotx_agents.agent[agent].task;
        text = task < AOTX_TASK_SLOTS ? aotx_agents.task[task].text_len : gear->message_len;
        if (gear->call.entry < AOTX_MODULE_SLOTS) {
            call = gear->call.over != 0u || format->kind == AOTX_CALL_NONE
                 ? gear->reply_len
                 : aotx_call_render(0, 0u, gear->call.entry,
                     (const unsigned char *)gear->call.pack, 0u, AOTX_TOOL_ARG_BYTES,
                     gear->call.at, gear->call.length);
            if (call > AOTX_SAY_BYTES) return 0u;
            call += wrap->length[AOTX_WRAP_ASSISTANT_HEAD] + wrap->length[AOTX_WRAP_ASSISTANT_TAIL];
            if (call > AOTX_SAY_BYTES) return 0u;
        }
    }
    unsigned int used = block + text + call + 128u + framing;
    return (AOTX_SAY_BYTES > used) ? (AOTX_SAY_BYTES - used) : 0u;
}

#endif
