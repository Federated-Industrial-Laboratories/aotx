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
#include "catalog/catalog.cuh"
#include "cli/prompt.cuh"
#include "tool/tool_state.cuh"

/* Add a text that ends with a zero byte to the prompt of a slot. */
__device__ __forceinline__ unsigned int aotx_agent_put(unsigned char *out, unsigned int at,
                                                       const char *text)
{
    for (unsigned int i = 0u; text[i] != '\0' && at < AOTX_SAY_BYTES; ++i) {
        out[at] = (unsigned char)text[i];
        at += 1u;
    }
    return at;
}

/* Add a run of bytes to the prompt of a slot. Bytes past the end of the table are dropped. */
__device__ __forceinline__ unsigned int aotx_agent_put_run(unsigned char *out,
                                                           unsigned int at,
                                                           const unsigned char *text,
                                                           unsigned int length)
{
    for (unsigned int i = 0u; i < length && at < AOTX_SAY_BYTES; ++i) {
        out[at] = text[i];
        at += 1u;
    }
    return at;
}

/* Add a run of bytes as the content of a JSON string. The three bytes the schema does not
 * take in a string get their escape. */
__device__ __forceinline__ unsigned int aotx_agent_put_json(unsigned char *out,
                                                            unsigned int at,
                                                            const char *text,
                                                            unsigned int length)
{
    for (unsigned int i = 0u; i < length && at + 2u <= AOTX_SAY_BYTES; ++i) {
        char byte = text[i];
        if (byte == '"' || byte == '\\') {
            out[at++] = (unsigned char)'\\';
            out[at++] = (unsigned char)byte;
        } else if (byte == '\n') {
            out[at++] = (unsigned char)'\\';
            out[at++] = (unsigned char)'n';
        } else if ((unsigned char)byte >= 0x20u) {
            out[at++] = (unsigned char)byte;
        }
    }
    return at;
}

/* Write the turn of the agent that made a tool call, as the chat template writes it. The
 * pieces are an assistant block with the call and a user block with the response. The
 * model needs the call it made in front of the response. Without that block the model
 * makes the same call again in place of an answer. */
__device__ __forceinline__ unsigned int aotx_agent_put_call(unsigned char *out,
                                                            unsigned int at,
                                                            const aotx_tool_call *call)
{
    at = aotx_agent_put(out, at, aotx_overlay_user_end);
    at = aotx_agent_put(out, at, "<|im_start|>assistant\n<tool_call>\n{\"name\": \"");
    if (call->entry < AOTX_MODULE_SLOTS) {
        at = aotx_agent_put_run(out, at, (const unsigned char *)
                                aotx_catalog.entry[call->entry].name,
                                aotx_catalog.entry[call->entry].name_len);
    }
    at = aotx_agent_put(out, at, "\", \"arguments\": {");
    if (call->tool == AOTX_TOOL_MEMORY_WRITE) {
        at = aotx_agent_put(out, at, "\"provenance\": \"");
        at = aotx_agent_put(out, at, aotx_tool_provenance_name(call->provenance));
        at = aotx_agent_put(out, at, "\", ");
    }
    at = aotx_agent_put(out, at, "\"");
    const aotx_catalog_run key = aotx_catalog_arg_key(call->entry, call->key);
    at = aotx_agent_put_run(out, at, aotx_catalog_arena + key.at, key.length);
    at = aotx_agent_put(out, at, "\": \"");
    at = aotx_agent_put_json(out, at, call->arg, call->arg_len);
    at = aotx_agent_put(out, at, "\"}}\n</tool_call>");
    at = aotx_agent_put(out, at, aotx_overlay_user_end);
    at = aotx_agent_put(out, at, aotx_overlay_user);
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
    unsigned char *out = aotx_say.prompt[agent];
    aotx_agent_work *gear = &aotx_agent_gear[agent];
    unsigned int role = aotx_agents.agent[agent].role;
    /* The system block starts with the duty sentence of the role, which is a run of the
     * arena. The bodies of the skills of the role follow it, then the two lists. */
    unsigned int at = aotx_agent_put(out, 0u, AOTX_OVERLAY_HEAD);
    if (role < AOTX_MODULE_SLOTS) {
        const aotx_catalog_run overlay = aotx_catalog.entry[role].role.overlay;
        at = aotx_agent_put_run(out, at, aotx_catalog_arena + overlay.at, overlay.length);
    }
    at = aotx_catalog_skill_bodies(out, at, role);
    at = aotx_catalog_tool_list(out, at, role);
    aotx_catalog_system_seen(role, at);
    at = aotx_agent_put(out, at, aotx_overlay_user);
    if (head != 0) {
        at = aotx_agent_put(out, at, head);
    }
    at = aotx_agent_put_run(out, at, first, first_len);
    if (middle != 0) {
        at = aotx_agent_put(out, at, middle);
    }
    at = aotx_agent_put_run(out, at, second, second_len);
    if (result != 0 && result_len != 0u) {
        if (gear->call.entry < AOTX_MODULE_SLOTS) {
            at = aotx_agent_put_call(out, at, &gear->call);
        }
        at = aotx_agent_put(out, at, aotx_overlay_result_head);
        at = aotx_agent_put_run(out, at, (const unsigned char *)result, result_len);
        at = aotx_agent_put(out, at, aotx_overlay_result_tail);
    }
    at = aotx_agent_put(out, at, aotx_overlay_user_end);
    at = aotx_agent_put(out, at, aotx_overlay_assistant);

    /* The console line of the conductor belongs to the command layer, so the line number
     * and the column of the slot are not touched here. */
    state->length = at;
    state->prompt = 0u;
    state->tokens = 0u;
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

/* Bytes of a tool result that a prompt of a role takes. The table holds the system block,
 * the text of the turn and the wrap beside it, so the result takes the room that is left.
 * The system block of a role is measured when the role builds a prompt. A role that has
 * not built one gives the bound of an overlay and the bound of a list. That figure
 * overstates the block and never understates it. */
__device__ __forceinline__ unsigned int aotx_agent_result_room(unsigned int role)
{
    unsigned int block = aotx_catalog_system_bytes(role);
    if (block == 0u) {
        block = AOTX_CATALOG_OVERLAY_BYTES + AOTX_CATALOG_LIST_BYTES;
    }
    unsigned int used = block + 2u * AOTX_TASK_TEXT_BYTES + 128u;
    return (AOTX_SAY_BYTES > used) ? (AOTX_SAY_BYTES - used) : 0u;
}

#endif
