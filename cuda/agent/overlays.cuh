/* Purpose: Give the constant prompt overlay of each role and the pieces of the chat wrap.
 * Owns: The overlay text of the three roles; nothing else.
 * Launch shape: Device functions; one call for each agent that builds a prompt.
 * Lifetime: The whole run; the text is constant. */
#ifndef AOTX_AGENT_OVERLAYS_CUH
#define AOTX_AGENT_OVERLAYS_CUH

#include "agent/agent.cuh"

/* Overlay numbers. A role row names one of them. */
#define AOTX_OVERLAY_CONDUCTOR  0u
#define AOTX_OVERLAY_WORKER     1u
#define AOTX_OVERLAY_VERIFIER   2u

/* The bytes of the longest overlay. The prompt table must hold the overlay, the message or
 * the task text, the result of a tool and the wrap. */
#define AOTX_OVERLAY_BYTES      1536u

/* The pieces of the system message that the chat template of the model file defines. The
 * template puts the duty of the role first, then the tool section, then the closing line.
 * Each tool is one JSON object between the tool tags, as the template writes it. */
#define AOTX_OVERLAY_HEAD "<|im_start|>system\n"

#define AOTX_OVERLAY_TOOLS_HEAD \
    "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n" \
    "You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n"

#define AOTX_OVERLAY_TOOL_RECALL \
    "{\"name\": \"memory_recall\", \"description\": \"Find the notes in memory that are " \
    "nearest to a text.\", \"parameters\": {\"type\": \"object\", \"properties\": " \
    "{\"text\": {\"type\": \"string\", \"description\": \"The text to look for.\"}}, " \
    "\"required\": [\"text\"]}}\n"

#define AOTX_OVERLAY_TOOL_WRITE \
    "{\"name\": \"memory_write\", \"description\": \"Put one note in memory with its " \
    "source.\", \"parameters\": {\"type\": \"object\", \"properties\": {\"provenance\": " \
    "{\"type\": \"string\", \"description\": \"One of computed, fetched, recalled, " \
    "testimony.\"}, \"text\": {\"type\": \"string\", \"description\": \"The note.\"}}, " \
    "\"required\": [\"provenance\", \"text\"]}}\n"

#define AOTX_OVERLAY_TOOL_READ \
    "{\"name\": \"fs_read\", \"description\": \"Read a file below the allowed root. The " \
    "operator must permit this tool.\", \"parameters\": {\"type\": \"object\", " \
    "\"properties\": {\"path\": {\"type\": \"string\", \"description\": \"The path of the " \
    "file.\"}}, \"required\": [\"path\"]}}\n"

#define AOTX_OVERLAY_TOOLS_TAIL \
    "</tools>\n\nFor each function call, return a json object with function name and " \
    "arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n" \
    "{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>" \
    "<|im_end|>\n"

/* The three overlays. Each one names the duty of the role and then lists the tools of that
 * role in the form the template gives them. */
__device__ static const char aotx_overlay_conductor[] =
    AOTX_OVERLAY_HEAD
    "You are the conductor. You answer the operator, you keep the work in order, and you "
    "give a short answer. Call a tool only when you need it."
    AOTX_OVERLAY_TOOLS_HEAD
    AOTX_OVERLAY_TOOL_RECALL
    AOTX_OVERLAY_TOOL_WRITE
    AOTX_OVERLAY_TOOL_READ
    AOTX_OVERLAY_TOOLS_TAIL;

__device__ static const char aotx_overlay_worker[] =
    AOTX_OVERLAY_HEAD
    "You are a worker. You do the task you receive and you give the result in a short "
    "answer. Call a tool only when you need it, and give the result when you have it."
    AOTX_OVERLAY_TOOLS_HEAD
    AOTX_OVERLAY_TOOL_RECALL
    AOTX_OVERLAY_TOOL_WRITE
    AOTX_OVERLAY_TOOL_READ
    AOTX_OVERLAY_TOOLS_TAIL;

__device__ static const char aotx_overlay_verifier[] =
    AOTX_OVERLAY_HEAD
    "You are a verifier. You receive a task and a result. Answer with exactly one word: "
    "uphold if the result answers the task, refute if it does not, uncertain if you cannot "
    "tell. Write that one word and nothing else."
    AOTX_OVERLAY_TOOLS_HEAD
    AOTX_OVERLAY_TOOL_RECALL
    AOTX_OVERLAY_TOOLS_TAIL;

/* The wrap of a turn. The user block carries the task text or the message. A turn that
 * follows a tool carries the result of that tool in the response block. The template puts
 * that block in a user block as well. The assistant header turns thinking off. */
__device__ static const char aotx_overlay_user[] = "<|im_start|>user\n";
__device__ static const char aotx_overlay_user_end[] = "<|im_end|>\n";
__device__ static const char aotx_overlay_result_head[] = "\n<tool_response>\n";
__device__ static const char aotx_overlay_result_tail[] = "\n</tool_response>";
__device__ static const char aotx_overlay_assistant[] =
    "<|im_start|>assistant\n<think>\n\n</think>\n\n";

/* The three words a verifier may answer, and the verdict of each one. */
#define AOTX_VERDICT_NONE       0u
#define AOTX_VERDICT_UPHOLD     1u
#define AOTX_VERDICT_REFUTE     2u
#define AOTX_VERDICT_UNCERTAIN  3u

/* Give the overlay of an overlay number, or the worker overlay when the number is not one
 * of the three. */
__device__ __forceinline__ const char *aotx_overlay_of(unsigned int index)
{
    if (index == AOTX_OVERLAY_CONDUCTOR) {
        return aotx_overlay_conductor;
    }
    if (index == AOTX_OVERLAY_VERIFIER) {
        return aotx_overlay_verifier;
    }
    return aotx_overlay_worker;
}

#endif
