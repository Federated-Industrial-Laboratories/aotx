/* Purpose: Give the tool markup and verdict constants of agent prompts.
 * Owns: The tool markup text; nothing else.
 * Launch shape: Device text; one read for each agent that builds a prompt.
 * Lifetime: The whole run; the text is constant.
 *
 * The duty sentence of a role is no longer text of this file. A role is a module of the
 * catalog, and its overlay is a run of the catalog arena. The tool list and the skill
 * list are built from the catalog beside it. A tool installed in one tick is therefore in
 * the prompt of the tick that follows. */
#ifndef AOTX_AGENT_OVERLAYS_CUH
#define AOTX_AGENT_OVERLAYS_CUH

#include "agent/agent.cuh"

/* Each tool is one JSON object between the tool tags. */

#define AOTX_OVERLAY_TOOLS_HEAD \
    "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n" \
    "You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n"

#define AOTX_OVERLAY_TOOLS_TAIL "</tools>\n"

/* The sentence that names the skills the catalog holds. The model reads the list and asks
 * for one body with skill_use. */
#define AOTX_OVERLAY_SKILLS_HEAD \
    "\nSkills you may ask for with skill_use:\n"

#define AOTX_OVERLAY_CALL_FORM \
    "\nFor each function call, return a json object with function name and " \
    "arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n" \
    "{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>"

__device__ static const char aotx_overlay_result_head[] = "\n<tool_response>\n";
__device__ static const char aotx_overlay_result_tail[] = "\n</tool_response>";

/* The three words a verifier may answer, and the verdict of each one. */
#define AOTX_VERDICT_NONE       0u
#define AOTX_VERDICT_UPHOLD     1u
#define AOTX_VERDICT_REFUTE     2u
#define AOTX_VERDICT_UNCERTAIN  3u

#endif
