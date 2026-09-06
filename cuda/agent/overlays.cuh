/* Purpose: Give skill list and verdict constants of agent prompts.
 * Owns: The skill list heading; nothing else.
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

/* The sentence that names the skills the catalog holds. The model reads the list and asks
 * for one body with skill_use. */
#define AOTX_OVERLAY_SKILLS_HEAD \
    "\nSkills you may ask for with skill_use:\n"


/* The three words a verifier may answer, and the verdict of each one. */
#define AOTX_VERDICT_NONE       0u
#define AOTX_VERDICT_UPHOLD     1u
#define AOTX_VERDICT_REFUTE     2u
#define AOTX_VERDICT_UNCERTAIN  3u

#endif
