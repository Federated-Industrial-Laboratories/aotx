/* Purpose: Resolve conversation tool choices and preserve the active turn selection.
 * Owns: The device policy table and its console status buffer.
 * Launch shape: Prompt threads capture their slots; the command thread changes choices.
 * Lifetime: The run; class A settings restore all conversation choices. */
#include "tool/policy.cuh"
#include "tool/tool.cuh"
#include "agent/agent.cuh"
#include "settings/settings.cuh"

__device__ aotx_tool_policy aotx_tool_policies[AOTX_SLOTS];
static __device__ aotx_cli_out aotx_tool_policy_out;

__device__ void aotx_tool_policy_reset(unsigned int agent)
{
    if (agent >= AOTX_SLOTS) return;
    aotx_tool_policy *policy = &aotx_tool_policies[agent];
    policy->choices = 0u;
    policy->selected = 0u;
    policy->ready = 0u;
    for (unsigned int i = 0u; i < AOTX_CATALOG_MASK_WORDS; ++i) policy->effective[i] = 0u;
}

__device__ unsigned int aotx_tool_policy_selected(unsigned int defaults, unsigned int choices)
{
    unsigned int selected = 0u;
    for (unsigned int group = 0u; group < AOTX_TOOL_POLICY_GROUPS; ++group) {
        unsigned int choice = (choices >> (group * 2u)) & 3u;
        if (choice == AOTX_TOOL_POLICY_ON
            || (choice == AOTX_TOOL_POLICY_INHERIT && (defaults & (1u << group)) != 0u))
            selected |= 1u << group;
    }
    return selected;
}

__device__ unsigned int aotx_tool_policy_group(unsigned int entry)
{
    if (entry >= AOTX_MODULE_SLOTS) return AOTX_TOOL_POLICY_GROUPS;
    unsigned int number = aotx_catalog.entry[entry].tool.built_in;
    if (number >= AOTX_TOOL_MEMORY_RECALL && number <= AOTX_TOOL_SKILL_USE) return number - 1u;
    return number == AOTX_TOOL_FS_STAT ? 8u : 9u;
}

__device__ void aotx_tool_policy_capture(unsigned int agent, unsigned int role)
{
    if (agent >= AOTX_SLOTS) return;
    aotx_tool_policy *policy = &aotx_tool_policies[agent];
    policy->selected = aotx_tool_policy_selected(aotx_setting_count(AOTX_SET_TOOLS_MASK), policy->choices);
    for (unsigned int word = 0u; word < AOTX_CATALOG_MASK_WORDS; ++word) policy->effective[word] = 0u;
    for (unsigned int entry = 0u; entry < AOTX_MODULE_SLOTS; ++entry) {
        unsigned int bit = 1u << aotx_tool_policy_group(entry);
        if ((policy->selected & bit) != 0u && aotx_catalog_may_call(role, entry)
            && aotx_tool_available(entry))
            policy->effective[entry / 32u] |= 1u << (entry % 32u);
    }
    policy->ready = 1u;
}

__device__ int aotx_tool_policy_enabled(unsigned int agent, unsigned int entry)
{
    if (agent >= AOTX_SLOTS || entry >= AOTX_MODULE_SLOTS) return 0;
    return (aotx_tool_policies[agent].selected & (1u << aotx_tool_policy_group(entry))) != 0u;
}

__device__ int aotx_tool_policy_allows(unsigned int agent, unsigned int entry)
{
    if (agent >= AOTX_SLOTS || entry >= AOTX_MODULE_SLOTS) return 0;
    return aotx_catalog_mask_has(aotx_tool_policies[agent].effective, entry);
}

__device__ int aotx_tool_policy_any(unsigned int agent)
{
    if (agent >= AOTX_SLOTS) return 0;
    for (unsigned int word = 0u; word < AOTX_CATALOG_MASK_WORDS; ++word)
        if (aotx_tool_policies[agent].effective[word] != 0u) return 1;
    return 0;
}

__device__ int aotx_tool_policy_key(const char *key, unsigned int length, unsigned int *agent)
{
    const char *prefix = "tools.agent.";
    if (length <= 12u || length > 15u) return 0;
    for (unsigned int i = 0u; i < 12u; ++i) if (key[i] != prefix[i]) return 0;
    unsigned int value = 0u;
    if (length > 13u && key[12] == '0') return 0;
    for (unsigned int i = 12u; i < length; ++i) {
        if (key[i] < '0' || key[i] > '9') return 0;
        value = value * 10u + (unsigned int)(key[i] - '0');
    }
    if (value >= AOTX_SLOTS) return 0;
    *agent = value;
    return 1;
}

__device__ int aotx_tool_policy_valid(long long choices)
{
    if (choices < 0ll || choices > AOTX_TOOL_POLICY_CHOICES) return 0;
    for (unsigned int i = 0u; i < AOTX_TOOL_POLICY_GROUPS; ++i)
        if (((unsigned int)choices >> (2u * i) & 3u) == 3u) return 0;
    return 1;
}

/* Report the next turn selection. An active turn retains its captured mask. */
__device__ void aotx_tool_policy_show(unsigned int agent)
{
    if (agent >= AOTX_SLOTS || aotx_agents.agent[agent].state == AOTX_AGENT_STATE_FREE) return;
    unsigned int defaults = aotx_setting_count(AOTX_SET_TOOLS_MASK);
    unsigned int choices = aotx_tool_policies[agent].choices;
    unsigned int selected = aotx_tool_policy_selected(defaults, choices);
    unsigned int effective = 0u;
    for (unsigned int entry = 0u; entry < AOTX_MODULE_SLOTS; ++entry) {
        unsigned int bit = 1u << aotx_tool_policy_group(entry);
        if ((selected & bit) != 0u && aotx_catalog_may_call(aotx_agents.agent[agent].role, entry)
            && aotx_tool_available(entry)) effective |= bit;
    }
    aotx_tool_policy_body body = {agent, defaults, choices, selected, effective};
    aotx_seam_write(AOTX_WRITER_SYSTEM, AOTX_CLASS_B, AOTX_REC_TOOL_POLICY, 0u,
                    &body, (unsigned int)sizeof body);
    aotx_cli_out *out = &aotx_tool_policy_out;
    aotx_cli_clear(out);
    aotx_cli_say(out, "tools: agent "); aotx_cli_num(out, agent);
    aotx_cli_say(out, " defaults "); aotx_cli_num(out, defaults);
    aotx_cli_say(out, " choices "); aotx_cli_num(out, choices);
    aotx_cli_say(out, " selected "); aotx_cli_num(out, selected);
    aotx_cli_say(out, " effective "); aotx_cli_num(out, effective);
    aotx_cli_console(out);
}

__device__ void aotx_tool_policy_show_all(void)
{
    for (unsigned int agent = 0u; agent < AOTX_SLOTS; ++agent) aotx_tool_policy_show(agent);
}
