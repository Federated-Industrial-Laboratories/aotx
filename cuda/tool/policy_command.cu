/* Purpose: Queue named tool choices through the existing settings journal path.
 * Owns: No state; pending settings hold command changes until tick commit.
 * Launch shape: The input consumer handles a batch of command lines in order.
 * Lifetime: Pending changes last until the next tick commit. */
#include "tool/policy.cuh"
#include "settings/settings.cuh"
#include "seam/seam.cuh"

static __device__ int aotx_tool_policy_equal(const char *text, unsigned int length, const char *word)
{
    unsigned int i = 0u;
    while (i < length && word[i] != '\0' && text[i] == word[i]) ++i;
    return i == length && word[i] == '\0';
}

__device__ void aotx_tool_policy_command(aotx_cli_out *out, unsigned int agent,
    const char *name, unsigned int name_len, const char *choice, unsigned int choice_len,
    unsigned int extra)
{
    if (name_len == 0u && choice_len == 0u && extra == 0u) {
        if (agent < AOTX_SLOTS) aotx_tool_policy_show(agent);
        else aotx_tool_policy_show_all();
        return;
    }
    if (aotx_seam.replaying != 0ull) return;
    const char *names[] = {
#define AOTX_TOOL_NAME(name) name,
        AOTX_TOOL_POLICY_NAMES(AOTX_TOOL_NAME)
#undef AOTX_TOOL_NAME
    };
    unsigned int group = AOTX_TOOL_POLICY_GROUPS;
    for (unsigned int i = 0u; i < AOTX_TOOL_POLICY_GROUPS; ++i)
        if (aotx_tool_policy_equal(name, name_len, names[i])) group = i;
    int all = aotx_tool_policy_equal(name, name_len, "all");
    unsigned int value = aotx_tool_policy_equal(choice, choice_len, "inherit") ? AOTX_TOOL_POLICY_INHERIT
        : aotx_tool_policy_equal(choice, choice_len, "off") ? AOTX_TOOL_POLICY_OFF
        : aotx_tool_policy_equal(choice, choice_len, "on") ? AOTX_TOOL_POLICY_ON : 3u;
    if (extra != 0u || (!all && group == AOTX_TOOL_POLICY_GROUPS) || value == 3u
        || (agent >= AOTX_SLOTS && value == AOTX_TOOL_POLICY_INHERIT)) {
        aotx_cli_say(out, "tools: give a tool name or all, then on or off; an agent also takes inherit");
        aotx_cli_console(out); aotx_cli_count.refused += 1u; return;
    }
    unsigned int count = aotx_setting_table.pending_count;
    if (count >= AOTX_SETTING_PENDING_MAX) {
        aotx_cli_say(out, "tools: the settings queue is full; send the command again");
        aotx_cli_console(out); aotx_cli_count.refused += 1u; return;
    }
    char key[AOTX_SETTING_WIRE_KEY_BYTES] = {};
    const char *prefix = agent < AOTX_SLOTS ? "tools.agent." : "tools.mask";
    unsigned int key_len = 0u;
    while (prefix[key_len] != '\0') { key[key_len] = prefix[key_len]; ++key_len; }
    if (agent < AOTX_SLOTS) {
        if (agent >= 100u) key[key_len++] = (char)('0' + agent / 100u);
        if (agent >= 10u) key[key_len++] = (char)('0' + agent / 10u % 10u);
        key[key_len++] = (char)('0' + agent % 10u);
    }
    unsigned int packed = agent < AOTX_SLOTS ? aotx_tool_policies[agent].choices
                                             : aotx_setting_count(AOTX_SET_TOOLS_MASK);
    for (unsigned int i = 0u; i < count; ++i) {
        const aotx_setting_pending *pending = &aotx_setting_table.pending[i];
        if (pending->key_len == key_len && aotx_tool_policy_equal(pending->key, key_len, key))
            packed = (unsigned int)pending->value;
    }
    for (unsigned int i = 0u; i < AOTX_TOOL_POLICY_GROUPS; ++i) {
        if (!all && i != group) continue;
        if (agent < AOTX_SLOTS) packed = (packed & ~(3u << (2u * i))) | (value << (2u * i));
        else if (value == AOTX_TOOL_POLICY_ON) packed |= 1u << i;
        else packed &= ~(1u << i);
    }
    aotx_setting_pending *pending = &aotx_setting_table.pending[count];
    pending->value = packed;
    pending->scale = 1u;
    pending->key_len = key_len;
    for (unsigned int i = 0u; i < sizeof pending->key; ++i) pending->key[i] = key[i];
    aotx_setting_table.pending_count = count + 1u;
    aotx_cli_say(out, "tools: the choice is pending for the next turn");
    aotx_cli_console(out);
}
