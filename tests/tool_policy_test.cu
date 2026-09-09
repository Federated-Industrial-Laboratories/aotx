/* Purpose: Check batched tool selection, frozen turns, refusal, and settings replay.
 * Owns: A bounded device ring, synthetic catalog rows, and distinct agent cases.
 * Launch shape: One thread for each agent, with serial journal commit and replay.
 * Lifetime: One test process; no model file or external tool executes. */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "agent/prompt.cuh"
#include "boot/check.h"
#include "tool/policy.cuh"
#include "wrap_fixture.h"

#define AOTX_POLICY_TEST_RING 8192u
#define AOTX_POLICY_TEST_ROLE (AOTX_MODULE_SLOTS - 1u)
#define AOTX_POLICY_TEST_IMPORTED (AOTX_MODULE_SLOTS - 3u)

struct aotx_policy_result {
    unsigned int selected;
    unsigned int effective;
    unsigned int choices;
    unsigned int frozen;
    unsigned int empty;
    unsigned int prompt;
    unsigned int refused;
    unsigned int restored;
    unsigned int reported;
};

static unsigned aotx_policy_applied;
static unsigned aotx_policy_failed;
static void aotx_policy_check(bool ok, const char *what)
{
    ++aotx_policy_applied;
    if (!ok) { ++aotx_policy_failed; std::printf("tool policy: FAILED %s\n", what); }
}

__global__ void aotx_policy_setup(unsigned int count)
{
    if (blockIdx.x != 0u || threadIdx.x != 0u) return;
    aotx_settings_reset();
    aotx_catalog_built_in();
    aotx_catalog_anchor();
    aotx_tool_embed.ready = 1u;
    aotx_time_tick = 7ull;
    aotx_catalog_entry *custom = &aotx_catalog.entry[AOTX_POLICY_TEST_IMPORTED];
    custom->state = AOTX_CATALOG_INSTALLED;
    custom->kind = AOTX_MODULE_TOOL;
    custom->tool.side = AOTX_CATALOG_SIDE_HOST;
    custom->tool.built_in = 0u;
    custom->import = 55u;
    const char name[] = "fixture_custom";
    custom->name_len = sizeof name - 1u;
    for (unsigned i = 0u; i < sizeof name; ++i) custom->name[i] = name[i];
    for (unsigned role = AOTX_POLICY_TEST_ROLE - 1u; role <= AOTX_POLICY_TEST_ROLE; ++role) {
        aotx_catalog.entry[role].state = AOTX_CATALOG_INSTALLED;
        aotx_catalog.entry[role].kind = AOTX_MODULE_ROLE;
        for (unsigned word = 0u; word < AOTX_CATALOG_MASK_WORDS; ++word)
            aotx_catalog.entry[role].role.tools[word] = ~0u;
    }
    unsigned denied = aotx_catalog_find("fs_write", 8u, AOTX_MODULE_TOOL);
    aotx_catalog.entry[AOTX_POLICY_TEST_ROLE - 1u].role.tools[denied / 32u] &= ~(1u << (denied % 32u));
    aotx_agents.live = count;
    for (unsigned agent = 0u; agent < AOTX_SLOTS; ++agent) {
        aotx_agents.agent[agent].state = agent < count ? AOTX_AGENT_STATE_IDLE : AOTX_AGENT_STATE_FREE;
        aotx_agents.agent[agent].role = AOTX_POLICY_TEST_ROLE - agent % 2u;
        aotx_agents.agent[agent].task = ~0u;
        aotx_transcript[agent].pages = AOTX_KV_PAGES_EACH;
        aotx_requests.slot[agent].request = 0u;
    }
    aotx_model[AOTX_MODEL_LANGUAGE].layers = 1u;
    aotx_kvl_make(&aotx_model_space[AOTX_MODEL_LANGUAGE].shape, 1u, 1u, 8u);
    aotx_setting_table.row[AOTX_SET_TOOLS_MASK].value = 341ll;
}

__global__ void aotx_policy_queue(unsigned int count, const unsigned int *choices)
{
    if (blockIdx.x != 0u || threadIdx.x != 0u) return;
    unsigned queued = 0u;
    for (unsigned agent = 0u; agent < count; ++agent) {
        aotx_setting_pending *pending = &aotx_setting_table.pending[queued];
        const char prefix[] = "tools.agent.";
        for (unsigned i = 0u; i < sizeof pending->key; ++i) pending->key[i] = 0;
        unsigned at = 0u;
        for (; at < sizeof prefix - 1u; ++at) pending->key[at] = prefix[at];
        if (agent >= 100u) pending->key[at++] = (char)('0' + agent / 100u);
        if (agent >= 10u) pending->key[at++] = (char)('0' + agent / 10u % 10u);
        pending->key[at++] = (char)('0' + agent % 10u);
        pending->key_len = at;
        pending->scale = 1u;
        pending->value = choices[agent];
        ++queued;
        if (queued == AOTX_SETTING_PENDING_MAX) {
            aotx_setting_table.pending_count = queued;
            aotx_settings_commit(7ull);
            queued = 0u;
        }
    }
    aotx_setting_table.pending_count = queued;
    aotx_settings_commit(7ull);
}

__global__ void aotx_policy_capture(unsigned count, aotx_policy_result *out)
{
    unsigned agent = blockIdx.x * blockDim.x + threadIdx.x;
    if (agent >= count) return;
    aotx_tool_policy_capture(agent, aotx_agents.agent[agent].role);
    out[agent].selected = aotx_tool_policies[agent].selected;
    out[agent].choices = aotx_tool_policies[agent].choices;
    unsigned effective = 0u;
    for (unsigned entry = 0u; entry < AOTX_MODULE_SLOTS; ++entry)
        if (aotx_tool_policy_allows(agent, entry)) effective |= 1u << aotx_tool_policy_group(entry);
    out[agent].effective = effective;
    unsigned reports = 0u;
    for (unsigned long long seq = 1ull; seq <= aotx_seam.dev.tail; ++seq) {
        const aotx_record_header *header = aotx_seam_slot(seq);
        if (header->type != AOTX_REC_TOOL_POLICY || header->body_len != sizeof(aotx_tool_policy_body)) continue;
        const aotx_tool_policy_body *body = (const aotx_tool_policy_body *)aotx_seam_body_of(seq);
        if (body->agent != agent) continue;
        if (header->cls == AOTX_CLASS_B && header->writer == AOTX_WRITER_SYSTEM
            && body->defaults == 341u && body->choices == out[agent].choices
            && body->selected == out[agent].selected && body->effective == effective) ++reports;
    }
    out[agent].reported = reports == 1u;
}

__global__ void aotx_policy_change(unsigned count)
{
    if (blockIdx.x != 0u || threadIdx.x != 0u) return;
    aotx_setting_table.row[AOTX_SET_TOOLS_MASK].value = 0ll;
    for (unsigned agent = 0u; agent < count; ++agent) aotx_tool_policies[agent].choices = 0u;
}

__global__ void aotx_policy_off(unsigned count, aotx_policy_result *out)
{
    unsigned agent = blockIdx.x * blockDim.x + threadIdx.x;
    if (agent >= count) return;
    unsigned held = 0u;
    for (unsigned entry = 0u; entry < AOTX_MODULE_SLOTS; ++entry)
        if (aotx_tool_policy_allows(agent, entry)) held |= 1u << aotx_tool_policy_group(entry);
    out[agent].frozen = held == out[agent].effective;
    aotx_tool_policy_capture(agent, aotx_agents.agent[agent].role);
    out[agent].empty = !aotx_tool_policy_any(agent)
        && aotx_catalog_tool_list(aotx_say.prompt[agent], 7u, aotx_agents.agent[agent].role, agent) == 7u;
    unsigned char message[] = {'u', 's', 'e', 'r', ' ', (unsigned char)('A' + agent % 26u)};
    unsigned bytes = aotx_agent_prompt(agent, 0, message, sizeof message, 0, 0, 0u, 0, 0u);
    bool tool_text = false;
    const char needle[] = "tool";
    for (unsigned at = 0u; at + 4u <= bytes; ++at) {
        bool same = true;
        for (unsigned i = 0u; i < 4u; ++i) same = same && aotx_say.prompt[agent][at + i] == needle[i];
        tool_text = tool_text || same;
    }
    out[agent].prompt = bytes != 0u && !tool_text;
    aotx_say.slot[agent].wanted = 0u;
    aotx_agent_work *gear = &aotx_agent_gear[agent];
    gear->kind = AOTX_AGENT_TURN_MESSAGE;
    gear->has_message = 1u;
    gear->console = 0u;
    gear->call.entry = agent % 2u == 0u ? AOTX_POLICY_TEST_IMPORTED
        : aotx_catalog_find("fs_write", 8u, AOTX_MODULE_TOOL);
    gear->call.tool = aotx_catalog_tool_number(gear->call.entry);
    gear->reply_len = 0u;
    gear->stop_requested = 0u;
    aotx_agents.agent[agent].state = AOTX_AGENT_STATE_POST;
    aotx_agents.agent[agent].turn = 1u;
}

__global__ void aotx_policy_refusal(unsigned count, aotx_policy_result *out)
{
    unsigned agent = blockIdx.x * blockDim.x + threadIdx.x;
    if (agent >= count) return;
    const aotx_request *slot = &aotx_requests.slot[agent];
    out[agent].refused = slot->status == AOTX_TOOL_REFUSED && slot->request != 0u
        && slot->call_seq == 0ull && slot->auth == AOTX_AUTH_NONE && slot->arg_len == 0u
        && aotx_tool_done[agent] != 0u && aotx_tool_embed.state[agent] == AOTX_TOOL_EMBED_NONE;
}

__global__ void aotx_policy_replay(unsigned count, aotx_policy_result *out)
{
    if (blockIdx.x != 0u || threadIdx.x != 0u) return;
    for (unsigned agent = 0u; agent < count; ++agent) aotx_tool_policy_reset(agent);
    aotx_seam.replaying = 1ull;
    unsigned long long tail = aotx_seam.dev.tail;
    unsigned took = 0u;
    for (unsigned long long seq = 1ull; seq <= tail; ++seq) {
        const aotx_record_header *header = aotx_seam_slot(seq);
        if (header->type == AOTX_REC_SETTING && header->body_len == sizeof(aotx_setting_body)) {
            aotx_setting_body body = *(const aotx_setting_body *)
                ((const unsigned char *)header + AOTX_HEADER_BYTES);
            if (aotx_settings_apply(&body, 8ull) == AOTX_SETTING_TOOK) ++took;
        }
    }
    for (unsigned agent = 0u; agent < count; ++agent)
        out[agent].restored = took == count && aotx_tool_policies[agent].choices == out[agent].choices;
    aotx_seam.replaying = 0ull;
}

static void aotx_policy_run(unsigned count)
{
    unsigned char *ring = nullptr;
    aotx_policy_result *result = nullptr;
    unsigned *choices = nullptr;
    aotx_check_runtime(cudaMallocManaged(&ring, AOTX_POLICY_TEST_RING * AOTX_SLOT_BYTES), "ring");
    aotx_check_runtime(cudaMallocManaged(&result, sizeof(*result) * count), "result");
    aotx_check_runtime(cudaMallocManaged(&choices, sizeof(*choices) * count), "choices");
    std::memset(ring, 0, AOTX_POLICY_TEST_RING * AOTX_SLOT_BYTES);
    std::memset(result, 0, sizeof(*result) * count);
    aotx_seam_state seam = {};
    seam.dev.base = ring; seam.dev.slot_count = AOTX_POLICY_TEST_RING;
    seam.dev.mask = AOTX_POLICY_TEST_RING - 1u; seam.apply.state_hash = AOTX_FNV_BASIS;
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_seam, &seam, sizeof seam), "seam");
    for (unsigned agent = 0u; agent < count; ++agent) {
        unsigned number = agent + 1u;
        choices[agent] = 0u;
        for (unsigned group = 0u; group < AOTX_TOOL_POLICY_GROUPS; ++group) {
            choices[agent] |= (number % 3u) << (group * 2u);
            number /= 3u;
        }
    }
    void *address = nullptr;
#define AOTX_POLICY_CLEAR(symbol) \
    aotx_check_runtime(cudaGetSymbolAddress(&address, symbol), "symbol"); \
    aotx_check_runtime(cudaMemset(address, 0, sizeof(symbol)), "clear")
    AOTX_POLICY_CLEAR(aotx_agent_gear);
    AOTX_POLICY_CLEAR(aotx_transcript);
    AOTX_POLICY_CLEAR(aotx_say);
    AOTX_POLICY_CLEAR(aotx_requests);
    AOTX_POLICY_CLEAR(aotx_tool_embed);
#undef AOTX_POLICY_CLEAR
    aotx_policy_setup<<<1, 1>>>(count);
    aotx_policy_queue<<<1, 1>>>(count, choices);
    aotx_policy_capture<<<1, AOTX_SLOTS>>>(count, result);
    aotx_policy_change<<<1, 1>>>(count);
    aotx_policy_off<<<1, AOTX_SLOTS>>>(count, result);
    aotx_agent_step<<<1, AOTX_SLOTS>>>(0ull);
    aotx_policy_refusal<<<1, AOTX_SLOTS>>>(count, result);
    aotx_policy_replay<<<1, 1>>>(count, result);
    aotx_check_runtime(cudaDeviceSynchronize(), "tool policy batch");
    for (unsigned agent = 0u; agent < count; ++agent) {
        unsigned expected = 0u;
        for (unsigned group = 0u; group < AOTX_TOOL_POLICY_GROUPS; ++group) {
            unsigned choice = choices[agent] / (1u << (group * 2u)) % 4u;
            if (choice == 2u || (choice == 0u && (341u / (1u << group)) % 2u != 0u))
                expected += 1u << group;
        }
        aotx_policy_check(result[agent].selected == expected, "conversation precedence");
        aotx_policy_check(result[agent].effective == (expected & (agent % 2u ? ~16u : ~0u)), "role cap");
        aotx_policy_check(result[agent].frozen, "active turn selection is fixed");
        aotx_policy_check(result[agent].empty, "all-off omits the complete tool list");
        aotx_policy_check(result[agent].prompt, "all-off prompt omits tool instructions");
        aotx_policy_check(result[agent].refused, "disabled generated calls execute no tool");
        aotx_policy_check(result[agent].restored, "settings replay restores distinct choices");
        aotx_policy_check(result[agent].reported, "the typed status matches the device policy");
    }
    std::printf("tool policy: N=%u complete\n", count);
    cudaFree(choices); cudaFree(result); cudaFree(ring);
}

int main()
{
    aotx_test_wrap_open();
    aotx_policy_run(1u);
    aotx_policy_run(AOTX_SLOTS);
    std::printf("tool policy: %u applied, %u failed\n", aotx_policy_applied, aotx_policy_failed);
    return aotx_policy_failed == 0u && aotx_policy_applied == 8u * (AOTX_SLOTS + 1u) ? 0 : 1;
}
