/* Purpose: Check result admission with distinct tool choices for the same role.
 * Owns: A temporary record ring and distinct per-agent prompts and results.
 * Launch shape: One thread per agent, at one and all profile slots.
 * Lifetime: One test process with fixed wrappers and no model weights. */
#include <stdio.h>
#include <string.h>
#include "agent/prompt.cuh"
#include "boot/check.h"
#include "wrap_fixture.h"

#define AOTX_ROOM_ROLE (AOTX_MODULE_SLOTS - 1u)
#define AOTX_ROOM_RING 4096u

struct aotx_room_result {
    unsigned int prompt, system, before, after, accepted, continuation, length;
};

__global__ void aotx_room_setup(unsigned int count)
{
    if (threadIdx.x != 0u) return;
    aotx_settings_reset();
    aotx_setting_table.row[AOTX_SET_RECALL_K].value = 0ll;
    aotx_setting_table.row[AOTX_SET_COMPACT_AT].value = 0ll;
    aotx_catalog_built_in();
    aotx_catalog_anchor();
    aotx_tool_embed.ready = 1u;
    aotx_catalog_entry *role = &aotx_catalog.entry[AOTX_ROOM_ROLE];
    role->state = AOTX_CATALOG_INSTALLED;
    role->kind = AOTX_MODULE_ROLE;
    for (unsigned int i = 0u; i < AOTX_CATALOG_MASK_WORDS; ++i) role->role.tools[i] = ~0u;
    aotx_agents.live = count;
    aotx_model[AOTX_MODEL_LANGUAGE].layers = 1u;
    aotx_kvl_make(&aotx_model_space[AOTX_MODEL_LANGUAGE].shape, 1u, 1u, 8u);
}

__global__ void aotx_room_prepare(unsigned int count)
{
    unsigned int agent = threadIdx.x;
    if (agent >= count) return;
    aotx_agents.agent[agent].state = AOTX_AGENT_STATE_IDLE;
    aotx_agents.agent[agent].role = AOTX_ROOM_ROLE;
    aotx_agents.agent[agent].task = ~0u;
    aotx_transcript[agent].pages = AOTX_KV_PAGES_EACH;
    aotx_tool_policies[agent].choices = agent % 2u == 0u ? 349525u : 699050u;
    aotx_agent_work *gear = &aotx_agent_gear[agent];
    gear->kind = AOTX_AGENT_TURN_MESSAGE;
    gear->source_seq = 4000ull + agent;
    gear->message_len = agent % 2u == 0u ? 4499u - agent : 17u + agent;
    gear->call.entry = AOTX_CATALOG_NO_ENTRY;
    for (unsigned int i = 0u; i < gear->message_len; ++i)
        gear->message[i] = (unsigned char)('a' + (i + agent) % 26u);
}

/* Build the long tools-off prompts first, then the tools-on prompts of the same role. */
__global__ void aotx_room_prompt(unsigned int count, unsigned int parity, aotx_room_result *out)
{
    unsigned int agent = threadIdx.x;
    if (agent >= count || agent % 2u != parity) return;
    aotx_agent_work *gear = &aotx_agent_gear[agent];
    out[agent].prompt = aotx_agent_prompt(agent, 0, gear->message, gear->message_len,
                                          0, 0, 0u, 0, 0u);
    aotx_say.slot[agent].wanted = 0u;
    out[agent].system = gear->system_bytes;
    out[agent].before = aotx_agent_result_room(agent);
}

__global__ void aotx_room_admit(unsigned int count, aotx_room_result *out)
{
    unsigned int agent = threadIdx.x;
    if (agent >= count) return;
    aotx_agent_work *gear = &aotx_agent_gear[agent];
    out[agent].after = aotx_agent_result_room(agent);
    aotx_request *slot = &aotx_requests.slot[agent];
    slot->agent = agent;
    slot->result_len = 64u + agent;
    for (unsigned int i = 0u; i < slot->result_len; ++i)
        slot->result[i] = (char)('A' + (i + agent) % 26u);
    out[agent].accepted = aotx_agent_cut_result(slot, out[agent].after);
    out[agent].length = slot->result_len;
    out[agent].continuation = aotx_agent_prompt(agent, 0, gear->message, gear->message_len,
                                               0, 0, 0u, slot->result, slot->result_len);
}

#define AOTX_ROOM_CLEAR(symbol) do { \
    void *address = NULL; \
    aotx_check_runtime(cudaGetSymbolAddress(&address, symbol), "cudaGetSymbolAddress"); \
    aotx_check_runtime(cudaMemset(address, 0, sizeof(symbol)), "cudaMemset"); \
} while (0)

int main(void)
{
    aotx_test_wrap_open();
    unsigned int checks = 0u, failed = 0u;
    for (unsigned int count = 1u; count <= AOTX_SLOTS; count *= AOTX_SLOTS) {
        AOTX_ROOM_CLEAR(aotx_agents);
        AOTX_ROOM_CLEAR(aotx_agent_gear);
        AOTX_ROOM_CLEAR(aotx_transcript);
        AOTX_ROOM_CLEAR(aotx_transcript_text);
        AOTX_ROOM_CLEAR(aotx_say);
        AOTX_ROOM_CLEAR(aotx_catalog);
        AOTX_ROOM_CLEAR(aotx_tool_policies);
        unsigned char *ring = NULL;
        aotx_room_result *out = NULL;
        aotx_check_runtime(cudaMallocManaged(&ring, AOTX_ROOM_RING * AOTX_SLOT_BYTES), "cudaMallocManaged");
        aotx_check_runtime(cudaMallocManaged(&out, sizeof(*out) * count), "cudaMallocManaged");
        memset(ring, 0, AOTX_ROOM_RING * AOTX_SLOT_BYTES);
        memset(out, 0, sizeof(*out) * count);
        aotx_seam_state seam = {};
        seam.dev.base = ring;
        seam.dev.slot_count = AOTX_ROOM_RING;
        seam.dev.mask = AOTX_ROOM_RING - 1u;
        aotx_check_runtime(cudaMemcpyToSymbol(aotx_seam, &seam, sizeof seam), "cudaMemcpyToSymbol");
        aotx_room_setup<<<1, 1>>>(count);
        aotx_room_prepare<<<1, AOTX_SLOTS>>>(count);
        aotx_room_prompt<<<1, AOTX_SLOTS>>>(count, 0u, out);
        aotx_room_prompt<<<1, AOTX_SLOTS>>>(count, 1u, out);
        aotx_room_admit<<<1, AOTX_SLOTS>>>(count, out);
        aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        for (unsigned int agent = 0u; agent < count; ++agent) {
            const bool ok[] = {
                out[agent].prompt != 0u && out[agent].system != 0u,
                out[agent].before > 128u && out[agent].after == out[agent].before,
                out[agent].accepted == 1u && out[agent].length == 64u + agent,
                out[agent].continuation > out[agent].prompt
                    && out[agent].continuation <= AOTX_SAY_BYTES,
                count == 1u || (agent % 2u == 0u ? out[agent].system < out[agent + 1u].system
                                                : out[agent].system > out[agent - 1u].system)
            };
            for (unsigned int i = 0u; i < sizeof ok / sizeof ok[0]; ++i) {
                ++checks;
                if (!ok[i]) {
                    ++failed;
                    printf("prompt policy: FAILED agent %u check %u room %u to %u prompt %u to %u\n",
                           agent, i, out[agent].before, out[agent].after,
                           out[agent].prompt, out[agent].continuation);
                }
            }
        }
        cudaFree(out);
        cudaFree(ring);
    }
    printf("prompt policy: %u checks, %u failed\n", checks, failed);
    return failed == 0u && checks == 5u * (1u + AOTX_SLOTS) ? 0 : 1;
}
