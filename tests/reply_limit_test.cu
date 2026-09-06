/* Purpose: Check reply continuation and the stated console form of a tool call.
 * Owns: One temporary device record ring and the state snapshots of each case.
 * Launch shape: One agent block and one-thread fixture kernels.
 * Lifetime: One test run. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "agent/prompt.cuh"
#include "agent/transcript.cuh"
#include "boot/check.h"
#include "cli/cli.cuh"
#include "seam/seam.cuh"
#include "settings/settings.cuh"

#define AOTX_REPLY_RING 4096ull

typedef struct reply_result {
    unsigned int state;
    unsigned int continuable;
    unsigned int has_message;
    unsigned int message_ok;
    unsigned int turns;
    unsigned int reply_limit;
    unsigned int transcript_full;
} reply_result;

static unsigned int applied;
static unsigned int failed;

static void check(int pass, const char *text)
{
    applied++;
    if (!pass) {
        failed++;
        printf("reply limit: FAILED %s\n", text);
    }
}

#define CLEAR(symbol, bytes) do {                                                \
    void *address = NULL;                                                        \
    aotx_check_runtime(cudaGetSymbolAddress(&address, symbol),                   \
                       "cudaGetSymbolAddress");                                \
    aotx_check_runtime(cudaMemset(address, 0, (bytes)), "cudaMemset");          \
} while (0)

static unsigned char *open_ring(void)
{
    unsigned char *ring = NULL;
    size_t bytes = (size_t)AOTX_REPLY_RING * AOTX_SLOT_BYTES;
    aotx_check_runtime(cudaMallocManaged(&ring, bytes), "cudaMallocManaged");
    memset(ring, 0, bytes);
    aotx_seam_state seam = {};
    seam.dev.base = ring;
    seam.dev.slot_count = AOTX_REPLY_RING;
    seam.dev.mask = AOTX_REPLY_RING - 1ull;
    seam.apply.state_hash = AOTX_FNV_BASIS;
    seam.boot_id = 14ull;
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_seam, &seam, sizeof seam),
                       "cudaMemcpyToSymbol");
    unsigned long long tick = 1ull;
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_time_tick, &tick, sizeof tick),
                       "cudaMemcpyToSymbol");
    return ring;
}

static void clear_state(void)
{
    CLEAR(aotx_transcript, sizeof(aotx_transcript_agent) * AOTX_SLOTS);
    CLEAR(aotx_transcript_text, (size_t)AOTX_SLOTS * AOTX_TRANSCRIPT_TEXT_BYTES);
    CLEAR(aotx_transcript_count, sizeof(aotx_transcript_counts));
    CLEAR(aotx_agents, sizeof(aotx_agent_table));
    CLEAR(aotx_agent_gear, sizeof(aotx_agent_work) * AOTX_SLOTS);
    CLEAR(aotx_say, sizeof(aotx_say_state));
    CLEAR(aotx_seqs, sizeof(aotx_seq_table));
    CLEAR(aotx_setting_table, sizeof(aotx_settings_state));
    CLEAR(aotx_console, sizeof(aotx_console_state));
    CLEAR(aotx_catalog, sizeof(aotx_catalog_state));
}

__global__ void reply_prepare(unsigned int automatic, unsigned int limited,
                              unsigned int turn)
{
    if (blockIdx.x != 0u || threadIdx.x != 0u) return;
    aotx_settings_reset();
    aotx_setting_table.row[AOTX_SET_AUTO_CONTINUE].value = (long long)automatic;
    aotx_agents.live = 1u;
    aotx_agent *agent = &aotx_agents.agent[0];
    agent->state = AOTX_AGENT_STATE_POST;
    agent->role = AOTX_ROLE_NONE;
    agent->task = ~0u;
    agent->turn = turn;
    aotx_transcript[0].pages = 1u;
    aotx_agent_work *gear = &aotx_agent_gear[0];
    const unsigned char question[] = "question";
    const unsigned char answer[] = "partial";
    for (unsigned int i = 0u; i < sizeof question - 1u; ++i) gear->message[i] = question[i];
    gear->message_len = (unsigned int)sizeof question - 1u;
    for (unsigned int i = 0u; i < sizeof answer - 1u; ++i) gear->reply[i] = answer[i];
    gear->reply_len = (unsigned int)sizeof answer - 1u;
    gear->kind = AOTX_AGENT_TURN_MESSAGE;
    gear->source_seq = turn;
    gear->out_tokens = 17u;
    gear->limit_end = limited;
    gear->call.entry = AOTX_CATALOG_NO_ENTRY;
    gear->call.tool = AOTX_TOOL_NONE;
}

__global__ void reply_next(unsigned int limited, unsigned int turn)
{
    if (blockIdx.x != 0u || threadIdx.x != 0u) return;
    aotx_agent *agent = &aotx_agents.agent[0];
    aotx_agent_work *gear = &aotx_agent_gear[0];
    const unsigned char answer[] = "next";
    agent->state = AOTX_AGENT_STATE_POST;
    agent->turn = turn;
    gear->has_message = 0u;
    for (unsigned int i = 0u; i < sizeof answer - 1u; ++i) gear->reply[i] = answer[i];
    gear->reply_len = (unsigned int)sizeof answer - 1u;
    gear->out_tokens = 17u;
    gear->limit_end = limited;
    gear->call.entry = AOTX_CATALOG_NO_ENTRY;
    gear->call.tool = AOTX_TOOL_NONE;
}

__global__ void reply_manual_continue(void)
{
    if (blockIdx.x != 0u || threadIdx.x != 0u) return;
    const unsigned char line[] = "continue";
    aotx_cli_line(line, (unsigned int)sizeof line - 1u, aotx_time_tick);
}

__global__ void reply_set_live(void)
{
    if (blockIdx.x != 0u || threadIdx.x != 0u) return;
    aotx_settings_reset();
    const unsigned char line[] = "set decode.reply_limit 17";
    aotx_cli_line(line, (unsigned int)sizeof line - 1u, aotx_time_tick);
    aotx_settings_commit(aotx_time_tick);
}

__global__ void reply_snapshot(reply_result *out)
{
    if (blockIdx.x != 0u || threadIdx.x != 0u) return;
    static const unsigned char want[] =
        "Continue the reply from where it stopped. Do not repeat earlier text.";
    const aotx_agent_work *gear = &aotx_agent_gear[0];
    out->state = aotx_agents.agent[0].state;
    out->continuable = gear->continuable;
    out->has_message = gear->has_message;
    out->turns = aotx_transcript[0].count;
    out->reply_limit = aotx_setting_count(AOTX_SET_REPLY_LIMIT);
    out->message_ok = gear->message_len == (unsigned int)sizeof want - 1u;
    for (unsigned int i = 0u; i < gear->message_len && out->message_ok != 0u; ++i) {
        out->message_ok &= gear->message[i] == want[i];
    }
}

__global__ void reply_markup(reply_result *out)
{
    if (blockIdx.x != 0u || threadIdx.x != 0u) return;
    static const unsigned char raw[] =
        "<tool_call>\n{\"name\":\"fs_read\",\"arguments\":{\"path\":\"README.md\"}}"
        "\n</tool_call>";
    aotx_catalog_entry *tool = &aotx_catalog.entry[0];
    tool->state = AOTX_CATALOG_INSTALLED;
    tool->kind = AOTX_MODULE_TOOL;
    tool->name_len = 7u;
    for (unsigned int i = 0u; i < 7u; ++i) tool->name[i] = "fs_read"[i];
    aotx_agent_work *gear = &aotx_agent_gear[0];
    gear->call.entry = 0u;
    gear->call.arg_len = 9u;
    for (unsigned int i = 0u; i < 9u; ++i) gear->call.arg[i] = "README.md"[i];
    for (unsigned int i = 0u; i < sizeof raw - 1u; ++i) gear->reply[i] = raw[i];
    gear->reply_len = (unsigned int)sizeof raw - 1u;
    gear->source_seq = 8ull;
    aotx_agents.agent[0].turn = 1u;
    aotx_transcript[0].pages = 1u;
    aotx_say.slot[0].at = aotx_console_start("conductor: ", 11u);
    aotx_say.slot[0].column = 1u;
    aotx_say_show(0u, raw, 5u);
    aotx_say_show(0u, raw + 5u, 8u);
    aotx_say_show(0u, raw + 13u, (unsigned int)sizeof raw - 14u);
    aotx_agent_call_line(0u);
    const unsigned char line[] = "read the file";
    aotx_transcript_finish(0u, line, (unsigned int)sizeof line - 1u, raw,
                           (unsigned int)sizeof raw - 1u, 17u, 9ull);
    const aotx_transcript_turn *turn = &aotx_transcript[0].turn[0];
    out->transcript_full = turn->reply_len == (unsigned int)sizeof raw - 1u;
    for (unsigned int i = 0u; i < turn->reply_len && out->transcript_full != 0u; ++i) {
        out->transcript_full &= aotx_transcript_text[0][turn->reply_at + i] == raw[i];
    }
}

static aotx_console_state console_state(void)
{
    aotx_console_state state;
    aotx_check_runtime(cudaMemcpyFromSymbol(&state, aotx_console, sizeof state),
                       "cudaMemcpyFromSymbol");
    return state;
}

static unsigned int console_exact(const aotx_console_state *state, const char *text)
{
    unsigned int count = 0u;
    size_t bytes = strlen(text);
    for (unsigned long long at = 1ull; at <= state->count; ++at) {
        const aotx_console_line *line = &state->line[(at - 1ull) & (AOTX_CONSOLE_LINES - 1u)];
        if (line->seq == at && line->length == bytes && memcmp(line->text, text, bytes) == 0) {
            count++;
        }
    }
    return count;
}

static reply_result snapshot(void)
{
    reply_result result = {};
    reply_result *device = NULL;
    aotx_check_runtime(cudaMalloc(&device, sizeof result), "cudaMalloc");
    reply_snapshot<<<1, 1>>>(device);
    aotx_check_runtime(cudaMemcpy(&result, device, sizeof result, cudaMemcpyDeviceToHost),
                       "cudaMemcpy");
    cudaFree(device);
    return result;
}

static void manual_case(void)
{
    clear_state();
    unsigned char *ring = open_ring();
    reply_prepare<<<1, 1>>>(0u, 1u, 1u);
    aotx_agent_step<<<1, AOTX_SLOTS>>>(2ull);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    reply_result first = snapshot();
    aotx_console_state console = console_state();
    check(first.state == AOTX_AGENT_STATE_IDLE && first.continuable == 1u,
          "a limited reply does not wait to continue");
    check(console_exact(&console,
          "reply: the limit of 17 tokens ended the reply; give continue to resume") == 1u,
          "the limit notice is not one exact console line");
    reply_manual_continue<<<1, 1>>>();
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    reply_result queued = snapshot();
    check(queued.has_message == 1u && queued.message_ok == 1u && queued.continuable == 0u,
          "the continue command does not queue the fixed follow-on turn");
    reply_next<<<1, 1>>>(0u, 2u);
    aotx_agent_step<<<1, AOTX_SLOTS>>>(3ull);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    reply_result done = snapshot();
    check(done.turns == 2u && done.continuable == 0u && done.has_message == 0u,
          "manual continuation does not end in the same transcript");
    cudaFree(ring);
}


static void live_setting_case(void)
{
    clear_state();
    unsigned char *ring = open_ring();
    reply_set_live<<<1, 1>>>();
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    reply_result result = snapshot();
    check(result.reply_limit == 17u, "the live reply limit is not the next sequence value");
    cudaFree(ring);
}

static void markup_case(void)
{
    clear_state();
    unsigned char *ring = open_ring();
    reply_result result = {};
    reply_result *device = NULL;
    aotx_check_runtime(cudaMalloc(&device, sizeof result), "cudaMalloc");
    aotx_check_runtime(cudaMemset(device, 0, sizeof result), "cudaMemset");
    reply_markup<<<1, 1>>>(device);
    aotx_check_runtime(cudaMemcpy(&result, device, sizeof result, cudaMemcpyDeviceToHost),
                       "cudaMemcpy");
    aotx_console_state console = console_state();
    check(console_exact(&console, "conductor: calls fs_read README.md") == 1u,
          "the console does not hold one stated tool call");
    check(console_exact(&console, "<tool_call>") == 0u,
          "the tool-call markup reaches the console");
    check(result.transcript_full == 1u, "the transcript does not keep the full tool call");
    cudaFree(device);
    cudaFree(ring);
}

#include "wrap_fixture.h"
#include "agent_bound.h"
int main(void)
{
    aotx_check_runtime(cudaSetDevice(0), "cudaSetDevice");
    aotx_test_wrap_open();
    manual_case();
    aotx_bound_case(1u, 0u, 0u);
    aotx_bound_case(1u, 0u, 1u);
    aotx_bound_case(1u, 3u, 0u);
    aotx_bound_case(1u, 3u, 1u);
    aotx_bound_case(AOTX_SLOTS, 0u, 0u);
    aotx_bound_case(AOTX_SLOTS, 0u, 1u);
    aotx_bound_case(AOTX_SLOTS, 3u, 0u);
    aotx_bound_case(AOTX_SLOTS, 3u, 1u);
    aotx_bound_stop_case(0u);
    aotx_bound_stop_case(1u);
    aotx_bound_stopped_case(1u);
    aotx_bound_stopped_case(AOTX_SLOTS);
    live_setting_case();
    markup_case();
    printf("reply limit: cases applied %u, failed %u\n", applied, failed);
    return failed == 0u ? 0 : 1;
}
