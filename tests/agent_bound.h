/* Purpose: Check bounded message turns through prompt, post, tool, and idle states.
 * Owns: Fixed turn results and per-agent observations for the reply test.
 * Launch shape: One thread per agent; host checks at one slot and the full batch.
 * Lifetime: One test case. */
#ifndef AOTX_TESTS_AGENT_BOUND_H
#define AOTX_TESTS_AGENT_BOUND_H

#include "tool/tool_state.cuh"

typedef struct aotx_bound_result {
    unsigned int state;
    unsigned int turns;
    unsigned int budget;
    unsigned int queued;
    unsigned int wanted;
    unsigned int request;
    unsigned int continuable;
    unsigned int reply_ok;
    unsigned int result_seen;
    unsigned int prompt_result;
} aotx_bound_result;

__global__ void aotx_bound_prepare(unsigned int count, unsigned int own)
{
    if (threadIdx.x != 0u) return;
    aotx_settings_reset();
    aotx_setting_table.row[AOTX_SET_AUTO_CONTINUE].value = 1ll;
#ifdef AOTX_AFFECT
    aotx_setting_table.row[AOTX_SET_QUALITY_ON].value = 0ll;
#endif
    aotx_catalog.entry[1].state = AOTX_CATALOG_INSTALLED;
    aotx_catalog.entry[1].kind = AOTX_MODULE_ROLE;
    aotx_catalog.entry[1].role.budget = own;
    aotx_catalog.entry[0].state = AOTX_CATALOG_INSTALLED;
    aotx_catalog.entry[0].kind = AOTX_MODULE_TOOL;
    aotx_catalog.entry[0].name_len = 7u;
    aotx_catalog.entry[0].tool.side = AOTX_CATALOG_SIDE_BUILT;
    aotx_catalog.entry[0].tool.built_in = AOTX_TOOL_FS_READ;
    for (unsigned int i = 0u; i < 7u; ++i) {
        aotx_catalog.entry[0].name[i] = "fs_read"[i];
    }
    aotx_agents.live = count;
    for (unsigned int a = 0u; a < count; ++a) {
        aotx_agents.agent[a].state = AOTX_AGENT_STATE_IDLE;
        aotx_agents.agent[a].role = 1u;
        aotx_agents.agent[a].task = ~0u;
        aotx_transcript[a].pages = AOTX_KV_PAGES_EACH;
        const unsigned char text[] = "question";
        aotx_agent_message(a, text, sizeof text - 1u, 1ull);
    }
}

/* Give a fixed completion only after the real begin path opened its prompt. */
__global__ void aotx_bound_complete(unsigned int count, unsigned int tool,
                                    unsigned int limited)
{
    unsigned int a = threadIdx.x;
    if (a >= count || aotx_agents.agent[a].state != AOTX_AGENT_STATE_PROMPT
        || aotx_say.slot[a].wanted == 0u) return;
    aotx_agent_work *gear = &aotx_agent_gear[a];
    const unsigned char answer[] = "partial";
    for (unsigned int i = 0u; i < sizeof answer - 1u; ++i) gear->reply[i] = answer[i];
    gear->reply_len = sizeof answer - 1u;
    gear->out_tokens = 17u;
    gear->limit_end = limited;
    gear->call.entry = tool ? 0u : AOTX_CATALOG_NO_ENTRY;
    gear->call.tool = tool ? AOTX_TOOL_FS_READ : AOTX_TOOL_NONE;
    if (tool) aotx_tool_outcome_arm(a, AOTX_TOOL_OK);
    aotx_say.slot[a].wanted = 0u;
    aotx_say.slot[a].live = 0u;
    aotx_agents.agent[a].state = AOTX_AGENT_STATE_POST;
}

__device__ int aotx_bound_contains(const unsigned char *text, unsigned int length,
                                   const char *word, unsigned int bytes)
{
    for (unsigned int i = 0u; i + bytes <= length; ++i) {
        unsigned int j = 0u;
        while (j < bytes && text[i + j] == (unsigned char)word[j]) ++j;
        if (j == bytes) return 1;
    }
    return 0;
}

__global__ void aotx_bound_snapshot(aotx_bound_result *out, unsigned int count)
{
    unsigned int a = threadIdx.x;
    if (a >= count) return;
    const aotx_agent_work *gear = &aotx_agent_gear[a];
    aotx_bound_result *row = &out[a];
    row->state = aotx_agents.agent[a].state;
    row->turns = aotx_agents.agent[a].turn;
    row->budget = aotx_agents.agent[a].budget_left;
    row->queued = gear->has_message;
    row->wanted = aotx_say.slot[a].wanted;
    row->request = aotx_agents.agent[a].request;
    row->continuable = gear->continuable;
    row->reply_ok = gear->reply_len == 7u
        && aotx_bound_contains(gear->reply, gear->reply_len, "partial", 7u);
    row->result_seen = 0u;
    const aotx_transcript_agent *hold = &aotx_transcript[a];
    if (hold->count != 0u) {
        unsigned int at = (hold->first + hold->count - 1u) % AOTX_MEMORY_TURNS;
        const aotx_transcript_turn *turn = &hold->turn[at];
        row->result_seen = aotx_bound_contains(aotx_transcript_text[a] + turn->extra_at,
            turn->extra_len, "the fixture tool ran", 20u);
    }
    row->prompt_result = aotx_bound_contains(aotx_say.prompt[a],
        aotx_say.slot[a].length, "the fixture tool ran", 20u);
}

__global__ void aotx_bound_input(unsigned int count, unsigned int manual)
{
    if (threadIdx.x != 0u) return;
    for (unsigned int a = 0u; a < count; ++a) {
        if (manual != 0u) {
            unsigned char line[48];
            const char *head = "agent ";
            unsigned int at = 0u;
            for (unsigned int i = 0u; head[i] != '\0'; ++i) line[at++] = head[i];
            at += aotx_text_utoa(a, (char *)line + at, sizeof line - at);
            const char *tail = " continue";
            for (unsigned int i = 0u; tail[i] != '\0'; ++i) line[at++] = tail[i];
            aotx_cli_line(line, at, aotx_time_tick);
        } else {
            const unsigned char text[] = "new question";
            aotx_agent_message(a, text, sizeof text - 1u, aotx_time_tick);
        }
    }
}

__global__ void aotx_bound_stop(void)
{
    if (threadIdx.x != 0u) return;
    const unsigned char text[] = "stop";
    aotx_cli_line(text, sizeof text - 1u, aotx_time_tick);
}

static void aotx_bound_read(aotx_bound_result *device, aotx_bound_result *out,
                            unsigned int count)
{
    aotx_bound_snapshot<<<1, AOTX_SLOTS>>>(device, count);
    aotx_check_runtime(cudaMemcpy(out, device, count * sizeof *out,
                                  cudaMemcpyDeviceToHost), "cudaMemcpy");
}

static void aotx_bound_case(unsigned int count, unsigned int own, unsigned int tool)
{
    clear_state();
    CLEAR(aotx_agent_count, sizeof(aotx_agent_counts));
    CLEAR(aotx_requests, sizeof(aotx_request_table));
    CLEAR(aotx_tool_embed, sizeof(aotx_tool_embed_batch));
    CLEAR(aotx_tool_done, sizeof(unsigned int) * AOTX_SLOTS);
    unsigned char *ring = open_ring();
    aotx_bound_result *device = NULL;
    aotx_bound_result rows[AOTX_SLOTS] = {};
    aotx_check_runtime(cudaMalloc(&device, sizeof rows), "cudaMalloc");
    unsigned int budget = own ? own : 8u;
    aotx_bound_prepare<<<1, 1>>>(count, own);
    aotx_agent_step<<<1, AOTX_SLOTS>>>(2ull);
    for (unsigned int turn = 1u; turn <= budget; ++turn) {
        aotx_bound_read(device, rows, count);
        for (unsigned int a = 0u; a < count; ++a) {
            check(rows[a].state == AOTX_AGENT_STATE_PROMPT && rows[a].wanted == 1u
                  && rows[a].turns == turn && rows[a].budget == budget - turn,
                  "a message turn does not spend its input budget once");
            if (tool && turn > 1u) check(rows[a].prompt_result != 0u,
                  "the next prompt does not contain the tool result");
        }
        aotx_bound_complete<<<1, AOTX_SLOTS>>>(count, tool, 1u);
        aotx_agent_step<<<1, AOTX_SLOTS>>>(3ull + turn * 2ull);
        if (tool) {
            aotx_bound_read(device, rows, count);
            for (unsigned int a = 0u; a < count; ++a) {
                check(rows[a].state == AOTX_AGENT_STATE_TOOL && rows[a].request != 0u,
                      "the tool call does not wait for its result");
            }
        }
        aotx_agent_step<<<1, AOTX_SLOTS>>>(4ull + turn * 2ull);
    }
    /* Idle ticks must not open a turn after the last automatic reply or tool result. */
    for (unsigned int i = 0u; i < 3u; ++i) aotx_agent_step<<<1, AOTX_SLOTS>>>(30ull + i);
    aotx_bound_read(device, rows, count);
    aotx_console_state console = console_state();
    for (unsigned int a = 0u; a < count; ++a) {
        check(rows[a].state == AOTX_AGENT_STATE_IDLE && rows[a].turns == budget
              && rows[a].queued == 0u && rows[a].wanted == 0u && rows[a].request == 0u,
              "the input does not end idle at its turn bound");
        check(rows[a].reply_ok != 0u, "the terminal notice replaces generated reply text");
        if (tool) check(rows[a].result_seen != 0u,
                        "the last tool result is absent from the transcript");
        char line[128];
        snprintf(line, sizeof line,
                 "agent %u: the turn budget is exhausted; give new input to resume", a);
        check(console_exact(&console, line) == 1u,
              "the exhausted input does not have one visible terminal reason");
    }
    aotx_bound_input<<<1, 1>>>(count, tool ? 0u : 1u);
    aotx_agent_step<<<1, AOTX_SLOTS>>>(40ull);
    aotx_bound_read(device, rows, count);
    for (unsigned int a = 0u; a < count; ++a) {
        check(rows[a].state == AOTX_AGENT_STATE_PROMPT && rows[a].turns == budget + 1u
              && rows[a].budget == budget - 1u,
              "explicit input does not open a fresh bounded turn");
    }
    aotx_bound_complete<<<1, AOTX_SLOTS>>>(count, 0u, 0u);
    aotx_agent_step<<<1, AOTX_SLOTS>>>(41ull);
    aotx_bound_read(device, rows, count);
    for (unsigned int a = 0u; a < count; ++a) {
        check(rows[a].state == AOTX_AGENT_STATE_IDLE && rows[a].queued == 0u
              && rows[a].continuable == 0u, "a natural reply end starts another turn");
    }
    printf("reply bound: N=%u budget=%u tool=%u\n", count, budget, tool);
    cudaFree(device);
    cudaFree(ring);
}

static void aotx_bound_stop_case(unsigned int queued)
{
    clear_state();
    unsigned char *ring = open_ring();
    aotx_bound_result *device = NULL;
    aotx_bound_result row = {};
    aotx_check_runtime(cudaMalloc(&device, sizeof row), "cudaMalloc");
    aotx_bound_prepare<<<1, 1>>>(1u, 3u);
    aotx_agent_step<<<1, AOTX_SLOTS>>>(2ull);
    aotx_bound_complete<<<1, 1>>>(1u, 0u, 1u);
    if (queued) aotx_agent_step<<<1, AOTX_SLOTS>>>(3ull);
    aotx_bound_stop<<<1, 1>>>();
    for (unsigned int i = 0u; i < 3u; ++i) aotx_agent_step<<<1, AOTX_SLOTS>>>(4ull + i);
    aotx_bound_read(device, &row, 1u);
    check(row.state == AOTX_AGENT_STATE_IDLE && row.turns == 1u
          && row.queued == 0u && row.continuable == 0u,
          "stop at the token limit permits an automatic follow-on");
    aotx_bound_input<<<1, 1>>>(1u, 0u);
    aotx_agent_step<<<1, AOTX_SLOTS>>>(8ull);
    aotx_bound_read(device, &row, 1u);
    check(row.state == AOTX_AGENT_STATE_PROMPT && row.turns == 2u && row.budget == 2u,
          "fresh input cannot start after stop");
    cudaFree(device);
    cudaFree(ring);
}

/* The sequence ended on the same token that reached its limit. */
__global__ void aotx_bound_stopped(unsigned int count)
{
    unsigned int a = threadIdx.x;
    if (a >= count || aotx_agents.agent[a].state != AOTX_AGENT_STATE_PROMPT) return;
    aotx_seq *seq = &aotx_seqs.slot[a];
    seq->state = AOTX_SEQ_STATE_DONE;
    seq->role = AOTX_MODEL_LANGUAGE;
    seq->sampled = 1u;
    seq->limit = 1u;
    seq->last = ~0u;
    seq->stop = 2u;
    seq->flags = AOTX_DECODE_MARK_STOP;
    aotx_seqs.tokens[a][0] = -1;
    aotx_say.slot[a].wanted = 0u;
    aotx_say.slot[a].ready = 1u;
}

static void aotx_bound_stopped_case(unsigned int count)
{
    clear_state();
    unsigned char *ring = open_ring();
    aotx_bound_result *device = NULL;
    aotx_bound_result rows[AOTX_SLOTS] = {};
    aotx_check_runtime(cudaMalloc(&device, sizeof rows), "cudaMalloc");
    aotx_bound_prepare<<<1, 1>>>(count, 3u);
    aotx_agent_step<<<1, AOTX_SLOTS>>>(2ull);
    aotx_bound_stopped<<<1, AOTX_SLOTS>>>(count);
    for (unsigned int i = 0u; i < 5u; ++i) aotx_agent_step<<<1, AOTX_SLOTS>>>(3ull + i);
    aotx_bound_read(device, rows, count);
    for (unsigned int a = 0u; a < count; ++a) {
        check(rows[a].state == AOTX_AGENT_STATE_IDLE && rows[a].turns == 1u
              && rows[a].queued == 0u && rows[a].continuable == 0u,
              "a stopped sequence at its limit opens a follow-on turn");
    }
    cudaFree(device);
    cudaFree(ring);
}

#endif
