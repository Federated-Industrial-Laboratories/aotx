/* Purpose: Check prompt bytes across the device paths and the quality tool renderer.
 * Owns: Explicit model wraps, fixed conversations and per-slot comparison buffers.
 * Launch shape: One thread for each prompt, at one and 64 slots.
 * Lifetime: One test process. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "agent/prompt.cuh"
#include "tools/quality_score.h"
#include "wrap_fixture.h"

#define AOTX_WRAP_TEST_BYTES 2048u

typedef struct aotx_wrap_test_row {
    char user[32], reply[32], next[32];
    unsigned char say[AOTX_WRAP_TEST_BYTES];
    unsigned char conversation[AOTX_WRAP_TEST_BYTES];
    unsigned char agent[AOTX_WRAP_TEST_BYTES];
    unsigned char agent_history[AOTX_WRAP_TEST_BYTES];
    unsigned int agent_history_length, history_hot;
    unsigned int say_length, conversation_length, agent_length, refused;
} aotx_wrap_test_row;

/* One page holds the stored turn; no cache allocation or inference runs. */
__global__ void aotx_wrap_test_model(void)
{
    aotx_model[AOTX_MODEL_LANGUAGE].layers = 1u;
    aotx_kvl_make(&aotx_model_space[AOTX_MODEL_LANGUAGE].shape, 1u, 1u, 32u);
}

__global__ void aotx_wrap_test_render(aotx_wrap_test_row *rows, unsigned int count)
{
    unsigned int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= count) return;
    aotx_wrap_test_row *row = &rows[slot];
    const aotx_wrap *wrap = aotx_wrap_active();
    unsigned int user_length = 0u, reply_length = 0u, next_length = 0u;
    while (row->user[user_length]) ++user_length;
    while (row->reply[reply_length]) ++reply_length;
    while (row->next[next_length]) ++next_length;
    aotx_say.slot[slot].wanted = 0u;
    aotx_say.slot[slot].live = 0u;
    row->refused = 0u;
    int status = aotx_say_ask(slot, (const unsigned char *)row->user, user_length);
    row->say_length = status ? 0u : aotx_say.slot[slot].length;
    for (unsigned int i = 0u; i < row->say_length && i < AOTX_WRAP_TEST_BYTES; ++i)
        row->say[i] = aotx_say.prompt[slot][i];
    aotx_say.slot[slot].wanted = 0u;
    row->refused += aotx_say_ask(slot, (const unsigned char *)row->user, AOTX_SAY_BYTES) != 0;
    row->refused += aotx_say.slot[slot].wanted == 0u;

    memset(&aotx_transcript[slot], 0, sizeof aotx_transcript[slot]);
    memset(&aotx_agent_gear[slot], 0, sizeof aotx_agent_gear[slot]);
    aotx_transcript_agent *hold = &aotx_transcript[slot];
    hold->count = 1u;
    hold->turn[0].text_live = 1u;
    hold->turn[0].tier = AOTX_MEMORY_HOT;
    hold->turn[0].text_len = user_length;
    hold->turn[0].reply_at = user_length;
    hold->turn[0].reply_len = reply_length;
    hold->pages = 1u;
    hold->turn[0].tokens = 4u;
    hold->turn[0].stored_len = user_length + reply_length;
    hold->text_used = user_length + reply_length;
    hold->text_head = hold->text_used;
    for (unsigned int i = 0u; i < user_length; ++i) aotx_transcript_text[slot][i] = row->user[i];
    for (unsigned int i = 0u; i < reply_length; ++i) aotx_transcript_text[slot][user_length + i] = row->reply[i];
    unsigned int at = aotx_wrap_prefix(row->conversation, 0u, AOTX_WRAP_TEST_BYTES, wrap);
    at = aotx_transcript_prompt(slot, row->conversation, at);
    at = aotx_wrap_put(row->conversation, at, AOTX_WRAP_TEST_BYTES, wrap, AOTX_WRAP_USER_HEAD);
    at = aotx_wrap_run(row->conversation, at, AOTX_WRAP_TEST_BYTES,
                        (const unsigned char *)row->next, next_length);
    at = aotx_wrap_put(row->conversation, at, AOTX_WRAP_TEST_BYTES, wrap, AOTX_WRAP_USER_TAIL);
    row->conversation_length = aotx_wrap_generation(row->conversation, at, AOTX_WRAP_TEST_BYTES, wrap);

    aotx_agents.agent[slot].role = AOTX_MODULE_SLOTS;
    aotx_agents.agent[slot].state = AOTX_AGENT_STATE_IDLE;
    aotx_agents.agent[slot].turn = 1u;
    row->agent_history_length = aotx_agent_prompt(slot, 0, (const unsigned char *)row->next,
                                                  next_length, 0, 0, 0u, 0, 0u);
    row->history_hot = hold->hot;
    for (unsigned int i = 0u; i < row->agent_history_length && i < AOTX_WRAP_TEST_BYTES; ++i)
        row->agent_history[i] = aotx_say.prompt[slot][i];
    aotx_say.slot[slot].wanted = 0u;

    memset(hold, 0, sizeof *hold);
    row->agent_length = aotx_agent_prompt(slot, 0, (const unsigned char *)row->next,
                                          next_length, 0, 0, 0u, 0, 0u);
    for (unsigned int i = 0u; i < row->agent_length && i < AOTX_WRAP_TEST_BYTES; ++i)
        row->agent[i] = aotx_say.prompt[slot][i];
    aotx_say.slot[slot].wanted = 0u;
    if (!wrap->usable) row->refused += status != 0 && row->agent_length == 0u;
}

static int same(const unsigned char *actual, unsigned int length, const char *expected, size_t count)
{
    return count < AOTX_WRAP_TEST_BYTES && length == count && memcmp(actual, expected, count) == 0;
}

static int run_case(aotx_wrap wrap, unsigned int count, unsigned int kind)
{
    aotx_test_wrap_upload(&wrap);
    aotx_wrap_test_row *host = (aotx_wrap_test_row *)calloc(count, sizeof *host), *device = 0;
    if (!host) return 1;
    for (unsigned int i = 0u; i < count; ++i) {
        snprintf(host[i].user, sizeof host[i].user, "question %u", i);
        snprintf(host[i].reply, sizeof host[i].reply, "answer %u", i);
        snprintf(host[i].next, sizeof host[i].next, "next %u", i);
    }
    aotx_check_runtime(cudaMalloc(&device, count * sizeof *device), "cudaMalloc");
    aotx_check_runtime(cudaMemcpy(device, host, count * sizeof *host, cudaMemcpyHostToDevice), "cudaMemcpy");
    aotx_wrap_test_render<<<(count + 63u) / 64u, 64u>>>(device, count);
    aotx_check_runtime(cudaMemcpy(host, device, count * sizeof *host, cudaMemcpyDeviceToHost), "cudaMemcpy");
    unsigned int bad = 0u;
    for (unsigned int i = 0u; i < count; ++i) {
        aotx_wrap_test_row *row = &host[i];
        char expected[AOTX_WRAP_TEST_BYTES], legacy[AOTX_WRAP_TEST_BYTES];
        if (!wrap.usable) {
            bad += row->say_length != 0u || row->agent_length != 0u
                || row->agent_history_length != 0u || row->refused != 3u;
            continue;
        }
        size_t at = aotx_score_query(expected, 0u, sizeof expected, &wrap, row->user, 1);
        bad += !same(row->say, row->say_length, expected, at) || row->refused != 2u;
        if (kind == 0u) {
            int n = snprintf(legacy, sizeof legacy, "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", row->user);
            bad += n < 0 || !same(row->say, row->say_length, legacy, (size_t)n);
        }
        at = aotx_score_prefix(expected, 0u, sizeof expected, &wrap);
        at = aotx_score_turn(expected, at, sizeof expected, &wrap, AOTX_WRAP_USER_HEAD, row->user);
        at = aotx_score_turn(expected, at, sizeof expected, &wrap, AOTX_WRAP_ASSISTANT_HEAD, row->reply);
        at = aotx_score_query(expected, at, sizeof expected, &wrap, row->next, 0);
        bad += !same(row->conversation, row->conversation_length, expected, at);
        if (kind == 0u) {
            int n = snprintf(legacy, sizeof legacy, "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n%s<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", row->user, row->reply, row->next);
            bad += n < 0 || !same(row->conversation, row->conversation_length, legacy, (size_t)n);
        }
        at = aotx_score_span(expected, 0u, sizeof expected, &wrap, AOTX_WRAP_SYSTEM_HEAD);
        const char *tools = AOTX_OVERLAY_TOOLS_HEAD AOTX_OVERLAY_TOOLS_TAIL AOTX_OVERLAY_CALL_FORM;
        at = aotx_score_put(expected, at, sizeof expected, tools, strlen(tools));
        at = aotx_score_span(expected, at, sizeof expected, &wrap, AOTX_WRAP_SYSTEM_TAIL);
        size_t system_end = at;
        at = aotx_score_turn(expected, at, sizeof expected, &wrap, AOTX_WRAP_USER_HEAD, row->user);
        at = aotx_score_turn(expected, at, sizeof expected, &wrap, AOTX_WRAP_ASSISTANT_HEAD, row->reply);
        at = aotx_score_query(expected, at, sizeof expected, &wrap, row->next, 0);
        bad += row->history_hot != 1u || !same(row->agent_history, row->agent_history_length, expected, at);
        if (kind == 0u) {
            int n = snprintf(legacy, sizeof legacy, "<|im_start|>system\n%s<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n%s<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", tools, row->user, row->reply, row->next);
            bad += n < 0 || !same(row->agent_history, row->agent_history_length, legacy, (size_t)n);
        }
        at = system_end;
        at = aotx_score_query(expected, at, sizeof expected, &wrap, row->next, 0);
        bad += !same(row->agent, row->agent_length, expected, at);
        if (kind == 0u) {
            int n = snprintf(legacy, sizeof legacy, "<|im_start|>system\n%s<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", tools, row->next);
            bad += n < 0 || !same(row->agent, row->agent_length, legacy, (size_t)n);
        }
        char guard[3] = { 'x', 'y', 'z' };
        bad += aotx_score_query(guard + 1u, 0u, 1u, &wrap, row->user, 1) != 1u;
        bad += guard[0] != 'x' || guard[2] != 'z';
    }
    printf("wrap prompt: kind %u, slots %u, failed %u\n", kind, count, bad);
    cudaFree(device);
    free(host);
    return bad != 0u;
}

int main(void)
{
    aotx_check_runtime(cudaSetDevice(0), "cudaSetDevice");
    aotx_wrap_test_model<<<1, 1>>>();
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_wrap wrap = aotx_test_wrap_table();
    int bad = run_case(wrap, 1u, 0u) | run_case(wrap, AOTX_SLOTS, 0u);
    static const char *const span[AOTX_WRAP_SPANS] = {
        "<bos><system>", "</system>", "<user>", "</user>",
        "<stored>", "</stored>", "<generate>", "", ""
    };
    memset(&wrap, 0, sizeof wrap);
    unsigned int at = 0u;
    for (unsigned int i = 0u; i < AOTX_WRAP_SPANS; ++i) {
        wrap.offset[i] = (uint16_t)at;
        wrap.length[i] = (uint8_t)strlen(span[i]);
        memcpy(wrap.bytes + at, span[i], wrap.length[i]);
        at += wrap.length[i];
    }
    wrap.prefix_length = 5u;
    wrap.think_open_id = UINT32_MAX;
    wrap.think_close_id = UINT32_MAX;
    wrap.usable = 1u;
    bad |= run_case(wrap, 1u, 1u) | run_case(wrap, AOTX_SLOTS, 1u);
    wrap.usable = 0u;
    bad |= run_case(wrap, 1u, 2u) | run_case(wrap, AOTX_SLOTS, 2u);
    return bad;
}
