/* Purpose: Check selected tool instructions, call history, and result framing.
 * Owns: Distinct call values and circular transcript fixtures.
 * Launch shape: One thread for each agent, at one and the profile slot count.
 * Lifetime: One test process. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "agent/prompt.cuh"
#include "wrap_fixture.h"

typedef struct aotx_call_prompt_row {
    char input[512];
    char result[64];
    aotx_tool_call call;
    aotx_tool_call parsed;
    unsigned char rendered[AOTX_SAY_BYTES];
    unsigned char history[AOTX_SAY_BYTES];
    unsigned char prompt[AOTX_SAY_BYTES];
    unsigned char list[AOTX_SAY_BYTES];
    unsigned char limits[AOTX_SAY_BYTES];
    unsigned int limits_len;
    unsigned int rendered_len, history_len, prompt_len, list_len;
    unsigned int input_ok, roundtrip, guard;
} aotx_call_prompt_row;

__global__ void aotx_call_prompt_model(void)
{
    aotx_model[AOTX_MODEL_LANGUAGE].layers = 1u;
    aotx_kvl_make(&aotx_model_space[AOTX_MODEL_LANGUAGE].shape, 1u, 1u, 32u);
    unsigned int role = AOTX_MODULE_SLOTS - 1u;
    unsigned int entry = aotx_catalog_find("fs_update", 9u, AOTX_MODULE_TOOL);
    aotx_catalog_mask_clear(aotx_catalog.entry[role].role.tools);
    aotx_catalog_mask_set(aotx_catalog.entry[role].role.tools, entry);
    unsigned int all = AOTX_MODULE_SLOTS - 2u;
    aotx_catalog_mask_clear(aotx_catalog.entry[all].role.tools);
    for (unsigned int i = 0u; i < AOTX_MODULE_SLOTS; ++i)
        if (aotx_catalog_is(i, AOTX_MODULE_TOOL))
            aotx_catalog_mask_set(aotx_catalog.entry[all].role.tools, i);
}

/* Store a tagged JSON call, then change the selected row before history is rendered. */
__global__ void aotx_call_prompt_store(aotx_call_prompt_row *rows, unsigned int count)
{
    unsigned int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= count) return;
    aotx_call_prompt_row *row = &rows[slot];
    unsigned int bytes = 0u;
    while (row->input[bytes]) ++bytes;
    row->input_ok = aotx_tool_parse((const unsigned char *)row->input, bytes, &row->call) == 1;
    memset(&aotx_transcript[slot], 0, sizeof aotx_transcript[slot]);
    memset(&aotx_agent_gear[slot], 0, sizeof aotx_agent_gear[slot]);
    aotx_agent_gear[slot].call = row->call;
    aotx_agents.agent[slot].role = AOTX_MODULE_SLOTS - 1u;
    aotx_agents.agent[slot].turn = 1u;
    aotx_transcript[slot].pages = AOTX_KV_PAGES_EACH;
    aotx_transcript[slot].text_head = AOTX_TRANSCRIPT_TEXT_BYTES - bytes - 12u;
    const unsigned char query[] = "change the file";
    aotx_transcript_finish(slot, query, sizeof query - 1u,
                            (const unsigned char *)row->input, bytes, 32u, 100u + slot);
    aotx_request *request = &aotx_requests.slot[slot];
    request->result_len = 0u;
    while (row->result[request->result_len]) {
        request->result[request->result_len] = row->result[request->result_len];
        ++request->result_len;
    }
    request->result_seq = 200u + slot;
    aotx_transcript_result(slot, request);
}

__global__ void aotx_call_prompt_render(aotx_call_prompt_row *rows, unsigned int count)
{
    unsigned int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= count) return;
    aotx_call_prompt_row *row = &rows[slot];
    row->rendered_len = aotx_call_render(row->rendered, 0u, row->call.entry,
        (const unsigned char *)row->call.pack, 0u, AOTX_TOOL_ARG_BYTES,
        row->call.at, row->call.length);
    row->roundtrip = row->rendered_len <= AOTX_SAY_BYTES
        && aotx_tool_parse(row->rendered, row->rendered_len, &row->parsed) == 1;
    row->history_len = aotx_transcript_prompt(slot, row->history, 0u);
    row->list_len = aotx_catalog_tool_list(row->list, 0u, AOTX_MODULE_SLOTS - 1u);
    row->limits_len = aotx_catalog_tool_list(row->limits, 0u, AOTX_MODULE_SLOTS - 2u);
    const unsigned char query[] = "change the file";
    aotx_say.slot[slot].wanted = 0u;
    unsigned int result_len = 0u;
    while (row->result[result_len]) ++result_len;
    row->prompt_len = aotx_agent_prompt(slot, 0, query, sizeof query - 1u, 0, 0, 0u,
                                       row->result, result_len);
    if (row->prompt_len <= AOTX_SAY_BYTES)
        memcpy(row->prompt, aotx_say.prompt[slot], row->prompt_len);
    row->rendered[AOTX_SAY_BYTES - 1u] = 0x5au;
    unsigned int end = aotx_call_render(row->rendered, AOTX_SAY_BYTES, row->call.entry,
        (const unsigned char *)row->call.pack, 0u, AOTX_TOOL_ARG_BYTES,
        row->call.at, row->call.length);
    row->guard = end == AOTX_SAY_BYTES + 1u && row->rendered[AOTX_SAY_BYTES - 1u] == 0x5au;
    row->guard &= aotx_call_render(0, 0u, row->call.entry,
        (const unsigned char *)row->call.pack, 0u, AOTX_TOOL_ARG_BYTES,
        row->call.at, row->call.length) == row->rendered_len;
}

static unsigned int applied, failed;
static void check(int yes, const char *what, unsigned int kind, unsigned int slot)
{
    ++applied;
    if (!yes) {
        ++failed;
        printf("call prompt: kind %u slot %u: %s\n", kind, slot, what);
    }
}

#include "call_history.h"

static void run(unsigned int kind, unsigned int count)
{
    aotx_call_prompt_row *host = (aotx_call_prompt_row *)calloc(count, sizeof *host), *device = NULL;
    if (host == NULL) exit(2);
    for (unsigned int i = 0u; i < count; ++i) {
        snprintf(host[i].input, sizeof host[i].input,
            "<tool_call>{\"name\":\"fs_update\",\"arguments\":{\"new\":\"new <%u>\","
            "\"old\":\"old\\t%u\\n\\\"\\\\\",\"path\":\"file-%u\"}}</tool_call>", i, i, i);
        snprintf(host[i].result, sizeof host[i].result, "saved %u\t\"ok\"\\", i);
    }
    aotx_check_runtime(cudaMalloc(&device, count * sizeof *device), "cudaMalloc");
    aotx_check_runtime(cudaMemcpy(device, host, count * sizeof *host, cudaMemcpyHostToDevice), "cudaMemcpy");
    aotx_test_call_upload(AOTX_CALL_HERMES);
    aotx_call_prompt_store<<<(count + 63u) / 64u, 64u>>>(device, count);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_test_call_upload(kind);
    aotx_call_prompt_render<<<(count + 63u) / 64u, 64u>>>(device, count);
    aotx_check_runtime(cudaMemcpy(host, device, count * sizeof *host, cudaMemcpyDeviceToHost), "cudaMemcpy");
    for (unsigned int i = 0u; i < count; ++i) {
        aotx_call_prompt_row *row = &host[i];
        char expected[1024], result[256];
        if (kind == AOTX_CALL_QWEN_XML) {
            snprintf(expected, sizeof expected,
                "<tool_call>\n<function=fs_update>\n<parameter=path>\nfile-%u\n</parameter>\n"
                "<parameter=old>\nold\t%u\n\"\\\n</parameter>\n<parameter=new>\nnew <%u>\n</parameter>\n"
                "</function>\n</tool_call>", i, i, i);
        } else {
            snprintf(expected, sizeof expected,
                "%s{\"name\": \"fs_update\", \"%s\": {\"path\": \"file-%u\", "
                "\"old\": \"old\\u0009%u\\u000a\\\"\\\\\", \"new\": \"new <%u>\"}}%s",
                kind == AOTX_CALL_HERMES ? "<tool_call>\n" : "",
                kind == AOTX_CALL_HERMES ? "arguments" : "parameters", i, i, i,
                kind == AOTX_CALL_HERMES ? "\n</tool_call>" : "");
        }
        if (kind == AOTX_CALL_LLAMA_JSON) {
            snprintf(result, sizeof result, "<|start_header_id|>ipython<|end_header_id|>\n\n"
                "\"saved %u\\u0009\\\"ok\\\"\\\\\"<|eot_id|>", i);
        } else {
            snprintf(result, sizeof result, "<|im_start|>user\n%s<tool_response>\n%s\n</tool_response><|im_end|>\n",
                kind == AOTX_CALL_HERMES ? "\n" : "", row->result);
        }
        check(row->input_ok && row->roundtrip, "rendered call parses", kind, i);
        check(row->rendered_len == strlen(expected) && !memcmp(row->rendered, expected, strlen(expected)),
                "call preserves every argument and control byte", kind, i);
        check(row->parsed.pack_len == row->call.pack_len && row->parsed.values == row->call.values,
                "round trip retains argument sizes", kind, i);
        for (unsigned int k = 0u; k < 3u; ++k) {
            check(row->parsed.length[k] == row->call.length[k]
                && !memcmp(row->parsed.pack + row->parsed.at[k], row->call.pack + row->call.at[k], row->call.length[k]),
                "round trip retains each key value", kind, i);
        }
        check(row->history_len < AOTX_SAY_BYTES && row->prompt_len > 0u && row->prompt_len < AOTX_SAY_BYTES,
                "history and prompt fit", kind, i);
        if (row->history_len < AOTX_SAY_BYTES) row->history[row->history_len] = 0u;
        if (row->prompt_len < AOTX_SAY_BYTES) row->prompt[row->prompt_len] = 0u;
        if (row->list_len < AOTX_SAY_BYTES) row->list[row->list_len] = 0u;
        if (row->limits_len < AOTX_SAY_BYTES) row->limits[row->limits_len] = 0u;
        check(strstr((const char *)row->history, expected) && strstr((const char *)row->prompt, expected),
                "stored call uses newly selected row", kind, i);
        check(strstr((const char *)row->history, result) && strstr((const char *)row->prompt, result),
                "stored result uses separate native tool turn", kind, i);
        const char *first_call = strstr((const char *)row->prompt, expected);
        check(first_call && !strstr(first_call + strlen(expected), expected),
                "continuation contains one prior call", kind, i);
        const char *instruction = kind == AOTX_CALL_HERMES ? "arguments within <tool_call>"
                                : kind == AOTX_CALL_LLAMA_JSON ? "parameters\": dictionary"
                                : "<function=example_function_name>";
        check(row->list_len <= AOTX_CATALOG_LIST_BYTES && strstr((const char *)row->list, instruction)
                && strstr((const char *)row->list, "\"fs_update\""), "native instruction and schema share bounded list", kind, i);
        check(row->limits_len <= AOTX_CATALOG_LIST_BYTES && strstr((const char *)row->limits, instruction),
                "full catalog keeps its instruction inside the list bound", kind, i);
        check(row->guard, "full prompt refuses another call", kind, i);
    }
    cudaFree(device);
    free(host);
}

int main(void)
{
    aotx_check_runtime(cudaSetDevice(0), "cudaSetDevice");
    aotx_catalog_open();
    aotx_test_wrap_open();
    aotx_call_prompt_model<<<1, 1>>>();
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    for (unsigned int kind = AOTX_CALL_HERMES; kind < AOTX_CALL_FORMAT_KINDS; ++kind) {
        run(kind, 1u);
        run(kind, AOTX_SLOTS);
    }
    for (unsigned int mode = 0u; mode < 3u; ++mode) {
        aotx_call_history_case(1u, mode);
        aotx_call_history_case(AOTX_SLOTS, mode);
    }
    printf("call storage: turn %zu schema %zu format %zu bytes\n",
           sizeof(aotx_transcript_turn), sizeof(aotx_call_schema), sizeof(aotx_call_format));
    printf("call prompt: %u checks, %u failed\n", applied, failed);
    return failed != 0u;
}
