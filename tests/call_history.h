/* Purpose: Check retained call identity and text-only history after model replacement.
 * Owns: Independent batches of stored calls and changed catalog rows.
 * Launch shape: One thread for each history; one thread changes the catalog fixture.
 * Lifetime: One call prompt test. */
#ifndef AOTX_TEST_CALL_HISTORY_H
#define AOTX_TEST_CALL_HISTORY_H

__global__ void aotx_call_history_reset(void)
{
    memset(&aotx_catalog, 0, sizeof aotx_catalog);
}

__global__ void aotx_call_history_change(unsigned int mode)
{
    unsigned int entry = aotx_catalog_find("fs_update", 9u, AOTX_MODULE_TOOL);
    if (entry >= AOTX_MODULE_SLOTS || mode == 2u) return;
    aotx_catalog_entry *row = &aotx_catalog.entry[entry];
    aotx_catalog_run first = row->tool.key[0];
    row->tool.key[0] = row->tool.key[2];
    row->tool.key[2] = first;
    if (mode == 1u) {
        const char name[] = "other_tool";
        memcpy(row->name, name, sizeof name);
        row->name_len = sizeof name - 1u;
        row->tool.arguments = 1u;
    }
}

__global__ void aotx_call_history_read(aotx_call_prompt_row *rows, unsigned int count)
{
    unsigned int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= count) return;
    aotx_call_prompt_row *row = &rows[slot];
    row->history_len = aotx_transcript_prompt(slot, row->history, 0u);
    row->list_len = aotx_catalog_tool_list(row->list, 0u, AOTX_MODULE_SLOTS - 1u);
    const unsigned char query[] = "change the file";
    unsigned int bytes = 0u;
    while (row->result[bytes]) ++bytes;
    aotx_say.slot[slot].wanted = 0u;
    row->prompt_len = aotx_agent_prompt(slot, 0, query, sizeof query - 1u, 0, 0, 0u,
                                       row->result, bytes);
    if (row->prompt_len <= AOTX_SAY_BYTES)
        memcpy(row->prompt, aotx_say.prompt[slot], row->prompt_len);
    row->guard = aotx_transcript[slot].turn[0].tier == AOTX_MEMORY_HOT;
    bytes = 0u;
    while (row->input[bytes]) ++bytes;
    row->roundtrip = aotx_tool_parse((const unsigned char *)row->input, bytes, &row->parsed);
}

static void aotx_call_history_case(unsigned int count, unsigned int mode)
{
    aotx_call_prompt_row *host = (aotx_call_prompt_row *)calloc(count, sizeof *host), *device = NULL;
    if (host == NULL) exit(2);
    aotx_call_history_reset<<<1, 1>>>();
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    if (aotx_catalog_open() != 0) exit(2);
    aotx_call_prompt_model<<<1, 1>>>();
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    for (unsigned int i = 0u; i < count; ++i) {
        snprintf(host[i].input, sizeof host[i].input,
            "<tool_call>{\"name\":\"fs_update\",\"arguments\":{\"path\":\"file-%u\",\"old\":\"old-%u\",\"new\":\"new-%u\"}}</tool_call>", i, i, i);
        snprintf(host[i].result, sizeof host[i].result, "result-%u", i);
    }
    aotx_check_runtime(cudaMalloc(&device, count * sizeof *device), "cudaMalloc");
    aotx_check_runtime(cudaMemcpy(device, host, count * sizeof *host, cudaMemcpyHostToDevice), "cudaMemcpy");
    aotx_test_call_upload(AOTX_CALL_HERMES);
    aotx_call_prompt_store<<<(count + 63u) / 64u, 64u>>>(device, count);
    aotx_call_history_change<<<1, 1>>>(mode);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_test_call_upload(mode == 2u ? AOTX_CALL_NONE : AOTX_CALL_LLAMA_JSON);
    aotx_call_history_read<<<(count + 63u) / 64u, 64u>>>(device, count);
    aotx_check_runtime(cudaMemcpy(host, device, count * sizeof *host, cudaMemcpyDeviceToHost), "cudaMemcpy");
    for (unsigned int i = 0u; i < count; ++i) {
        aotx_call_prompt_row *row = &host[i];
        char expected[512], result[256];
        if (mode == 2u) {
            snprintf(expected, sizeof expected, "%s", row->input);
            snprintf(result, sizeof result, "[tool result]\n%s", row->result);
        } else {
            snprintf(expected, sizeof expected,
                "{\"name\": \"fs_update\", \"parameters\": {\"path\": \"file-%u\", \"old\": \"old-%u\", \"new\": \"new-%u\"}}", i, i, i);
            snprintf(result, sizeof result, "<|start_header_id|>ipython<|end_header_id|>\n\n\"%s\"<|eot_id|>", row->result);
        }
        check(row->input_ok, "source call is accepted before replacement", mode, i);
        check(row->history_len < AOTX_SAY_BYTES && row->prompt_len > 0u && row->prompt_len < AOTX_SAY_BYTES,
              "changed catalog and no-tool history remain renderable", mode, i);
        if (row->history_len < AOTX_SAY_BYTES) row->history[row->history_len] = 0u;
        if (row->prompt_len < AOTX_SAY_BYTES) row->prompt[row->prompt_len] = 0u;
        check(strstr((const char *)row->history, expected) && strstr((const char *)row->prompt, expected),
              "history retains original tool identity and key meanings", mode, i);
        if (i == 0u && (!strstr((const char *)row->history, expected)
                        || !strstr((const char *)row->prompt, expected))) {
            printf("expected call: %s\nhistory: %s\nprompt: %s\n", expected, row->history, row->prompt);
        }
        check(strstr((const char *)row->history, result) && strstr((const char *)row->prompt, result),
              "history retains the result after replacement", mode, i);
        check(row->guard, "replacement does not evict fitting tool history", mode, i);
        if (mode == 2u) check(row->list_len == 0u && row->roundtrip == 0u,
              "text-only history does not enable new calls", mode, i);
    }
    cudaFree(device);
    free(host);
}

#endif
