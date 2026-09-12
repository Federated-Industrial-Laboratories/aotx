/* Purpose: Check complete tool chains and repeated operator inputs in prompt history.
 * Owns: Distinct stored turns, results, and selected-memory fixtures.
 * Launch shape: One thread for each agent, at one and the profile slot count.
 * Lifetime: One call prompt test. */
#ifndef AOTX_TEST_CALL_CHAIN_H
#define AOTX_TEST_CALL_CHAIN_H

__global__ void aotx_call_chain_store(aotx_call_prompt_row *rows, unsigned int count,
                                      unsigned int calls)
{
    unsigned int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= count) return;
    aotx_call_prompt_row *row = &rows[slot];
    unsigned int bytes = 0u, query_len = 0u, result_len = 0u;
    while (row->input[bytes]) ++bytes;
    while (row->query[query_len]) ++query_len;
    while (row->result[result_len]) ++result_len;
    row->input_ok = aotx_tool_parse((const unsigned char *)row->input, bytes, &row->call) == 1;
    memset(&aotx_transcript[slot], 0, sizeof aotx_transcript[slot]);
    memset(&aotx_agent_gear[slot], 0, sizeof aotx_agent_gear[slot]);
    aotx_transcript[slot].pages = AOTX_KV_PAGES_EACH;
    aotx_transcript[slot].text_head = AOTX_TRANSCRIPT_TEXT_BYTES - query_len - 8u;
    aotx_agent_gear[slot].source_seq = 1000ull + slot;
    for (unsigned int turn = 0u; turn < calls; ++turn) {
        aotx_agents.agent[slot].turn = turn + 1u;
        aotx_agent_gear[slot].call = row->call;
        aotx_transcript_finish(slot, (const unsigned char *)row->query, query_len,
            (const unsigned char *)row->input, bytes, 32u, 2000ull + turn * count + slot);
        aotx_request *request = &aotx_requests.slot[slot];
        memcpy(request->result, row->result, result_len);
        request->result_len = result_len;
        request->result_seq = 3000ull + turn * count + slot;
        aotx_transcript_result(slot, request);
    }
    const unsigned char answer[] = "The file is saved.";
    aotx_tool_parse(answer, sizeof answer - 1u, &aotx_agent_gear[slot].call);
    aotx_agents.agent[slot].turn = calls + 1u;
    aotx_transcript_finish(slot, (const unsigned char *)row->query, query_len,
        answer, sizeof answer - 1u, 16u, 4000ull + slot);
}

__global__ void aotx_call_chain_read(aotx_call_prompt_row *rows, unsigned int count,
                                     unsigned int calls)
{
    unsigned int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= count) return;
    aotx_call_prompt_row *row = &rows[slot];
    unsigned int bytes = 0u, query_len = 0u;
    while (row->input[bytes]) ++bytes;
    while (row->query[query_len]) ++query_len;
    row->history_len = aotx_transcript_prompt(slot, row->history, 0u);
    row->rendered_len = aotx_agent_put_call(row->rendered, 0u, &row->call,
        (const unsigned char *)row->input, bytes, aotx_model_default_language());
    aotx_agent_gear[slot].source_seq = 5000ull + slot;
    aotx_agents.agent[slot].turn = calls + 2u;
    const unsigned char answer[] = "The same request is complete.";
    aotx_transcript_finish(slot, (const unsigned char *)row->query, query_len,
        answer, sizeof answer - 1u, 16u, 6000ull + slot);
    row->prompt_len = aotx_transcript_prompt(slot, row->prompt, 0u);
    aotx_transcript_agent *hold = &aotx_transcript[slot];
    for (unsigned int i = 0u; i < hold->count; ++i) hold->turn[i].tier = AOTX_MEMORY_WARM;
    hold->selected_count = 1u;
    hold->selected[0] = calls;
    row->limits_len = aotx_transcript_prompt(slot, row->limits, 0u);
    row->guard = row->call.prefix_len;
}

static unsigned int aotx_call_occurrences(const unsigned char *text, const char *part)
{
    unsigned int count = 0u;
    const char *at = (const char *)text;
    while ((at = strstr(at, part)) != NULL) {
        ++count;
        at += strlen(part);
    }
    return count;
}

static void aotx_call_chain_case(unsigned int count, unsigned int kind, unsigned int calls)
{
    aotx_call_prompt_row *host = (aotx_call_prompt_row *)calloc(count, sizeof *host), *device = NULL;
    if (host == NULL) exit(2);
    for (unsigned int i = 0u; i < count; ++i) {
        snprintf(host[i].query, sizeof host[i].query, "change file %u", i);
        snprintf(host[i].prefix, sizeof host[i].prefix, "The file %u will change.\n", i);
        snprintf(host[i].input, sizeof host[i].input,
            "%s<tool_call>{\"name\":\"fs_update\",\"arguments\":{\"path\":\"file-%u\","
            "\"old\":\"old-%u\",\"new\":\"new-%u\"}}</tool_call>", host[i].prefix, i, i, i);
        snprintf(host[i].result, sizeof host[i].result, "saved file %u", i);
    }
    aotx_call_history_reset<<<1, 1>>>();
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    if (aotx_catalog_open() != 0) exit(2);
    aotx_call_prompt_model<<<1, 1>>>();
    aotx_check_runtime(cudaMalloc(&device, count * sizeof *device), "cudaMalloc");
    aotx_check_runtime(cudaMemcpy(device, host, count * sizeof *host, cudaMemcpyHostToDevice), "cudaMemcpy");
    aotx_test_call_upload(AOTX_CALL_HERMES);
    aotx_call_chain_store<<<(count + 63u) / 64u, 64u>>>(device, count, calls);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_test_call_upload(kind);
    aotx_call_chain_read<<<(count + 63u) / 64u, 64u>>>(device, count, calls);
    aotx_check_runtime(cudaMemcpy(host, device, count * sizeof *host, cudaMemcpyDeviceToHost), "cudaMemcpy");
    for (unsigned int i = 0u; i < count; ++i) {
        aotx_call_prompt_row *row = &host[i];
        int fits = row->history_len < AOTX_SAY_BYTES && row->prompt_len < AOTX_SAY_BYTES
            && row->limits_len < AOTX_SAY_BYTES && row->rendered_len < AOTX_SAY_BYTES;
        check(fits, "complete chain fits the prompt", kind, i);
        if (!fits) continue;
        row->history[row->history_len] = 0u;
        row->prompt[row->prompt_len] = 0u;
        row->limits[row->limits_len] = 0u;
        row->rendered[row->rendered_len] = 0u;
        check(row->input_ok && row->guard == strlen(row->prefix),
              "parser retains the text before the call", kind, i);
        check(aotx_call_occurrences(row->history, row->query) == 1u,
              "one input occurs once through the complete tool chain", kind, i);
        check(aotx_call_occurrences(row->history, row->prefix) == calls,
              "each stored call keeps its assistant prefix", kind, i);
        check(aotx_call_occurrences(row->rendered, row->prefix) == 1u,
              "a call outside hot history keeps its assistant prefix", kind, i);
        check(aotx_call_occurrences(row->prompt, row->query) == 2u,
              "equal text from a new input remains a separate user turn", kind, i);
        check(aotx_call_occurrences(row->limits, row->query) == 1u,
              "an isolated recalled continuation retains its user context", kind, i);
        check(aotx_call_occurrences(row->history, row->result) == calls,
              "each call keeps one result", kind, i);
        if (kind == AOTX_CALL_LLAMA_JSON) {
            check(strstr((const char *)row->history, "\"arguments\"") == NULL
                && strstr((const char *)row->rendered, "\"arguments\"") == NULL,
                "a retained prefix does not reintroduce old call grammar", kind, i);
        }
    }
    cudaFree(device);
    free(host);
}

#endif
