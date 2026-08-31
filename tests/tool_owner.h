/* Purpose: Check that an agent states a disk-tool refusal on the console. */
#ifndef AOTX_TEST_TOOL_OWNER_H
#define AOTX_TEST_TOOL_OWNER_H

__global__ void aotx_tool_owner_refusal(void)
{
    if (blockIdx.x != 0u || threadIdx.x != 0u) {
        return;
    }
    static const char reason[] = "no root is set";
    unsigned int entry = aotx_catalog_built_entry(AOTX_TOOL_FS_READ);
    aotx_agent *agent = &aotx_agents.agent[0];
    aotx_request *request = &aotx_requests.slot[0];
    agent->state = AOTX_AGENT_STATE_TOOL;
    agent->task = ~0u;
    agent->request = 77u;
    agent->tool = entry;
    agent->budget_left = 0u;
    request->request = 77u;
    request->agent = 0u;
    request->entry = entry;
    request->tool = AOTX_TOOL_FS_READ;
    request->status = AOTX_TOOL_REFUSED;
    request->result_len = (unsigned int)sizeof reason - 1u;
    for (unsigned int i = 0u; i < request->result_len; ++i) {
        request->result[i] = reason[i];
    }
    aotx_tool_done[0] = 1u;
}

static void aotx_tool_owner_case(unsigned int *applied, unsigned int *failed)
{
    aotx_console_state *console = (aotx_console_state *)calloc(1, sizeof *console);
    aotx_tool_owner_refusal<<<1, 1>>>();
    aotx_agent_step<<<1, AOTX_SLOTS>>>(10ull);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_check_runtime(cudaMemcpyFromSymbol(console, aotx_console, sizeof *console),
                       "cudaMemcpyFromSymbol");
    int found = 0;
    for (unsigned long long at = 1ull; at <= console->count; ++at) {
        const aotx_console_line *line =
            &console->line[(at - 1ull) & (AOTX_CONSOLE_LINES - 1u)];
        static const char want[] = "fs_read: no root is set";
        if (line->seq == at && line->length == (unsigned int)sizeof want - 1u
            && memcmp(line->text, want, sizeof want - 1u) == 0) {
            found = 1;
        }
    }
    *applied += 1u;
    if (!found) {
        *failed += 1u;
        printf("tool: the no-root refusal did not reach the console\n");
    }
    free(console);
}

#endif
