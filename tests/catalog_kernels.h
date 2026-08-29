/* Purpose: Give the catalog check its kernels, its module texts and its waits.
 * Owns: The scratch entry of the reader case and the buffer of the list case.
 * Threading: One host thread drives the cases; the kernels take one thread.
 * Lifetime: One run of the test program. */
#ifndef AOTX_TESTS_CATALOG_KERNELS_H
#define AOTX_TESTS_CATALOG_KERNELS_H

#include "agent/agent_state.cuh"
#include "agent/prompt.cuh"
#include "boot/check.h"
#include "catalog/catalog.cuh"
#include "cli/cli.cuh"
#include "sched/sched.cuh"
#include "tool/tool_state.cuh"

#include "catalog_feed.h"

#define AOTX_CATALOG_TEST_TOOL \
    "kind: tool\n" \
    "name: word_count\n" \
    "version: 0.1\n" \
    "description: Counts the words of a text.\n" \
    "side: device\n" \
    "arguments: text\n" \
    "authorise: always\n" \
    "deadline: 400\n" \
    "timeout: 20\n" \
    "module: word_count.ptx\n" \
    "entry: aotx_tool_word_count\n" \
    "example: text=one two three\n" \
    "sha256: 00\n"

/* The same tool as a host tool. A tool of side device names a module file, and the pump
 * loads that file at the capture that follows the import. The cases of the catalog hold no
 * module file, so the tool they install runs on the disk side. */
#define AOTX_CATALOG_TEST_TOOL_HOST \
    "kind: tool\n" \
    "name: word_count\n" \
    "version: 0.1\n" \
    "description: Counts the words of a text.\n" \
    "side: host\n" \
    "arguments: text\n" \
    "authorise: always\n" \
    "deadline: 400\n" \
    "timeout: 20\n" \
    "program: word_count.sh\n" \
    "example: text=one two three\n"

#define AOTX_CATALOG_TEST_ROLE \
    "kind: role\n" \
    "name: scribe\n" \
    "version: 2\n" \
    "description: Writes things down.\n" \
    "model: language-q4\n" \
    "tools: memory_recall,memory_write\n" \
    "authorise: memory_write\n" \
    "budget: 5\n" \
    "pages: 3\n" \
    "skills:\n" \
    "body: overlay.txt\n"

#define AOTX_CATALOG_TEST_SKILL \
    "kind: skill\n" \
    "name: how_to_count\n" \
    "version: 1\n" \
    "description: How to count things.\n" \
    "body: SKILL.md\n"

/* The entry the reader case fills. The entry stands beside the catalog, so a text the
 * reader refuses leaves the catalog as it was. */
__device__ aotx_catalog_entry aotx_catalog_test_row;

/* The bytes the list case builds. The buffer is a prompt table of one slot. */
__device__ unsigned char aotx_catalog_test_block[AOTX_SAY_BYTES];
__device__ unsigned int aotx_catalog_test_block_len;

/* Read one manifest text. The text goes in the arena, as an import puts it there, and the
 * run goes back when the read ends. */
__global__ void aotx_catalog_test_read_one(const char *text, unsigned int length,
                                           unsigned int kind, const char *head,
                                           unsigned int head_len, unsigned int *out)
{
    if (blockIdx.x != 0u || threadIdx.x != 0u) {
        return;
    }
    aotx_catalog_run run;
    if (aotx_catalog_take_run(length, &run) != 0) {
        out[0] = AOTX_CATALOG_WHY_ARENA;
        out[1] = 0u;
        return;
    }
    for (unsigned int i = 0u; i < length; ++i) {
        aotx_catalog_arena[run.at + i] = (unsigned char)text[i];
    }
    aotx_catalog_entry *row = &aotx_catalog_test_row;
    row->name_len = head_len;
    for (unsigned int i = 0u; i < AOTX_CATALOG_NAME_BYTES; ++i) {
        row->name[i] = (i < head_len) ? head[i] : '\0';
    }
    row->manifest = run;
    unsigned int figure = 0u;
    out[0] = aotx_catalog_manifest_read(row, run.at, length, kind, &figure);
    out[1] = figure;
    out[2] = row->tool.arguments;
    out[3] = row->role.skills;
    out[4] = row->description.length;
    out[5] = row->version.length;
    out[6] = row->unknown;
    out[7] = row->tool.deadline;
    out[8] = row->tool.timeout;
    out[9] = row->tool.side;
    out[10] = row->tool.authorize;
    out[11] = row->role.budget;
    out[12] = row->role.model;
    out[13] = row->role.pages;
    aotx_catalog_free_run(run);
}

/* Build the system block of a role, as a prompt of that role does. */
__global__ void aotx_catalog_test_build_block(unsigned int role)
{
    if (blockIdx.x != 0u || threadIdx.x != 0u) {
        return;
    }
    unsigned int at = 0u;
    for (unsigned int i = 0u; i < AOTX_SAY_BYTES; ++i) {
        aotx_catalog_test_block[i] = 0u;
    }
    if (role < AOTX_MODULE_SLOTS) {
        aotx_catalog_run overlay = aotx_catalog.entry[role].role.overlay;
        for (unsigned int i = 0u; i < overlay.length && at < AOTX_SAY_BYTES; ++i) {
            aotx_catalog_test_block[at++] = aotx_catalog_arena[overlay.at + i];
        }
    }
    at = aotx_catalog_skill_bodies(aotx_catalog_test_block, at, role);
    at = aotx_catalog_tool_list(aotx_catalog_test_block, at, role);
    aotx_catalog_test_block_len = at;
}

/* Give every entry that is not a built-in tool back, and give its arena runs back with
 * it. The free list must then hold one run, which proves the runs join again. */
__global__ void aotx_catalog_test_clear(unsigned int *frees)
{
    if (blockIdx.x != 0u || threadIdx.x != 0u) {
        return;
    }
    for (unsigned int i = 0u; i < AOTX_CATALOG_ARRIVING_MAX; ++i) {
        if (aotx_catalog.arriving[i].import != 0u) {
            aotx_catalog_free_run(aotx_catalog.arriving[i].run[0]);
            aotx_catalog_free_run(aotx_catalog.arriving[i].run[1]);
            aotx_catalog.arriving[i].import = 0u;
        }
    }
    for (unsigned int i = 0u; i < AOTX_MODULE_SLOTS; ++i) {
        aotx_catalog_entry *row = &aotx_catalog.entry[i];
        if (row->state == AOTX_CATALOG_FREE
            || (row->kind == AOTX_MODULE_TOOL
                && row->tool.side == AOTX_CATALOG_SIDE_BUILT)) {
            continue;
        }
        aotx_catalog_release(row);
        row->state = AOTX_CATALOG_FREE;
        row->name_len = 0u;
        row->kind = 0u;
        row->why = AOTX_CATALOG_WHY_NONE;
    }
    aotx_catalog_anchor();
    *frees = aotx_catalog.frees;
}

/* Open one skill_use request for a run of agents, as the agent step does. */
__global__ void aotx_catalog_test_ask(unsigned int count, const char *name,
                                      unsigned int length, unsigned int *out,
                                      unsigned long long tick)
{
    if (blockIdx.x != 0u || threadIdx.x != 0u) {
        return;
    }
    aotx_tool_call call;
    call.entry = aotx_catalog_find("skill_use", 9u, AOTX_MODULE_TOOL);
    call.tool = AOTX_TOOL_SKILL_USE;
    call.key = 0u;
    call.provenance = 0u;
    call.arg_len = length;
    for (unsigned int i = 0u; i < length && i < AOTX_TOOL_ARG_BYTES; ++i) {
        call.arg[i] = name[i];
        call.pack[i] = name[i];
    }
    /* The pack of a call holds the value of every key, and the request writes the argument
     * line from it. A call the check builds by hand fills the pack of its one key. */
    call.values = 1u;
    call.pack_len = length;
    for (unsigned int k = 0u; k < AOTX_CATALOG_ARGS; ++k) {
        call.at[k] = 0u;
        call.length[k] = (k == 0u) ? length : 0u;
    }
    for (unsigned int a = 0u; a < count; ++a) {
        aotx_requests.slot[a].request = 0u;
        out[a] = aotx_tool_request(a, &call, 0u, tick);
        /* The turn of an agent keeps the call it made, because the prompt that carries
         * the result writes the call in front of it. */
        aotx_agent_gear[a].call = call;
    }
}

/* Build the prompt of one turn of an agent, as the agent step does. The result of the
 * request of that agent goes in the prompt when the case asks for it. */
__global__ void aotx_catalog_test_turn(unsigned int agent, const char *message,
                                       unsigned int message_len, unsigned int with_result,
                                       unsigned long long *tick, unsigned int *length)
{
    if (blockIdx.x != 0u || threadIdx.x != 0u) {
        return;
    }
    aotx_say.slot[agent].wanted = 0u;
    const aotx_request *hold = &aotx_requests.slot[agent];
    const char *result = (with_result != 0u) ? hold->result : 0;
    unsigned int bytes = (with_result != 0u) ? hold->result_len : 0u;
    *length = aotx_agent_prompt(agent, 0, (const unsigned char *)message, message_len,
                                0, 0, 0u, result, bytes);
    *tick = aotx_time_tick;
    /* The case reads the bytes of the prompt and asks for no sequence. This run holds no
     * model file, so the tokenize nodes of a tick must not take the slot. */
    aotx_say.slot[agent].wanted = 0u;
    aotx_agents.agent[agent].state = AOTX_AGENT_STATE_IDLE;
}

/* Read the prompt bytes of one slot. */
__global__ void aotx_catalog_test_prompt(unsigned int agent, unsigned char *out)
{
    unsigned int at = blockIdx.x * blockDim.x + threadIdx.x;
    if (at < AOTX_SAY_BYTES) {
        out[at] = aotx_say.prompt[agent][at];
    }
}

/* Open one request for a tool entry, so a remove of that tool meets a request in flight. */
__global__ void aotx_catalog_test_hold(unsigned int agent, unsigned int entry,
                                       unsigned long long tick, unsigned int *out)
{
    if (blockIdx.x != 0u || threadIdx.x != 0u) {
        return;
    }
    aotx_tool_call call;
    call.entry = entry;
    call.tool = AOTX_TOOL_NONE;
    call.key = 0u;
    call.provenance = 0u;
    call.arg_len = 4u;
    call.arg[0] = 'w';
    call.arg[1] = 'o';
    call.arg[2] = 'r';
    call.arg[3] = 'd';
    call.values = 1u;
    call.pack_len = 4u;
    call.pack[0] = 'w';
    call.pack[1] = 'o';
    call.pack[2] = 'r';
    call.pack[3] = 'd';
    for (unsigned int k = 0u; k < AOTX_CATALOG_ARGS; ++k) {
        call.at[k] = 0u;
        call.length[k] = (k == 0u) ? 4u : 0u;
    }
    aotx_requests.slot[agent].request = 0u;
    *out = aotx_tool_request(agent, &call, 0u, tick);
}

/* Open one skill_use request for one agent. */
__global__ void aotx_catalog_test_ask_one(unsigned int agent, const char *name,
                                          unsigned int length, unsigned int *out,
                                          unsigned long long tick)
{
    if (blockIdx.x != 0u || threadIdx.x != 0u) {
        return;
    }
    aotx_tool_call call;
    call.entry = aotx_catalog_find("skill_use", 9u, AOTX_MODULE_TOOL);
    call.tool = AOTX_TOOL_SKILL_USE;
    call.key = 0u;
    call.provenance = 0u;
    call.arg_len = length;
    for (unsigned int i = 0u; i < length && i < AOTX_TOOL_ARG_BYTES; ++i) {
        call.arg[i] = name[i];
    }
    aotx_requests.slot[agent].request = 0u;
    *out = aotx_tool_request(agent, &call, 0u, tick);
    aotx_agent_gear[agent].call = call;
}

/* Read the result of one request. */
__global__ void aotx_catalog_test_result_one(unsigned int agent, unsigned int *length,
                                             char *bytes)
{
    unsigned int at = blockIdx.x * blockDim.x + threadIdx.x;
    if (at == 0u) {
        *length = aotx_requests.slot[agent].result_len;
    }
    for (unsigned int i = at; i < AOTX_TOOL_RESULT_BYTES; i += gridDim.x * blockDim.x) {
        bytes[i] = aotx_requests.slot[agent].result[i];
    }
}

/* Read the result of a run of requests. */
__global__ void aotx_catalog_test_results(unsigned int count, unsigned int *status,
                                          unsigned int *length, char *bytes)
{
    unsigned int at = blockIdx.x * blockDim.x + threadIdx.x;
    if (at >= count) {
        return;
    }
    status[at] = aotx_requests.slot[at].status;
    length[at] = aotx_requests.slot[at].result_len;
    for (unsigned int i = 0u; i < AOTX_TOOL_RESULT_BYTES; ++i) {
        bytes[(size_t)at * AOTX_TOOL_RESULT_BYTES + i] = aotx_requests.slot[at].result[i];
    }
}

/* Give the request table back empty. */
__global__ void aotx_catalog_test_free_requests(void)
{
    unsigned int at = blockIdx.x * blockDim.x + threadIdx.x;
    if (at < AOTX_SLOTS) {
        aotx_requests.slot[at].request = 0u;
        aotx_requests.slot[at].auth = AOTX_AUTH_NONE;
        aotx_tool_done[at] = 0u;
        aotx_agents.agent[at].state = AOTX_AGENT_STATE_FREE;
        aotx_agents.agent[at].role = AOTX_ROLE_NONE;
        aotx_agents.agent[at].tool = AOTX_CATALOG_NO_ENTRY;
    }
    if (at == 0u) {
        aotx_agents.live = 0u;
    }
}

/* Spawn one agent of a role, as the command layer does. */
__global__ void aotx_catalog_test_spawn(unsigned int role, unsigned int *out,
                                        unsigned long long tick)
{
    if (blockIdx.x == 0u && threadIdx.x == 0u) {
        *out = aotx_agent_spawn(role, ~0u, tick);
    }
}

/* Cut the result of the request of an agent to the room a prompt of its role holds. The
 * agent step makes this call before the turn that carries the result. */
__global__ void aotx_catalog_test_cut(unsigned int agent, unsigned int *room)
{
    if (blockIdx.x != 0u || threadIdx.x != 0u) {
        return;
    }
    *room = aotx_agent_result_room(aotx_agents.agent[agent].role);
    aotx_agent_cut_result(&aotx_requests.slot[agent], *room);
}

/* Report whether the free list of the arena is sound. */
__global__ void aotx_catalog_test_arena(unsigned int *out)
{
    if (blockIdx.x == 0u && threadIdx.x == 0u) {
        *out = (unsigned int)aotx_catalog_arena_sound();
    }
}

/* Copy the body of the newest record of a type in the device ring. The mark is 1 when the
 * ring holds such a record. */
__global__ void aotx_catalog_test_last(unsigned int type, unsigned char *out,
                                       unsigned int *mark)
{
    if (blockIdx.x != 0u || threadIdx.x != 0u) {
        return;
    }
    *mark = 0u;
    unsigned long long seq = aotx_cli_last(type);
    if (seq == 0ull) {
        return;
    }
    const volatile unsigned char *body = (const volatile unsigned char *)aotx_cli_slot(seq)
                                       + AOTX_HEADER_BYTES;
    for (unsigned int i = 0u; i < AOTX_BODY_BYTES; ++i) {
        out[i] = body[i];
    }
    *mark = 1u;
}

/* Take device memory for a case and give it back. */
static void *aotx_catalog_test_take(size_t bytes)
{
    void *at = NULL;
    aotx_check_runtime(cudaMalloc(&at, bytes), "cudaMalloc");
    aotx_check_runtime(cudaMemset(at, 0, bytes), "cudaMemset");
    return at;
}

/* Read the console buffer of the device. */
static void aotx_catalog_test_console(aotx_console_state *out)
{
    aotx_check_runtime(cudaMemcpyFromSymbol(out, aotx_console, sizeof *out),
                       "cudaMemcpyFromSymbol");
}

/* The line the console has put in last. */
static unsigned long long aotx_catalog_test_mark(void)
{
    aotx_console_state *console = (aotx_console_state *)malloc(sizeof *console);
    aotx_catalog_test_console(console);
    unsigned long long at = console->count;
    free(console);
    return at;
}

/* Report whether a line the console put in after the mark holds the text. */
static int aotx_catalog_test_said(unsigned long long mark, const char *text, int want)
{
    aotx_console_state *console = (aotx_console_state *)malloc(sizeof *console);
    int found = 0;
    aotx_catalog_test_console(console);
    unsigned long long first = mark + 1ull;
    if (console->count > AOTX_CONSOLE_LINES
        && first < console->count - AOTX_CONSOLE_LINES + 1ull) {
        first = console->count - AOTX_CONSOLE_LINES + 1ull;
    }
    for (unsigned long long at = first; at <= console->count && found == 0; ++at) {
        const aotx_console_line *line =
            &console->line[(at - 1ull) & (AOTX_CONSOLE_LINES - 1u)];
        if (line->seq != at || line->length == 0u) {
            continue;
        }
        char held[AOTX_CONSOLE_COLS + 1u];
        unsigned int span = (line->length < AOTX_CONSOLE_COLS) ? line->length
                                                               : AOTX_CONSOLE_COLS;
        memcpy(held, line->text, span);
        held[span] = '\0';
        found = (strstr(held, text) != NULL) ? 1 : 0;
    }
    if (found != want) {
        printf("catalog: the lines of the tick %s hold '%s'\n",
               (found != 0) ? "still" : "do not", text);
    }
    free(console);
    return (found == want) ? 1 : 0;
}

/* Run a run of ticks. */
static void aotx_catalog_test_ticks(aotx_pump *pump, unsigned int count)
{
    for (unsigned int i = 0u; i < count; ++i) {
        aotx_pump_tick(pump);
    }
}

/* Report one case. The pair of a guard is the case that passes and the case that fails,
 * so a guard that cannot fail is not a guard. */
static void aotx_catalog_test_check(int good, const char *what, unsigned int *applied,
                                    unsigned int *failed)
{
    *applied += 1u;
    if (good == 0) {
        printf("catalog: FAILED %s\n", what);
        *failed += 1u;
    }
}

#endif
