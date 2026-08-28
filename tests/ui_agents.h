/* Purpose: Check the agents panel over the agent table, its requests and the focus title.
 * Owns: The agent, sequence and request fixtures of the panel check.
 * Threading: One thread; the panel check calls these one at a time.
 * Lifetime: The program.
 *
 * The file is a part of the panel check. It reads the helpers of that check, so it comes
 * after them in the same translation unit. */
#ifndef AOTX_TEST_UI_AGENTS_H
#define AOTX_TEST_UI_AGENTS_H

/* The sample fixture. The period is 17 ms, which the pump never paces to. The step is the
 * reply tokens one sample adds for slot zero. */
#define AOTX_TEST_PERIOD    17000000ull
#define AOTX_TEST_BASE_NS   4000000000ull
#define AOTX_TEST_STEP      3u

/* Fill the agent table and the sequence table with content that differs from agent to
 * agent, so a row on the wrong agent cannot pass. The stride leaves free slots between the
 * ones that are taken. The row of an agent is then not its identity, and a panel that reads
 * one for the other fails. */
__global__ void aotx_test_agents_fill(unsigned int count, unsigned int stride,
                                      unsigned long long tick)
{
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= AOTX_AGENT_SLOTS) {
        return;
    }
    if (id == 0u) {
        aotx_agents.live = count;
        aotx_agents.tasks = count * 2u;
    }
    aotx_agent *agent = &aotx_agents.agent[id];
    aotx_seq *seq = &aotx_seqs.slot[id];
    if (stride == 0u || id % stride != 0u || id / stride >= count) {
        agent->state = AOTX_AGENT_STATE_FREE;
        seq->state = AOTX_SEQ_STATE_FREE;
        seq->sampled = 0u;
        for (unsigned int i = 0u; i < AOTX_SAY_WINDOW; ++i) {
            aotx_say_window[id][i].tick = 0ull;
        }
        return;
    }
    agent->state = AOTX_AGENT_STATE_IDLE + (id % 5u);
    agent->role = id % AOTX_ROLE_COUNT;
    agent->task = (id % 3u == 0u) ? ~0u : (id + 7u);
    agent->tool = (id % 4u == 0u) ? AOTX_TOOL_NONE : (1u + (id % 3u));
    agent->request = (id % 4u == 0u) ? 0u : (100u + id);
    agent->turn = id % 8u;
    seq->state = AOTX_SEQ_STATE_DECODE;
    seq->sampled = AOTX_TEST_STEP * (id + 1u) * (AOTX_SAY_WINDOW - 1u);
    seq->opened = tick - (unsigned long long)(10u * (id + 1u));

    /* The window the rate reads. The samples carry a period of AOTX_TEST_PERIOD ns. The
     * pump never paces to that period, so a rate taken from the pace cannot pass. */
    for (unsigned int i = 0u; i < AOTX_SAY_WINDOW; ++i) {
        aotx_say_sample *at = &aotx_say_window[id][(tick - (AOTX_SAY_WINDOW - 1u) + i)
                                                   % AOTX_SAY_WINDOW];
        at->tick = tick - (AOTX_SAY_WINDOW - 1u) + i;
        at->ns = AOTX_TEST_BASE_NS + (unsigned long long)i * AOTX_TEST_PERIOD;
        at->opened = seq->opened;
        at->sampled = AOTX_TEST_STEP * (id + 1u) * i;
    }
}

/* Fill the request table with requests that wait for the operator. The number falls as the
 * slot rises. A panel that takes the slot order for the number order then puts another
 * request at the top, and the case fails. */
__global__ void aotx_test_requests_fill(unsigned int count)
{
    unsigned int at = blockIdx.x * blockDim.x + threadIdx.x;
    if (at >= AOTX_REQUEST_SLOTS) {
        return;
    }
    aotx_request *slot = &aotx_requests.slot[at];
    if (at >= count) {
        slot->request = 0u;
        slot->auth = AOTX_AUTH_NONE;
        return;
    }
    slot->request = (count - at) * 10u;
    slot->agent = at + 3u;
    slot->tool = AOTX_TOOL_FS_READ;
    slot->auth = AOTX_AUTH_PENDING;
    /* The argument is longer than a row shows, so the cut of the row has bytes to cut. */
    unsigned int put = 0u;
    const char *head = "models/notes/request-";
    for (unsigned int i = 0u; head[i] != '\0'; ++i) {
        slot->arg[put++] = head[i];
    }
    put += aotx_text_utoa((unsigned long long)slot->request, slot->arg + put,
                          AOTX_TOOL_ARG_BYTES - put);
    while (put < 60u) {
        slot->arg[put] = (char)('a' + (char)(put % 26u));
        put += 1u;
    }
    slot->arg_len = put;
}

/* Put the focus where the case asks for it. */
__global__ void aotx_test_set_focus(unsigned int focus)
{
    aotx_cli_focus = focus;
}

/* Give the attribute of one cell of a panel. */
static unsigned int aotx_test_row_attr(unsigned int panel, unsigned int row,
                                       unsigned int col)
{
    const aotx_ui_panel *at = &aotx_test_panels[panel];
    unsigned int cell = ((unsigned int)at->row + row) * AOTX_UI_COLS
                      + (unsigned int)at->col + col;
    return aotx_test_grid[cell].attr;
}

static const char *aotx_test_role_of(unsigned int role)
{
    static const char *names[AOTX_ROLE_COUNT] = { "conductor", "worker", "verifier" };
    return (role < AOTX_ROLE_COUNT) ? names[role] : "-";
}

static const char *aotx_test_agent_state_of(unsigned int state)
{
    switch (state) {
    case AOTX_AGENT_STATE_FREE:   return "free";
    case AOTX_AGENT_STATE_IDLE:   return "idle";
    case AOTX_AGENT_STATE_PROMPT: return "prompt";
    case AOTX_AGENT_STATE_RUN:    return "run";
    case AOTX_AGENT_STATE_TOOL:   return "tool";
    case AOTX_AGENT_STATE_POST:   return "post";
    default:                      return "-";
    }
}

static const char *aotx_test_tool_of(unsigned int tool)
{
    switch (tool) {
    case AOTX_TOOL_MEMORY_RECALL: return "memory_recall";
    case AOTX_TOOL_MEMORY_WRITE:  return "memory_write";
    case AOTX_TOOL_FS_READ:       return "fs_read";
    default:                      return "-";
    }
}

/* Build the row the panel must hold for one agent, from the same fields the fixture wrote.
 * The rate is the tokens of the window over the time of the window. Both come from the two
 * samples the fixture wrote, so no part of the figure comes from the pace of the pump. */
static void aotx_test_agent_row(unsigned int id, char *out, size_t max)
{
    unsigned int sampled = AOTX_TEST_STEP * (id + 1u) * (AOTX_SAY_WINDOW - 1u);
    unsigned long long span = (unsigned long long)(AOTX_SAY_WINDOW - 1u) * AOTX_TEST_PERIOD;
    unsigned long long rate = (unsigned long long)sampled * 1000000000ull / span;
    char task[16];
    char request[16];
    if (id % 3u == 0u) {
        snprintf(task, sizeof task, "-");
    } else {
        snprintf(task, sizeof task, "%u", id + 7u);
    }
    if (id % 4u == 0u) {
        snprintf(request, sizeof request, "-");
    } else {
        snprintf(request, sizeof request, "%u", 100u + id);
    }
    snprintf(out, max, "%u %s %s %s %s %s %u %u %llu", id,
             aotx_test_role_of(id % AOTX_ROLE_COUNT),
             aotx_test_agent_state_of(AOTX_AGENT_STATE_IDLE + (id % 5u)), task,
             aotx_test_tool_of((id % 4u == 0u) ? AOTX_TOOL_NONE : (1u + (id % 3u))),
             request, id % 8u, sampled, rate);
}

/* Build the row the panel must hold for one request, and give the column of its argument. */
static unsigned int aotx_test_request_row(unsigned int slot, unsigned int count, char *out,
                                          size_t max)
{
    unsigned int number = (count - slot) * 10u;
    unsigned int agent = slot + 3u;
    char arg[AOTX_TOOL_ARG_BYTES];
    unsigned int put = (unsigned int)snprintf(arg, sizeof arg, "models/notes/request-%u",
                                              number);
    unsigned int head = 0u;
    while (put < 60u) {
        arg[put] = (char)('a' + (char)(put % 26u));
        put += 1u;
    }
    arg[60u] = '\0';
    head = (unsigned int)snprintf(out, max, "%u %u fs_read ", number, agent);
    snprintf(out + head, max - head, "%.*s", (int)AOTX_UI_REQUEST_ARG, arg);
    return 1u + head;
}

/* The agents panel holds one row for each agent that is not free, and then the requests
 * that wait. The check runs at one agent and at 64. A table with more agents than the panel
 * holds keeps its last agent row for the count that is left. */
static void aotx_test_agents_panel(unsigned int count, unsigned int stride,
                                   unsigned int requests)
{
    const aotx_ui_panel *panel = &aotx_test_panels[AOTX_UI_AGENTS];
    unsigned long long tick = 4096ull;
    unsigned int rows = AOTX_UI_AGENT_ROWS;
    unsigned int names = 2u + rows;
    unsigned int shown = (count > rows) ? (rows - 1u) : count;
    unsigned int request_rows = (requests > AOTX_UI_REQUEST_ROWS)
                              ? (AOTX_UI_REQUEST_ROWS - 1u) : requests;
    unsigned int matched = 0u;
    unsigned int right = 0u;
    char want[160];

    aotx_check_runtime(cudaMemcpyToSymbol(aotx_time_tick, &tick, sizeof tick),
                       "cudaMemcpyToSymbol");
    aotx_test_agents_fill<<<1, AOTX_AGENT_SLOTS>>>(count, stride, tick);
    aotx_test_requests_fill<<<1, AOTX_REQUEST_SLOTS>>>(requests);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_ui_agents<<<1, AOTX_UI_PANEL_THREADS>>>();
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_test_read_grid();

    for (unsigned int rank = 0u; rank < shown; ++rank) {
        unsigned int id = rank * stride;
        aotx_test_agent_row(id, want, sizeof want);
        if (aotx_test_row_says(AOTX_UI_AGENTS, rank + 2u, 1u, want)) {
            matched += 1u;
        } else {
            printf("ui: the agents row %u is not '%s'\n", rank + 2u, want);
        }
    }
    aotx_test_check(matched == shown, "every row of the agents panel holds its own agent");
    aotx_test_check(aotx_test_row_says(AOTX_UI_AGENTS, 1u, 1u,
                                       "id role state task tool request turn tokens rate"),
                    "the agents panel names its columns");
    if (count > rows) {
        snprintf(want, sizeof want, "and %u more", count - shown);
        aotx_test_check(aotx_test_row_says(AOTX_UI_AGENTS, names - 1u, 1u, want),
                        "the last agent row states the agents the panel could not hold");
    } else {
        aotx_test_check(aotx_test_row_blank(AOTX_UI_AGENTS, names - 1u),
                        "a table that fits leaves the last agent row empty");
    }

    /* The requests that wait, the lowest number first, which is the last slot filled. */
    for (unsigned int rank = 0u; rank < request_rows; ++rank) {
        unsigned int slot = requests - 1u - rank;
        unsigned int col = aotx_test_request_row(slot, requests, want, sizeof want);
        if (aotx_test_row_says(AOTX_UI_AGENTS, names + 1u + rank, 1u, want)) {
            right += 1u;
        } else {
            printf("ui: the requests row %u is not '%s'\n", names + 1u + rank, want);
        }
        /* The row shows the first bytes of the argument and no more. */
        aotx_test_check(aotx_test_row_says(AOTX_UI_AGENTS, names + 1u + rank,
                                           col + AOTX_UI_REQUEST_ARG, " "),
                        "the request row cuts the argument at its bound");
    }
    aotx_test_check(right == request_rows, "every request row holds its own request");
    if (requests == 0u) {
        aotx_test_check(aotx_test_row_says(AOTX_UI_AGENTS, names, 1u, "requests none"),
                        "a panel with no request that waits says so");
    } else {
        aotx_test_check(aotx_test_row_says(AOTX_UI_AGENTS, names, 1u,
                                           "requests id agent tool argument"),
                        "the requests list names its columns");
    }
    if (requests > AOTX_UI_REQUEST_ROWS) {
        snprintf(want, sizeof want, "and %u more",
                 requests - (AOTX_UI_REQUEST_ROWS - 1u));
        aotx_test_check(aotx_test_row_says(AOTX_UI_AGENTS,
                                           (unsigned int)panel->rows - 1u, 1u, want),
                        "the last row states the requests the panel could not hold");
    }

    /* The tick panel states the agents, the tasks and the requests that wait. */
    aotx_ui_tick<<<1, AOTX_UI_PANEL_THREADS>>>();
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_test_read_grid();
    snprintf(want, sizeof want, "agents %u tasks %u pending %u", count, count * 2u,
             requests);
    aotx_test_check(aotx_test_row_says(AOTX_UI_TICK, 11u, 1u, want),
                    "the tick panel states the agents, the tasks and the requests");
    printf("ui: %u agents at a stride of %u gave %u rows, and %u requests gave %u rows\n",
           count, stride, matched, requests, right);

    /* Give the tables back empty, so a later case sees the panel with nothing in it. */
    aotx_test_agents_fill<<<1, AOTX_AGENT_SLOTS>>>(0u, stride, tick);
    aotx_test_requests_fill<<<1, AOTX_REQUEST_SLOTS>>>(0u);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
}

/* The two panels that take the focus show it in the title. The panel with the focus shows a
 * bright title and the other one shows a dim title. */
static void aotx_test_focus_title(void)
{
    aotx_test_set_focus<<<1, 1>>>(AOTX_CLI_FOCUS_CONSOLE);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_ui_console<<<1, AOTX_UI_PANEL_THREADS>>>();
    aotx_ui_agents<<<1, AOTX_UI_PANEL_THREADS>>>();
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_test_read_grid();
    aotx_test_check(aotx_test_row_attr(AOTX_UI_CONSOLE, 0u, 1u) == AOTX_UI_HIGH,
                    "the console title is bright while the console has the focus");
    aotx_test_check(aotx_test_row_attr(AOTX_UI_AGENTS, 0u, 1u) == AOTX_UI_DIM,
                    "the agents title is dim while the console has the focus");

    aotx_test_set_focus<<<1, 1>>>(AOTX_CLI_FOCUS_AGENTS);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_ui_console<<<1, AOTX_UI_PANEL_THREADS>>>();
    aotx_ui_agents<<<1, AOTX_UI_PANEL_THREADS>>>();
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_test_read_grid();
    aotx_test_check(aotx_test_row_attr(AOTX_UI_AGENTS, 0u, 1u) == AOTX_UI_HIGH,
                    "the agents title is bright while the panel has the focus");
    aotx_test_check(aotx_test_row_attr(AOTX_UI_CONSOLE, 0u, 1u) == AOTX_UI_DIM,
                    "the console title is dim while the panel has the focus");
    aotx_test_set_focus<<<1, 1>>>(AOTX_CLI_FOCUS_CONSOLE);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    printf("ui: the focus title swapped between the console and the agents panel\n");
}

#endif
