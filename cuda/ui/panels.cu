/* Purpose: Fill the console, agents and bus panels from the records the device holds.
 * Owns: Nothing; the kernels write the cells of their own panel.
 * Launch shape: One block for each panel; one thread for each row and for each cell.
 * Lifetime: One node of every frame. */
#include "bus/bus.cuh"
#include "ui/ui.cuh"

/* Put the title on the first row of a panel. */
static __device__ __forceinline__ void aotx_ui_title(const aotx_ui_panel *panel,
                                                     const char *name)
{
    aotx_ui_say(panel, 0u, 1u, name, AOTX_UI_HIGH);
}

/* Put the title of a panel that takes the focus. The panel with the focus shows a bright
 * title and the other one shows a dim title. A panel that never takes the focus keeps the
 * bright title of the function above. */
static __device__ __forceinline__ void aotx_ui_focus_title(const aotx_ui_panel *panel,
                                                           const char *name,
                                                           unsigned int which)
{
    aotx_ui_say(panel, 0u, 1u, name,
                (aotx_cli_focus == which) ? AOTX_UI_HIGH : AOTX_UI_DIM);
}

/* Put a value on a row, or a dash when the value is the one that stands for nothing. */
static __device__ __forceinline__ unsigned int aotx_ui_or_dash(const aotx_ui_panel *panel,
                                                               unsigned int row,
                                                               unsigned int col,
                                                               unsigned int value,
                                                               unsigned int none)
{
    if (value == none) {
        return aotx_ui_say(panel, row, col, "-", AOTX_UI_DIM);
    }
    return aotx_ui_number(panel, row, col, (unsigned long long)value, AOTX_UI_NORMAL);
}

/* Put the command line on the last row of the console panel, with the cursor cell bright.
 * A cursor at the end of the line shows an underscore. */
static __device__ __forceinline__ void aotx_ui_command(const aotx_ui_panel *panel)
{
    unsigned int row = (unsigned int)panel->rows - 1u;
    unsigned int col = aotx_ui_say(panel, row, 1u, "> ", AOTX_UI_DIM);
    unsigned int room = (unsigned int)panel->cols - col - 1u;
    unsigned int length = aotx_cli.length;
    if (length > room) {
        length = room;
    }
    for (unsigned int i = 0u; i < length; ++i) {
        aotx_ui_put(panel, row, col + i, aotx_ui_glyph(aotx_cli.line[i]), AOTX_UI_NORMAL);
    }
    unsigned int cursor = aotx_cli.cursor;
    if (cursor > length) {
        cursor = length;
    }
    unsigned char glyph = (cursor < length) ? aotx_ui_glyph(aotx_cli.line[cursor])
                                            : aotx_ui_glyph((unsigned int)'_');
    aotx_ui_put(panel, row, col + cursor, glyph, AOTX_UI_HIGH);
}

/* The console shows the last lines of the console buffer in order, the newest on the row
 * above the command line. Each thread takes one row and reads the line of that row. The
 * panel reads the buffer and not the record ring. At a high record rate the ring holds a
 * console record for a fraction of a second. */
__global__ void aotx_ui_console(void)
{
    const aotx_ui_panel *panel = &aotx_ui_panel_table[AOTX_UI_CONSOLE];
    aotx_ui_blank(panel);
    __syncthreads();

    unsigned int lines = (unsigned int)panel->rows - 2u;
    if (lines > AOTX_UI_LINE_MAX) {
        lines = AOTX_UI_LINE_MAX;
    }
    unsigned long long count = aotx_console.count;
    for (unsigned int i = threadIdx.x; i < lines; i += blockDim.x) {
        if ((unsigned long long)i >= count) {
            continue;
        }
        aotx_ui_line(panel, lines - i, 1u, count - (unsigned long long)i);
    }
    if (threadIdx.x == 0u) {
        aotx_ui_focus_title(panel, "console", AOTX_CLI_FOCUS_CONSOLE);
        aotx_ui_command(panel);
    }
}

/* Put one agent on one row. The row holds the identity, the role and the state. It then
 * holds the task in hand, the tool of a request that waits and the number of that request.
 * It ends with the turns taken, the reply tokens and the reply tokens each second. An agent
 * owns the sequence slot of its identity, so the tokens come from that slot. */
static __device__ __forceinline__ void aotx_ui_agent_row(const aotx_ui_panel *panel,
                                                         unsigned int row, unsigned int id)
{
    const aotx_agent *agent = &aotx_agents.agent[id];
    unsigned int col = aotx_ui_number(panel, row, 1u, (unsigned long long)id,
                                      AOTX_UI_NORMAL);
    col = aotx_ui_say(panel, row, col + 1u, aotx_cli_role_name(agent->role), AOTX_UI_DIM);
    col = aotx_ui_say(panel, row, col + 1u, aotx_cli_agent_state_name(agent->state),
                      AOTX_UI_HIGH);
    col = aotx_ui_or_dash(panel, row, col + 1u, agent->task, 0xffffffffu);
    col = aotx_ui_say(panel, row, col + 1u, aotx_cli_tool_name(agent->tool), AOTX_UI_DIM);
    col = aotx_ui_or_dash(panel, row, col + 1u, agent->request, 0u);
    col = aotx_ui_number(panel, row, col + 1u, (unsigned long long)agent->turn,
                         AOTX_UI_NORMAL);
    col = aotx_ui_number(panel, row, col + 1u,
                         (unsigned long long)aotx_seqs.slot[id].sampled, AOTX_UI_NORMAL);
    aotx_ui_number(panel, row, col + 1u, aotx_say_rate(id), AOTX_UI_NORMAL);
}

/* Put one request that waits for the operator on one row: the number, the agent, the tool
 * and the first bytes of the argument. The operator answers the first row with the keys y
 * and n. The authorise and refuse commands answer any row. */
static __device__ __forceinline__ void aotx_ui_request_row(const aotx_ui_panel *panel,
                                                           unsigned int row,
                                                           unsigned int at)
{
    const aotx_request *slot = &aotx_requests.slot[at];
    unsigned int col = aotx_ui_number(panel, row, 1u, (unsigned long long)slot->request,
                                      AOTX_UI_HIGH);
    col = aotx_ui_number(panel, row, col + 1u, (unsigned long long)slot->agent,
                         AOTX_UI_NORMAL);
    col = aotx_ui_say(panel, row, col + 1u, aotx_cli_tool_name(slot->tool), AOTX_UI_DIM);
    col += 1u;
    unsigned int length = slot->arg_len;
    if (length > AOTX_TOOL_ARG_BYTES) {
        length = AOTX_TOOL_ARG_BYTES;
    }
    if (length > AOTX_UI_REQUEST_ARG) {
        length = AOTX_UI_REQUEST_ARG;
    }
    if (col + length > panel->cols) {
        length = (col < panel->cols) ? ((unsigned int)panel->cols - col) : 0u;
    }
    aotx_ui_text(panel, row, col, slot->arg, length, AOTX_UI_NORMAL);
}

/* The agents panel shows one row for each agent that is not free, and then the requests
 * that wait for the operator. Each thread takes a share of the agents and finds the row of
 * its agent from the agents below it. A table with more agents than the panel holds keeps
 * its last agent row for the count that is left. */
__global__ void aotx_ui_agents(void)
{
    const aotx_ui_panel *panel = &aotx_ui_panel_table[AOTX_UI_AGENTS];
    aotx_ui_blank(panel);
    __syncthreads();

    const unsigned int rows = AOTX_UI_AGENT_ROWS;
    const unsigned int first = 2u;
    const unsigned int names = first + rows;
    unsigned int live = 0u;
    for (unsigned int i = 0u; i < AOTX_SLOTS; ++i) {
        if (aotx_agents.agent[i].state != AOTX_AGENT_STATE_FREE) {
            live += 1u;
        }
    }
    unsigned int shown = (live > rows) ? (rows - 1u) : rows;

    for (unsigned int id = threadIdx.x; id < AOTX_SLOTS; id += blockDim.x) {
        if (aotx_agents.agent[id].state == AOTX_AGENT_STATE_FREE) {
            continue;
        }
        unsigned int rank = 0u;
        for (unsigned int i = 0u; i < id; ++i) {
            if (aotx_agents.agent[i].state != AOTX_AGENT_STATE_FREE) {
                rank += 1u;
            }
        }
        if (rank < shown) {
            aotx_ui_agent_row(panel, rank + first, id);
        }
    }

    /* The requests that wait, the lowest number first. The keys y and n answer the first
     * one of them. A list longer than the rows keeps the last row for the count that is
     * left, so no thread writes a request row there. */
    unsigned int waiting = aotx_cli_pending_count();
    unsigned int request_rows = (waiting > AOTX_UI_REQUEST_ROWS)
                              ? (AOTX_UI_REQUEST_ROWS - 1u) : AOTX_UI_REQUEST_ROWS;
    for (unsigned int rank = threadIdx.x; rank < request_rows; rank += blockDim.x) {
        unsigned int at = aotx_cli_pending_at(rank);
        if (at < AOTX_SLOTS) {
            aotx_ui_request_row(panel, names + 1u + rank, at);
        }
    }

    if (threadIdx.x == 0u) {
        unsigned int last = (unsigned int)panel->rows - 1u;
        aotx_ui_focus_title(panel, "agents", AOTX_CLI_FOCUS_AGENTS);
        aotx_ui_say(panel, 1u, 1u, "id role state task tool request turn tokens rate",
                    AOTX_UI_DIM);
        if (live == 0u) {
            aotx_ui_say(panel, first, 1u, "no agents", AOTX_UI_NORMAL);
        } else if (live > rows) {
            unsigned int col = aotx_ui_say(panel, names - 1u, 1u, "and", AOTX_UI_DIM);
            col = aotx_ui_number(panel, names - 1u, col + 1u,
                                 (unsigned long long)(live - shown), AOTX_UI_NORMAL);
            aotx_ui_say(panel, names - 1u, col + 1u, "more", AOTX_UI_DIM);
        }
        if (waiting == 0u) {
            aotx_ui_say(panel, names, 1u, "requests none", AOTX_UI_DIM);
        } else {
            unsigned int col = aotx_ui_say(panel, names, 1u, "requests", AOTX_UI_HIGH);
            aotx_ui_say(panel, names, col + 1u, "id agent tool argument", AOTX_UI_DIM);
        }
        if (waiting > AOTX_UI_REQUEST_ROWS) {
            unsigned int col = aotx_ui_say(panel, last, 1u, "and", AOTX_UI_DIM);
            col = aotx_ui_number(panel, last, col + 1u,
                                 (unsigned long long)(waiting
                                                      - (AOTX_UI_REQUEST_ROWS - 1u)),
                                 AOTX_UI_NORMAL);
            aotx_ui_say(panel, last, col + 1u, "more", AOTX_UI_DIM);
        }
    }
}

/* Put one message of the bus buffer on one row: the writer, the kind, the source and the
 * text. The message number is read again after the copy. A message that a new message took
 * the place of therefore shows nothing, and not a mix of two messages. */
static __device__ __forceinline__ void aotx_ui_message(const aotx_ui_panel *panel,
                                                       unsigned int row,
                                                       unsigned long long at)
{
    const volatile aotx_bus_line *line = aotx_bus_line_at(at);
    if (line == 0) {
        return;
    }
    unsigned int col = aotx_ui_number(panel, row, 1u, (unsigned long long)line->writer,
                                      AOTX_UI_DIM);
    col = aotx_ui_say(panel, row, col + 1u, aotx_cli_kind_name(line->kind), AOTX_UI_HIGH);
    col = aotx_ui_say(panel, row, col + 1u, aotx_cli_source_name(line->provenance),
                      AOTX_UI_DIM);
    col += 1u;
    unsigned int length = line->text_len;
    if (length > AOTX_BUS_TEXT_BYTES) {
        length = AOTX_BUS_TEXT_BYTES;
    }
    if (col + length > panel->cols) {
        length = (col < panel->cols) ? ((unsigned int)panel->cols - col) : 0u;
    }
    for (unsigned int i = 0u; i < length; ++i) {
        aotx_ui_put(panel, row, col + i, aotx_ui_glyph((unsigned char)line->text[i]),
                    AOTX_UI_NORMAL);
    }
    if (line->at != at) {
        aotx_ui_blank_row(panel, row);
    }
}

/* The bus panel shows the last messages of every kind, the newest first. Each thread takes
 * one row and reads the message of that row. The panel reads the buffer and not the record
 * ring. At a high record rate the ring holds a bus record for a fraction of a second. */
__global__ void aotx_ui_bus(void)
{
    const aotx_ui_panel *panel = &aotx_ui_panel_table[AOTX_UI_BUS];
    unsigned int rows = (unsigned int)panel->rows - 2u;
    if (rows > AOTX_UI_BUS_MAX) {
        rows = AOTX_UI_BUS_MAX;
    }
    aotx_ui_blank(panel);
    __syncthreads();

    unsigned long long count = aotx_bus_lines.count;
    for (unsigned int i = threadIdx.x; i < rows; i += blockDim.x) {
        if ((unsigned long long)i >= count) {
            continue;
        }
        aotx_ui_message(panel, i + 2u, count - (unsigned long long)i);
    }
    if (threadIdx.x == 0u) {
        aotx_ui_title(panel, "bus");
        if (count == 0ull) {
            aotx_ui_say(panel, 1u, 1u, "no messages", AOTX_UI_DIM);
        } else {
            aotx_ui_say(panel, 1u, 1u, "writer kind source text", AOTX_UI_DIM);
        }
    }
}
