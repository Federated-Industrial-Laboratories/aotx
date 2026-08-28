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
        aotx_ui_title(panel, "console");
        aotx_ui_command(panel);
    }
}

/* Put one sequence on one row. The row holds the slot, the role and the state. It then
 * holds the tokens the key value cache holds, the reply tokens, and the reply tokens each
 * second. The rate comes from two samples of the window, which carry the device clock. A
 * sequence is the decode half of an agent, and a later version wraps it in an agent
 * record. */
static __device__ __forceinline__ void aotx_ui_sequence(const aotx_ui_panel *panel,
                                                        unsigned int row, unsigned int slot,
                                                        const aotx_seq *seq)
{
    unsigned int col = aotx_ui_number(panel, row, 1u, (unsigned long long)slot,
                                      AOTX_UI_NORMAL);
    col = aotx_ui_say(panel, row, col + 1u, aotx_say_role_name(seq->role), AOTX_UI_DIM);
    col = aotx_ui_say(panel, row, col + 1u, aotx_say_state_name(seq->state), AOTX_UI_HIGH);
    col = aotx_ui_number(panel, row, col + 1u, (unsigned long long)seq->held,
                         AOTX_UI_NORMAL);
    col = aotx_ui_number(panel, row, col + 1u, (unsigned long long)seq->sampled,
                         AOTX_UI_NORMAL);
    aotx_ui_number(panel, row, col + 1u, aotx_say_rate(slot), AOTX_UI_NORMAL);
}

/* The agents panel shows one row for each sequence slot that is not free. Each thread takes
 * a share of the slots and finds the row of its slot from the slots below it. A table with
 * more sequences than the panel holds keeps its last row for the count that is left. */
__global__ void aotx_ui_agents(void)
{
    const aotx_ui_panel *panel = &aotx_ui_panel_table[AOTX_UI_AGENTS];
    aotx_ui_blank(panel);
    __syncthreads();

    const unsigned int rows = (unsigned int)panel->rows - 2u;
    unsigned int live = 0u;
    for (unsigned int i = 0u; i < AOTX_SEQ_SLOTS; ++i) {
        if (aotx_seqs.slot[i].state != AOTX_SEQ_STATE_FREE) {
            live += 1u;
        }
    }
    unsigned int shown = (live > rows) ? (rows - 1u) : rows;

    for (unsigned int slot = threadIdx.x; slot < AOTX_SEQ_SLOTS; slot += blockDim.x) {
        const aotx_seq *seq = &aotx_seqs.slot[slot];
        if (seq->state == AOTX_SEQ_STATE_FREE) {
            continue;
        }
        unsigned int rank = 0u;
        for (unsigned int i = 0u; i < slot; ++i) {
            if (aotx_seqs.slot[i].state != AOTX_SEQ_STATE_FREE) {
                rank += 1u;
            }
        }
        if (rank < shown) {
            aotx_ui_sequence(panel, rank + 2u, slot, seq);
        }
    }

    if (threadIdx.x == 0u) {
        aotx_ui_title(panel, "agents");
        aotx_ui_say(panel, 1u, 1u, "slot role state position reply rate", AOTX_UI_DIM);
        if (live == 0u) {
            aotx_ui_say(panel, 2u, 1u, "no agents", AOTX_UI_NORMAL);
        } else if (live > rows) {
            unsigned int col = aotx_ui_say(panel, rows + 1u, 1u, "and", AOTX_UI_DIM);
            col = aotx_ui_number(panel, rows + 1u, col + 1u,
                                 (unsigned long long)(live - shown), AOTX_UI_NORMAL);
            aotx_ui_say(panel, rows + 1u, col + 1u, "more", AOTX_UI_DIM);
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
