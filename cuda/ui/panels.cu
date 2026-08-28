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

/* The agents panel shows one row for each agent. No agent is resident in this version, so
 * the panel shows the header and says that the table is empty. */
__global__ void aotx_ui_agents(void)
{
    const aotx_ui_panel *panel = &aotx_ui_panel_table[AOTX_UI_AGENTS];
    aotx_ui_blank(panel);
    __syncthreads();
    if (threadIdx.x == 0u) {
        aotx_ui_title(panel, "agents");
        aotx_ui_say(panel, 1u, 1u, "id role state task tokens", AOTX_UI_DIM);
        aotx_ui_say(panel, 2u, 1u, "no agents", AOTX_UI_NORMAL);
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
