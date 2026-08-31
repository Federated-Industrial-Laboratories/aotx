/* Purpose: Compare every grid row with the expected wrap at one and sixty-four writers. */
#include <stdio.h>
#include <string.h>

#include "boot/check.h"
#include "ui/ui.cuh"

typedef struct editor_result {
    unsigned int length_before_send;
    unsigned int lines_before_send;
    unsigned int break_at;
    unsigned int length_after_send;
    unsigned int lines_after_send;
    unsigned char sent[7];
} editor_result;

__global__ void aotx_wrap_fill(unsigned int count)
{
    unsigned int id = threadIdx.x;
    if (blockIdx.x != 0u || id >= count) {
        return;
    }
    unsigned char text[AOTX_CONSOLE_COLS];
    for (unsigned int i = 0u; i < AOTX_CONSOLE_COLS; ++i) {
        text[i] = (unsigned char)('a' + (id + i) % 26u);
    }
    text[0] = 'r'; text[1] = 'e'; text[2] = 'p'; text[3] = 'l'; text[4] = 'y';
    text[5] = (unsigned char)('0' + id / 10u);
    text[6] = (unsigned char)('0' + id % 10u);
    text[7] = ' ';
    aotx_console_put(text, AOTX_CONSOLE_COLS);
}

/* Fill the editor with marked logical rows. The marks make a wrong scroll head visible. */
__global__ void editor_fill(unsigned int length)
{
    if (blockIdx.x != 0u || threadIdx.x != 0u) {
        return;
    }
    aotx_cli.length = length;
    aotx_cli.cursor = length;
    for (unsigned int i = 0u; i < length; ++i) {
        aotx_cli.line[i] = (unsigned char)('A' + (i / 96u));
    }
}

/* Take the break and send keys through the same editor entry point as the run. */
__global__ void editor_keys(editor_result *result)
{
    if (blockIdx.x != 0u || threadIdx.x != 0u) {
        return;
    }
    const char first[] = "abc";
    const char second[] = "def";
    aotx_cli.length = 0u;
    aotx_cli.cursor = 0u;
    aotx_cli.lines = 0u;
    aotx_cli.history_count = 0u;
    aotx_cli.history_first = 0u;
    aotx_cli.history_at = 0u;
    aotx_cli_focus = AOTX_CLI_FOCUS_CONSOLE;
    aotx_key_body key = {};
    key.action = AOTX_CLI_PRESS;
    for (unsigned int i = 0u; i < 3u; ++i) {
        key.codepoint = (unsigned int)first[i];
        key.key = 0u;
        key.mods = 0u;
        aotx_cli_key(&key, 1ull);
    }
    key.codepoint = 0u;
    key.key = AOTX_CLI_KEY_ENTER;
    key.mods = 0x0004u;
    aotx_cli_key(&key, 1ull);
    for (unsigned int i = 0u; i < 3u; ++i) {
        key.codepoint = (unsigned int)second[i];
        key.key = 0u;
        key.mods = 0u;
        aotx_cli_key(&key, 1ull);
    }
    result->length_before_send = aotx_cli.length;
    result->lines_before_send = aotx_cli.lines;
    result->break_at = aotx_cli.line[3];
    key.codepoint = 0u;
    key.key = AOTX_CLI_KEY_ENTER;
    key.mods = 0u;
    aotx_cli_key(&key, 2ull);
    result->length_after_send = aotx_cli.length;
    result->lines_after_send = aotx_cli.lines;
    for (unsigned int i = 0u; i < 7u; ++i) {
        result->sent[i] = aotx_cli.taken[i];
    }
}

static unsigned char glyph(unsigned char byte)
{
    return (byte < AOTX_UI_GLYPH_FIRST || byte > AOTX_UI_GLYPH_LAST)
         ? (unsigned char)AOTX_UI_GLYPH_BOX
         : (unsigned char)(byte - AOTX_UI_GLYPH_FIRST);
}

static void put(aotx_ui_cell *grid, unsigned int row, unsigned int col,
                unsigned char byte, unsigned int attr)
{
    grid[row * AOTX_UI_COLS + col].glyph = glyph(byte);
    grid[row * AOTX_UI_COLS + col].attr = (unsigned char)attr;
}

static int wrap_case(unsigned int count)
{
    aotx_console_state console = {};
    aotx_cli_state editor = {};
    aotx_ui_cell got[AOTX_UI_CELLS];
    aotx_ui_cell want[AOTX_UI_CELLS];
    for (unsigned int i = 0u; i < AOTX_UI_CELLS; ++i) {
        want[i].glyph = glyph('X');
        want[i].attr = AOTX_UI_HIGH;
    }
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_console, &console, sizeof console),
                       "cudaMemcpyToSymbol");
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_cli, &editor, sizeof editor),
                       "cudaMemcpyToSymbol");
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_ui_grid, want, sizeof want),
                       "cudaMemcpyToSymbol");
    aotx_wrap_fill<<<1, count>>>(count);
    aotx_ui_console<<<1, AOTX_UI_PANEL_THREADS>>>();
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_check_runtime(cudaMemcpyFromSymbol(&console, aotx_console, sizeof console),
                       "cudaMemcpyFromSymbol");
    aotx_check_runtime(cudaMemcpyFromSymbol(got, aotx_ui_grid, sizeof got),
                       "cudaMemcpyFromSymbol");

    for (unsigned int row = 0u; row < 34u; ++row) {
        for (unsigned int col = 0u; col < 100u; ++col) {
            want[row * AOTX_UI_COLS + col].glyph = AOTX_UI_GLYPH_SPACE;
            want[row * AOTX_UI_COLS + col].attr = AOTX_UI_DIM;
        }
    }
    const char title[] = "console";
    for (unsigned int i = 0u; i < sizeof title - 1u; ++i) {
        put(want, 0u, 1u + i, (unsigned char)title[i], AOTX_UI_HIGH);
    }
    put(want, 33u, 1u, '>', AOTX_UI_DIM);
    put(want, 33u, 2u, ' ', AOTX_UI_DIM);
    put(want, 33u, 3u, '_', AOTX_UI_HIGH);
    unsigned int row = 33u;
    for (unsigned long long back = 0ull; back < console.count && row > 1u; ++back) {
        const aotx_console_line *line =
            &console.line[(console.count - back - 1ull) & (AOTX_CONSOLE_LINES - 1u)];
        unsigned int wraps = (line->length + 97u) / 98u;
        for (unsigned int part = wraps; part > 0u && row > 1u; --part) {
            row -= 1u;
            unsigned int start = (part - 1u) * 98u;
            unsigned int bytes = line->length - start;
            if (bytes > 98u) bytes = 98u;
            for (unsigned int i = 0u; i < bytes; ++i) {
                put(want, row, 1u + i, line->text[start + i], AOTX_UI_NORMAL);
            }
        }
    }
    unsigned int bad = 0u;
    for (unsigned int i = 0u; i < AOTX_UI_CELLS; ++i) {
        bad += (got[i].glyph != want[i].glyph || got[i].attr != want[i].attr) ? 1u : 0u;
    }
    printf("wrap: N=%u compared 50 rows, %u cells differ\n", count, bad);
    return bad == 0u ? 0 : 1;
}

static int cell_is(const aotx_ui_cell *grid, unsigned int row, unsigned int col,
                   unsigned char byte, unsigned int attr)
{
    const aotx_ui_cell *cell = &grid[row * AOTX_UI_COLS + col];
    return cell->glyph == glyph(byte) && cell->attr == attr;
}

/* Check two rows, four rows and the scroll which follows the cursor past four rows. */
static int editor_grid(void)
{
    aotx_console_state console = {};
    aotx_cli_state editor = {};
    aotx_ui_cell grid[AOTX_UI_CELLS];
    aotx_ui_cell guard[AOTX_UI_CELLS];
    int failed = 0;
    for (unsigned int i = 0u; i < AOTX_UI_CELLS; ++i) {
        guard[i].glyph = glyph('X');
        guard[i].attr = AOTX_UI_HIGH;
    }
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_console, &console, sizeof console),
                       "cudaMemcpyToSymbol");
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_cli, &editor, sizeof editor),
                       "cudaMemcpyToSymbol");
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_ui_grid, guard, sizeof guard),
                       "cudaMemcpyToSymbol");
    editor_fill<<<1, 1>>>(142u);
    aotx_ui_console<<<1, AOTX_UI_PANEL_THREADS>>>();
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_check_runtime(cudaMemcpyFromSymbol(grid, aotx_ui_grid, sizeof grid),
                       "cudaMemcpyFromSymbol");
    failed += !cell_is(grid, 32u, 3u, 'A', AOTX_UI_NORMAL);
    failed += !cell_is(grid, 33u, 3u, 'B', AOTX_UI_NORMAL);
    failed += !cell_is(grid, 33u, 49u, '_', AOTX_UI_HIGH);
    const char count[] = "142/4000";
    for (unsigned int i = 0u; i < sizeof count - 1u; ++i) {
        failed += !cell_is(grid, 33u, 91u + i, (unsigned char)count[i], AOTX_UI_DIM);
    }
    failed += memcmp(&grid[34u * AOTX_UI_COLS], &guard[34u * AOTX_UI_COLS],
                     AOTX_UI_COLS * sizeof(aotx_ui_cell)) != 0;

    aotx_check_runtime(cudaMemcpyToSymbol(aotx_ui_grid, guard, sizeof guard),
                       "cudaMemcpyToSymbol");
    editor_fill<<<1, 1>>>(401u);
    aotx_ui_console<<<1, AOTX_UI_PANEL_THREADS>>>();
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_check_runtime(cudaMemcpyFromSymbol(grid, aotx_ui_grid, sizeof grid),
                       "cudaMemcpyFromSymbol");
    failed += !cell_is(grid, 30u, 3u, 'B', AOTX_UI_NORMAL);
    failed += !cell_is(grid, 31u, 3u, 'C', AOTX_UI_NORMAL);
    failed += !cell_is(grid, 32u, 3u, 'D', AOTX_UI_NORMAL);
    failed += !cell_is(grid, 33u, 3u, 'E', AOTX_UI_NORMAL);
    failed += !cell_is(grid, 33u, 20u, '_', AOTX_UI_HIGH);
    failed += memcmp(&grid[34u * AOTX_UI_COLS], &guard[34u * AOTX_UI_COLS],
                     AOTX_UI_COLS * sizeof(aotx_ui_cell)) != 0;
    printf("editor: two and four rows, 142/4000, and cursor scroll give %d differences\n",
           failed);
    return failed == 0 ? 0 : 1;
}

static int editor_send(void)
{
    const unsigned char want[7] = { 'a', 'b', 'c', '\n', 'd', 'e', 'f' };
    unsigned char *ring = NULL;
    editor_result *device = NULL;
    editor_result result = {};
    const unsigned long long slots = 64ull;
    aotx_check_runtime(cudaMalloc(&ring, (size_t)slots * AOTX_SLOT_BYTES), "cudaMalloc");
    aotx_check_runtime(cudaMemset(ring, 0, (size_t)slots * AOTX_SLOT_BYTES), "cudaMemset");
    aotx_seam_state seam = {};
    seam.dev.base = ring;
    seam.dev.slot_count = slots;
    seam.dev.mask = slots - 1ull;
    seam.apply.state_hash = AOTX_FNV_BASIS;
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_seam, &seam, sizeof seam),
                       "cudaMemcpyToSymbol");
    aotx_check_runtime(cudaMalloc(&device, sizeof *device), "cudaMalloc");
    editor_keys<<<1, 1>>>(device);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_check_runtime(cudaMemcpy(&result, device, sizeof result, cudaMemcpyDeviceToHost),
                       "cudaMemcpy");
    cudaFree(device);
    cudaFree(ring);
    int failed = result.length_before_send != 7u || result.lines_before_send != 0u
              || result.break_at != (unsigned int)'\n' || result.length_after_send != 0u
              || result.lines_after_send != 1u || memcmp(result.sent, want, sizeof want) != 0;
    printf("editor: Alt-Enter keeps seven bytes and Enter sends once: %s\n",
           failed ? "FAILED" : "passed");
    return failed ? 1 : 0;
}

int main(void)
{
    int failed = 0;
    aotx_check_runtime(cudaSetDevice(0), "cudaSetDevice");
    failed += wrap_case(1u);
    failed += wrap_case(64u);
    failed += editor_grid();
    failed += editor_send();
    return failed == 0 ? 0 : 1;
}
