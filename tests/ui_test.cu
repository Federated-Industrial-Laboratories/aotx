/* Purpose: Check the panels at one record and at 64 records, and check the raster.
 * Owns: The record fixtures, the grid copy and the counts of the cases.
 * Launch shape: A grid writes the records; one block fills a panel; a grid rasterizes.
 * Lifetime: One run of the test program. */
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda.h>

#include "boot/check.h"
#include "bus/bus.cuh"
#include "mem/mem.cuh"
#include "seam/seam.cuh"
#include "ui/ui.cuh"

#define AOTX_TEST_FOUND   1024u
#define AOTX_TEST_TEXT    64u
#define AOTX_TEST_SLOTS   64u

typedef struct aotx_test_line {
    unsigned long long seq;
    unsigned int length;
    unsigned char body[AOTX_BODY_BYTES];
} aotx_test_line;

static unsigned int aotx_test_applied;
static unsigned int aotx_test_failed;
static aotx_ui_cell aotx_test_grid[AOTX_UI_CELLS];
static aotx_ui_panel aotx_test_panels[AOTX_UI_PANELS];
static unsigned char aotx_test_font[AOTX_UI_GLYPHS][AOTX_UI_GLYPH_ROWS];

static void aotx_test_check(int ok, const char *what)
{
    aotx_test_applied += 1u;
    if (!ok) {
        aotx_test_failed += 1u;
        printf("ui: FAILED %s\n", what);
    }
}

/* Write one console record for each thread. Every body carries its own index and tag, so
 * no two records hold the same text and a wrong index cannot hide. */
__global__ void aotx_test_console(unsigned int count, unsigned int tag)
{
    unsigned int lane = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = gridDim.x * blockDim.x;
    for (unsigned int i = lane; i < count; i += stride) {
        char text[AOTX_TEST_TEXT];
        unsigned int at = 0u;
        const char *head = "line ";
        for (unsigned int b = 0u; head[b] != '\0'; ++b) {
            text[at++] = head[b];
        }
        at += aotx_text_utoa(tag, text + at, AOTX_TEST_TEXT - at);
        text[at++] = ' ';
        at += aotx_text_utoa(i, text + at, AOTX_TEST_TEXT - at);
        /* The product path: one function writes the record and fills the console buffer. */
        aotx_console_write(text, at);
    }
}

/* Append one bus message for each thread, with a distinct text and a distinct kind. */
__global__ void aotx_test_bus(unsigned int count, unsigned int tag)
{
    unsigned int lane = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = gridDim.x * blockDim.x;
    for (unsigned int i = lane; i < count; i += stride) {
        char text[AOTX_TEST_TEXT];
        unsigned int at = 0u;
        const char *head = "message ";
        for (unsigned int b = 0u; head[b] != '\0'; ++b) {
            text[at++] = head[b];
        }
        at += aotx_text_utoa(tag, text + at, AOTX_TEST_TEXT - at);
        text[at++] = ' ';
        at += aotx_text_utoa(i, text + at, AOTX_TEST_TEXT - at);
        unsigned int kind = (i % 2u == 0u) ? AOTX_BUS_NOTE : AOTX_BUS_FINDING;
        unsigned int source = (kind == AOTX_BUS_FINDING) ? AOTX_PROV_COMPUTED : 0u;
        aotx_bus_append(AOTX_WRITER_CONSOLE, kind, source, text, at, 0ull, 0ull, 0.0f,
                        aotx_time_tick);
    }
}

/* Write records of another type, so a console line has records after it in the ring. */
__global__ void aotx_test_notes(unsigned int count)
{
    unsigned int lane = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = gridDim.x * blockDim.x;
    for (unsigned int i = lane; i < count; i += stride) {
        unsigned long long body[8];
        for (unsigned int b = 0u; b < 8u; ++b) {
            body[b] = (unsigned long long)i * 8ull + (unsigned long long)b;
        }
        aotx_seam_write(AOTX_WRITER_SYSTEM, AOTX_CLASS_B, AOTX_REC_NOTE, 0u, body, 64u);
    }
}

/* Gather the records of a type from the device ring, one thread for one sequence. */
__global__ void aotx_test_gather(unsigned int type, aotx_test_line *out, unsigned int max,
                                 unsigned int *count)
{
    unsigned long long tail = aotx_seam.dev.tail;
    unsigned long long stride = (unsigned long long)(gridDim.x * blockDim.x);
    for (unsigned long long seq = 1ull + (unsigned long long)(blockIdx.x * blockDim.x
                                                              + threadIdx.x);
         seq <= tail; seq += stride) {
        const volatile aotx_record_header *header = aotx_cli_slot(seq);
        if (!aotx_cli_holds(header, seq, type)) {
            continue;
        }
        unsigned int at = atomicAdd(count, 1u);
        if (at >= max) {
            continue;
        }
        unsigned int length = header->body_len;
        if (length > AOTX_BODY_BYTES) {
            length = AOTX_BODY_BYTES;
        }
        out[at].seq = seq;
        out[at].length = length;
        const volatile unsigned char *body = (const volatile unsigned char *)header
                                           + AOTX_HEADER_BYTES;
        for (unsigned int b = 0u; b < length; ++b) {
            out[at].body[b] = body[b];
        }
    }
}

static int aotx_test_order(const void *left, const void *right)
{
    const aotx_test_line *a = (const aotx_test_line *)left;
    const aotx_test_line *b = (const aotx_test_line *)right;
    return (a->seq < b->seq) ? -1 : ((a->seq > b->seq) ? 1 : 0);
}

static unsigned int aotx_test_records(unsigned int type, aotx_test_line *out, unsigned int max)
{
    aotx_test_line *device = NULL;
    unsigned int *counter = NULL;
    unsigned int count = 0u;
    aotx_check_runtime(cudaMalloc(&device, (size_t)max * sizeof *device), "cudaMalloc");
    aotx_check_runtime(cudaMalloc(&counter, sizeof *counter), "cudaMalloc");
    aotx_check_runtime(cudaMemset(counter, 0, sizeof *counter), "cudaMemset");
    aotx_test_gather<<<64, 128>>>(type, device, max, counter);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_check_runtime(cudaMemcpy(&count, counter, sizeof count, cudaMemcpyDeviceToHost),
                       "cudaMemcpy");
    if (count > max) {
        count = max;
    }
    aotx_check_runtime(cudaMemcpy(out, device, (size_t)count * sizeof *out,
                                  cudaMemcpyDeviceToHost), "cudaMemcpy");
    cudaFree(device);
    cudaFree(counter);
    qsort(out, count, sizeof *out, aotx_test_order);
    return count;
}

/* The regions the table holds. The arena panel puts its totals under them. */
static unsigned int aotx_mem_test_regions(void)
{
    aotx_mem_table table;
    aotx_check_runtime(cudaMemcpyFromSymbol(&table, aotx_mem_region_table, sizeof table),
                       "cudaMemcpyFromSymbol");
    return table.count;
}

static aotx_console_state aotx_test_lines;

static void aotx_test_read_console(void)
{
    aotx_check_runtime(cudaMemcpyFromSymbol(&aotx_test_lines, aotx_console,
                                            sizeof aotx_test_lines), "cudaMemcpyFromSymbol");
}

/* Give the line of a line number from the copy of the console buffer. */
static const aotx_console_line *aotx_test_line_at(unsigned long long at)
{
    const aotx_console_line *line =
        &aotx_test_lines.line[(at - 1ull) & (AOTX_CONSOLE_LINES - 1u)];
    return (line->seq == at) ? line : NULL;
}

/* Report whether a console record of the ring holds the bytes of a console buffer line. */
static int aotx_test_line_in(const aotx_test_line *found, unsigned int have,
                             const aotx_console_line *line)
{
    for (unsigned int i = 0u; i < have; ++i) {
        if (found[i].length != line->length) {
            continue;
        }
        if (memcmp(found[i].body, line->text, line->length) == 0) {
            return 1;
        }
    }
    return 0;
}

static aotx_bus_buffer aotx_test_messages;

static void aotx_test_read_bus(void)
{
    aotx_check_runtime(cudaMemcpyFromSymbol(&aotx_test_messages, aotx_bus_lines,
                                            sizeof aotx_test_messages),
                       "cudaMemcpyFromSymbol");
}

/* Give the message of a message number from the copy of the bus buffer. */
static const aotx_bus_line *aotx_test_message_at(unsigned long long at)
{
    const aotx_bus_line *line = &aotx_test_messages.line[(at - 1ull) & (AOTX_BUS_LINES - 1u)];
    return (line->at == at) ? line : NULL;
}

/* Report whether a bus record of the ring holds the message of a bus buffer line. The
 * record sequence of the line names the record, so a text that two lines share cannot
 * make this true by chance. */
static int aotx_test_message_in(const aotx_test_line *found, unsigned int have,
                                const aotx_bus_line *line)
{
    for (unsigned int i = 0u; i < have; ++i) {
        const aotx_bus_body *body = (const aotx_bus_body *)found[i].body;
        if (found[i].seq != line->seq) {
            continue;
        }
        if (body->text_len == line->text_len
            && memcmp(body->text, line->text, line->text_len) == 0) {
            return 1;
        }
    }
    return 0;
}

/* Write the row that the bus panel makes for one message of the buffer. */
static void aotx_test_message_row(const aotx_bus_line *line, char *out, size_t max)
{
    const char *kind = (line->kind == AOTX_BUS_FINDING) ? "finding" : "note";
    const char *source = (line->kind == AOTX_BUS_FINDING) ? "computed" : "-";
    unsigned int length = line->text_len;
    if (length > AOTX_BUS_TEXT_BYTES) {
        length = AOTX_BUS_TEXT_BYTES;
    }
    snprintf(out, max, "%u %s %s %.*s", line->writer, kind, source, (int)length, line->text);
}

static void aotx_test_read_grid(void)
{
    aotx_check_runtime(cudaMemcpyFromSymbol(aotx_test_grid, aotx_ui_grid,
                                            sizeof aotx_test_grid), "cudaMemcpyFromSymbol");
}

/* Give the glyph of a code point, as the panels give it. */
static unsigned char aotx_test_glyph(unsigned int code)
{
    if (code < AOTX_UI_GLYPH_FIRST || code > AOTX_UI_GLYPH_LAST) {
        return (unsigned char)AOTX_UI_GLYPH_BOX;
    }
    return (unsigned char)(code - AOTX_UI_GLYPH_FIRST);
}

/* Report whether a row of a panel starts with the text, at the column given. */
static int aotx_test_row_is(unsigned int panel, unsigned int row, unsigned int col,
                            const unsigned char *text, unsigned int length)
{
    const aotx_ui_panel *at = &aotx_test_panels[panel];
    if (row >= at->rows || col + length > at->cols) {
        return 0;
    }
    for (unsigned int i = 0u; i < length; ++i) {
        unsigned int cell = ((unsigned int)at->row + row) * AOTX_UI_COLS
                          + (unsigned int)at->col + col + i;
        if (aotx_test_grid[cell].glyph != aotx_test_glyph(text[i])) {
            return 0;
        }
    }
    return 1;
}

static int aotx_test_row_says(unsigned int panel, unsigned int row, unsigned int col,
                              const char *text)
{
    return aotx_test_row_is(panel, row, col, (const unsigned char *)text,
                            (unsigned int)strlen(text));
}

/* Report whether every cell of a row of a panel is the space. */
static int aotx_test_row_blank(unsigned int panel, unsigned int row)
{
    const aotx_ui_panel *at = &aotx_test_panels[panel];
    for (unsigned int col = 0u; col < at->cols; ++col) {
        unsigned int cell = ((unsigned int)at->row + row) * AOTX_UI_COLS
                          + (unsigned int)at->col + col;
        if (aotx_test_grid[cell].glyph != (unsigned char)AOTX_UI_GLYPH_SPACE) {
            return 0;
        }
    }
    return 1;
}

/* The console panel puts the newest line on the row above the command line, and the lines
 * before it on the rows above that. The panel reads the console buffer, so the check reads
 * the same buffer. The check reads the records of the ring as well, to show that the two
 * agree. */
static void aotx_test_console_panel(unsigned int count, unsigned int tag)
{
    aotx_test_line *found = (aotx_test_line *)malloc(AOTX_TEST_FOUND * sizeof *found);
    unsigned int have = 0u;
    unsigned int lines = (unsigned int)aotx_test_panels[AOTX_UI_CONSOLE].rows - 2u;
    unsigned int matched = 0u;
    unsigned int agreed = 0u;
    unsigned int shown = 0u;
    unsigned long long held = 0ull;

    aotx_test_console<<<8, 32>>>(count, tag);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_ui_console<<<1, AOTX_UI_PANEL_THREADS>>>();
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_test_read_grid();
    aotx_test_read_console();

    have = aotx_test_records(AOTX_REC_CONSOLE, found, AOTX_TEST_FOUND);
    held = aotx_test_lines.count;
    shown = (held < (unsigned long long)lines) ? (unsigned int)held : lines;
    for (unsigned int i = 0u; i < shown; ++i) {
        /* The line at held-i is the ith from the newest. It goes i rows above the row that
         * carries the command line. */
        const aotx_console_line *line = aotx_test_line_at(held - (unsigned long long)i);
        if (line == NULL) {
            continue;
        }
        if (aotx_test_row_is(AOTX_UI_CONSOLE, lines - i, 1u, line->text, line->length)) {
            matched += 1u;
        }
        if (aotx_test_line_in(found, have, line)) {
            agreed += 1u;
        }
    }
    aotx_test_check(have >= count, "the console records reached the ring");
    aotx_test_check(held >= (unsigned long long)count, "the console buffer took the lines");
    aotx_test_check(shown > 0u && matched == shown, "every console row shows its line");
    aotx_test_check(shown > 0u && agreed == shown,
                    "every line of the buffer is a record of the ring as well");
    aotx_test_check(aotx_test_row_says(AOTX_UI_CONSOLE, 0u, 1u, "console"),
                    "the console panel carries its title");
    aotx_test_check(aotx_test_row_says(AOTX_UI_CONSOLE,
                                       (unsigned int)aotx_test_panels[AOTX_UI_CONSOLE].rows
                                       - 1u, 1u, "> "),
                    "the command line sits at the bottom of the console panel");
    printf("ui: console at %u records, %u rows shown, %u matched, %u agree with the ring\n",
           count, shown, matched, agreed);
    free(found);
}

/* The bus panel puts the newest message first, under the header row. The panel reads the
 * bus buffer, so the check reads the same buffer. The check reads the records of the ring
 * as well, to show that the two agree. */
static void aotx_test_bus_panel(unsigned int count, unsigned int tag)
{
    aotx_test_line *found = (aotx_test_line *)malloc(AOTX_TEST_FOUND * sizeof *found);
    unsigned int have = 0u;
    unsigned int rows = (unsigned int)aotx_test_panels[AOTX_UI_BUS].rows - 2u;
    unsigned int matched = 0u;
    unsigned int agreed = 0u;
    unsigned int shown = 0u;
    unsigned long long held = 0ull;

    if (rows > AOTX_UI_BUS_MAX) {
        rows = AOTX_UI_BUS_MAX;
    }
    aotx_test_bus<<<8, 32>>>(count, tag);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_ui_bus<<<1, AOTX_UI_PANEL_THREADS>>>();
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_test_read_grid();
    aotx_test_read_bus();

    have = aotx_test_records(AOTX_REC_BUS, found, AOTX_TEST_FOUND);
    held = aotx_test_messages.count;
    shown = (held < (unsigned long long)rows) ? (unsigned int)held : rows;
    for (unsigned int i = 0u; i < shown; ++i) {
        /* The message at held-i is the ith from the newest. It goes on row i+2, because
         * the title takes row 0 and the header of the columns takes row 1. */
        const aotx_bus_line *line = aotx_test_message_at(held - (unsigned long long)i);
        char row[AOTX_TEST_TEXT * 2];
        if (line == NULL) {
            continue;
        }
        aotx_test_message_row(line, row, sizeof row);
        if (aotx_test_row_says(AOTX_UI_BUS, i + 2u, 1u, row)) {
            matched += 1u;
        } else if (i == 0u) {
            printf("ui: the first bus row does not read '%s'\n", row);
        }
        if (aotx_test_message_in(found, have, line)) {
            agreed += 1u;
        }
    }
    aotx_test_check(have >= count, "the bus messages reached the ring");
    aotx_test_check(held >= (unsigned long long)count, "the bus buffer took the messages");
    aotx_test_check(shown > 0u && matched == shown, "every bus row shows its message");
    aotx_test_check(shown > 0u && agreed == shown,
                    "every message of the buffer is a record of the ring as well");
    aotx_test_check(aotx_test_row_says(AOTX_UI_BUS, 0u, 1u, "bus"),
                    "the bus panel carries its title");
    printf("ui: bus at %u messages, %u rows shown, %u matched, %u agree with the ring\n",
           count, shown, matched, agreed);
    free(found);
}

/* Every panel writes its own rectangle and no other. */
static void aotx_test_all_panels(void)
{
    static const char *titles[AOTX_UI_PANELS] = { "console", "agents", "bus", "arena",
                                                  "tick", "seam" };
    unsigned int titled = 0u;
    aotx_ui_console<<<1, AOTX_UI_PANEL_THREADS>>>();
    aotx_ui_agents<<<1, AOTX_UI_PANEL_THREADS>>>();
    aotx_ui_bus<<<1, AOTX_UI_PANEL_THREADS>>>();
    aotx_ui_arena<<<1, AOTX_UI_PANEL_THREADS>>>();
    aotx_ui_tick<<<1, AOTX_UI_PANEL_THREADS>>>();
    aotx_ui_seam<<<1, AOTX_UI_PANEL_THREADS>>>();
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_test_read_grid();
    for (unsigned int i = 0u; i < AOTX_UI_PANELS; ++i) {
        if (aotx_test_row_says(i, 0u, 1u, titles[i])) {
            titled += 1u;
        } else {
            printf("ui: panel %u has no title\n", i);
        }
    }
    aotx_test_check(titled == AOTX_UI_PANELS, "every panel carries its title");
    aotx_test_check(aotx_test_row_says(AOTX_UI_AGENTS, 2u, 1u, "no agents"),
                    "the agents panel says that no agent is resident");
    aotx_test_check(aotx_test_row_says(AOTX_UI_ARENA, 2u, 1u, "ring"),
                    "the arena panel names the record ring region");
    aotx_test_check(aotx_test_row_says(AOTX_UI_ARENA,
                                       aotx_mem_test_regions() + 8u, 1u, "kv pages mapped"),
                    "the arena panel states the pages the key and value cache holds");
    aotx_test_check(aotx_test_row_says(AOTX_UI_TICK, 1u, 1u, "tick"),
                    "the tick panel names the tick");
    aotx_test_check(aotx_test_row_says(AOTX_UI_SEAM, 3u, 1u, "lag bytes"),
                    "the seam panel names the lag in bytes");
}

/* Fold the pixel buffer as the test folds it: FNV-1a 64 over the bytes. */
static unsigned long long aotx_test_fold(const unsigned char *bytes, size_t count)
{
    unsigned long long hash = AOTX_FNV_BASIS;
    for (size_t i = 0u; i < count; ++i) {
        hash ^= (unsigned long long)bytes[i];
        hash *= AOTX_FNV_PRIME;
    }
    return hash;
}

/* Compose the pixel buffer on the host from the same cells, the same font and the same
 * rule. The figure is therefore computed twice, once on each side. */
static unsigned long long aotx_test_expect(const aotx_ui_cell *cells,
                                           const unsigned char font[][AOTX_UI_GLYPH_ROWS])
{
    unsigned int *pixels = (unsigned int *)malloc(AOTX_UI_PIXEL_BYTES);
    unsigned long long hash = 0ull;
    for (unsigned int at = 0u; at < AOTX_UI_PIXELS; ++at) {
        unsigned int x = at % AOTX_UI_WIDTH;
        unsigned int y = at / AOTX_UI_WIDTH;
        unsigned int cell = (y / AOTX_UI_CELL_HEIGHT) * AOTX_UI_COLS
                          + (x / AOTX_UI_CELL_WIDTH);
        unsigned int glyph = cells[cell].glyph;
        unsigned int attr = cells[cell].attr;
        unsigned int color = AOTX_UI_COLOR_NORMAL;
        if (glyph >= AOTX_UI_GLYPHS) {
            glyph = AOTX_UI_GLYPH_BOX;
        }
        if (attr == AOTX_UI_HIGH) {
            color = AOTX_UI_COLOR_HIGH;
        } else if (attr == AOTX_UI_DIM) {
            color = AOTX_UI_COLOR_DIM;
        }
        unsigned int bits = font[glyph][y % AOTX_UI_CELL_HEIGHT];
        unsigned int lit = (bits >> (7u - (x % AOTX_UI_CELL_WIDTH))) & 1u;
        pixels[at] = lit ? color : AOTX_UI_BACK;
    }
    hash = aotx_test_fold((const unsigned char *)pixels, AOTX_UI_PIXEL_BYTES);
    free(pixels);
    return hash;
}

/* The raster of a known cell buffer gives one figure, and one flipped glyph bit gives
 * another. Both figures are computed twice: once on the device, once on the host. */
static void aotx_test_raster(void)
{
    aotx_ui_cell *cells = (aotx_ui_cell *)malloc(sizeof aotx_test_grid);
    unsigned int *pixels = (unsigned int *)malloc(AOTX_UI_PIXEL_BYTES);
    unsigned long long on_device = 0ull;
    unsigned long long on_host = 0ull;
    unsigned long long flipped_device = 0ull;
    unsigned long long flipped_host = 0ull;
    unsigned char byte = 0u;
    unsigned char changed = 0u;
    const unsigned int glyph = 41u;   /* the glyph of a capital letter */
    const unsigned int row = 5u;

    for (unsigned int i = 0u; i < AOTX_UI_CELLS; ++i) {
        cells[i].glyph = (unsigned char)(i % AOTX_UI_GLYPHS);
        cells[i].attr = (unsigned char)(i % AOTX_UI_ATTRS);
    }
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_ui_grid, cells, sizeof aotx_test_grid),
                       "cudaMemcpyToSymbol");
    aotx_ui_raster<<<AOTX_UI_RASTER_BLOCKS, AOTX_UI_RASTER_THREADS>>>();
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_check_runtime(cudaMemcpyFromSymbol(pixels, aotx_ui_pixel, AOTX_UI_PIXEL_BYTES),
                       "cudaMemcpyFromSymbol");
    on_device = aotx_test_fold((const unsigned char *)pixels, AOTX_UI_PIXEL_BYTES);
    on_host = aotx_test_expect(cells, aotx_test_font);

    byte = aotx_test_font[glyph][row];
    changed = (unsigned char)(byte ^ 0x01u);
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_ui_font, &changed, sizeof changed,
                                          (size_t)glyph * AOTX_UI_GLYPH_ROWS + row),
                       "cudaMemcpyToSymbol");
    aotx_ui_raster<<<AOTX_UI_RASTER_BLOCKS, AOTX_UI_RASTER_THREADS>>>();
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_check_runtime(cudaMemcpyFromSymbol(pixels, aotx_ui_pixel, AOTX_UI_PIXEL_BYTES),
                       "cudaMemcpyFromSymbol");
    flipped_device = aotx_test_fold((const unsigned char *)pixels, AOTX_UI_PIXEL_BYTES);
    aotx_test_font[glyph][row] = changed;
    flipped_host = aotx_test_expect(cells, aotx_test_font);
    aotx_test_font[glyph][row] = byte;
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_ui_font, &byte, sizeof byte,
                                          (size_t)glyph * AOTX_UI_GLYPH_ROWS + row),
                       "cudaMemcpyToSymbol");

    aotx_test_check(on_device == on_host, "the raster gives the figure the host computes");
    aotx_test_check(flipped_device == flipped_host,
                    "the raster with one flipped bit gives the figure the host computes");
    aotx_test_check(on_device != flipped_device, "one flipped glyph bit changes the figure");
    printf("ui: raster %016llx, one bit flipped %016llx, host %016llx and %016llx\n",
           on_device, flipped_device, on_host, flipped_host);
    free(cells);
    free(pixels);
}

/* The raster graph holds the six panel kernels and the raster kernel, on the stream of the
 * highest priority of this device. */
static void aotx_test_graph(void)
{
    aotx_ui_graph graph;
    size_t nodes = 0u;
    int low = 0;
    int high = 0;
    aotx_check_runtime(cudaDeviceGetStreamPriorityRange(&low, &high),
                       "cudaDeviceGetStreamPriorityRange");
    aotx_test_check(aotx_ui_graph_build(&graph) == 0, "the raster graph builds");
    aotx_check_runtime(cudaGraphGetNodes(graph.graph, 0, &nodes), "cudaGraphGetNodes");
    aotx_test_check(nodes == (size_t)(AOTX_UI_PANELS + 1u),
                    "the graph holds one node for each panel and one for the raster");
    aotx_test_check(graph.priority == high, "the raster stream takes the highest priority");
    aotx_ui_graph_run(&graph);
    aotx_test_read_grid();
    aotx_test_check(aotx_test_row_says(AOTX_UI_CONSOLE, 0u, 1u, "console"),
                    "one launch of the graph fills the panels");

    cudaEvent_t start;
    cudaEvent_t end;
    float spent = 0.0f;
    aotx_check_runtime(cudaEventCreate(&start), "cudaEventCreate");
    aotx_check_runtime(cudaEventCreate(&end), "cudaEventCreate");
    aotx_check_runtime(cudaEventRecord(start), "cudaEventRecord");
    for (unsigned int i = 0u; i < 10u; ++i) {
        aotx_ui_graph_run(&graph);
    }
    aotx_check_runtime(cudaEventRecord(end), "cudaEventRecord");
    aotx_check_runtime(cudaEventSynchronize(end), "cudaEventSynchronize");
    aotx_check_runtime(cudaEventElapsedTime(&spent, start, end), "cudaEventElapsedTime");
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    /* The sanitizer makes every kernel far slower than the period of the display. A run
     * with AOTX_SANITIZER set leaves this case out, as it leaves the frame case out. */
    if (getenv("AOTX_SANITIZER") == NULL) {
        aotx_test_check((double)spent / 10.0 < 16.7,
                        "the raster graph fits in the period of the display");
    }
    printf("ui: graph nodes %u, priority %d of the range %d to %d, %.3f ms for each frame\n",
           (unsigned int)nodes, graph.priority, low, high, (double)spent / 10.0);
    aotx_ui_graph_close(&graph);
}

/* Records of other types that the check writes after the console lines. The device ring
 * holds AOTX_DEVICE_RING_SLOTS records, so this count writes over every console record of
 * the ring more than once. */
#define AOTX_TEST_FLOOD  100000u

/* A console line stays on the panel while the console buffer holds it, whatever the record
 * ring does. The check writes lines from one writer and from 64 writers. It then writes
 * 100,000 records of other types and reads the panel. The row check can fail: a line number
 * that the buffer no longer holds gives an empty row. The ring held the same lines, and
 * holds none of them after the flood. */
static void aotx_test_scrollback(unsigned int writers)
{
    aotx_test_line *found = (aotx_test_line *)malloc(AOTX_TEST_FOUND * sizeof *found);
    unsigned int lines = (unsigned int)aotx_test_panels[AOTX_UI_CONSOLE].rows - 2u;
    unsigned int made = 64u;
    unsigned int tag = 40u + writers;
    unsigned int matched = 0u;
    unsigned int ordered = 0u;
    unsigned int distinct = 0u;
    unsigned int have = 0u;
    unsigned long long held = 0ull;
    unsigned long long first = 0ull;

    aotx_test_read_console();
    first = aotx_test_lines.count;
    aotx_test_console<<<1, writers>>>(made, tag);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_test_notes<<<64, 128>>>(AOTX_TEST_FLOOD);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_ui_console<<<1, AOTX_UI_PANEL_THREADS>>>();
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_test_read_grid();
    aotx_test_read_console();
    held = aotx_test_lines.count;

    for (unsigned int i = 0u; i < lines; ++i) {
        const aotx_console_line *line = aotx_test_line_at(held - (unsigned long long)i);
        char want[AOTX_TEST_TEXT];
        if (line == NULL) {
            continue;
        }
        if (aotx_test_row_is(AOTX_UI_CONSOLE, lines - i, 1u, line->text, line->length)) {
            matched += 1u;
        }
        /* One writer puts the lines in one order, so the text of each row is known. */
        snprintf(want, sizeof want, "line %u %u", tag, made - 1u - i);
        if (writers == 1u && aotx_test_row_says(AOTX_UI_CONSOLE, lines - i, 1u, want)) {
            ordered += 1u;
        }
        distinct += 1u;
        for (unsigned int b = 0u; b < i; ++b) {
            const aotx_console_line *other = aotx_test_line_at(held - (unsigned long long)b);
            if (other != NULL && other->length == line->length
                && memcmp(other->text, line->text, line->length) == 0) {
                distinct -= 1u;
                break;
            }
        }
    }
    have = aotx_test_records(AOTX_REC_CONSOLE, found, AOTX_TEST_FOUND);

    aotx_test_check(held - first == (unsigned long long)made,
                    "the console buffer took one line for each line written");
    aotx_test_check(matched == lines,
                    "every row of the console holds its line after 100000 other records");
    aotx_test_check(distinct == lines, "the rows of the console hold 32 different lines");
    if (writers == 1u) {
        aotx_test_check(ordered == lines, "one writer gives the rows the order of the lines");
    }
    /* The ring no longer holds the records of these lines. A panel that reads the ring
     * therefore shows nothing, which is the defect the buffer answers. */
    aotx_test_check(have == 0u, "the record ring no longer holds a console record");

    /* The row check can fail: a wrong line number in the buffer empties the row. */
    unsigned long long spoiled = 0ull;
    size_t at = offsetof(aotx_console_state, line)
              + (size_t)((held - 1ull) & (AOTX_CONSOLE_LINES - 1u)) * sizeof(aotx_console_line)
              + offsetof(aotx_console_line, seq);
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_console, &spoiled, sizeof spoiled, at),
                       "cudaMemcpyToSymbol");
    aotx_ui_console<<<1, AOTX_UI_PANEL_THREADS>>>();
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_test_read_grid();
    aotx_test_check(aotx_test_row_blank(AOTX_UI_CONSOLE, lines),
                    "a line number the buffer does not hold gives an empty row");
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_console, &held, sizeof held, at),
                       "cudaMemcpyToSymbol");

    printf("ui: scrollback at %u writers, %u lines, %u rows matched, %u in order, %u"
           " different, %u console records left in the ring\n", writers, made, matched,
           ordered, distinct, have);
    free(found);
}

/* A bus message stays on the panel while the bus buffer holds it, whatever the record ring
 * does. The check appends messages from one writer and from 64 writers. It then writes
 * 100,000 records of other types and reads the panel. The ring held the same messages, and
 * holds none of them after the flood. */
static void aotx_test_bus_scrollback(unsigned int writers)
{
    aotx_test_line *found = (aotx_test_line *)malloc(AOTX_TEST_FOUND * sizeof *found);
    unsigned int rows = (unsigned int)aotx_test_panels[AOTX_UI_BUS].rows - 2u;
    unsigned int made = (unsigned int)AOTX_BUS_LINES;
    unsigned int tag = 70u + writers;
    unsigned int matched = 0u;
    unsigned int ordered = 0u;
    unsigned int distinct = 0u;
    unsigned int have = 0u;
    unsigned long long held = 0ull;
    unsigned long long first = 0ull;

    if (rows > AOTX_UI_BUS_MAX) {
        rows = AOTX_UI_BUS_MAX;
    }
    aotx_test_read_bus();
    first = aotx_test_messages.count;
    aotx_test_bus<<<1, writers>>>(made, tag);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_test_notes<<<64, 128>>>(AOTX_TEST_FLOOD);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_ui_bus<<<1, AOTX_UI_PANEL_THREADS>>>();
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_test_read_grid();
    aotx_test_read_bus();
    held = aotx_test_messages.count;

    for (unsigned int i = 0u; i < rows; ++i) {
        const aotx_bus_line *line = aotx_test_message_at(held - (unsigned long long)i);
        char row[AOTX_TEST_TEXT * 2];
        char want[AOTX_TEST_TEXT];
        if (line == NULL) {
            continue;
        }
        aotx_test_message_row(line, row, sizeof row);
        if (aotx_test_row_says(AOTX_UI_BUS, i + 2u, 1u, row)) {
            matched += 1u;
        }
        /* One writer puts the messages in one order, so the text of each row is known. */
        snprintf(want, sizeof want, "message %u %u", tag, made - 1u - i);
        if (writers == 1u && line->text_len == (unsigned int)strlen(want)
            && memcmp(line->text, want, line->text_len) == 0) {
            ordered += 1u;
        }
        distinct += 1u;
        for (unsigned int b = 0u; b < i; ++b) {
            const aotx_bus_line *other =
                aotx_test_message_at(held - (unsigned long long)b);
            if (other != NULL && other->text_len == line->text_len
                && memcmp(other->text, line->text, line->text_len) == 0) {
                distinct -= 1u;
                break;
            }
        }
    }
    have = aotx_test_records(AOTX_REC_BUS, found, AOTX_TEST_FOUND);

    aotx_test_check(held - first == (unsigned long long)made,
                    "the bus buffer took one message for each message appended");
    aotx_test_check(matched == rows,
                    "every row of the bus panel holds its message after 100000 other"
                    " records");
    aotx_test_check(distinct == rows, "the rows of the bus panel hold different messages");
    if (writers == 1u) {
        aotx_test_check(ordered == rows,
                        "one writer gives the rows the order of the messages");
    }
    /* The ring no longer holds the records of these messages. A panel that reads the ring
     * therefore shows nothing, which is the defect the buffer answers. */
    aotx_test_check(have == 0u, "the record ring no longer holds a bus record");

    /* The row check can fail: a wrong message number in the buffer empties the row. */
    unsigned long long spoiled = 0ull;
    size_t at = offsetof(aotx_bus_buffer, line)
              + (size_t)((held - 1ull) & (AOTX_BUS_LINES - 1u)) * sizeof(aotx_bus_line)
              + offsetof(aotx_bus_line, at);
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_bus_lines, &spoiled, sizeof spoiled, at),
                       "cudaMemcpyToSymbol");
    aotx_ui_bus<<<1, AOTX_UI_PANEL_THREADS>>>();
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_test_read_grid();
    aotx_test_check(aotx_test_row_blank(AOTX_UI_BUS, 2u),
                    "a message number the buffer does not hold gives an empty row");
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_bus_lines, &held, sizeof held, at),
                       "cudaMemcpyToSymbol");

    printf("ui: bus buffer at %u writers, %u messages, %u rows matched, %u in order, %u"
           " different, %u bus records left in the ring\n", writers, made, matched,
           ordered, distinct, have);
    free(found);
}

/* A line goes from the panel when AOTX_CONSOLE_LINES newer lines take its place. */
static void aotx_test_scroll_out(void)
{
    unsigned int lines = (unsigned int)aotx_test_panels[AOTX_UI_CONSOLE].rows - 2u;
    unsigned long long held = 0ull;
    aotx_test_read_console();
    held = aotx_test_lines.count;
    aotx_test_console<<<1, 32>>>((unsigned int)AOTX_CONSOLE_LINES, 99u);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_ui_console<<<1, AOTX_UI_PANEL_THREADS>>>();
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_test_read_grid();
    aotx_test_read_console();
    aotx_test_check(aotx_test_line_at(held) == NULL,
                    "a line goes from the buffer when 256 newer lines take its place");
    aotx_test_check(!aotx_test_row_blank(AOTX_UI_CONSOLE, lines),
                    "the newest of the lines that came in is on the panel");
    printf("ui: the buffer holds %llu lines and shows the newest %u\n",
           (unsigned long long)AOTX_CONSOLE_LINES, lines);
}

/* The frame of a full ring is the worst case of the walk the other panels make. It must fit
 * in the period of a display at 60 Hz, which is 16.7 ms. */
static void aotx_test_frame_cost(void)
{
    aotx_ui_graph graph;
    cudaEvent_t start;
    cudaEvent_t end;
    float frame = 0.0f;
    aotx_check_runtime(cudaEventCreate(&start), "cudaEventCreate");
    aotx_check_runtime(cudaEventCreate(&end), "cudaEventCreate");
    aotx_ui_graph_build(&graph);
    aotx_ui_graph_run(&graph);
    aotx_check_runtime(cudaEventRecord(start), "cudaEventRecord");
    for (unsigned int i = 0u; i < 10u; ++i) {
        aotx_ui_graph_run(&graph);
    }
    aotx_check_runtime(cudaEventRecord(end), "cudaEventRecord");
    aotx_check_runtime(cudaEventSynchronize(end), "cudaEventSynchronize");
    aotx_check_runtime(cudaEventElapsedTime(&frame, start, end), "cudaEventElapsedTime");
    aotx_ui_graph_close(&graph);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    /* The sanitizer makes every kernel far slower than the period of the display. A run
     * with AOTX_SANITIZER set leaves the frame case out and states that it did. */
    if (getenv("AOTX_SANITIZER") != NULL) {
        printf("ui: AOTX_SANITIZER is set: the frame period case was left out\n");
    } else {
        aotx_test_check((double)frame / 10.0 < 16.7,
                        "the raster graph of a full ring fits in the period of the display");
    }
    printf("ui: the raster graph of a full ring took %.3f ms for each frame\n",
           (double)frame / 10.0);
}

#include "ui_agents.h"

int main(void)
{
    CUdevice device;
    CUcontext context;
    aotx_mem_map map;
    aotx_seam_rings rings;
    unsigned long long boot_id = 0x0f1de5c0ull;

    aotx_check_driver(cuInit(0), "cuInit");
    aotx_check_driver(cuDeviceGet(&device, 0), "cuDeviceGet");
    aotx_check_driver(cuDevicePrimaryCtxRetain(&context, device), "cuDevicePrimaryCtxRetain");
    aotx_check_driver(cuCtxSetCurrent(context), "cuCtxSetCurrent");
    if (aotx_mem_reserve(&map) != 0 || aotx_seam_open(&rings, boot_id) != 0) {
        printf("ui: the memory map or the rings did not open\n");
        return 1;
    }
    aotx_seam_bind(&rings, map.ring, map.ring_bytes, boot_id);
    aotx_check_runtime(cudaMemcpyFromSymbol(aotx_test_panels, aotx_ui_panel_table,
                                            sizeof aotx_test_panels), "cudaMemcpyFromSymbol");
    aotx_check_runtime(cudaMemcpyFromSymbol(aotx_test_font, aotx_ui_font,
                                            sizeof aotx_test_font), "cudaMemcpyFromSymbol");

    aotx_test_console_panel(1u, 1u);
    aotx_test_console_panel(64u, 2u);
    aotx_test_bus_panel(1u, 3u);
    aotx_test_bus_panel(64u, 4u);
    aotx_test_all_panels();
    aotx_test_agents_panel(1u, 1u, 0u);
    aotx_test_agents_panel(1u, 1u, 1u);
    aotx_test_agents_panel(AOTX_TEST_SLOTS / 2u, 2u, 3u);
    aotx_test_agents_panel(AOTX_TEST_SLOTS, 1u, AOTX_TEST_SLOTS);
    aotx_test_focus_title();
    aotx_test_graph();
    aotx_test_raster();
    aotx_test_scrollback(1u);
    aotx_test_scrollback(64u);
    aotx_test_bus_scrollback(1u);
    aotx_test_bus_scrollback(64u);
    aotx_test_scroll_out();
    aotx_test_frame_cost();

    aotx_seam_close(&rings);
    aotx_mem_release(&map);
    printf("ui: %u cases applied, %u passed, %u failed\n", aotx_test_applied,
           aotx_test_applied - aotx_test_failed, aotx_test_failed);
    if (aotx_test_applied == 0u) {
        printf("ui: no case ran\n");
        return 1;
    }
    return (aotx_test_failed == 0u) ? 0 : 1;
}
