/* Purpose: Read a snapshot of the mirror and write the cells that changed to the terminal.
 * Owns: The picture the program wants and the picture the terminal holds.
 * Threading: One thread; the read of the mirror follows the sequence rule and takes no lock.
 * Lifetime: The whole run. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "disk/tui/tui.h"

#include <string.h>

/* The glyph of a cell is an index into the font of the window. Index 0 is the space, and
 * the last index is the replacement box, which a terminal shows as a question mark. */
#define AOTX_TUI_GLYPH_FIRST 32u
#define AOTX_TUI_GLYPH_BOX   95u

/* The rows the frame keeps: one status line above the work area and one key bar below it.
 * The work area holds every column, because the panel borders of the grid are its edge. */
#define AOTX_TUI_FRAME_ROWS  2u

/* The code point a terminal shows for one glyph index of the font of the window. */
static unsigned int glyph_code(unsigned char glyph)
{
    if (glyph >= AOTX_TUI_GLYPH_BOX) {
        return (unsigned int)'?';
    }
    return AOTX_TUI_GLYPH_FIRST + glyph;
}

unsigned int aotx_paint_view_rows(const aotx_paint *p)
{
    return (p->rows > AOTX_TUI_FRAME_ROWS) ? p->rows - AOTX_TUI_FRAME_ROWS : 0u;
}

unsigned int aotx_paint_view_cols(const aotx_paint *p)
{
    return p->cols;
}

void aotx_paint_size(aotx_paint *p, unsigned int cols, unsigned int rows)
{
    p->cols = (cols > AOTX_TUI_COLS_MAX) ? AOTX_TUI_COLS_MAX : cols;
    p->rows = (rows > AOTX_TUI_ROWS_MAX) ? AOTX_TUI_ROWS_MAX : rows;
    p->full = 1;
    memset(p->drawn, 0, sizeof(p->drawn));
    aotx_paint_clear(p);
}

void aotx_paint_clear(aotx_paint *p)
{
    unsigned int at;
    unsigned int cells = p->cols * p->rows;
    for (at = 0; at < cells; at++) {
        p->want[at].code = (uint16_t)' ';
        p->want[at].attribute = (uint8_t)AOTX_TUI_PLAIN;
        p->want[at].reserved = 0;
    }
}

void aotx_paint_put(aotx_paint *p, unsigned int row, unsigned int col,
                    unsigned int code, unsigned int attribute)
{
    aotx_tui_cell *cell;
    if (row >= p->rows || col >= p->cols) {
        return;
    }
    cell = &p->want[row * p->cols + col];
    cell->code = (uint16_t)code;
    cell->attribute = (uint8_t)attribute;
    cell->reserved = 0;
}

void aotx_paint_text(aotx_paint *p, unsigned int row, unsigned int col,
                     const char *text, unsigned int attribute)
{
    unsigned int at;
    for (at = 0; text[at] != '\0'; at++) {
        unsigned char byte = (unsigned char)text[at];
        /* The screens take the printing bytes of the ascii set and nothing else. A byte
         * of another set thus never reaches the terminal from a name on the disk. */
        if (byte < AOTX_TUI_GLYPH_FIRST || byte > 126u) {
            byte = ' ';
        }
        aotx_paint_put(p, row, col + at, byte, attribute);
    }
}

void aotx_paint_fill(aotx_paint *p, unsigned int row, unsigned int col,
                     unsigned int cols, unsigned int code, unsigned int attribute)
{
    unsigned int at;
    for (at = 0; at < cols; at++) {
        aotx_paint_put(p, row, col + at, code, attribute);
    }
}

/* The box is drawn with the ascii characters by default. The panel borders of the grid
 * are ascii, and a terminal with no wide character set shows them right. */
void aotx_paint_box(aotx_paint *p, unsigned int row, unsigned int col,
                    unsigned int rows, unsigned int cols, int utf8)
{
    unsigned int r;
    unsigned int corner = (unsigned int)'+';
    unsigned int across = (unsigned int)'-';
    unsigned int down = (unsigned int)'|';
    if (rows < 2u || cols < 2u) {
        return;
    }
    /* The wide character set is not in the font of the grid. The box thus keeps the same
     * characters, and the setting names the intent for a later version. */
    (void)utf8;
    aotx_paint_fill(p, row, col + 1u, cols - 2u, across, AOTX_TUI_DIM);
    aotx_paint_fill(p, row + rows - 1u, col + 1u, cols - 2u, across, AOTX_TUI_DIM);
    aotx_paint_put(p, row, col, corner, AOTX_TUI_DIM);
    aotx_paint_put(p, row, col + cols - 1u, corner, AOTX_TUI_DIM);
    aotx_paint_put(p, row + rows - 1u, col, corner, AOTX_TUI_DIM);
    aotx_paint_put(p, row + rows - 1u, col + cols - 1u, corner, AOTX_TUI_DIM);
    for (r = 1u; r + 1u < rows; r++) {
        aotx_paint_put(p, row + r, col, down, AOTX_TUI_DIM);
        aotx_paint_put(p, row + r, col + cols - 1u, down, AOTX_TUI_DIM);
        aotx_paint_fill(p, row + r, col + 1u, cols - 2u, (unsigned int)' ',
                        AOTX_TUI_PLAIN);
    }
}

/* Holds the pan inside the grid. A view that is wider than the grid pans no further than
 * zero, so a wide terminal shows the whole picture. */
static void clamp_pan(aotx_paint *p)
{
    unsigned int rows = aotx_paint_view_rows(p);
    unsigned int cols = aotx_paint_view_cols(p);
    unsigned int most_row = (AOTX_MIRROR_ROWS > rows) ? AOTX_MIRROR_ROWS - rows : 0u;
    unsigned int most_col = (AOTX_MIRROR_COLS > cols) ? AOTX_MIRROR_COLS - cols : 0u;
    if (p->pan_row > most_row) {
        p->pan_row = most_row;
    }
    if (p->pan_col > most_col) {
        p->pan_col = most_col;
    }
}

void aotx_paint_pan(aotx_paint *p, int rows, int cols)
{
    int row = (int)p->pan_row + rows;
    int col = (int)p->pan_col + cols;
    p->pan_row = (row < 0) ? 0u : (unsigned int)row;
    p->pan_col = (col < 0) ? 0u : (unsigned int)col;
    clamp_pan(p);
}

void aotx_paint_panel(aotx_paint *p, const aotx_mirror_snapshot *shot, unsigned int panel)
{
    if (panel >= AOTX_MIRROR_PANELS) {
        return;
    }
    p->pan_row = shot->head.panel[panel].row;
    p->pan_col = shot->head.panel[panel].col;
    clamp_pan(p);
}

void aotx_paint_picture(aotx_paint *p, const aotx_mirror_snapshot *shot)
{
    unsigned int rows = aotx_paint_view_rows(p);
    unsigned int cols = aotx_paint_view_cols(p);
    unsigned int r;
    clamp_pan(p);
    for (r = 0; r < rows; r++) {
        unsigned int from_row = p->pan_row + r;
        unsigned int c;
        if (from_row >= AOTX_MIRROR_ROWS) {
            break;
        }
        for (c = 0; c < cols; c++) {
            unsigned int from_col = p->pan_col + c;
            const aotx_mirror_cell *cell;
            if (from_col >= AOTX_MIRROR_COLS) {
                break;
            }
            cell = &shot->cell[from_row * AOTX_MIRROR_COLS + from_col];
            aotx_paint_put(p, r + 1u, c, glyph_code(cell->glyph), cell->attribute);
        }
    }
    /* The cursor of the editor stands where the window shows the bright cell, when that
     * cell is on view. */
    if (shot->head.cursor_row >= p->pan_row && shot->head.cursor_col >= p->pan_col) {
        unsigned int row = shot->head.cursor_row - p->pan_row;
        unsigned int col = shot->head.cursor_col - p->pan_col;
        if (row < rows && col < cols) {
            p->cursor_row = row + 1u;
            p->cursor_col = col;
            p->cursor_on = (shot->head.focus == 0u) ? 1 : 0;
            return;
        }
    }
    p->cursor_on = 0;
}

/* The sequence of a slot, read with an acquire load. */
static uint64_t slot_sequence(const unsigned char *slot)
{
    const aotx_mirror_snapshot *shot = (const aotx_mirror_snapshot *)slot;
    return __atomic_load_n(&shot->head.sequence, __ATOMIC_ACQUIRE);
}

/* Copies one slot and keeps the copy when the sequence did not move. Returns 1 or 0. */
static int take_slot(const unsigned char *slot, aotx_mirror_snapshot *out)
{
    uint64_t first = slot_sequence(slot);
    if (first == 0) {
        return 0;
    }
    memcpy(out, slot, sizeof(*out));
    return (slot_sequence(slot) == first && out->head.sequence == first) ? 1 : 0;
}

int aotx_mirror_take(const unsigned char *mirror, aotx_mirror_snapshot *out)
{
    const aotx_mirror_preamble *pre = (const aotx_mirror_preamble *)mirror;
    const unsigned char *base = mirror + sizeof(aotx_mirror_preamble);
    unsigned int slots;
    unsigned int best = AOTX_MIRROR_SLOTS;
    uint64_t best_sequence = 0;
    unsigned int i;
    if (pre->magic != AOTX_MIRROR_MAGIC || pre->layout != AOTX_MIRROR_LAYOUT
        || pre->slot_bytes < sizeof(aotx_mirror_snapshot)) {
        return 0;
    }
    slots = (pre->slots < AOTX_MIRROR_SLOTS) ? pre->slots : AOTX_MIRROR_SLOTS;
    for (i = 0; i < slots; i++) {
        uint64_t sequence = slot_sequence(base + (size_t)i * pre->slot_bytes);
        if (sequence != 0 && sequence > best_sequence) {
            best_sequence = sequence;
            best = i;
        }
    }
    if (best == AOTX_MIRROR_SLOTS) {
        return 0;
    }
    if (take_slot(base + (size_t)best * pre->slot_bytes, out) != 0) {
        return 1;
    }
    /* The newest slot was written while it was read. The other slot holds a whole frame
     * that is one behind, which is a frame lost and never a frame torn. */
    for (i = 0; i < slots; i++) {
        if (i != best && take_slot(base + (size_t)i * pre->slot_bytes, out) != 0) {
            return 1;
        }
    }
    return 0;
}

/* Writes one code point as utf8. Returns the bytes written. */
static unsigned int put_code(char *out, unsigned int code)
{
    if (code < 0x80u) {
        out[0] = (char)code;
        return 1u;
    }
    if (code < 0x800u) {
        out[0] = (char)(0xc0u | (code >> 6));
        out[1] = (char)(0x80u | (code & 0x3fu));
        return 2u;
    }
    out[0] = (char)(0xe0u | (code >> 12));
    out[1] = (char)(0x80u | ((code >> 6) & 0x3fu));
    out[2] = (char)(0x80u | (code & 0x3fu));
    return 3u;
}

/* Reports whether the cell at `at` needs a write. */
static int cell_changed(const aotx_paint *p, unsigned int at)
{
    if (p->full != 0) {
        return 1;
    }
    return (p->want[at].code != p->drawn[at].code
            || p->want[at].attribute != p->drawn[at].attribute) ? 1 : 0;
}

void aotx_paint_flush(aotx_paint *p, aotx_term *t, int color)
{
    unsigned int row;
    char run[AOTX_TUI_COLS_MAX * 3u];
    t->color = color;
    aotx_term_cursor(t, 0);
    for (row = 0; row < p->rows; row++) {
        unsigned int col = 0;
        while (col < p->cols) {
            unsigned int at = row * p->cols + col;
            unsigned int fill = 0;
            unsigned int cells = 0;
            unsigned int attribute;
            if (cell_changed(p, at) == 0) {
                col++;
                continue;
            }
            /* One cursor move covers the whole run of cells that changed, so a live
             * console costs a few hundred bytes and not a whole frame. */
            attribute = p->want[at].attribute;
            aotx_term_move(t, row + 1u, col + 1u);
            aotx_term_rendition(t, attribute);
            while (col < p->cols) {
                at = row * p->cols + col;
                if (p->want[at].attribute != attribute || cell_changed(p, at) == 0) {
                    break;
                }
                fill += put_code(run + fill, p->want[at].code);
                p->drawn[at] = p->want[at];
                cells++;
                col++;
            }
            aotx_term_put(t, run, fill);
            t->col += cells;
            p->cells += cells;
        }
    }
    if (p->cursor_on != 0) {
        aotx_term_move(t, p->cursor_row + 1u, p->cursor_col + 1u);
        aotx_term_cursor(t, 1);
    }
    p->full = 0;
    p->frames++;
    aotx_term_flush(t);
}
