/* Purpose: Read the splash art, draw it with its line, and dissolve it into the picture.
 * Owns: The code points of the art and the cell order of the dissolve.
 * Threading: One thread; the art is read one time at each size the terminal takes.
 * Lifetime: From the start of the program to its exit. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "disk/tui/tui.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* The sizes the tool writes. The program picks the largest one that fits and rescales
 * nothing, because a scaled figure of characters is a different figure. */
static const unsigned int aotx_splash_sizes[AOTX_TUI_SPLASH_SIZES][2] = {
    { 160u, 40u }, { 120u, 30u }, { 80u, 20u }, { 60u, 12u }, { 40u, 8u }
};

/* The name the program shows when no art fits or no art reads. */
#define AOTX_SPLASH_NAME "AOTX-1"

/* The seed of the dissolve. One seed gives one order, so every run dissolves the same way
 * and a check counts the frames. */
#define AOTX_SPLASH_SEED 0x9e3779b9u

/* Reads one code point of a utf8 line. Returns the bytes it took, or 0 at the end. */
static unsigned int take_code(const char *line, unsigned int at, unsigned int bytes,
                              unsigned int *code)
{
    unsigned char first;
    if (at >= bytes) {
        return 0;
    }
    first = (unsigned char)line[at];
    if (first < 0x80u) {
        *code = first;
        return 1u;
    }
    if ((first & 0xe0u) == 0xc0u && at + 1u < bytes) {
        *code = ((unsigned int)(first & 0x1fu) << 6)
              | (unsigned int)((unsigned char)line[at + 1u] & 0x3fu);
        return 2u;
    }
    if ((first & 0xf0u) == 0xe0u && at + 2u < bytes) {
        *code = ((unsigned int)(first & 0x0fu) << 12)
              | ((unsigned int)((unsigned char)line[at + 1u] & 0x3fu) << 6)
              | (unsigned int)((unsigned char)line[at + 2u] & 0x3fu);
        return 3u;
    }
    /* A byte that starts no form this build reads gives one space. The width of the line
     * thus stays the width the name says, and the check below sees it. */
    *code = (unsigned int)' ';
    return 1u;
}

/* Reads one file of art into the state. Returns 0, or 1 with the reason. */
static int read_file(aotx_splash *s, const char *path, unsigned int cols, unsigned int rows)
{
    char line[AOTX_TUI_COLS_MAX * 4u + 8u];
    FILE *file = fopen(path, "r");
    unsigned int row = 0;
    if (file == NULL) {
        return 1;
    }
    while (row < rows && fgets(line, (int)sizeof(line), file) != NULL) {
        unsigned int bytes = (unsigned int)strlen(line);
        unsigned int at = 0;
        unsigned int col = 0;
        while (bytes > 0 && (line[bytes - 1u] == '\n' || line[bytes - 1u] == '\r')) {
            bytes--;
        }
        while (col < cols) {
            unsigned int code = (unsigned int)' ';
            unsigned int took = take_code(line, at, bytes, &code);
            if (took == 0) {
                break;
            }
            at += took;
            s->code[row * cols + col] = (uint16_t)code;
            col++;
        }
        if (col != cols || at != bytes) {
            /* A file of the wrong width is refused with its name. The program never shows
             * art that does not fit the cells the name says. */
            fclose(file);
            return 1;
        }
        row++;
    }
    fclose(file);
    if (row != rows) {
        return 1;
    }
    s->cols = cols;
    s->rows = rows;
    return 0;
}

/* Reports whether the locale of this program names the wide character set. */
static int locale_is_utf8(void)
{
    const char *names[3];
    unsigned int i;
    names[0] = getenv("LC_ALL");
    names[1] = getenv("LC_CTYPE");
    names[2] = getenv("LANG");
    for (i = 0; i < 3u; i++) {
        if (names[i] != NULL && names[i][0] != '\0') {
            return (strstr(names[i], "UTF-8") != NULL || strstr(names[i], "utf8") != NULL
                    || strstr(names[i], "UTF8") != NULL || strstr(names[i], "utf-8") != NULL)
                   ? 1 : 0;
        }
    }
    return 0;
}

int aotx_splash_read(aotx_splash *s, const char *dir, const char *mode,
                     unsigned int cols, unsigned int rows)
{
    unsigned int i;
    int braille;
    memset(s, 0, sizeof(*s));
    if (mode != NULL && strcmp(mode, "off") == 0) {
        snprintf(s->reason, sizeof(s->reason), "the splash is off");
        return 0;
    }
    braille = 1;
    if (mode != NULL && strcmp(mode, "ascii") == 0) {
        braille = 0;
    } else if (mode == NULL || strcmp(mode, "braille") != 0) {
        braille = locale_is_utf8();
    }
    /* One row under the art holds the line that says what the system does. */
    if (rows < 2u) {
        snprintf(s->reason, sizeof(s->reason), "the terminal is too small for the splash");
        return 0;
    }
    for (i = 0; i < AOTX_TUI_SPLASH_SIZES; i++) {
        char path[AOTX_PATH_BYTES];
        unsigned int art_cols = aotx_splash_sizes[i][0];
        unsigned int art_rows = aotx_splash_sizes[i][1];
        if (art_cols > cols || art_rows + 1u > rows) {
            continue;
        }
        snprintf(path, sizeof(path), "%s/%ux%u.%s", dir, art_cols, art_rows,
                 braille ? "braille" : "ascii");
        snprintf(s->name, sizeof(s->name), "%ux%u.%s", art_cols, art_rows,
                 braille ? "braille" : "ascii");
        if (read_file(s, path, art_cols, art_rows) == 0) {
            s->braille = braille;
            s->held = 1;
            s->reason[0] = '\0';
            return 1;
        }
        snprintf(s->reason, sizeof(s->reason), "the art %s does not read at its width",
                 s->name);
    }
    if (s->reason[0] == '\0') {
        snprintf(s->reason, sizeof(s->reason), "no splash art fits this terminal");
    }
    s->held = 0;
    return 0;
}

/* The first row and the first column of the art inside the work area. */
static void place(const aotx_splash *s, const aotx_paint *p, unsigned int top,
                  unsigned int rows, unsigned int *art_row, unsigned int *art_col)
{
    unsigned int cols = aotx_paint_view_cols(p);
    *art_row = top + ((rows > s->rows + 1u) ? (rows - s->rows - 1u) / 2u : 0u);
    *art_col = (cols > s->cols) ? (cols - s->cols) / 2u : 0u;
}

void aotx_splash_draw(const aotx_splash *s, aotx_paint *p, unsigned int top,
                      unsigned int rows, const char *state)
{
    unsigned int art_row;
    unsigned int art_col;
    unsigned int r;
    unsigned int line_row;
    unsigned int line_col;
    unsigned int cols = aotx_paint_view_cols(p);
    size_t bytes = strlen(state);
    if (s->held == 0) {
        /* With no art the name stands alone, and the reason is under it. */
        unsigned int middle = top + (rows / 2u);
        line_col = (cols > strlen(AOTX_SPLASH_NAME))
                   ? (cols - (unsigned int)strlen(AOTX_SPLASH_NAME)) / 2u : 0u;
        aotx_paint_text(p, middle, line_col, AOTX_SPLASH_NAME, AOTX_TUI_BRIGHT);
        line_col = (cols > bytes) ? (cols - (unsigned int)bytes) / 2u : 0u;
        aotx_paint_text(p, middle + 1u, line_col, state, AOTX_TUI_DIM);
        return;
    }
    place(s, p, top, rows, &art_row, &art_col);
    for (r = 0; r < s->rows; r++) {
        unsigned int c;
        for (c = 0; c < s->cols; c++) {
            aotx_paint_put(p, art_row + r, art_col + c, s->code[r * s->cols + c],
                           AOTX_TUI_PLAIN);
        }
    }
    line_row = art_row + s->rows;
    line_col = (cols > bytes) ? (cols - (unsigned int)bytes) / 2u : 0u;
    aotx_paint_text(p, line_row, line_col, state, AOTX_TUI_DIM);
}

void aotx_splash_dissolve_open(aotx_splash *s, unsigned int cols, unsigned int rows,
                               unsigned int frames)
{
    unsigned int count = cols * rows;
    unsigned int seed = AOTX_SPLASH_SEED;
    unsigned int i;
    if (count > AOTX_TUI_CELLS) {
        count = AOTX_TUI_CELLS;
    }
    for (i = 0; i < count; i++) {
        s->order[i] = i;
    }
    /* One seed gives one order. The exchange runs from the end, so each cell has the same
     * chance of each place and the order is the same at every run. */
    for (i = count; i > 1u; i--) {
        unsigned int at;
        unsigned int hold;
        seed = (seed * 1664525u) + 1013904223u;
        at = seed % i;
        hold = s->order[i - 1u];
        s->order[i - 1u] = s->order[at];
        s->order[at] = hold;
    }
    s->order_count = count;
    s->step = 0;
    (void)frames;
}

int aotx_splash_dissolve_step(aotx_splash *s, aotx_paint *p, unsigned int top,
                              unsigned int rows, unsigned int frames)
{
    unsigned int art_row;
    unsigned int art_col;
    unsigned int gone;
    unsigned int i;
    if (frames == 0 || s->held == 0 || s->order_count == 0) {
        return 0;
    }
    if (s->step >= frames) {
        return 0;
    }
    s->step++;
    place(s, p, top, rows, &art_row, &art_col);
    gone = (unsigned int)(((uint64_t)s->order_count * s->step) / frames);
    for (i = gone; i < s->order_count; i++) {
        unsigned int cell = s->order[i];
        unsigned int r = cell / s->cols;
        unsigned int c = cell % s->cols;
        aotx_paint_put(p, art_row + r, art_col + c, s->code[cell], AOTX_TUI_PLAIN);
    }
    return (s->step < frames) ? 1 : 0;
}
