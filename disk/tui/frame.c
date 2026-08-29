/* Purpose: Draw the status line and the key bar that hold the work area between them.
 * Owns: Nothing; the caller owns the picture and the state the frame reads.
 * Threading: One thread.
 * Lifetime: One frame. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "disk/tui/tui.h"

#include <stdio.h>
#include <string.h>

#ifndef AOTX_VERSION
#define AOTX_VERSION "0.0.0"
#endif

/* The two forms of the key bar. The wide form has two spaces between the fields; the
 * narrow form drops the letter of the key and one space, and fits eighty columns. */
static unsigned int keybar_width(int wide)
{
    unsigned int width = 0;
    unsigned int i;
    for (i = 0; aotx_tui_keys[i].key != NULL; i++) {
        const char *key = aotx_tui_keys[i].key;
        unsigned int bytes = (unsigned int)strlen(aotx_tui_keys[i].label);
        bytes += (unsigned int)strlen(key) - (wide ? 0u : 1u);
        width += bytes + (wide ? 3u : 1u);
    }
    return (width > 0) ? width - (wide ? 2u : 1u) : 0u;
}

void aotx_frame_keybar(aotx_tui *tui)
{
    aotx_paint *p = &tui->paint;
    unsigned int row = (p->rows > 0) ? p->rows - 1u : 0u;
    unsigned int col = 0;
    int wide = (keybar_width(1) <= p->cols) ? 1 : 0;
    unsigned int i;
    aotx_paint_fill(p, row, 0, p->cols, (unsigned int)' ', AOTX_TUI_DIM);
    for (i = 0; aotx_tui_keys[i].key != NULL; i++) {
        const char *key = aotx_tui_keys[i].key;
        const char *label = aotx_tui_keys[i].label;
        unsigned int open = (tui->screen == i) ? 1u : 0u;
        unsigned int attribute = open ? AOTX_TUI_REVERSE : AOTX_TUI_DIM;
        if (wide == 0) {
            key = key + 1;  /* the number alone, without the letter */
        }
        aotx_paint_text(p, row, col, key, attribute);
        col += (unsigned int)strlen(key);
        if (wide != 0) {
            aotx_paint_put(p, row, col++, (unsigned int)' ', attribute);
        }
        aotx_paint_text(p, row, col, label, open ? AOTX_TUI_REVERSE : AOTX_TUI_PLAIN);
        col += (unsigned int)strlen(label);
        col += wide ? 2u : 1u;
        if (col >= p->cols) {
            break;
        }
    }
}

/* The name of the resident language model, as a role number or a dash. */
static void language_text(const aotx_mirror_snapshot *shot, char *out, size_t bytes)
{
    unsigned int i;
    if (shot->head.language == ~0u) {
        snprintf(out, bytes, "-");
        return;
    }
    for (i = 0; i < AOTX_MIRROR_MODEL_ROWS; i++) {
        if (shot->tables.model[i].role == shot->head.language
            && shot->tables.model[i].name[0] != '\0') {
            snprintf(out, bytes, "%.*s", (int)bytes - 1, shot->tables.model[i].name);
            return;
        }
    }
    snprintf(out, bytes, "role %u", shot->head.language);
}

void aotx_frame_status(aotx_tui *tui)
{
    aotx_paint *p = &tui->paint;
    char line[AOTX_TUI_LINE_BYTES];
    if (tui->have_shot != 0) {
        char language[AOTX_MIRROR_TEXT_BYTES];
        const aotx_mirror_head *head = &tui->shot.head;
        language_text(&tui->shot, language, sizeof(language));
        snprintf(line, sizeof(line),
                 "aotx %s %.15s %.7s boot %llx tick %llu lag %llums held %llu"
                 " model %.20s agents %u/%u wait %u",
                 AOTX_VERSION, head->profile, head->arch,
                 (unsigned long long)head->boot_id, (unsigned long long)head->tick,
                 (unsigned long long)head->drain_lag_ms, (unsigned long long)head->held,
                 language, head->agents_live, head->slots, head->requests_waiting);
        /* A narrow terminal takes the short line, so the figures of the run stay whole
         * and none of them is cut in the middle. */
        if (strlen(line) > p->cols) {
            snprintf(line, sizeof(line),
                     "aotx %s %.15s tick %llu lag %llums agents %u/%u wait %u",
                     AOTX_VERSION, head->profile, (unsigned long long)head->tick,
                     (unsigned long long)head->drain_lag_ms, head->agents_live,
                     head->slots, head->requests_waiting);
        }
    } else {
        snprintf(line, sizeof(line), "aotx %s   %.60s", AOTX_VERSION, tui->state);
    }
    aotx_paint_fill(p, 0, 0, p->cols, (unsigned int)' ', AOTX_TUI_REVERSE);
    aotx_paint_text(p, 0, 0, line, AOTX_TUI_REVERSE);
}

void aotx_frame_draw(aotx_tui *tui)
{
    aotx_paint *p = &tui->paint;
    unsigned int rows = aotx_paint_view_rows(p);
    unsigned int at;
    aotx_paint_clear(p);
    if (rows == 0) {
        return;
    }
    /* The work area holds the picture, the splash, or a screen over them. */
    if (tui->screen == AOTX_TUI_SCREEN_NONE) {
        if (tui->have_shot != 0 && tui->dissolve == 0) {
            aotx_paint_picture(p, &tui->shot);
        } else if (tui->have_shot != 0) {
            aotx_paint_picture(p, &tui->shot);
            tui->dissolve = (unsigned int)aotx_splash_dissolve_step(
                &tui->splash, p, 1u, rows, AOTX_TUI_DISSOLVE);
        } else {
            p->cursor_on = 0;
            aotx_splash_draw(&tui->splash, p, 1u, rows, tui->state);
        }
    } else {
        p->cursor_on = 0;
        if (tui->have_shot != 0) {
            aotx_paint_picture(p, &tui->shot);
            p->cursor_on = 0;
        }
        aotx_screen_draw(tui, 1u, rows);
    }
    /* The last row of the work area holds what the program has to say, so a refusal is
     * where the key was pressed. */
    if (tui->socket_closed != 0 || tui->says[0] != '\0') {
        const char *notice = (tui->socket_closed != 0)
                           ? "the system closed the socket" : tui->says;
        at = rows;
        aotx_paint_fill(p, at, 0, p->cols, (unsigned int)' ', AOTX_TUI_BRIGHT);
        aotx_paint_text(p, at, 0u, notice, AOTX_TUI_BRIGHT);
    }
    aotx_frame_status(tui);
    aotx_frame_keybar(tui);
}
