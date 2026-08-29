/* Purpose: Draw one agent transcript and take the keys of its multi-line editor.
 * Owns: The rows read from one transcript file for the current frame.
 * Threading: One terminal thread.
 * Lifetime: The Session screen. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "disk/tui/tui.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define AOTX_SESSION_ROWS 128u
#define AOTX_SESSION_EDITOR_ROWS 4u

static char transcript_rows[AOTX_SESSION_ROWS][AOTX_TUI_LINE_BYTES];
static char result_text[AOTX_TUI_LINE_BYTES * 4u];
static int have_result;

static const aotx_mirror_agent_row *selected_agent(const aotx_tui *tui)
{
    unsigned int i;
    for (i = 0u; i < AOTX_MIRROR_AGENT_ROWS; i++) {
        const aotx_mirror_agent_row *row = &tui->shot.tables.agent[i];
        if (row->role_name[0] != '\0' && row->id == tui->session_agent) {
            return row;
        }
    }
    return NULL;
}

/* Makes one terminal row from one transcript element. */
static void element_row(const char *line, char *out, size_t bytes)
{
    char kind[24];
    char text[AOTX_TUI_LINE_BYTES];
    char tool[48];
    char status[32];
    uint64_t request = 0;
    uint64_t turn = 0;
    size_t i;
    if (!aotx_json_text(line, "\"kind\":\"", kind, sizeof(kind))) {
        snprintf(out, bytes, "? a transcript line did not read");
        return;
    }
    if (!aotx_json_text(line, "\"status\":\"", status, sizeof(status))) {
        status[0] = '\0';
    }
    aotx_json_number(line, "\"request\":", &request);
    aotx_json_number(line, "\"turn\":", &turn);
    if (aotx_json_text(line, "\"tool\":\"", tool, sizeof(tool))
        && aotx_json_text(line, "\"text\":\"", text, sizeof(text))) {
        for (i = 0u; text[i] != '\0'; i++) {
            if ((unsigned char)text[i] < 0x20u || text[i] == 0x7f) text[i] = ' ';
        }
        snprintf(out, bytes, "t%llu %-9s %s %.130s request %llu  %s",
                 (unsigned long long)turn, kind, tool, text,
                 (unsigned long long)request, status);
    } else if (aotx_json_text(line, "\"text\":\"", text, sizeof(text))) {
        for (i = 0u; text[i] != '\0'; i++) {
            if ((unsigned char)text[i] < 0x20u || text[i] == 0x7f) {
                text[i] = ' ';
            }
        }
        snprintf(out, bytes, "t%llu %-9s %.180s%s%s", (unsigned long long)turn, kind,
                 text, (status[0] != '\0') ? "  " : "", status);
    } else if (aotx_json_text(line, "\"tool\":\"", tool, sizeof(tool))) {
        snprintf(out, bytes, "t%llu %-9s %s request %llu  %s",
                 (unsigned long long)turn, kind, tool, (unsigned long long)request, status);
    } else {
        snprintf(out, bytes, "t%llu %-9s %s", (unsigned long long)turn, kind, status);
    }
}

static unsigned int read_transcript(const aotx_tui *tui)
{
    char path[AOTX_PATH_BYTES + 80];
    char *line = NULL;
    size_t cap = 0;
    ssize_t got;
    unsigned int count = 0u;
    FILE *file;
    have_result = 0;
    result_text[0] = '\0';
    if (tui->session.journal[0] == '\0' || tui->have_shot == 0) {
        return 0u;
    }
    snprintf(path, sizeof(path), "%s/%016llx/transcript/%u.jsonl", tui->session.journal,
             (unsigned long long)tui->shot.head.boot_id, tui->session_agent);
    file = fopen(path, "r");
    if (file == NULL) {
        return 0u;
    }
    while ((got = getline(&line, &cap, file)) > 0) {
        unsigned int at = count % AOTX_SESSION_ROWS;
        (void)got;
        element_row(line, transcript_rows[at], sizeof(transcript_rows[at]));
        {
            char kind[24];
            if (aotx_json_text(line, "\"kind\":\"", kind, sizeof(kind))
                && strcmp(kind, "result") == 0
                && aotx_json_text(line, "\"text\":\"", result_text,
                                  sizeof(result_text))) {
                have_result = 1;
            }
        }
        count++;
    }
    free(line);
    fclose(file);
    return (count > AOTX_SESSION_ROWS) ? AOTX_SESSION_ROWS : count;
}

/* Reads the newest nonempty row of the live console panel. */
static int live_console(const aotx_tui *tui, char *out, size_t bytes)
{
    unsigned int panel;
    for (panel = 0u; panel < AOTX_MIRROR_PANELS; panel++) {
        const aotx_mirror_panel *p = &tui->shot.head.panel[panel];
        unsigned int r;
        if (strcmp(p->name, "console") != 0) continue;
        for (r = p->rows; r > 0u; r--) {
            unsigned int row = p->row + r - 1u;
            unsigned int c;
            size_t used = 0u;
            if (row >= AOTX_MIRROR_ROWS) continue;
            for (c = 0u; c < p->cols && used + 1u < bytes; c++) {
                unsigned int col = p->col + c;
                unsigned int code;
                if (col >= AOTX_MIRROR_COLS) break;
                code = 32u + tui->shot.cell[row * AOTX_MIRROR_COLS + col].glyph;
                out[used++] = (code >= 32u && code < 127u) ? (char)code : ' ';
            }
            while (used > 0u && out[used - 1u] == ' ') used--;
            out[used] = '\0';
            if (used > 0u) return 1;
        }
    }
    out[0] = '\0';
    return 0;
}

static void draw_result(aotx_paint *p, unsigned int row, unsigned int rows,
                        unsigned int cols)
{
    unsigned int at = 0u;
    unsigned int r;
    unsigned int room = (cols > 6u) ? cols - 6u : 1u;
    for (r = 0u; r < rows && result_text[at] != '\0'; r++) {
        char line[AOTX_TUI_LINE_BYTES];
        unsigned int used = 0u;
        while (result_text[at] != '\0' && result_text[at] != '\n' && used < room) {
            line[used++] = result_text[at++];
        }
        if (result_text[at] == '\n') at++;
        line[used] = '\0';
        aotx_paint_text(p, row + r, 2u, line, AOTX_TUI_PLAIN);
    }
}

static void draw_editor(aotx_tui *tui, unsigned int row, unsigned int rows,
                        unsigned int cols)
{
    aotx_paint *p = &tui->paint;
    unsigned int room = (cols > 4u) ? cols - 4u : 1u;
    unsigned int r = 0u;
    unsigned int c = 0u;
    unsigned int i;
    const char *label = (tui->session_mode == 1u) ? "pages: "
                      : ((tui->session_mode == 2u) ? "role: " : "write: ");
    aotx_paint_text(p, row, 2u, label, AOTX_TUI_BRIGHT);
    c = (unsigned int)strlen(label);
    for (i = 0u; i < tui->edit_fill && r < rows; i++) {
        unsigned char b = (unsigned char)tui->edit[i];
        if (b == '\n' || c >= room) {
            r++;
            c = 0u;
            if (b == '\n') {
                continue;
            }
        }
        if (r < rows) {
            aotx_paint_put(p, row + r, 2u + c, (b >= 32u && b < 127u) ? b : ' ',
                           AOTX_TUI_BRIGHT);
            c++;
        }
    }
    p->cursor_row = row + ((r < rows) ? r : rows - 1u);
    p->cursor_col = 2u + c;
    p->cursor_on = 1;
}

void aotx_screen_session_draw(aotx_tui *tui, unsigned int top, unsigned int rows)
{
    aotx_paint *p = &tui->paint;
    const aotx_mirror_agent_row *agent = selected_agent(tui);
    unsigned int cols = aotx_paint_view_cols(p);
    unsigned int editor_rows = (rows > 12u) ? AOTX_SESSION_EDITOR_ROWS : 2u;
    unsigned int transcript_room = rows - editor_rows - 5u;
    unsigned int count = read_transcript(tui);
    int live;
    unsigned int start;
    unsigned int i;
    char line[AOTX_TUI_LINE_BYTES];
    if (rows < 8u || cols < 20u) {
        return;
    }
    aotx_paint_box(p, top, 0u, rows, cols, tui->utf8_box);
    snprintf(line, sizeof(line), " Session, card %u of %u, agent %u ", tui->card + 1u,
             tui->other_count + 1u, tui->session_agent);
    aotx_paint_text(p, top, 2u, line, AOTX_TUI_BRIGHT);
    if (agent != NULL) {
        snprintf(line, sizeof(line), "agent %u  role %.15s  state %u  pages %u  turn %u",
                 agent->id, agent->role_name, agent->state, agent->pages, agent->turn);
    } else {
        snprintf(line, sizeof(line), "agent %u is not in the live table", tui->session_agent);
    }
    aotx_paint_text(p, top + 1u, 2u, line, AOTX_TUI_REVERSE);
    for (i = 0u; i < AOTX_MIRROR_REQUEST_ROWS; i++) {
        const aotx_mirror_request_row *request = &tui->shot.tables.request[i];
        if (request->request != 0u && request->agent == tui->session_agent) {
            snprintf(line, sizeof(line), "request %u waits: %.15s %.80s  y allow  n refuse",
                     request->request, request->tool_name, request->argument);
            aotx_paint_text(p, top + 2u, 2u, line, AOTX_TUI_BRIGHT);
            break;
        }
    }
    live = live_console(tui, line, sizeof(line));
    if (live != 0 && transcript_room > 1u) transcript_room--;
    start = (count > transcript_room + tui->session_scroll)
          ? count - transcript_room - tui->session_scroll : 0u;
    if (tui->session_result != 0 && have_result != 0) {
        draw_result(p, top + 3u, transcript_room, cols);
    } else {
        for (i = 0u; i < transcript_room && start + i < count; i++) {
            snprintf(line, sizeof(line), "%.*s", (int)((cols > 4u) ? cols - 4u : 1u),
                     transcript_rows[(start + i) % AOTX_SESSION_ROWS]);
            aotx_paint_text(p, top + 3u + i, 2u, line, AOTX_TUI_PLAIN);
        }
        if (live != 0) {
            char tail[AOTX_TUI_LINE_BYTES];
            live_console(tui, line, sizeof(line));
            snprintf(tail, sizeof(tail), "live %.220s", line);
            aotx_paint_text(p, top + 3u + transcript_room, 2u, tail, AOTX_TUI_REVERSE);
        }
    }
    draw_editor(tui, top + rows - editor_rows - 1u, editor_rows, cols);
    snprintf(line, sizeof(line),
             "Left/Right card  Enter new line  Ctrl/Alt-Enter send  p pages  c compact  s spawn");
    aotx_paint_text(p, top + rows - 2u, 2u, line, AOTX_TUI_DIM);
}

static uint32_t waiting_request(const aotx_tui *tui)
{
    unsigned int i;
    for (i = 0u; i < AOTX_MIRROR_REQUEST_ROWS; i++) {
        const aotx_mirror_request_row *row = &tui->shot.tables.request[i];
        if (row->request != 0u && row->agent == tui->session_agent) {
            return row->request;
        }
    }
    return 0u;
}

static void send_editor(aotx_tui *tui)
{
    char line[AOTX_TUI_EDIT_BYTES];
    int used;
    if (tui->edit_fill == 0u) {
        return;
    }
    if (tui->session_mode == 1u) {
        used = snprintf(line, sizeof(line), "agent %u pages %s", tui->session_agent,
                        tui->edit);
    } else if (tui->session_mode == 2u) {
        used = snprintf(line, sizeof(line), "spawn %s", tui->edit);
    } else if (tui->session_agent == 0u) {
        used = snprintf(line, sizeof(line), "say %s", tui->edit);
    } else {
        used = snprintf(line, sizeof(line), "task %u %s", tui->session_agent, tui->edit);
    }
    if (used < 0 || (size_t)used >= sizeof(line)) {
        snprintf(tui->says, sizeof(tui->says), "the command is longer than the input bound");
        return;
    }
    aotx_screen_send(tui, line);
    tui->edit_fill = 0u;
    tui->edit[0] = '\0';
    tui->session_mode = 0u;
}

int aotx_screen_session_key(aotx_tui *tui, const aotx_tui_key *key)
{
    if (key->code == AOTX_TUI_KEY_ESCAPE) {
        tui->screen = AOTX_TUI_SCREEN_NONE;
        tui->edit_fill = 0u;
        tui->edit[0] = '\0';
        tui->session_mode = 0u;
        return 1;
    }
    if (key->code == AOTX_TUI_KEY_PAGE_UP) {
        tui->session_scroll++;
        return 1;
    }
    if (key->code == AOTX_TUI_KEY_PAGE_DN) {
        if (tui->session_scroll > 0u) tui->session_scroll--;
        return 1;
    }
    if (key->code == AOTX_TUI_KEY_LEFT || key->code == AOTX_TUI_KEY_RIGHT) {
        if (aotx_tui_select_card(tui, (key->code == AOTX_TUI_KEY_LEFT) ? -1 : 1) != 0) {
            tui->session_scroll = 0u;
        }
        return 1;
    }
    if (key->code == AOTX_TUI_KEY_UP) {
        if (tui->session_agent > 0u) tui->session_agent--;
        tui->session_scroll = 0u;
        return 1;
    }
    if (key->code == AOTX_TUI_KEY_DOWN) {
        if (tui->session_agent + 1u < AOTX_MIRROR_AGENT_ROWS) tui->session_agent++;
        tui->session_scroll = 0u;
        return 1;
    }
    if (key->code == AOTX_TUI_KEY_BACK) {
        if (tui->edit_fill > 0u) tui->edit[--tui->edit_fill] = '\0';
        return 1;
    }
    if (key->code == AOTX_TUI_KEY_ENTER
        && (key->mods & (AOTX_TUI_MOD_CONTROL | AOTX_TUI_MOD_ALT)) != 0u) {
        send_editor(tui);
        return 1;
    }
    if (key->code == AOTX_TUI_KEY_ENTER) {
        if (tui->edit_fill == 0u && tui->session_scroll > 0u && have_result != 0) {
            tui->session_result = !tui->session_result;
            return 1;
        }
        if (tui->edit_fill + 1u < sizeof(tui->edit)) {
            tui->edit[tui->edit_fill++] = '\n';
            tui->edit[tui->edit_fill] = '\0';
        }
        return 1;
    }
    if (key->code == 0u && key->mods == 0u
        && (key->codepoint == 'y' || key->codepoint == 'n')) {
        uint32_t request = waiting_request(tui);
        char line[64];
        if (request != 0u) {
            snprintf(line, sizeof(line), "%s %u",
                     (key->codepoint == 'y') ? "authorize" : "refuse", request);
            aotx_screen_send(tui, line);
            return 1;
        }
    }
    if (key->code == 0u && key->mods == 0u && key->codepoint == 'c'
        && tui->edit_fill == 0u) {
        char line[64];
        snprintf(line, sizeof(line), "agent %u compact", tui->session_agent);
        aotx_screen_send(tui, line);
        return 1;
    }
    if (key->code == 0u && key->mods == 0u && key->codepoint == 'p'
        && tui->edit_fill == 0u) {
        tui->session_mode = 1u;
        return 1;
    }
    if (key->code == 0u && key->mods == 0u && key->codepoint == 's'
        && tui->edit_fill == 0u) {
        tui->session_mode = 2u;
        return 1;
    }
    if (key->code == 0u && key->codepoint >= 32u && key->codepoint < 127u
        && tui->edit_fill + 1u < sizeof(tui->edit)) {
        tui->edit[tui->edit_fill++] = (char)key->codepoint;
        tui->edit[tui->edit_fill] = '\0';
        return 1;
    }
    return 1;
}
