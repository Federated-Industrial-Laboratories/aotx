/* Purpose: Draw the open screen, take its keys, and send the line that each action names.
 * Owns: The rows of the open screen and the field that a screen edits.
 * Threading: One thread.
 * Lifetime: One frame for the rows; the state of the screen lives in the program.  */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "disk/tui/tui.h"

#include <stdio.h>
#include <string.h>

/* The rows of the open screen. One screen is open at a time, so one table serves them
 * all. The table is large, so it lives beside the program and not on a stack. */
static char aotx_screen_rows[AOTX_TUI_ROWS_LIST][AOTX_TUI_LINE_BYTES];

/* A screen that reads the disk fills its rows this often at the most. The frame is drawn
 * many times a second, and a directory does not change at that rate. */
#define AOTX_TUI_DISK_MS 500u

/* The screens by their place in the table of actions.h. */
#define AOTX_SCREEN_HELP     0u
#define AOTX_SCREEN_MENU     1u
#define AOTX_SCREEN_AGENTS   2u
#define AOTX_SCREEN_BUS      3u
#define AOTX_SCREEN_MODELS   4u
#define AOTX_SCREEN_TOOLS    5u
#define AOTX_SCREEN_SKILLS   6u
#define AOTX_SCREEN_SETTINGS 7u
#define AOTX_SCREEN_SYSTEM   8u
#define AOTX_SCREEN_QUIT     9u
#define AOTX_SCREEN_PICKER   10u

const char *aotx_screen_name(unsigned int screen)
{
    unsigned int i;
    for (i = 0; aotx_tui_screens[i].name != NULL; i++) {
        if (i == screen) {
            return aotx_tui_screens[i].name;
        }
    }
    return "-";
}

unsigned int aotx_screen_of_key(const aotx_tui_key *key)
{
    unsigned int i;
    char token[8];
    if (key->code < AOTX_TUI_KEY_F1 || key->code > AOTX_TUI_KEY_F12) {
        return AOTX_TUI_SCREEN_NONE;
    }
    snprintf(token, sizeof(token), "F%u", key->code - AOTX_TUI_KEY_F1 + 1u);
    for (i = 0; aotx_tui_screens[i].name != NULL; i++) {
        if (strcmp(aotx_tui_screens[i].key, token) == 0) {
            return i;
        }
    }
    return AOTX_TUI_SCREEN_NONE;
}

/* The token of a key, as the table of actions spells it. */
static void key_token(const aotx_tui_key *key, char *out, size_t bytes)
{
    if (key->code == AOTX_TUI_KEY_ENTER) {
        snprintf(out, bytes, "Enter");
        return;
    }
    if (key->code == 0 && key->codepoint >= 32u && key->codepoint < 127u
        && key->mods == 0) {
        out[0] = (char)key->codepoint;
        out[1] = '\0';
        return;
    }
    out[0] = '\0';
}

const char *aotx_screen_action(unsigned int screen, const aotx_tui_key *key)
{
    char token[8];
    const char *name = aotx_screen_name(screen);
    unsigned int i;
    key_token(key, token, sizeof(token));
    if (token[0] == '\0') {
        return NULL;
    }
    for (i = 0; aotx_tui_actions[i].screen != NULL; i++) {
        if (strcmp(aotx_tui_actions[i].screen, name) == 0
            && strcmp(aotx_tui_actions[i].key, token) == 0) {
            return aotx_tui_actions[i].line;
        }
    }
    return NULL;
}

int aotx_tui_join(char *out, size_t bytes, const char *dir, const char *name)
{
    size_t dir_bytes = strlen(dir);
    size_t name_bytes = strlen(name);
    if (bytes == 0) {
        return -1;
    }
    out[0] = '\0';
    if (dir_bytes + 1u + name_bytes + 1u > bytes) {
        return -1;
    }
    memcpy(out, dir, dir_bytes);
    out[dir_bytes] = '/';
    memcpy(out + dir_bytes + 1u, name, name_bytes);
    out[dir_bytes + 1u + name_bytes] = '\0';
    return 0;
}

/* Takes the first word of a row, which every screen writes as the argument of its line. */
static void first_word(const char *row, char *out, size_t bytes)
{
    size_t at = 0;
    while (row[at] == ' ') {
        at++;
    }
    out[0] = '\0';
    while (row[at] != '\0' && row[at] != ' ' && (at + 1u) < bytes) {
        out[at] = row[at];
        at++;
        out[at] = '\0';
    }
}

/* Builds a command line from a template. The first part between angle brackets takes the
 * first argument, and the second takes the second. The line the operator sees is thus the
 * line the parser reads. */
static void build_line(const char *pattern, const char *one, const char *two,
                       char *out, size_t bytes)
{
    size_t at = 0;
    unsigned int part = 0;
    size_t i = 0;
    while (pattern[i] != '\0' && at + 1u < bytes) {
        if (pattern[i] == '<') {
            const char *value = (part == 0) ? one : two;
            while (pattern[i] != '\0' && pattern[i] != '>') {
                i++;
            }
            if (pattern[i] == '>') {
                i++;
            }
            part++;
            if (value != NULL) {
                while (*value != '\0' && at + 1u < bytes) {
                    out[at++] = *value++;
                }
            }
            continue;
        }
        out[at++] = pattern[i++];
    }
    out[at] = '\0';
}

void aotx_screen_send(aotx_tui *tui, const char *line)
{
    if (line == NULL || line[0] == '\0') {
        return;
    }
    if (tui->session.fd < 0) {
        snprintf(tui->says, sizeof(tui->says),
                 "no system runs; the line %.120s needs one", line);
        return;
    }
    if (aotx_session_line(&tui->session, line) != 0) {
        snprintf(tui->says, sizeof(tui->says), "the line did not go out: %.150s",
                 tui->session.reason);
        return;
    }
    snprintf(tui->says, sizeof(tui->says), "sent %.200s", line);
}

/* The rows of the small screens, which need no file and no table. */
static unsigned int rows_menu(char *out, unsigned int rows, unsigned int cols)
{
    unsigned int count = 0;
    unsigned int i;
    for (i = 0; aotx_tui_menu[i].label != NULL && count < rows; i++) {
        snprintf(out + (size_t)count * cols, cols, "%s", aotx_tui_menu[i].label);
        count++;
    }
    return count;
}

static unsigned int rows_bus(char *out, unsigned int rows, unsigned int cols)
{
    unsigned int count = 0;
    unsigned int i;
    if (rows > 0u) {
        snprintf(out, cols, "all");
        count++;
    }
    for (i = 0; aotx_tui_bus_kinds[i].word != NULL && count < rows; i++) {
        snprintf(out + (size_t)count * cols, cols, "%s", aotx_tui_bus_kinds[i].word);
        count++;
    }
    return count;
}

static unsigned int rows_quit(const aotx_tui *tui, char *out, unsigned int rows,
                              unsigned int cols)
{
    unsigned int count = 0;
    if (rows == 0) {
        return 0;
    }
    snprintf(out, cols, "quit this terminal");
    count = 1;
    if (count < rows) {
        snprintf(out + (size_t)count * cols, cols,
                 (tui->session.fd >= 0) ? "the system goes on; x on the System screen stops it"
                                        : "no system runs");
        count++;
    }
    return count;
}

/* Reports whether a screen reads the disk to fill its rows. */
static int reads_disk(unsigned int screen)
{
    return (screen == AOTX_SCREEN_MODELS || screen == AOTX_SCREEN_TOOLS
            || screen == AOTX_SCREEN_SKILLS || screen == AOTX_SCREEN_PICKER
            || screen == AOTX_SCREEN_SYSTEM) ? 1 : 0;
}

/* Fills the rows of the open screen. Returns the count. */
static unsigned int screen_rows(aotx_tui *tui, unsigned int rows)
{
    /* The whole table is the object the rows are written into, so the cast
     * names the table and not its first row. */
    char *out = (char *)aotx_screen_rows;
    unsigned int cols = AOTX_TUI_LINE_BYTES;
    if (rows > AOTX_TUI_ROWS_LIST) {
        rows = AOTX_TUI_ROWS_LIST;
    }
    memset(aotx_screen_rows, 0, sizeof(aotx_screen_rows));
    switch (tui->screen) {
    case AOTX_SCREEN_HELP:     return aotx_rows_help(tui, out, rows, cols);
    case AOTX_SCREEN_MENU:     return rows_menu(out, rows, cols);
    case AOTX_SCREEN_AGENTS:   return aotx_rows_agents(tui, out, rows, cols);
    case AOTX_SCREEN_BUS:      return rows_bus(out, rows, cols);
    case AOTX_SCREEN_MODELS:   return aotx_rows_models(tui, out, rows, cols);
    case AOTX_SCREEN_TOOLS:    return aotx_rows_modules(tui, 0, out, rows, cols);
    case AOTX_SCREEN_SKILLS:   return aotx_rows_modules(tui, 1, out, rows, cols);
    case AOTX_SCREEN_SETTINGS: return aotx_rows_settings(tui, out, rows, cols);
    case AOTX_SCREEN_SYSTEM:   return aotx_rows_system(tui, out, rows, cols);
    case AOTX_SCREEN_QUIT:     return rows_quit(tui, out, rows, cols);
    case AOTX_SCREEN_PICKER:   return aotx_rows_picker(tui, out, rows, cols);
    default:                   return 0;
    }
}

/* The line of hints under the rows: every action of the open screen. A hint that does not
 * fit the line stops the write, so the buffer is never passed. */
static void hints(unsigned int screen, char *out, size_t bytes)
{
    const char *name = aotx_screen_name(screen);
    unsigned int i;
    size_t at = 0;
    out[0] = '\0';
    for (i = 0; aotx_tui_actions[i].screen != NULL; i++) {
        const char *line = aotx_tui_actions[i].line;
        int wrote;
        if (strcmp(aotx_tui_actions[i].screen, name) != 0 || line[0] == '\0') {
            continue;
        }
        wrote = snprintf(out + at, bytes - at, "%s%s %s", (at > 0) ? "  " : "",
                         aotx_tui_actions[i].key, line);
        if (wrote < 0 || (size_t)wrote >= bytes - at) {
            return;
        }
        at += (size_t)wrote;
    }
    snprintf(out + at, bytes - at, "%sEsc closes", (at > 0) ? "  " : "");
}

void aotx_screen_draw(aotx_tui *tui, unsigned int top, unsigned int rows)
{
    aotx_paint *p = &tui->paint;
    unsigned int cols = aotx_paint_view_cols(p);
    unsigned int inside;
    unsigned int count;
    unsigned int room;
    unsigned int i;
    uint64_t now;
    char hint[AOTX_TUI_LINE_BYTES];
    char title[64];
    if (rows < 4u || cols < 8u) {
        return;
    }
    aotx_paint_box(p, top, 0u, rows, cols, tui->utf8_box);
    snprintf(title, sizeof(title), " %s ", aotx_screen_name(tui->screen));
    aotx_paint_text(p, top, 2u, title, AOTX_TUI_BRIGHT);
    inside = rows - 3u;  /* the two border rows and the row of hints */
    now = aotx_wall_ns();
    if (tui->rows_screen != tui->screen || reads_disk(tui->screen) == 0
        || now >= tui->rows_ns + ((uint64_t)AOTX_TUI_DISK_MS * 1000000ull)) {
        tui->rows_count = screen_rows(tui, inside + AOTX_TUI_ROWS_LIST);
        tui->rows_screen = tui->screen;
        tui->rows_ns = now;
    }
    count = tui->rows_count;
    if (tui->cursor >= count) {
        tui->cursor = (count > 0) ? count - 1u : 0u;
    }
    if (tui->cursor < tui->top) {
        tui->top = tui->cursor;
    }
    if (tui->cursor >= tui->top + inside) {
        tui->top = tui->cursor - inside + 1u;
    }
    /* The text of a row stops one column short of the border, so a long name never
     * writes over the box. */
    room = (cols > 4u) ? cols - 4u : 1u;
    if (room >= sizeof(hint)) {
        room = (unsigned int)sizeof(hint) - 1u;
    }
    for (i = 0; i < inside; i++) {
        unsigned int index = tui->top + i;
        unsigned int attribute = (index == tui->cursor) ? AOTX_TUI_REVERSE : AOTX_TUI_PLAIN;
        if (index >= count) {
            break;
        }
        snprintf(hint, sizeof(hint), "%.*s", (int)room, aotx_screen_rows[index]);
        aotx_paint_fill(p, top + 1u + i, 1u, cols - 2u, (unsigned int)' ', attribute);
        aotx_paint_text(p, top + 1u + i, 1u, hint, attribute);
    }
    if (tui->editing != 0) {
        snprintf(hint, sizeof(hint), "value: %.*s", (int)room, tui->edit);
        aotx_paint_fill(p, top + rows - 2u, 1u, cols - 2u, (unsigned int)' ',
                        AOTX_TUI_BRIGHT);
        aotx_paint_text(p, top + rows - 2u, 1u, hint, AOTX_TUI_BRIGHT);
        p->cursor_row = top + rows - 2u;
        p->cursor_col = 1u + 7u + tui->edit_fill;
        p->cursor_on = 1;
        return;
    }
    hints(tui->screen, hint, sizeof(hint));
    hint[room] = '\0';
    aotx_paint_text(p, top + rows - 2u, 1u, hint, AOTX_TUI_DIM);
}

/* Moves the cursor of the open screen. Returns 1 when the key moved it. */
static int move_cursor(aotx_tui *tui, const aotx_tui_key *key)
{
    unsigned int page = (tui->paint.rows > 6u) ? tui->paint.rows - 6u : 1u;
    switch (key->code) {
    case AOTX_TUI_KEY_UP:
        if (tui->cursor > 0) {
            tui->cursor--;
        }
        return 1;
    case AOTX_TUI_KEY_DOWN:
        tui->cursor++;
        return 1;
    case AOTX_TUI_KEY_PAGE_UP:
        tui->cursor = (tui->cursor > page) ? tui->cursor - page : 0u;
        return 1;
    case AOTX_TUI_KEY_PAGE_DN:
        tui->cursor += page;
        return 1;
    case AOTX_TUI_KEY_HOME:
        tui->cursor = 0;
        return 1;
    case AOTX_TUI_KEY_END:
        tui->cursor = AOTX_TUI_ROWS_LIST - 1u;
        return 1;
    default:
        return 0;
    }
}

/* Takes one key while a value is edited. Returns 1 when the key was taken. */
static int edit_key(aotx_tui *tui, const aotx_tui_key *key)
{
    if (key->code == AOTX_TUI_KEY_ESCAPE) {
        tui->editing = 0;
        tui->edit_fill = 0;
        tui->edit[0] = '\0';
        return 1;
    }
    if (key->code == AOTX_TUI_KEY_BACK) {
        if (tui->edit_fill > 0) {
            tui->edit[--tui->edit_fill] = '\0';
        }
        return 1;
    }
    if (key->code == AOTX_TUI_KEY_ENTER) {
        return 0;  /* the screen commits the value */
    }
    if (key->code == 0 && key->codepoint >= 32u && key->codepoint < 127u
        && tui->edit_fill + 1u < sizeof(tui->edit)) {
        tui->edit[tui->edit_fill++] = (char)key->codepoint;
        tui->edit[tui->edit_fill] = '\0';
        return 1;
    }
    return 1;
}

/* The action of the Enter key or a character key, with the argument of the row. */
static int take_action(aotx_tui *tui, const aotx_tui_key *key)
{
    const char *pattern = aotx_screen_action(tui->screen, key);
    char argument[AOTX_TUI_LINE_BYTES];
    char line[AOTX_TUI_LINE_BYTES];
    char path[AOTX_PATH_BYTES];
    if (pattern == NULL) {
        return 0;
    }
    first_word(aotx_screen_rows[tui->cursor], argument, sizeof(argument));
    switch (tui->screen) {
    case AOTX_SCREEN_AGENTS:
        if ((key->codepoint == (unsigned int)'y' || key->codepoint == (unsigned int)'n')
            && (argument[0] < '0' || argument[0] > '9')) {
            snprintf(tui->says, sizeof(tui->says),
                     "put the cursor on a request that waits");
            return 1;
        }
        break;
    case AOTX_SCREEN_MENU:
        if (key->code == AOTX_TUI_KEY_ENTER) {
            const aotx_tui_menu_row *row = &aotx_tui_menu[tui->cursor];
            unsigned int screen = 0u;
            if (row->panel[0] != '\0') {
                unsigned int panel;
                for (panel = 0u; panel < AOTX_MIRROR_PANELS; panel++) {
                    if (strcmp(tui->shot.head.panel[panel].name, row->panel) == 0) {
                        aotx_paint_panel(&tui->paint, &tui->shot, panel);
                        break;
                    }
                }
                tui->screen = AOTX_TUI_SCREEN_NONE;
            } else {
                while (aotx_tui_screens[screen].name != NULL
                       && strcmp(aotx_tui_screens[screen].name, row->screen) != 0) {
                    screen++;
                }
                tui->screen = (aotx_tui_screens[screen].name != NULL)
                              ? screen : AOTX_TUI_SCREEN_NONE;
            }
            tui->cursor = 0;
            tui->top = 0;
        }
        return 1;
    case AOTX_SCREEN_BUS:
        if (strcmp(argument, "all") == 0) {
            aotx_screen_send(tui, "bus");
            return 1;
        }
        break;
    case AOTX_SCREEN_QUIT:
        tui->quit = 1;
        return 1;
    case AOTX_SCREEN_SETTINGS:
        if (key->code == AOTX_TUI_KEY_ENTER && tui->editing == 0) {
            tui->editing = 1;
            tui->edit_fill = 0;
            tui->edit[0] = '\0';
            return 1;
        }
        if (key->code == AOTX_TUI_KEY_ENTER) {
            char reason[AOTX_SETTINGS_REASON_BYTES];
            tui->editing = 0;
            if (tui->session.fd >= 0) {
                build_line(pattern, argument, tui->edit, line, sizeof(line));
                aotx_screen_send(tui, line);
                return 1;
            }
            /* The value is judged before the file takes it, so a file never holds a
             * value that the reader refuses. */
            if (aotx_settings_set(argument, tui->edit, &tui->settings, reason) != 0) {
                snprintf(tui->says, sizeof(tui->says), "%.60s does not take %.40s: %.100s",
                         argument, tui->edit, reason);
                return 1;
            }
            if (aotx_settings_write_key(tui->settings_path, argument, tui->edit,
                                        reason) != 0) {
                snprintf(tui->says, sizeof(tui->says),
                         "the file does not take %.60s: %.120s", argument, reason);
                return 1;
            }
            snprintf(tui->says, sizeof(tui->says), "%.60s = %.60s written to %.100s",
                     argument, tui->edit, tui->settings_path);
            return 1;
        }
        break;
    case AOTX_SCREEN_TOOLS:
    case AOTX_SCREEN_SKILLS:
        if (key->codepoint == (unsigned int)'p') {
            const char *root = tui->settings.text[AOTX_SET_TOOLS_ROOT];
            if (root[0] == '\0') {
                root = tui->settings.text[AOTX_SET_MODULES_DIR];
            }
            aotx_picker_open(tui, root);
            tui->screen = AOTX_SCREEN_PICKER;
            return 1;
        }
        if (key->code == AOTX_TUI_KEY_ENTER) {
            if (aotx_module_path(tui->cursor, path, sizeof(path)) != 0) {
                snprintf(tui->says, sizeof(tui->says), "this row names no module directory");
                return 1;
            }
            build_line(pattern, path, NULL, line, sizeof(line));
            aotx_screen_send(tui, line);
            return 1;
        }
        break;
    case AOTX_SCREEN_PICKER:
        if (key->code == AOTX_TUI_KEY_ENTER) {
            if (aotx_picker_enter(tui, path, sizeof(path)) != 0) {
                build_line(pattern, path, NULL, line, sizeof(line));
                aotx_screen_send(tui, line);
            }
            return 1;
        }
        break;
    case AOTX_SCREEN_SYSTEM:
        if (key->code == AOTX_TUI_KEY_ENTER) {
            return 0;  /* the System screen holds its own keys */
        }
        break;
    default:
        break;
    }
    build_line(pattern, argument, NULL, line, sizeof(line));
    aotx_screen_send(tui, line);
    return 1;
}

int aotx_screen_key(aotx_tui *tui, const aotx_tui_key *key)
{
    if (tui->screen == AOTX_TUI_SCREEN_NONE) {
        return 0;
    }
    if (tui->editing != 0 && edit_key(tui, key) != 0) {
        return 1;
    }
    if (key->code == AOTX_TUI_KEY_ESCAPE) {
        tui->screen = AOTX_TUI_SCREEN_NONE;
        tui->cursor = 0;
        tui->top = 0;
        return 1;
    }
    if (move_cursor(tui, key) != 0) {
        return 1;
    }
    if (tui->screen == AOTX_SCREEN_SYSTEM && aotx_screen_system_key(tui, key) != 0) {
        return 1;
    }
    return take_action(tui, key);
}
