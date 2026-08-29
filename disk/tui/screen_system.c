/* Purpose: Show the start form of the System screen and the entries of the picker.
 * Owns: The entries of the directory that the picker shows.
 * Threading: One thread; the screens read the disk and start one child.
 * Lifetime: One frame for the rows; the entries last until the next fill. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "disk/tui/tui.h"

#include <dirent.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>

/* The lines of the boot that the screen shows under the form. */
#define AOTX_SYSTEM_LOG_ROWS 8u

/* The entries of the directory the picker shows, in row order. A directory carries the
 * mark, so the enter knows to go into it and not to take it. */
static char aotx_picker_names[AOTX_TUI_ROWS_LIST][256];
static unsigned char aotx_picker_kind[AOTX_TUI_ROWS_LIST];
static unsigned int aotx_picker_count;

#define AOTX_PICKER_FILE   0u
#define AOTX_PICKER_DIR    1u
#define AOTX_PICKER_MODULE 2u

unsigned int aotx_rows_system(aotx_tui *tui, char *out, unsigned int rows,
                              unsigned int cols)
{
    static char log[AOTX_SYSTEM_LOG_ROWS][AOTX_TUI_LINE_BYTES];
    unsigned int count = 0;
    unsigned int lines;
    unsigned int i;
    int status = 0;
    int state;
    if (rows == 0) {
        return 0;
    }
    snprintf(out, cols, "- Enter starts a system, r restores the newest journal, x stops");
    count = 1;
    if (count < rows) {
        /* The text comes from the state, which read it one time at the start. The screen
         * is drawn at the rate of the frames, so a program started here would be started
         * many times a second. */
        snprintf(out + (size_t)count * cols, cols, "- build   %s", tui->version);
        count++;
    }
    if (count < rows) {
        snprintf(out + (size_t)count * cols, cols, "- program %s", tui->program);
        count++;
    }
    if (count < rows) {
        snprintf(out + (size_t)count * cols, cols, "- settings %s", tui->settings_path);
        count++;
    }
    if (count < rows) {
        snprintf(out + (size_t)count * cols, cols, "- journal %s", tui->journal);
        count++;
    }
    if (count < rows) {
        snprintf(out + (size_t)count * cols, cols, "- models  %s, roles %s",
                 tui->settings.text[AOTX_SET_MODELS_DIR],
                 (tui->settings.text[AOTX_SET_MODELS_ROLES][0] != '\0')
                 ? tui->settings.text[AOTX_SET_MODELS_ROLES] : "the roles of the profile");
        count++;
    }
    if (count < rows) {
        snprintf(out + (size_t)count * cols, cols, "- modules %s",
                 tui->settings.text[AOTX_SET_MODULES_DIR]);
        count++;
    }
    if (count < rows) {
        snprintf(out + (size_t)count * cols, cols, "- root    %s",
                 (tui->settings.text[AOTX_SET_TOOLS_ROOT][0] != '\0')
                 ? tui->settings.text[AOTX_SET_TOOLS_ROOT] : "none");
        count++;
    }
    if (count < rows) {
        snprintf(out + (size_t)count * cols, cols, "- window  %s",
                 (tui->settings.number[AOTX_SET_WINDOW_ON] != 0) ? "on" : "off");
        count++;
    }
    state = aotx_session_boot_state(&tui->session, &status);
    if (count < rows) {
        snprintf(out + (size_t)count * cols, cols, "- state   %s",
                 (tui->session.fd >= 0) ? "a system is attached"
                 : (state > 0) ? "the boot of this terminal runs"
                 : (state == 0) ? "the boot of this terminal ended" : "no system runs");
        count++;
    }
    if (state == 0 && count < rows) {
        snprintf(out + (size_t)count * cols, cols, "- the boot ended with the status %d",
                 status);
        count++;
    }
    /* The last lines of the boot go under the form, so a start that fails states its
     * reason where the key was pressed. */
    lines = aotx_session_boot_log(&tui->session, (char *)log, AOTX_SYSTEM_LOG_ROWS,
                                  AOTX_TUI_LINE_BYTES);
    for (i = 0; i < lines && count < rows; i++) {
        snprintf(out + (size_t)count * cols, cols, "- %s", log[i]);
        count++;
    }
    return count;
}

int aotx_screen_system_key(aotx_tui *tui, const aotx_tui_key *key)
{
    int restore;
    if (key->code != AOTX_TUI_KEY_ENTER
        && !(key->code == 0 && key->codepoint == (unsigned int)'r')) {
        return 0;
    }
    if (tui->session.fd >= 0) {
        snprintf(tui->says, sizeof(tui->says),
                 "a system is attached already; x stops it");
        return 1;
    }
    restore = (key->code == 0) ? 1 : 0;
    if (aotx_session_start(&tui->session, tui->program, tui->settings_path, tui->journal,
                           restore) != 0) {
        snprintf(tui->says, sizeof(tui->says), "the start did not happen: %s",
                 tui->session.reason);
        return 1;
    }
    snprintf(tui->says, sizeof(tui->says), "the boot runs; the terminal waits to attach");
    snprintf(tui->state, sizeof(tui->state), "starting the system");
    return 1;
}

void aotx_picker_open(aotx_tui *tui, const char *dir)
{
    snprintf(tui->picker_dir, sizeof(tui->picker_dir), "%s", dir);
    tui->cursor = 0;
    tui->top = 0;
    /* The directory changed, so the rows of the last fill name another place. */
    tui->rows_ns = 0;
    tui->rows_screen = AOTX_TUI_SCREEN_NONE;
}

/* Reports whether a directory is a module directory, which holds its own manifest. */
static int is_module(const char *path)
{
    char manifest[AOTX_PATH_BYTES];
    struct stat state;
    if (aotx_tui_join(manifest, sizeof(manifest), path, "module.manifest") != 0) {
        return 0;
    }
    return (stat(manifest, &state) == 0 && S_ISREG(state.st_mode)) ? 1 : 0;
}

unsigned int aotx_rows_picker(aotx_tui *tui, char *out, unsigned int rows,
                              unsigned int cols)
{
    DIR *open_dir;
    struct dirent *entry;
    unsigned int count = 0;
    memset(aotx_picker_names, 0, sizeof(aotx_picker_names));
    memset(aotx_picker_kind, 0, sizeof(aotx_picker_kind));
    aotx_picker_count = 0;
    if (rows == 0) {
        return 0;
    }
    if (tui->picker_dir[0] == '\0') {
        snprintf(tui->picker_dir, sizeof(tui->picker_dir), "%s",
                 tui->settings.text[AOTX_SET_MODULES_DIR]);
    }
    open_dir = opendir(tui->picker_dir);
    if (open_dir == NULL) {
        snprintf(out, cols, "- the directory %s does not open", tui->picker_dir);
        return 1;
    }
    snprintf(aotx_picker_names[count], sizeof(aotx_picker_names[count]), "..");
    aotx_picker_kind[count] = (unsigned char)AOTX_PICKER_DIR;
    snprintf(out + (size_t)count * cols, cols, ".. %s", tui->picker_dir);
    count++;
    while ((entry = readdir(open_dir)) != NULL && count < rows
           && count < AOTX_TUI_ROWS_LIST) {
        char path[AOTX_PATH_BYTES];
        struct stat state;
        unsigned int kind;
        if (entry->d_name[0] == '.') {
            continue;
        }
        if (aotx_tui_join(path, sizeof(path), tui->picker_dir, entry->d_name) != 0) {
            continue;
        }
        if (stat(path, &state) != 0) {
            continue;
        }
        if (S_ISDIR(state.st_mode) != 0) {
            kind = is_module(path) ? AOTX_PICKER_MODULE : AOTX_PICKER_DIR;
        } else {
            kind = AOTX_PICKER_FILE;
        }
        snprintf(aotx_picker_names[count], sizeof(aotx_picker_names[count]), "%s",
                 entry->d_name);
        aotx_picker_kind[count] = (unsigned char)kind;
        snprintf(out + (size_t)count * cols, cols, "%s %s", entry->d_name,
                 (kind == AOTX_PICKER_MODULE) ? "a module directory"
                 : (kind == AOTX_PICKER_DIR) ? "a directory" : "a file");
        count++;
    }
    closedir(open_dir);
    aotx_picker_count = count;
    return count;
}

int aotx_picker_enter(aotx_tui *tui, char *out, size_t bytes)
{
    unsigned int at = tui->cursor;
    char path[AOTX_PATH_BYTES];
    if (at >= aotx_picker_count) {
        return 0;
    }
    if (strcmp(aotx_picker_names[at], "..") == 0) {
        char *end;
        snprintf(path, sizeof(path), "%s", tui->picker_dir);
        end = strrchr(path, '/');
        if (end != NULL && end != path) {
            *end = '\0';
        } else {
            snprintf(path, sizeof(path), "%s", (path[0] == '/') ? "/" : "..");
        }
        aotx_picker_open(tui, path);
        return 0;
    }
    if (aotx_tui_join(path, sizeof(path), tui->picker_dir,
                      aotx_picker_names[at]) != 0) {
        return 0;
    }
    if (aotx_picker_kind[at] == (unsigned char)AOTX_PICKER_MODULE) {
        snprintf(out, bytes, "%s", path);
        return 1;
    }
    if (aotx_picker_kind[at] == (unsigned char)AOTX_PICKER_DIR) {
        aotx_picker_open(tui, path);
        return 0;
    }
    snprintf(tui->says, sizeof(tui->says), "a module is a directory, not a file");
    return 0;
}
