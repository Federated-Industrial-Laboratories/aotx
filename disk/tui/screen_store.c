/* Purpose: Fill the rows of the Tools and Skills screens.
 * Owns: The paths of the module directories that the last fill found.
 * Threading: One thread; the screens read the disk and change nothing on it.
 * Lifetime: One frame for the rows; the paths last until the next fill. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "disk/tui/tui.h"

#include <dirent.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>

/* The manifest of a module directory. */
#define AOTX_MODULE_MANIFEST "module.manifest"

/* The paths of the module directories of the last fill, in row order. */
static char aotx_module_paths[AOTX_TUI_ROWS_LIST][AOTX_PATH_BYTES];
static unsigned int aotx_module_count;

int aotx_module_path(unsigned int index, char *out, size_t bytes)
{
    if (index >= aotx_module_count || aotx_module_paths[index][0] == '\0') {
        return -1;
    }
    snprintf(out, bytes, "%s", aotx_module_paths[index]);
    return 0;
}

/* Reads one field of a module manifest. Returns 1 when the field is there. */
static int module_field(const char *path, const char *key, char *out, size_t bytes)
{
    char line[AOTX_MANIFEST_LINE_BYTES];
    FILE *file = fopen(path, "r");
    size_t key_bytes = strlen(key);
    int found = 0;
    out[0] = '\0';
    if (file == NULL) {
        return 0;
    }
    while (fgets(line, (int)sizeof(line), file) != NULL) {
        size_t at = 0;
        size_t end;
        if (strncmp(line, key, key_bytes) != 0 || line[key_bytes] != ':') {
            continue;
        }
        at = key_bytes + 1u;
        while (line[at] == ' ') {
            at++;
        }
        end = strlen(line);
        while (end > at && (line[end - 1u] == '\n' || line[end - 1u] == '\r'
                            || line[end - 1u] == ' ')) {
            end--;
        }
        snprintf(out, bytes, "%.*s", (int)(end - at), line + at);
        found = 1;
        break;
    }
    fclose(file);
    return found;
}

/* The state of a module of a running system, from the catalog table of the mirror. */
static const char *module_state(const aotx_tui *tui, const char *name, const char **reason)
{
    unsigned int i;
    *reason = "";
    if (tui->have_shot == 0) {
        return "no system runs";
    }
    for (i = 0; i < AOTX_MIRROR_MODULE_ROWS; i++) {
        const aotx_mirror_module_row *row = &tui->shot.tables.module[i];
        if (row->name[0] == '\0' || strncmp(row->name, name, AOTX_MIRROR_TEXT_BYTES) != 0) {
            continue;
        }
        if (row->state == 0u) {
            *reason = row->reason;
            return "refused";
        }
        return "installed";
    }
    return "not imported";
}

/* Adds one module directory to the rows when its kind is the one the screen shows. */
static unsigned int take_module(aotx_tui *tui, const char *path, int skills, char *out,
                                unsigned int count, unsigned int rows, unsigned int cols)
{
    char manifest[AOTX_PATH_BYTES];
    char kind[64];
    char name[128];
    const char *reason;
    const char *state;
    if (aotx_tui_join(manifest, sizeof(manifest), path, AOTX_MODULE_MANIFEST) != 0) {
        return count;
    }
    if (module_field(manifest, "kind", kind, sizeof(kind)) == 0) {
        return count;
    }
    if (skills != 0 && strcmp(kind, "skill") != 0) {
        return count;
    }
    if (skills == 0 && strcmp(kind, "skill") == 0) {
        return count;
    }
    if (module_field(manifest, "name", name, sizeof(name)) == 0) {
        return count;
    }
    if (count >= rows || count >= AOTX_TUI_ROWS_LIST) {
        return count;
    }
    state = module_state(tui, name, &reason);
    snprintf(out + (size_t)count * cols, cols, "%s %s %s%s%s", name, kind, state,
             (reason[0] != '\0') ? ", " : "", reason);
    snprintf(aotx_module_paths[count], sizeof(aotx_module_paths[count]), "%s", path);
    return count + 1u;
}

unsigned int aotx_rows_modules(aotx_tui *tui, int skills, char *out, unsigned int rows,
                               unsigned int cols)
{
    const char *dir = tui->settings.text[AOTX_SET_MODULES_DIR];
    DIR *open_dir;
    struct dirent *entry;
    unsigned int count = 0;
    memset(aotx_module_paths, 0, sizeof(aotx_module_paths));
    aotx_module_count = 0;
    if (rows == 0) {
        return 0;
    }
    open_dir = opendir(dir);
    if (open_dir == NULL) {
        snprintf(out, cols, "- the directory %s does not open", dir);
        return 1;
    }
    /* A module directory holds its manifest. A directory that holds none is a directory of
     * module directories, and the walk goes one level into it. */
    while ((entry = readdir(open_dir)) != NULL && count < rows) {
        char path[AOTX_PATH_BYTES];
        struct stat state;
        unsigned int before = count;
        if (entry->d_name[0] == '.') {
            continue;
        }
        if (aotx_tui_join(path, sizeof(path), dir, entry->d_name) != 0) {
            continue;
        }
        if (stat(path, &state) != 0 || S_ISDIR(state.st_mode) == 0) {
            continue;
        }
        count = take_module(tui, path, skills, out, count, rows, cols);
        if (count != before) {
            continue;
        }
        {
            DIR *inner = opendir(path);
            struct dirent *at;
            if (inner == NULL) {
                continue;
            }
            while ((at = readdir(inner)) != NULL && count < rows) {
                char deep[AOTX_PATH_BYTES];
                struct stat inner_state;
                if (at->d_name[0] == '.') {
                    continue;
                }
                if (aotx_tui_join(deep, sizeof(deep), path, at->d_name) != 0) {
                    continue;
                }
                if (stat(deep, &inner_state) != 0 || S_ISDIR(inner_state.st_mode) == 0) {
                    continue;
                }
                count = take_module(tui, deep, skills, out, count, rows, cols);
            }
            closedir(inner);
        }
    }
    closedir(open_dir);
    aotx_module_count = count;
    if (count == 0 && rows > 0) {
        snprintf(out, cols, "- %s holds no module of this kind", dir);
        return 1;
    }
    return count;
}
