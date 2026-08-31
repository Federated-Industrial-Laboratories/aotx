/* Purpose: Read the command line of the boot program and write its options.
 * Owns: Nothing; the caller holds the options.
 * Launch shape: Host glue only; no kernels.
 * Lifetime: The start of the program. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "boot/boot.cuh"
#include "disk/settings/settings.h"

void aotx_boot_usage(void)
{
    printf("aotx_boot --journal <dir> [--models <dir>] [--roles <list>] [--restore]\n");
    printf("          [--window] [--tui] [--tui-attached] [--ticks <n>]\n");
    printf("          [--workload <n>] [--blocks <n>]\n");
    printf("          [--records <n>] [--derive <list>] [--root <dir>] [--solo]\n");
    printf("          [--modules <dir>]\n");
    printf("          [--settings <file>] [--clock-only] [--version]\n");
    printf("  --journal    the directory the journal goes in\n");
    printf("  --models     the directory the model files are in\n");
    printf("  --roles      roles of the model file list to load, with commas between\n");
    printf("  --root       the one directory a file read tool may reach\n");
    printf("  --modules    the directory of module directories to import at the start\n");
    printf("  --restore    replay the journal before the first input\n");
    printf("  --window     show the panels in a window on the display\n");
    printf("  --tui        start the terminal program beside the system\n");
    printf("  --tui-attached a terminal started this run and is attached already\n");
    printf("  --ticks      run this many ticks, then stop; zero runs on\n");
    printf("  --workload   records the tick load writes for each tick\n");
    printf("  --blocks     blocks of the tick load\n");
    printf("  --records    stop a run that has no tick count at this record count\n");
    printf("  --derive     types the drain makes lines from, with commas between them\n");
    printf("  --settings   the settings file; the default is beside the journal\n");
    printf("  --solo       run with no disk side programs\n");
    printf("  --version    write the version, the profile and the build, then stop\n");
    printf("  --clock-only run the clock module check and stop\n");
}

int aotx_boot_parse(int argc, char **argv, aotx_boot_options *options)
{
    memset(options, 0, sizeof *options);
    options->records = 1000000ull;
    options->blocks = 64u;
    for (int i = 1; i < argc; ++i) {
        int last = (i + 1 >= argc);
        if (strcmp(argv[i], "--journal") == 0 && !last) {
            options->journal = argv[++i];
        } else if (strcmp(argv[i], "--models") == 0 && !last) {
            options->models = argv[++i];
        } else if (strcmp(argv[i], "--roles") == 0 && !last) {
            options->roles = argv[++i];
        } else if (strcmp(argv[i], "--root") == 0 && !last) {
            options->root = argv[++i];
        } else if (strcmp(argv[i], "--modules") == 0 && !last) {
            options->modules = argv[++i];
        } else if (strcmp(argv[i], "--settings") == 0 && !last) {
            options->settings = argv[++i];
        } else if (strcmp(argv[i], "--ticks") == 0 && !last) {
            options->ticks = strtoull(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--workload") == 0 && !last) {
            options->workload = strtoull(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--records") == 0 && !last) {
            options->records = strtoull(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--derive") == 0 && !last) {
            options->derive = argv[++i];
        } else if (strcmp(argv[i], "--blocks") == 0 && !last) {
            options->blocks = (unsigned int)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--restore") == 0) {
            options->restore = 1;
        } else if (strcmp(argv[i], "--window") == 0) {
            options->window = 1;
        } else if (strcmp(argv[i], "--tui") == 0) {
            options->tui = 1;
        } else if (strcmp(argv[i], "--tui-attached") == 0) {
            options->tui_attached = 1;
        } else if (strcmp(argv[i], "--solo") == 0) {
            options->solo = 1;
        } else if (strcmp(argv[i], "--version") == 0) {
            options->version = 1;
        } else if (strcmp(argv[i], "--clock-only") == 0) {
            options->clock_only = 1;
        } else {
            fprintf(stderr, "the option %s is not known\n", argv[i]);
            aotx_boot_usage();
            return 2;
        }
    }
    return 0;
}


/* Take one text of the table when the file gave it and the command line did not. An empty
 * text is the default of the program that reads it, so it is not taken. */
static void aotx_boot_take_text(const char **option, const aotx_settings *table,
                                unsigned int index)
{
    if (*option != NULL || table->text_given[index] == 0u || table->text[index][0] == '\0') {
        return;
    }
    *option = table->text[index];
}

int aotx_boot_settings(aotx_boot_options *options, struct aotx_settings *table,
                       char *path, unsigned int bytes)
{
    aotx_settings *hold = (aotx_settings *)table;
    int named = (options->settings != NULL) ? 1 : 0;
    if (options->settings != NULL) {
        snprintf(path, bytes, "%s", options->settings);
    } else if (options->journal != NULL) {
        snprintf(path, bytes, "%s/../%s", options->journal, AOTX_SETTINGS_FILE_DEFAULT);
    } else {
        snprintf(path, bytes, "%s", AOTX_SETTINGS_FILE_DEFAULT);
    }
    int state = aotx_settings_read(path, hold);
    for (unsigned int i = 0u; i < hold->refused_count && i < AOTX_SETTINGS_REFUSALS; ++i) {
        fprintf(stderr, "settings: line %u: %s\n", hold->refused[i].line,
                hold->refused[i].reason);
    }
    if (state == 2) {
        return 1;
    }
    /* A default file can be absent. An operator who names a file asks for that file, so
     * the system must not start from defaults when the name is absent. */
    if (named != 0 && access(path, F_OK) != 0) {
        return 2;
    }

    /* A command line option wins over the file for the same key. */
    aotx_boot_take_text(&options->journal, hold, AOTX_SET_JOURNAL_DIR);
    aotx_boot_take_text(&options->models, hold, AOTX_SET_MODELS_DIR);
    aotx_boot_take_text(&options->roles, hold, AOTX_SET_MODELS_ROLES);
    aotx_boot_take_text(&options->root, hold, AOTX_SET_TOOLS_ROOT);
    aotx_boot_take_text(&options->modules, hold, AOTX_SET_MODULES_DIR);
    /* The modules of the repository stand behind the build. A run that names no
     * directory imports them, so the roles of the run are there. */
    if (options->modules == NULL) {
        options->modules = AOTX_MODULES_DIR;
    }
    aotx_boot_take_text(&options->derive, hold, AOTX_SET_DERIVE_LIST);
    if (options->window == 0 && hold->number_given[AOTX_SET_WINDOW_ON] != 0u) {
        options->window = (hold->number[AOTX_SET_WINDOW_ON] != 0) ? 1 : 0;
    }
    /* The terminal surface: the boot starts the terminal program beside it and the
     * program attaches to the mirror. */
    if (options->tui == 0 && hold->number_given[AOTX_SET_TUI_ON] != 0u) {
        options->tui = (hold->number[AOTX_SET_TUI_ON] != 0) ? 1 : 0;
    }
    return 0;
}
