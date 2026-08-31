/* Purpose: Check the console commands that read the catalog and take a module out.
 * Owns: The module texts of the command check.
 * Threading: One thread; the command line check calls these one at a time.
 * Lifetime: The program.
 *
 * The file is a part of the command line check. It reads the helpers of that check, so it
 * comes after them in the same translation unit. */
#ifndef AOTX_TEST_CLI_MODULES_H
#define AOTX_TEST_CLI_MODULES_H

/* The tick commit node writes the records of the remove lines; this stands in for it. */
__global__ void aotx_test_commit_modules(void)
{
    if (threadIdx.x == 0u && blockIdx.x == 0u) {
        aotx_catalog_commit(aotx_time_tick);
    }
}

/* The line the console has put in last. A case reads it before a command, and then looks
 * at the lines that came after it alone. */
static unsigned long long aotx_test_console_mark(void)
{
    aotx_console_state *console = (aotx_console_state *)malloc(sizeof *console);
    aotx_test_console_state(console);
    unsigned long long at = console->count;
    free(console);
    return at;
}

/* Report whether a line the console put in after the mark holds the text. */
static int aotx_test_console_since(unsigned long long mark, const char *text, int want)
{
    aotx_console_state *console = (aotx_console_state *)malloc(sizeof *console);
    int found = 0;
    aotx_test_console_state(console);
    unsigned long long first = mark + 1ull;
    if (console->count > AOTX_CONSOLE_LINES
        && first < console->count - AOTX_CONSOLE_LINES + 1ull) {
        first = console->count - AOTX_CONSOLE_LINES + 1ull;
    }
    for (unsigned long long at = first; at <= console->count && found == 0; ++at) {
        const aotx_console_line *line = aotx_test_at(console, at);
        if (line == NULL || line->length == 0u) {
            continue;
        }
        char held[AOTX_CONSOLE_COLS + 1u];
        unsigned int span = (line->length < AOTX_CONSOLE_COLS) ? line->length
                                                               : AOTX_CONSOLE_COLS;
        memcpy(held, line->text, span);
        held[span] = '\0';
        found = (strstr(held, text) != NULL) ? 1 : 0;
    }
    if (found != want) {
        printf("cli: the lines of the command %s hold '%s'\n",
               (found != 0) ? "still" : "do not", text);
    }
    free(console);
    return (found == want) ? 1 : 0;
}

/* Count the lines the console put in after the mark that hold the text. */
static unsigned int aotx_test_console_count(unsigned long long mark, const char *text)
{
    aotx_console_state *console = (aotx_console_state *)malloc(sizeof *console);
    unsigned int found = 0u;
    aotx_test_console_state(console);
    unsigned long long first = mark + 1ull;
    if (console->count > AOTX_CONSOLE_LINES
        && first < console->count - AOTX_CONSOLE_LINES + 1ull) {
        first = console->count - AOTX_CONSOLE_LINES + 1ull;
    }
    for (unsigned long long at = first; at <= console->count; ++at) {
        const aotx_console_line *line = aotx_test_at(console, at);
        if (line == NULL || line->length == 0u) {
            continue;
        }
        char held[AOTX_CONSOLE_COLS + 1u];
        unsigned int span = (line->length < AOTX_CONSOLE_COLS) ? line->length
                                                               : AOTX_CONSOLE_COLS;
        memcpy(held, line->text, span);
        held[span] = '\0';
        found += (strstr(held, text) != NULL) ? 1u : 0u;
    }
    free(console);
    return found;
}

/* The catalog commands: the lists, one module in full, the remove and the import line. */
static void aotx_test_modules_commands(void)
{
    aotx_test_module module;
    unsigned long long mark = 0ull;

    /* One skill of the check, put in with no ring, as the setup of the check does. */
    aotx_test_module_text(&module, AOTX_MODULE_SKILL, "table_notes",
                          "kind: skill\nname: table_notes\nversion: 3\n"
                          "description: How to read a table.\n",
                          "Read the head of the table first.\nThen read the rows.");
    aotx_test_import_direct(&module, 201u);
    aotx_test_module_free(&module);

    mark = aotx_test_console_mark();
    aotx_test_one("modules");
    aotx_test_check(aotx_test_console_since(mark, "modules: name kind state version", 1),
                    "the modules command names its columns");
    aotx_test_check(aotx_test_console_since(mark, "memory_recall tool installed", 1)
                    && aotx_test_console_since(mark, "table_notes skill installed", 1),
                    "the modules command shows a built-in tool and an imported skill");
    aotx_test_check(aotx_test_console_since(mark, "arena: bytes", 1),
                    "the modules command states the bytes of the arena");

    mark = aotx_test_console_mark();
    aotx_test_one("modules skill");
    aotx_test_check(aotx_test_console_since(mark, "table_notes skill installed", 1)
                    && aotx_test_console_since(mark, "memory_recall tool installed", 0),
                    "the modules command of one kind shows that kind alone");

    aotx_test_one("modules wibble");
    aotx_test_check(aotx_test_last_says("modules: the kind is not known; give skill, role "
                                        "or tool"),
                    "a kind the catalog does not know is refused with the three kinds");

    mark = aotx_test_console_mark();
    aotx_test_one("skills");
    aotx_test_check(aotx_test_console_since(mark, "table_notes skill installed", 1)
                    && aotx_test_console_since(mark, "conductor role installed", 0),
                    "the skills command shows the skills of the catalog");
    mark = aotx_test_console_mark();
    aotx_test_one("tools");
    aotx_test_check(aotx_test_console_since(mark, "skill_use tool installed", 1),
                    "the tools command shows the tools of the catalog");
    mark = aotx_test_console_mark();
    aotx_test_one("roles");
    aotx_test_check(aotx_test_console_since(mark, "conductor role installed", 1),
                    "the roles command shows the roles of the catalog");

    mark = aotx_test_console_mark();
    aotx_test_one("module table_notes");
    aotx_test_check(aotx_test_console_since(mark, "description: How to read a table.", 1)
                    && aotx_test_console_since(mark, "| Read the head of the table first.",
                                               1),
                    "the module command shows the manifest and the first lines of a body");
    mark = aotx_test_console_mark();
    aotx_test_one("module fs_read");
    aotx_test_check(aotx_test_console_since(mark, "side: host", 1),
                    "the module command shows the manifest of a built-in tool");
    aotx_test_one("module wibble");
    aotx_test_check(aotx_test_last_says("module: the name is not in the catalog"),
                    "a module command of a name the catalog does not hold is refused");
    aotx_test_one("module");
    aotx_test_check(aotx_test_last_says("module: give the name of one module"),
                    "a module command with no name is refused");

    mark = aotx_test_console_mark();
    aotx_test_one("import modules/roles/worker");
    aotx_test_check(aotx_test_console_count(mark, "import:") == 0u,
                    "an accepted import request states no result before it arrives");
    char want[128];
    snprintf(want, sizeof want, "import: give a path of 1 to %u bytes to a module "
             "directory", (unsigned int)AOTX_TOOL_ARG_BYTES);
    aotx_test_one("import");
    aotx_test_check(aotx_test_last_says(want),
                    "an import line with no path is refused with the bound");
    aotx_test_one("import /a/path refused: the file is not there");
    aotx_test_check(aotx_test_last_says("import: the directory is not readable: the file is "
                                        "not there"),
                    "the report of a refused import states its reason");

    aotx_test_one("remove fs_read");
    aotx_test_check(aotx_test_last_says("remove: fs_read: a built-in tool cannot be removed"),
                    "a remove of a built-in tool is refused");
    aotx_test_one("remove wibble");
    aotx_test_check(aotx_test_last_says("remove: wibble: the module name is not in the catalog"),
                    "a remove of a name the catalog does not hold is refused");
    aotx_test_one("remove");
    aotx_test_check(aotx_test_last_says("remove: give the name of one module"),
                    "a remove with no name is refused");

    aotx_test_one("remove table_notes");
    aotx_test_check(aotx_test_last_says("remove: table_notes removal is pending"),
                    "a remove line remains pending until the tick commit");
    aotx_test_check(aotx_test_catalog_entry("table_notes", AOTX_MODULE_SKILL)
                    < AOTX_MODULE_SLOTS,
                    "the module is still in the catalog before the commit");
    aotx_test_commit_modules<<<1, 1>>>();
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_test_check(aotx_test_catalog_entry("table_notes", AOTX_MODULE_SKILL)
                    >= AOTX_MODULE_SLOTS,
                    "the commit of the tick removes the module");
    aotx_test_check(aotx_test_last_says("remove: table_notes is out of the catalog"),
                    "the commit states that the module is out");

    mark = aotx_test_console_mark();
    aotx_test_one("help");
    /* The help writes the lines of its switch and the line of its default, and no line
     * twice. A count that is over the lines of the switch repeats the default. */
    aotx_test_check(aotx_test_console_count(mark, "stop the run") == 1u,
                    "the help writes the line of its default one time");
    aotx_test_check(aotx_test_console_since(mark, "modules [kind]", 1)
                    && aotx_test_console_since(mark, "module <name>", 1)
                    && aotx_test_console_since(mark, "remove <name>", 1)
                    && aotx_test_console_since(mark, "import <path>", 1),
                    "the help names the commands of the catalog");
    printf("cli: the six catalog commands and the import line answered on the console\n");
}

#endif
