/* Purpose: Check the table of host tool programs that the feeder keeps beside the journal.
 * Owns: One temporary tree of module directories and one inbound ring for each case.
 * Threading: Two processes; the check reads the ring while the feeder writes it.
 * Lifetime: The run of one case. */
#ifndef AOTX_TESTS_FEED_MODULES_H
#define AOTX_TESTS_FEED_MODULES_H

#include "disk/feed/modules.h"

/* The bytes of the table file that a case reads back. */
#define AOTX_TABLE_TEXT 32768

static char table_text[AOTX_TABLE_TEXT];

/* Builds one module directory of a host tool. Half of them name a timeout of their own.
 * A third of them ask the operator. A row with the wrong figure thus shows. */
static void tool_module(const char *root, int i)
{
    char path[512];
    char file[768];
    char text[512];
    int len;
    snprintf(path, sizeof(path), "%s/a%02d", root, i);
    module_dir(path);
    len = snprintf(text, sizeof(text),
                   "kind: tool\nname: a%02d\nversion: 1\n"
                   "description: One host tool of the check.\n"
                   "side: host\narguments: text\nauthorise: %s\nprogram: run.sh\n%s",
                   i, ((i % 3) == 0) ? "always" : "never",
                   ((i % 2) == 0) ? "timeout: 5\n" : "");
    snprintf(file, sizeof(file), "%s/%s", path, AOTX_IMPORT_MANIFEST);
    module_file(file, text, (size_t)len);
    len = snprintf(text, sizeof(text), "#!/bin/sh\necho tool %d\n", i);
    snprintf(file, sizeof(file), "%s/run.sh", path);
    module_file(file, text, (size_t)len);
}

/* Builds the modules that must add no row: two skills and one device tool. */
static void other_modules(const char *root)
{
    char path[512];
    char file[768];
    char text[256];
    int len;
    int i;
    for (i = 0; i < 2; i++) {
        snprintf(path, sizeof(path), "%s/z%d", root, i);
        module_dir(path);
        len = snprintf(text, sizeof(text), "kind: skill\nname: z%d\nbody: skill.txt\n", i);
        snprintf(file, sizeof(file), "%s/%s", path, AOTX_IMPORT_MANIFEST);
        module_file(file, text, (size_t)len);
        snprintf(file, sizeof(file), "%s/skill.txt", path);
        module_file(file, "the body of a skill", 19);
    }
    snprintf(path, sizeof(path), "%s/y0", root);
    module_dir(path);
    len = snprintf(text, sizeof(text),
                   "kind: tool\nname: y0\nside: device\nmodule: tool.ptx\n"
                   "entry: aotx_tool_y0\n");
    snprintf(file, sizeof(file), "%s/%s", path, AOTX_IMPORT_MANIFEST);
    module_file(file, text, (size_t)len);
    snprintf(file, sizeof(file), "%s/tool.ptx", path);
    module_file(file, ".version 8.0\n", 13);
}

/* Consumes every slot the feeder published, so the ring cannot fill. */
static void table_drain(aotx_inbound_ring *ring)
{
    uint64_t consumed = aotx_inbound_consumed(ring);
    uint64_t head = aotx_inbound_head(ring);
    while (consumed < head) {
        consumed++;
        aotx_store_release(&ring->pre->consumed, consumed);
    }
}

/* Waits until the table file holds the count of lines and reads it back. */
static int table_wait(const char *dir, int want)
{
    char path[512];
    uint64_t deadline = aotx_wall_ns() + AOTX_WAIT_NS;
    snprintf(path, sizeof(path), "%s/%s", dir, AOTX_MODULE_TABLE);
    for (;;) {
        uint64_t backoff = 0;
        int lines = 0;
        int fd = open(path, O_RDONLY);
        ssize_t got = 0;
        int i;
        table_text[0] = '\0';
        if (fd >= 0) {
            got = read(fd, table_text, sizeof(table_text) - 1u);
            close(fd);
        }
        table_text[(got > 0) ? (size_t)got : 0] = '\0';
        for (i = 0; i < (int)got; i++) {
            if (table_text[i] == '\n') {
                lines++;
            }
        }
        if (lines >= want || aotx_wall_ns() >= deadline) {
            return lines;
        }
        aotx_pause(&backoff);
    }
}

/* Runs one feeder over the tree of modules and gives back the count of table lines. */
static int table_run(const char *dir, const char *root, const char *mods,
                     const char *requests, int want)
{
    aotx_map map;
    aotx_inbound_ring ring;
    char fd_text[16];
    char *args[10];
    int child;
    int lines;
    uint64_t deadline;
    CHECK(aotx_inbound_create(256u, &map, &ring) == 0, "the ring does not open");
    snprintf(fd_text, sizeof(fd_text), "%d", map.fd);
    args[0] = arguments[1];
    args[1] = (char *)"--inbound-fd";
    args[2] = fd_text;
    args[3] = (char *)"--root";
    args[4] = (char *)root;
    args[5] = (char *)"--requests";
    args[6] = (char *)requests;
    args[7] = (char *)"--modules";
    args[8] = (char *)mods;
    args[9] = NULL;
    child = aotx_spawn(args, -1, -1);
    CHECK(child > 0, "the feeder does not start");
    /* The ring must not fill while the imports go out, so the check consumes as it waits. */
    deadline = aotx_wall_ns() + AOTX_WAIT_NS;
    for (;;) {
        uint64_t backoff = 0;
        table_drain(&ring);
        lines = table_wait(dir, 0);
        if (lines >= want || aotx_wall_ns() >= deadline) {
            break;
        }
        aotx_pause(&backoff);
    }
    table_drain(&ring);
    aotx_store_release16(&ring.pre->closed, 1);
    CHECK(aotx_wait(child) == 0, "the feeder does not end with a clean status");
    aotx_map_release(&map);
    return lines;
}

/* One import of a host tool writes one line of the table. A skill and a device tool write
 * none, because the feeder runs no program for them. A second feeder reads the file from
 * its first byte and gives every further import a number that the file does not hold. */
static void table_arm(int n)
{
    char dir[256];
    char root[320];
    char mods[400];
    char requests[400];
    char want[768];
    int lines;
    int i;

    CHECK(aotx_temp_dir(dir, sizeof(dir)) == 0, "the temporary directory does not open");
    snprintf(root, sizeof(root), "%s/root", dir);
    CHECK(aotx_make_dir(root) == 0, "the root does not open");
    snprintf(mods, sizeof(mods), "%s/mods", dir);
    module_dir(mods);
    snprintf(requests, sizeof(requests), "%s/requests.jsonl", dir);
    for (i = 0; i < n; i++) {
        tool_module(mods, i);
    }
    other_modules(mods);

    lines = table_run(dir, root, mods, requests, n);
    CHECK(lines == n, "the table holds %d lines and %d were asked for", lines, n);
    for (i = 0; i < n; i++) {
        snprintf(want, sizeof(want),
                 "{\"name\":\"a%02d\",\"kind\":\"tool\",\"side\":\"host\",\"dir\":\"%s/a%02d\","
                 "\"program\":\"run.sh\",\"timeout\":%u,\"authorize\":\"%s\","
                 "\"import\":%d,\"number\":%d}",
                 i, mods, i, ((i % 2) == 0) ? 5u : (unsigned)AOTX_MODULE_TIMEOUT,
                 ((i % 3) == 0) ? "always" : "never", i + 1,
                 (int)AOTX_TOOL_MODULE_BASE + i + 1);
        CHECK(strstr(table_text, want) != NULL, "the table holds no row %s", want);
    }
    CHECK(strstr(table_text, "\"name\":\"z0\"") == NULL, "a skill wrote a row of the table");
    CHECK(strstr(table_text, "\"name\":\"y0\"") == NULL,
          "a device tool wrote a row of the table");

    /* The second feeder reads the file first. Every import of it thus takes a number above
     * every number the file holds, and no two modules of one journal share a number. */
    lines = table_run(dir, root, mods, requests, 2 * n);
    CHECK(lines == 2 * n, "the second run left %d lines and %d were asked for", lines, 2 * n);
    for (i = 0; i < n; i++) {
        snprintf(want, sizeof(want), "\"name\":\"a%02d\",\"kind\":\"tool\",\"side\":\"host\","
                 "\"dir\":\"%s/a%02d\",\"program\":\"run.sh\",\"timeout\":%u,"
                 "\"authorize\":\"%s\",\"import\":%d,\"number\":%d}",
                 i, mods, i, ((i % 2) == 0) ? 5u : (unsigned)AOTX_MODULE_TIMEOUT,
                 ((i % 3) == 0) ? "always" : "never", n + i + 1,
                 (int)AOTX_TOOL_MODULE_BASE + n + i + 1);
        CHECK(strstr(table_text, want) != NULL,
              "the second run holds no row of the number %d",
              (int)AOTX_TOOL_MODULE_BASE + n + i + 1);
    }
    printf("table %d: lines %d\n", n, lines);
    aotx_remove_tree(dir);
}

#endif
