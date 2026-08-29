/* Purpose: Check that the feeder runs a host tool module as a program and answers with it.
 * Owns: One temporary tree with module directories, one ring and one feeder for each case.
 * Threading: Two processes; the check reads the ring while the feeder writes it.
 * Lifetime: The run of the program. */
#include "disk/feed/modules.h"
#include "tests/tool_harness.h"

#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

/* The seconds a built-in command may run. A module takes the figure of its manifest. */
#define AOTX_CASE_TIMEOUT "5"

/* The modules the tree holds, in name order. The order gives each module its import number
 * and thus the tool number the ring carries. */
#define AOTX_MODULE_COUNT 7

static char **arguments;

/* Writes one module directory: the manifest, the program, and a marker file the program
 * reads to prove the working directory. */
static void make_module(const char *dir, const char *name, const char *body,
                        unsigned timeout, const char *authorize)
{
    char path[640];
    char text[768];
    int n;
    snprintf(path, sizeof(path), "%s/%s", dir, name);
    CHECK(aotx_make_dir(path) == 0, "the module directory %s does not open", name);
    n = snprintf(text, sizeof(text),
                 "kind: tool\nname: %s\nversion: 1\n"
                 "description: One check of the host tool contract.\n"
                 "side: host\narguments: text\nauthorise: %s\ntimeout: %u\n"
                 "program: run.sh\nexample: text=one\n",
                 name, authorize, timeout);
    snprintf(path, sizeof(path), "%s/%s/module.manifest", dir, name);
    th_file(path, text, (size_t)n);
    n = snprintf(text, sizeof(text), "#!/bin/sh\n%s\n", body);
    snprintf(path, sizeof(path), "%s/%s/run.sh", dir, name);
    th_file(path, text, (size_t)n);
    CHECK(chmod(path, 0755) == 0, "the program of %s does not take the run bits", name);
    snprintf(path, sizeof(path), "%s/%s/marker.txt", dir, name);
    snprintf(text, sizeof(text), "the marker of %s", name);
    th_file(path, text, strlen(text));
}

/* Makes every module of the tree. */
static void make_tree(const char *dir)
{
    CHECK(aotx_make_dir(dir) == 0, "the module tree does not open");
    make_module(dir, "big_tool", "head -c 5000 /dev/zero | tr '\\0' z", 10u, "never");
    make_module(dir, "count_tool", "echo row $AOTX_REQUEST", 10u, "never");
    make_module(dir, "cwd_tool", "cat marker.txt", 10u, "never");
    make_module(dir, "echo_line", "head -c 24 /dev/stdin", 10u, "never");
    make_module(dir, "env_tool", "echo $AOTX_REQUEST $AOTX_AGENT $AOTX_TOOL", 10u, "always");
    make_module(dir, "fail_tool", "echo the cause 1>&2; exit 7", 10u, "never");
    make_module(dir, "slow_tool", "sleep 60", 1u, "never");
}

/* Waits until the module table holds the count of lines, and reads the file back. Returns
 * the count of lines the file holds. */
static int table_lines(const char *dir, int want, char *out, size_t bytes)
{
    char path[512];
    uint64_t deadline = aotx_wall_ns() + AOTX_TH_WAIT_NS;
    snprintf(path, sizeof(path), "%s/%s", dir, AOTX_MODULE_TABLE);
    for (;;) {
        uint64_t backoff = 0;
        int lines = 0;
        long got = th_read(path, out, bytes);
        long i;
        for (i = 0; i < got; i++) {
            if (out[i] == '\n') {
                lines++;
            }
        }
        if (lines >= want || aotx_wall_ns() >= deadline) {
            return lines;
        }
        aotx_pause(&backoff);
    }
}

/* Gives the tool number of one module from the table file, or zero. */
static uint32_t number_of(const char *table, const char *name)
{
    char want[128];
    const char *at;
    uint64_t number = 0;
    snprintf(want, sizeof(want), "\"name\":\"%s\"", name);
    at = strstr(table, want);
    if (at == NULL || !aotx_json_number(at, "\"number\":", &number)) {
        return 0;
    }
    return (uint32_t)number;
}

/* ---- the programs of the catalog ---- */

static void programs(int n)
{
    th_run t;
    char mods[400];
    char table[8192];
    char arg[AOTX_TH_ARG_TEXT];
    char want[768];
    uint64_t start_ns;
    int i;

    th_tree(&t);
    snprintf(mods, sizeof(mods), "%s/mods", t.dir);
    make_tree(mods);
    th_spawn(&t, arguments[1], AOTX_CASE_TIMEOUT, mods, -1);
    CHECK(table_lines(t.dir, AOTX_MODULE_COUNT, table, sizeof(table)) == AOTX_MODULE_COUNT,
          "the module table holds %d rows and %d were asked for",
          table_lines(t.dir, 0, table, sizeof(table)), AOTX_MODULE_COUNT);
    /* Every row states the directory as an absolute path, the program and the timeout. */
    snprintf(want, sizeof(want), "\"dir\":\"%s/count_tool\"", mods);
    CHECK(strstr(table, want) != NULL, "the table names no absolute directory");
    CHECK(strstr(table, "\"program\":\"run.sh\"") != NULL, "the table names no program");
    CHECK(strstr(table, "\"timeout\":1,") != NULL, "the table holds no timeout of one second");
    CHECK(strstr(table, "\"authorize\":\"always\"") != NULL,
          "the table states no tool that the operator must permit");
    CHECK(number_of(table, "count_tool") >= AOTX_TOOL_MODULE_BASE,
          "the table gives the tool number %u", number_of(table, "count_tool"));

    /* Every request of the group carries an identity of its own. The group runs at one
     * time, so a reply that names the wrong request shows. */
    for (i = 0; i < n; i++) {
        th_one_arg(arg, sizeof(arg), "text", "one");
        th_request(&t, (uint32_t)(900 + i), (uint32_t)(i % 64), "count_tool",
                   number_of(table, "count_tool"), arg);
    }
    /* The group runs at one time and fills every place of the table at N of the slots.
     * The cases that follow thus wait for it. */
    th_wait(&t, n);
    th_request(&t, 1000u, 3u, "echo_line", number_of(table, "echo_line"), "one");
    th_request(&t, 1001u, 4u, "env_tool", number_of(table, "env_tool"), "one");
    th_request(&t, 1002u, 5u, "fail_tool", number_of(table, "fail_tool"), "one");
    th_request(&t, 1003u, 6u, "cwd_tool", number_of(table, "cwd_tool"), "one");
    th_request(&t, 1004u, 7u, "big_tool", number_of(table, "big_tool"), "one");
    /* A tool that the line names by its number alone. The drain writes such a line for a
     * tool of the catalog. */
    th_request(&t, 1005u, 8u, "module", number_of(table, "count_tool"), "one");
    /* A name that no module of the run holds. */
    th_request(&t, 1006u, 9u, "no_such_tool", AOTX_TOOL_MODULE_BASE + 99u, "one");
    start_ns = aotx_wall_ns();
    th_request(&t, 1007u, 10u, "slow_tool", number_of(table, "slow_tool"), "one");
    th_wait(&t, n + 8);

    for (i = 0; i < n; i++) {
        th_reply *e = th_entry((uint32_t)(900 + i));
        snprintf(want, sizeof(want), "row %d\n", 900 + i);
        CHECK(e->status == AOTX_TOOL_OK, "the program %d gives the status %u", i, e->status);
        CHECK(e->agent == (uint32_t)(i % 64), "the program %d names the agent %u", i,
              e->agent);
        CHECK(strcmp(e->bytes, want) == 0, "the program %d wrote %s", i, e->bytes);
    }
    CHECK(strncmp(th_entry(1000u)->bytes, "{\"request\":1000", 15) == 0,
          "the standard input of the program holds no requests line: %s",
          th_entry(1000u)->bytes);
    CHECK(strcmp(th_entry(1001u)->bytes, "1001 4 env_tool\n") == 0,
          "the environment of the program reads %s", th_entry(1001u)->bytes);
    CHECK(th_entry(1002u)->status == AOTX_TOOL_ERROR, "a program that fails gives %u",
          th_entry(1002u)->status);
    CHECK(strstr(th_entry(1002u)->reason, "status 7") != NULL,
          "the reason states no exit status: %s", th_entry(1002u)->reason);
    CHECK(strstr(th_entry(1002u)->reason, "the cause") != NULL,
          "the reason holds no error output: %s", th_entry(1002u)->reason);
    CHECK(strcmp(th_entry(1003u)->bytes, "the marker of cwd_tool") == 0,
          "the working directory is not the module directory: %s", th_entry(1003u)->bytes);
    CHECK(th_entry(1004u)->len == AOTX_FS_CAP, "the long output gives %u bytes",
          th_entry(1004u)->len);
    CHECK(strstr(th_entry(1004u)->reason, "longer than the cap") != NULL,
          "the cut reason reads %s", th_entry(1004u)->reason);
    CHECK(strcmp(th_entry(1005u)->bytes, "row 1005\n") == 0,
          "a tool named by its number alone wrote %s", th_entry(1005u)->bytes);
    CHECK(th_entry(1006u)->status == AOTX_TOOL_ERROR, "a name of no module gives %u",
          th_entry(1006u)->status);
    CHECK(strstr(th_entry(1006u)->reason, "not a host tool") != NULL,
          "the reason reads %s", th_entry(1006u)->reason);
    CHECK(th_entry(1007u)->status == AOTX_TOOL_ERROR, "a program that never ends gives %u",
          th_entry(1007u)->status);
    CHECK(strstr(th_entry(1007u)->reason, "timeout of 1 seconds") != NULL,
          "the reason reads %s", th_entry(1007u)->reason);
    /* The timeout of the manifest ends the program, and not the timeout of a built-in
     * tool, which this run set to five seconds. */
    CHECK(aotx_wall_ns() - start_ns < 4000000000ull,
          "the program of a timeout of one second ran for %llu milliseconds",
          (unsigned long long)((aotx_wall_ns() - start_ns) / 1000000ull));
    printf("programs %d: replies %d, table rows %d\n", n, th_collected.count,
           AOTX_MODULE_COUNT);
    th_stop(&t);
}

/* ---- a feeder that follows another over one journal ---- */

/* The table on disk lets a second feeder run the programs of the run it continues. It
 * also keeps two modules of one journal from taking one number. */
static void again(int n)
{
    th_run t;
    char mods[400];
    char table[8192];
    char path[512];
    char line[1024];
    char want[128];
    int pipe_fds[2];
    uint32_t first;
    int i;

    th_tree(&t);
    snprintf(mods, sizeof(mods), "%s/mods", t.dir);
    make_tree(mods);
    th_spawn(&t, arguments[1], AOTX_CASE_TIMEOUT, mods, -1);
    CHECK(table_lines(t.dir, AOTX_MODULE_COUNT, table, sizeof(table)) == AOTX_MODULE_COUNT,
          "the first feeder wrote no whole table");
    first = number_of(table, "count_tool");
    th_close(&t);

    /* The second feeder gets no module directory, as a feeder after a restore does. The
     * journal replays the imports to the device and the table gives the feeder the
     * programs. */
    CHECK(pipe(pipe_fds) == 0, "the line pipe does not open");
    th_spawn(&t, arguments[1], AOTX_CASE_TIMEOUT, NULL, pipe_fds[0]);
    close(pipe_fds[0]);
    for (i = 0; i < n; i++) {
        th_request(&t, (uint32_t)(1100 + i), (uint32_t)(i % 64), "count_tool", first, "one");
    }
    /* The group fills every place of the table at N of the slots, so the module that comes
     * later waits for it. */
    th_wait(&t, n);
    /* One more module, imported by a line of the standard input. Its number goes on from
     * the highest the table holds. */
    snprintf(path, sizeof(path), "%s/later_tool", mods);
    make_module(mods, "later_tool", "echo the later tool", 10u, "never");
    snprintf(line, sizeof(line), "import %s\n", path);
    CHECK(write(pipe_fds[1], line, strlen(line)) == (ssize_t)strlen(line),
          "the import line does not write");
    CHECK(table_lines(t.dir, AOTX_MODULE_COUNT + 1, table, sizeof(table)) ==
          AOTX_MODULE_COUNT + 1, "the second feeder wrote no row for the later module");
    snprintf(want, sizeof(want), "\"import\":%u,\"number\":%u}",
             (unsigned)(AOTX_MODULE_COUNT + 1),
             (unsigned)(AOTX_TOOL_MODULE_BASE + AOTX_MODULE_COUNT + 1));
    CHECK(strstr(table, want) != NULL,
          "the later module took a number the table already held");
    th_request(&t, 1200u, 1u, "later_tool", number_of(table, "later_tool"), "one");
    th_wait(&t, n + 1);

    for (i = 0; i < n; i++) {
        th_reply *e = th_entry((uint32_t)(1100 + i));
        snprintf(want, sizeof(want), "row %d\n", 1100 + i);
        CHECK(e->status == AOTX_TOOL_OK, "the program %d of the second feeder gives %u", i,
              e->status);
        CHECK(strcmp(e->bytes, want) == 0, "the program %d of the second feeder wrote %s",
              i, e->bytes);
    }
    CHECK(strcmp(th_entry(1200u)->bytes, "the later tool\n") == 0,
          "the module of the second feeder gives the status %u and the content %s",
          th_entry(1200u)->status, th_entry(1200u)->bytes);
    close(pipe_fds[1]);
    printf("again %d: replies %d, first number %u\n", n, th_collected.count, first);
    th_stop(&t);
}

int main(int argc, char **argv)
{
    arguments = argv;
    if (argc < 2) {
        printf("usage: aotx_run_tool_test <feeder>\n");
        return 1;
    }
    programs(1);
    programs(64);
    again(1);
    again(64);
    return aotx_report("run_tool_test", 200);
}
