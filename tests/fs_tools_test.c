/* Purpose: Check the four built-in host tools of the feeder over a fixture tree.
 * Owns: One temporary tree, one inbound ring and one feeder for each case.
 * Threading: Two processes; the check reads the ring while the feeder writes it.
 * Lifetime: The run of the program. */
#include "disk/feed/run_tool.h"
#include "tests/tool_harness.h"

#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

/* The seconds a command of a case may run. A case that proves the timeout waits this long
 * and no longer, so the whole program stays short. */
#define AOTX_CASE_TIMEOUT "2"

/* The entries of the directory that proves the cut of the entry table. The count is above
 * the table and the lines are short, so the byte cap is not reached first. */
#define AOTX_MANY_ENTRIES 300

/* The bytes one line of a listing may take, with the name in it. */
#define AOTX_LIST_LINE 320

/* The programs the feeder must run at one time, which is the request slot count of the
 * reference profile. The figure stands here and not as the constant under check. */
#define AOTX_FULL_PROGRAMS 64

static char **arguments;

/* ---- fs_list ---- */

/* Reports whether the lines of a listing are in name order. */
static int in_order(const char *text)
{
    char last[AOTX_LIST_LINE];
    const char *at = text;
    last[0] = '\0';
    while (*at != '\0') {
        char name[AOTX_LIST_LINE];
        const char *space = strchr(at, ' ');
        const char *end = strchr(at, '\n');
        size_t take;
        if (space == NULL || end == NULL || space > end) {
            return 0;
        }
        take = (size_t)(space - at);
        if (take >= sizeof(name)) {
            return 0;
        }
        memcpy(name, at, take);
        name[take] = '\0';
        if (last[0] != '\0' && strcmp(last, name) > 0) {
            return 0;
        }
        snprintf(last, sizeof(last), "%s", name);
        at = end + 1;
    }
    return 1;
}

static void list_case(int n)
{
    th_run t;
    char path[512];
    char arg[AOTX_TH_ARG_TEXT];
    char want[128];
    int i;

    th_tree(&t);
    th_spawn(&t, arguments[1], AOTX_CASE_TIMEOUT, NULL, -1);
    snprintf(path, sizeof(path), "%s/list", t.root);
    CHECK(aotx_make_dir(path) == 0, "the listed directory does not open");
    snprintf(path, sizeof(path), "%s/list/sub", t.root);
    CHECK(aotx_make_dir(path) == 0, "the directory under it does not open");
    /* Every file holds a length of its own, so a line that names the wrong file shows. */
    for (i = 0; i < n; i++) {
        char content[128];
        int len = i + 1;
        memset(content, 'a' + (i % 26), (size_t)len);
        snprintf(path, sizeof(path), "%s/list/entry-%03d.txt", t.root, i);
        th_file(path, content, (size_t)len);
    }
    snprintf(path, sizeof(path), "%s/list/link", t.root);
    CHECK(symlink("../../outside.txt", path) == 0, "the link does not open");

    /* A directory with more entries than the table holds, and lines short enough that the
     * byte cap is not the cut. */
    snprintf(path, sizeof(path), "%s/many", t.root);
    CHECK(aotx_make_dir(path) == 0, "the full directory does not open");
    for (i = 0; i < AOTX_MANY_ENTRIES; i++) {
        snprintf(path, sizeof(path), "%s/many/%c%c", t.root, 'a' + (i / 26), 'a' + (i % 26));
        th_file(path, "", 0);
    }

    th_one_arg(arg, sizeof(arg), "path", "list");
    th_request(&t, 100u, 7u, "fs_list", AOTX_TOOL_FS_LIST, arg);
    th_one_arg(arg, sizeof(arg), "path", "many");
    th_request(&t, 101u, 8u, "fs_list", AOTX_TOOL_FS_LIST, arg);
    /* The one value shape of a call with one argument still reads as the path. */
    th_request(&t, 102u, 9u, "fs_list", AOTX_TOOL_FS_LIST, "list");
    /* Every refusal of the walk, over the tool that lists. */
    th_request(&t, 110u, 1u, "fs_list", AOTX_TOOL_FS_LIST, "../");
    th_request(&t, 111u, 1u, "fs_list", AOTX_TOOL_FS_LIST, "/etc");
    th_request(&t, 112u, 1u, "fs_list", AOTX_TOOL_FS_LIST, "list/link");
    th_request(&t, 113u, 1u, "fs_list", AOTX_TOOL_FS_LIST, "gone");
    th_request(&t, 114u, 1u, "fs_list", AOTX_TOOL_FS_LIST, "list/entry-000.txt");
    th_one_arg(arg, sizeof(arg), "folder", "list");
    th_request(&t, 115u, 1u, "fs_list", AOTX_TOOL_FS_LIST, arg);
    th_wait(&t, 9);

    {
        th_reply *whole = th_entry(100u);
        th_reply *many = th_entry(101u);
        th_reply *bare = th_entry(102u);
        CHECK(whole->status == AOTX_TOOL_OK, "the listing gives the status %u", whole->status);
        CHECK(whole->agent == 7u, "the listing names the agent %u", whole->agent);
        CHECK(in_order(whole->bytes) == 1, "the listing is not in name order");
        for (i = 0; i < n; i++) {
            snprintf(want, sizeof(want), "entry-%03d.txt file %d\n", i, i + 1);
            CHECK(strstr(whole->bytes, want) != NULL, "the listing holds no line %s", want);
        }
        CHECK(strstr(whole->bytes, "sub directory ") != NULL,
              "the listing names no directory");
        CHECK(strstr(whole->bytes, "link link ") != NULL, "the listing names no link");
        CHECK(strstr(whole->bytes, " .\n") == NULL && strstr(whole->bytes, " ..\n") == NULL,
              "the listing names the directory or the one above it");
        CHECK(bare->status == AOTX_TOOL_OK && bare->len == whole->len,
              "the one value shape gives another listing");
        CHECK(many->status == AOTX_TOOL_ERROR, "a full directory gives the status %u",
              many->status);
        CHECK(strstr(many->reason, "more entries") != NULL, "the cut reason reads %s",
              many->reason);
        CHECK(many->content > 0, "a cut listing gives no line of content");
    }
    {
        CHECK(th_entry(110u)->status == AOTX_TOOL_REFUSED, "a path of two dots is not refused");
        CHECK(strstr(th_entry(110u)->reason, "two dots") != NULL, "the reason reads %s",
              th_entry(110u)->reason);
        CHECK(th_entry(111u)->status == AOTX_TOOL_REFUSED, "an absolute path is not refused");
        CHECK(th_entry(112u)->status == AOTX_TOOL_REFUSED, "a link is not refused");
        CHECK(strstr(th_entry(112u)->reason, "symbolic link") != NULL, "the reason reads %s",
              th_entry(112u)->reason);
        CHECK(th_entry(113u)->status == AOTX_TOOL_ERROR, "a directory that is not there"
              " gives %u", th_entry(113u)->status);
        CHECK(th_entry(114u)->status == AOTX_TOOL_ERROR, "a file that is not a directory"
              " gives %u", th_entry(114u)->status);
        CHECK(th_entry(115u)->status == AOTX_TOOL_REFUSED, "an unknown key is not refused");
        CHECK(strstr(th_entry(115u)->reason, "folder") != NULL,
              "the reason of an unknown key names no key: %s", th_entry(115u)->reason);
    }
    printf("list %d: replies %d, listing bytes %u\n", n, th_collected.count,
           th_entry(100u)->len);
    th_stop(&t);
}

/* ---- fs_write ---- */

static void write_case(int n)
{
    th_run t;
    char path[512];
    char arg[AOTX_TH_ARG_TEXT];
    char text[256];
    char got[AOTX_FS_CAP];
    const char *keys[2];
    const char *values[2];
    int i;

    keys[0] = "path";
    keys[1] = "text";
    th_tree(&t);
    th_spawn(&t, arguments[1], AOTX_CASE_TIMEOUT, NULL, -1);
    snprintf(path, sizeof(path), "%s/sub", t.root);
    CHECK(aotx_make_dir(path) == 0, "the directory does not open");
    snprintf(path, sizeof(path), "%s/outside.txt", t.dir);
    th_file(path, "the file outside the root", 25);
    snprintf(path, sizeof(path), "%s/wlink", t.root);
    CHECK(symlink("../outside.txt", path) == 0, "the link does not open");
    snprintf(path, sizeof(path), "%s/held.txt", t.root);
    th_file(path, "the text that stands", 20);

    /* Every write carries content of its own, so a reply that names the wrong request
     * shows in the file it left. */
    for (i = 0; i < n; i++) {
        char name[64];
        snprintf(name, sizeof(name), "written-%03d.txt", i);
        snprintf(text, sizeof(text), "the content of the file %d", i);
        values[0] = name;
        values[1] = text;
        th_args(arg, sizeof(arg), keys, values, 2);
        th_request(&t, (uint32_t)(200 + i), (uint32_t)(i % 64), "fs_write",
                   AOTX_TOOL_FS_WRITE, arg);
    }
    /* A file that stands is replaced whole. */
    values[0] = "held.txt";
    values[1] = "the text that came after";
    th_args(arg, sizeof(arg), keys, values, 2);
    th_request(&t, 300u, 1u, "fs_write", AOTX_TOOL_FS_WRITE, arg);
    /* A file under a directory of the root, which proves the walk of a whole path. */
    values[0] = "sub/deep.txt";
    values[1] = "the text under a directory";
    th_args(arg, sizeof(arg), keys, values, 2);
    th_request(&t, 301u, 1u, "fs_write", AOTX_TOOL_FS_WRITE, arg);
    /* The refusals. */
    values[0] = "sub";
    values[1] = "text";
    th_args(arg, sizeof(arg), keys, values, 2);
    th_request(&t, 310u, 1u, "fs_write", AOTX_TOOL_FS_WRITE, arg);
    values[0] = "gone/x.txt";
    th_args(arg, sizeof(arg), keys, values, 2);
    th_request(&t, 311u, 1u, "fs_write", AOTX_TOOL_FS_WRITE, arg);
    values[0] = "wlink";
    th_args(arg, sizeof(arg), keys, values, 2);
    th_request(&t, 312u, 1u, "fs_write", AOTX_TOOL_FS_WRITE, arg);
    values[0] = "../outside.txt";
    th_args(arg, sizeof(arg), keys, values, 2);
    th_request(&t, 313u, 1u, "fs_write", AOTX_TOOL_FS_WRITE, arg);
    values[0] = "/tmp/x.txt";
    th_args(arg, sizeof(arg), keys, values, 2);
    th_request(&t, 314u, 1u, "fs_write", AOTX_TOOL_FS_WRITE, arg);
    /* A call with no text key names a file and no content. */
    th_one_arg(arg, sizeof(arg), "path", "no-text.txt");
    th_request(&t, 315u, 1u, "fs_write", AOTX_TOOL_FS_WRITE, arg);
    th_wait(&t, n + 8);

    for (i = 0; i < n; i++) {
        th_reply *e = th_entry((uint32_t)(200 + i));
        snprintf(path, sizeof(path), "%s/written-%03d.txt", t.root, i);
        snprintf(text, sizeof(text), "the content of the file %d", i);
        CHECK(e->status == AOTX_TOOL_OK, "the write %d gives the status %u", i, e->status);
        CHECK(th_read(path, got, sizeof(got)) == (long)strlen(text) &&
              strcmp(got, text) == 0, "the file %d holds %s", i, got);
        CHECK(strstr(e->bytes, "bytes") != NULL, "the reply of a write states no count: %s",
              e->bytes);
    }
    snprintf(path, sizeof(path), "%s/held.txt", t.root);
    CHECK(th_read(path, got, sizeof(got)) == 24 && strcmp(got, "the text that came after") == 0,
          "the file that stood holds %s", got);
    snprintf(path, sizeof(path), "%s/sub/deep.txt", t.root);
    CHECK(th_read(path, got, sizeof(got)) == 26, "the file under a directory holds %s", got);
    snprintf(path, sizeof(path), "%s/outside.txt", t.dir);
    CHECK(th_read(path, got, sizeof(got)) == 25,
          "a refused write changed the file outside the root");
    CHECK(th_entry(310u)->status == AOTX_TOOL_REFUSED, "a directory is not refused");
    CHECK(strstr(th_entry(310u)->reason, "directory") != NULL, "the reason reads %s",
          th_entry(310u)->reason);
    CHECK(th_entry(311u)->status != AOTX_TOOL_OK, "a parent that is not there is not refused");
    CHECK(strstr(th_entry(311u)->reason, "not there") != NULL, "the reason reads %s",
          th_entry(311u)->reason);
    CHECK(th_entry(312u)->status == AOTX_TOOL_REFUSED, "a link is not refused");
    CHECK(th_entry(313u)->status == AOTX_TOOL_REFUSED, "a path of two dots is not refused");
    CHECK(th_entry(314u)->status == AOTX_TOOL_REFUSED, "an absolute path is not refused");
    CHECK(th_entry(315u)->status == AOTX_TOOL_REFUSED, "a call with no text is not refused");
    CHECK(strstr(th_entry(315u)->reason, "text") != NULL,
          "the reason names no missing key: %s", th_entry(315u)->reason);
    /* The temporary name is gone: a rename left no file behind it. */
    {
        char line[AOTX_TH_ARG_TEXT];
        th_one_arg(line, sizeof(line), "path", ".");
        th_request(&t, 400u, 1u, "fs_list", AOTX_TOOL_FS_LIST, line);
        th_wait(&t, n + 9);
        CHECK(strstr(th_entry(400u)->bytes, ".aotx-") == NULL,
              "a temporary file stands in the root");
    }
    printf("write %d: replies %d\n", n, th_collected.count);
    th_stop(&t);
}

/* ---- fs_update ---- */

static void update_case(int n)
{
    th_run t;
    char path[512];
    char arg[AOTX_TH_ARG_TEXT];
    char text[256];
    char got[AOTX_FS_CAP];
    const char *keys[3];
    const char *values[3];
    static char big[AOTX_FS_FILE_CAP + 32];
    int i;

    keys[0] = "path";
    keys[1] = "old";
    keys[2] = "new";
    th_tree(&t);
    th_spawn(&t, arguments[1], AOTX_CASE_TIMEOUT, NULL, -1);
    /* Every file holds a run of its own, so a reply that names the wrong file shows. */
    for (i = 0; i < n; i++) {
        snprintf(path, sizeof(path), "%s/update-%03d.txt", t.root, i);
        snprintf(text, sizeof(text), "head old-%03d tail", i);
        th_file(path, text, strlen(text));
    }
    snprintf(path, sizeof(path), "%s/none.txt", t.root);
    th_file(path, "the file holds no such run", 26);
    snprintf(path, sizeof(path), "%s/twice.txt", t.root);
    th_file(path, "one word and one word again", 27);
    memset(big, 'b', sizeof(big));
    snprintf(path, sizeof(path), "%s/big.txt", t.root);
    th_file(path, big, sizeof(big));

    for (i = 0; i < n; i++) {
        char name[64];
        char old_text[32];
        char new_text[32];
        snprintf(name, sizeof(name), "update-%03d.txt", i);
        snprintf(old_text, sizeof(old_text), "old-%03d", i);
        snprintf(new_text, sizeof(new_text), "new-%03d-x", i);
        values[0] = name;
        values[1] = old_text;
        values[2] = new_text;
        th_args(arg, sizeof(arg), keys, values, 3);
        th_request(&t, (uint32_t)(500 + i), (uint32_t)(i % 64), "fs_update",
                   AOTX_TOOL_FS_UPDATE, arg);
    }
    values[0] = "none.txt";
    values[1] = "absent";
    values[2] = "x";
    th_args(arg, sizeof(arg), keys, values, 3);
    th_request(&t, 600u, 1u, "fs_update", AOTX_TOOL_FS_UPDATE, arg);
    values[0] = "twice.txt";
    values[1] = "one word";
    values[2] = "x";
    th_args(arg, sizeof(arg), keys, values, 3);
    th_request(&t, 601u, 1u, "fs_update", AOTX_TOOL_FS_UPDATE, arg);
    values[0] = "big.txt";
    values[1] = "bbbb";
    values[2] = "x";
    th_args(arg, sizeof(arg), keys, values, 3);
    th_request(&t, 602u, 1u, "fs_update", AOTX_TOOL_FS_UPDATE, arg);
    values[0] = "none.txt";
    values[1] = "";
    values[2] = "x";
    th_args(arg, sizeof(arg), keys, values, 3);
    th_request(&t, 603u, 1u, "fs_update", AOTX_TOOL_FS_UPDATE, arg);
    values[0] = "../outside.txt";
    values[1] = "a";
    values[2] = "x";
    th_args(arg, sizeof(arg), keys, values, 3);
    th_request(&t, 604u, 1u, "fs_update", AOTX_TOOL_FS_UPDATE, arg);
    th_wait(&t, n + 5);

    for (i = 0; i < n; i++) {
        th_reply *e = th_entry((uint32_t)(500 + i));
        snprintf(path, sizeof(path), "%s/update-%03d.txt", t.root, i);
        snprintf(text, sizeof(text), "head new-%03d-x tail", i);
        CHECK(e->status == AOTX_TOOL_OK, "the update %d gives the status %u", i, e->status);
        CHECK(th_read(path, got, sizeof(got)) == (long)strlen(text) &&
              strcmp(got, text) == 0, "the file %d holds %s", i, got);
    }
    CHECK(th_entry(600u)->status == AOTX_TOOL_REFUSED, "a run that is not there is not"
          " refused");
    CHECK(strstr(th_entry(600u)->reason, "0 times") != NULL, "the reason reads %s",
          th_entry(600u)->reason);
    CHECK(th_entry(601u)->status == AOTX_TOOL_REFUSED, "a run that is there two times is"
          " not refused");
    CHECK(strstr(th_entry(601u)->reason, "2 times") != NULL, "the reason reads %s",
          th_entry(601u)->reason);
    snprintf(path, sizeof(path), "%s/twice.txt", t.root);
    CHECK(th_read(path, got, sizeof(got)) == 27, "a refused update changed the file");
    CHECK(th_entry(602u)->status != AOTX_TOOL_OK, "a file over the bound is not refused");
    CHECK(strstr(th_entry(602u)->reason, "bound") != NULL, "the reason reads %s",
          th_entry(602u)->reason);
    CHECK(th_entry(603u)->status == AOTX_TOOL_REFUSED, "an old text of no bytes is not"
          " refused");
    /* The reason names the cause. A run count of zero is another cause, and a reason that
     * states it would send an agent to read the file again for nothing. */
    CHECK(strstr(th_entry(603u)->reason, "no byte") != NULL, "the reason reads %s",
          th_entry(603u)->reason);
    CHECK(th_entry(604u)->status == AOTX_TOOL_REFUSED, "a path of two dots is not refused");
    printf("update %d: replies %d\n", n, th_collected.count);
    th_stop(&t);
}

/* ---- run ---- */

static void command_case(int n)
{
    th_run t;
    char path[512];
    char arg[AOTX_TH_ARG_TEXT];
    char want[128];
    static char big[5000];
    int i;

    th_tree(&t);
    th_spawn(&t, arguments[1], AOTX_CASE_TIMEOUT, NULL, -1);
    snprintf(path, sizeof(path), "%s/marker.txt", t.root);
    th_file(path, "the marker of the root", 22);
    memset(big, 'z', sizeof(big));
    snprintf(path, sizeof(path), "%s/big.txt", t.root);
    th_file(path, big, sizeof(big));

    /* Every command writes text of its own, so a reply that names the wrong request
     * shows. The whole group runs at one time. */
    for (i = 0; i < n; i++) {
        char command[128];
        snprintf(command, sizeof(command), "echo out-%03d", i);
        th_one_arg(arg, sizeof(arg), "command", command);
        th_request(&t, (uint32_t)(700 + i), (uint32_t)(i % 64), "run", AOTX_TOOL_RUN, arg);
    }
    /* The group runs at one time and fills every place of the table at N of the slots.
     * The cases that follow thus wait for it. */
    th_wait(&t, n);
    /* The working directory is the allowed root. */
    th_one_arg(arg, sizeof(arg), "command", "cat marker.txt");
    th_request(&t, 800u, 1u, "run", AOTX_TOOL_RUN, arg);
    /* A command that fails states its status and the first bytes of its error output. */
    th_one_arg(arg, sizeof(arg), "command", "echo the cause 1>&2; exit 3");
    th_request(&t, 801u, 1u, "run", AOTX_TOOL_RUN, arg);
    /* Output over the cap states the cut. */
    th_one_arg(arg, sizeof(arg), "command", "cat big.txt");
    th_request(&t, 802u, 1u, "run", AOTX_TOOL_RUN, arg);
    /* A command that never ends is killed at the timeout. */
    th_one_arg(arg, sizeof(arg), "command", "sleep 60");
    th_request(&t, 803u, 1u, "run", AOTX_TOOL_RUN, arg);
    /* The environment of the child names the request, the agent and the tool. */
    th_one_arg(arg, sizeof(arg), "command",
            "echo $AOTX_REQUEST $AOTX_AGENT $AOTX_TOOL; head -c 20 /dev/stdin");
    th_request(&t, 804u, 5u, "run", AOTX_TOOL_RUN, arg);
    th_wait(&t, n + 5);


    for (i = 0; i < n; i++) {
        th_reply *e = th_entry((uint32_t)(700 + i));
        snprintf(want, sizeof(want), "out-%03d\n", i);
        CHECK(e->status == AOTX_TOOL_OK, "the command %d gives the status %u", i, e->status);
        CHECK(e->agent == (uint32_t)(i % 64), "the command %d names the agent %u", i,
              e->agent);
        CHECK(strcmp(e->bytes, want) == 0, "the command %d wrote %s", i, e->bytes);
    }
    CHECK(strcmp(th_entry(800u)->bytes, "the marker of the root") == 0,
          "the working directory is not the root: %s", th_entry(800u)->bytes);
    CHECK(th_entry(801u)->status == AOTX_TOOL_ERROR, "a command that fails gives %u",
          th_entry(801u)->status);
    CHECK(strstr(th_entry(801u)->reason, "status 3") != NULL,
          "the reason states no exit status: %s", th_entry(801u)->reason);
    CHECK(strstr(th_entry(801u)->reason, "the cause") != NULL,
          "the reason holds no error output: %s", th_entry(801u)->reason);
    CHECK(th_entry(802u)->len == AOTX_FS_CAP, "the long output gives %u bytes",
          th_entry(802u)->len);
    CHECK(th_entry(802u)->status == AOTX_TOOL_ERROR, "the cut of an output is not stated");
    CHECK(strstr(th_entry(802u)->reason, "longer than the cap") != NULL,
          "the cut reason reads %s", th_entry(802u)->reason);
    CHECK(th_entry(803u)->status == AOTX_TOOL_ERROR, "a command that never ends gives %u",
          th_entry(803u)->status);
    CHECK(strstr(th_entry(803u)->reason, "timeout") != NULL, "the reason reads %s",
          th_entry(803u)->reason);
    CHECK(th_entry(803u)->len == 0, "a command the timeout ended gives content");
    CHECK(strstr(th_entry(804u)->bytes, "804 5 run\n") != NULL,
          "the environment of the child reads %s", th_entry(804u)->bytes);
    CHECK(strstr(th_entry(804u)->bytes, "{\"request\":804") != NULL,
          "the standard input of the child holds no requests line: %s",
          th_entry(804u)->bytes);
    printf("run %d: replies %d\n", n, th_collected.count);
    th_stop(&t);
}

/* The feeder runs as many programs at one time as the system holds request slots. One
 * more request finds no place and is answered with that reason. Every command of the group
 * holds its place for a second, so no place can come free while the lines are read.
 *
 * The count stands here as a figure of its own and not as the constant under check. A
 * check written against that constant grows with it and can never fail. */
static void full_case(void)
{
    th_run t;
    char arg[AOTX_TH_ARG_TEXT];
    uint32_t i;

    CHECK(AOTX_TOOL_PROGRAMS_MAX == (uint32_t)AOTX_FULL_PROGRAMS,
          "the feeder holds %u places for programs and %d are the request slots",
          (unsigned)AOTX_TOOL_PROGRAMS_MAX, AOTX_FULL_PROGRAMS);
    th_tree(&t);
    th_spawn(&t, arguments[1], AOTX_CASE_TIMEOUT, NULL, -1);
    th_one_arg(arg, sizeof(arg), "command", "sleep 1");
    for (i = 0; i < (uint32_t)AOTX_FULL_PROGRAMS; i++) {
        th_request(&t, 2000u + i, 1u, "run", AOTX_TOOL_RUN, arg);
    }
    th_one_arg(arg, sizeof(arg), "command", "echo one more");
    th_request(&t, 2900u, 1u, "run", AOTX_TOOL_RUN, arg);
    th_wait(&t, AOTX_FULL_PROGRAMS + 1);
    CHECK(th_entry(2900u)->status == AOTX_TOOL_ERROR,
          "a request over the place count gives %u", th_entry(2900u)->status);
    CHECK(strstr(th_entry(2900u)->reason, "as many programs") != NULL,
          "the reason reads %s", th_entry(2900u)->reason);
    for (i = 0; i < (uint32_t)AOTX_FULL_PROGRAMS; i++) {
        CHECK(th_entry(2000u + i)->status == AOTX_TOOL_OK,
              "the program %u of a full table gives %u", i, th_entry(2000u + i)->status);
    }
    printf("full: replies %d\n", th_collected.count);
    th_stop(&t);
}

int main(int argc, char **argv)
{
    arguments = argv;
    if (argc < 2) {
        printf("usage: aotx_fs_tools_test <feeder>\n");
        return 1;
    }
    list_case(1);
    list_case(64);
    write_case(1);
    write_case(64);
    update_case(1);
    update_case(64);
    command_case(1);
    command_case(64);
    full_case();
    return aotx_report("fs_tools_test", 400);
}
