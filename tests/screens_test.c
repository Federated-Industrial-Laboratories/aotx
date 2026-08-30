/* Purpose: Check that each screen sends the line the table of actions names for its key.
 * Owns: One terminal state, one socket pair and one settings file for each case.
 * Threading: One thread.
 * Lifetime: The run of the program. */
#include "tests/disk_fake.h"

#include "disk/feed/attach.h"
#include "disk/tui/tui.h"

#include <fcntl.h>
#include <sys/socket.h>
#include <sys/stat.h>

/* The state is large, so the test holds it beside the program. */
static aotx_tui aotx_test_tui;

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
#define AOTX_SCREEN_SESSION  10u
#define AOTX_SCREEN_PICKER   11u

static void press(aotx_tui *tui, unsigned int code, unsigned int codepoint);

static aotx_tui_key key_of(unsigned int code, unsigned int codepoint)
{
    aotx_tui_key key;
    key.code = code;
    key.codepoint = codepoint;
    key.mods = 0;
    return key;
}

static void press(aotx_tui *tui, unsigned int code, unsigned int codepoint)
{
    aotx_tui_key key = key_of(code, codepoint);
    aotx_screen_key(tui, &key);
}

/* Makes a fixture of two module directories, one tool and one skill. */
static void make_modules(aotx_tui *tui, const char dir[128])
{
    char path[256];
    const char *kinds[2];
    const char *names[2];
    unsigned int i;
    kinds[0] = "tool";
    kinds[1] = "skill";
    names[0] = "count_words";
    names[1] = "read_first";
    snprintf(path, sizeof(path), "%s/modules", dir);
    mkdir(path, 0700);
    for (i = 0; i < 2u; i++) {
        FILE *file;
        snprintf(path, sizeof(path), "%s/modules/%s", dir, names[i]);
        mkdir(path, 0700);
        snprintf(path, sizeof(path), "%s/modules/%s/module.manifest", dir, names[i]);
        file = fopen(path, "w");
        if (file != NULL) {
            fprintf(file, "kind: %s\nname: %s\nversion: 1\n", kinds[i], names[i]);
            fclose(file);
        }
    }
    snprintf(tui->settings.text[AOTX_SET_MODULES_DIR],
             sizeof(tui->settings.text[AOTX_SET_MODULES_DIR]), "%s/modules", dir);
}

/* Opens the terminal state with a socket pair in place of the feeder. */
static int open_state(aotx_tui *tui, const char *settings, int *feeder)
{
    int pair[2];
    memset(tui, 0, sizeof(*tui));
    tui->session.fd = -1;
    tui->session.mirror_fd = -1;
    tui->session.boot_pid = -1;
    tui->model_pid = -1;
    tui->model_fd = -1;
    tui->screen = AOTX_TUI_SCREEN_NONE;
    snprintf(tui->settings_path, sizeof(tui->settings_path), "%s", settings);
    aotx_settings_defaults(&tui->settings);
    aotx_paint_size(&tui->paint, 80u, 24u);
    if (feeder == NULL) {
        return 0;
    }
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, pair) != 0) {
        return -1;
    }
    tui->session.fd = pair[0];
    *feeder = pair[1];
    return 0;
}

/* Reads the line that the terminal sent, without its frame. Returns 1 or 0. */
static int taken_line(int feeder, char *out, size_t bytes)
{
    unsigned char frame[5u + AOTX_BODY_BYTES];
    ssize_t got = recv(feeder, frame, sizeof(frame), MSG_DONTWAIT);
    unsigned int length;
    out[0] = '\0';
    if (got < 5 || frame[0] != (unsigned char)AOTX_ATTACH_LINE) {
        return 0;
    }
    length = (unsigned int)frame[1] | ((unsigned int)frame[2] << 8);
    if (length > (unsigned int)got - 5u) {
        length = (unsigned int)got - 5u;
    }
    if (length >= bytes) {
        length = (unsigned int)bytes - 1u;
    }
    memcpy(out, frame + 5, length);
    out[length] = '\0';
    return 1;
}

/* The first word of a line. */
static void first_word(const char *line, char *out, size_t bytes)
{
    size_t at = 0;
    while (line[at] != '\0' && line[at] != ' ' && at + 1u < bytes) {
        out[at] = line[at];
        at++;
    }
    out[at] = '\0';
}

/* The line the table names for one screen and one key. */
static const char *table_line(const char *screen, const char *key)
{
    unsigned int i;
    for (i = 0; aotx_tui_actions[i].screen != NULL; i++) {
        if (strcmp(aotx_tui_actions[i].screen, screen) == 0
            && strcmp(aotx_tui_actions[i].key, key) == 0) {
            return aotx_tui_actions[i].line;
        }
    }
    return NULL;
}

/* Each screen with a line for its Enter key sends that line. The first word of the row
 * fills the part that the table names. */
static void enter_of_each(void)
{
    static const unsigned int screens[] = {
        AOTX_SCREEN_HELP, AOTX_SCREEN_AGENTS, AOTX_SCREEN_BUS, AOTX_SCREEN_TOOLS,
        AOTX_SCREEN_SKILLS, AOTX_SCREEN_SETTINGS
    };
    char settings[192];
    char dir[128];
    unsigned int i;
    int feeder = -1;
    CHECK(aotx_temp_dir(dir, sizeof(dir)) == 0, "the temporary directory does not open");
    snprintf(settings, sizeof(settings), "%s/aotx.settings", dir);
    for (i = 0; i < sizeof(screens) / sizeof(screens[0]); i++) {
        aotx_tui *tui = &aotx_test_tui;
        aotx_tui_key enter = key_of(AOTX_TUI_KEY_ENTER, 0);
        const char *name;
        const char *want;
        char line[AOTX_TUI_LINE_BYTES];
        char one[64];
        char two[64];
        CHECK(open_state(tui, settings, &feeder) == 0, "the state does not open");
        make_modules(tui, dir);
        tui->screen = screens[i];
        name = aotx_screen_name(tui->screen);
        want = table_line(name, "Enter");
        CHECK(want != NULL, "the table names no line for the Enter key of %s", name);
        aotx_screen_draw(tui, 1u, 22u);
        CHECK(strcmp(aotx_screen_action(tui->screen, &enter), want) == 0,
              "the screen %s does not find its own row", name);
        if (want == NULL || want[0] == '\0') {
            close(feeder);
            aotx_session_detach(&tui->session);
            continue;
        }
        if (tui->screen == AOTX_SCREEN_SETTINGS) {
            /* The Settings screen edits the value first, and the second Enter sends it. */
            aotx_screen_key(tui, &enter);
            CHECK(tui->editing == 1, "the Settings screen did not open the field");
            press(tui, 0, (unsigned int)'2');
            press(tui, 0, (unsigned int)'0');
        }
        aotx_screen_key(tui, &enter);
        CHECK(taken_line(feeder, line, sizeof(line)) == 1,
              "the screen %s sent no line for its Enter key", name);
        first_word(want, one, sizeof(one));
        first_word(line, two, sizeof(two));
        CHECK(strcmp(one, two) == 0, "the screen %s sent the command %s, not %s", name,
              two, one);
        close(feeder);
        aotx_session_detach(&tui->session);
    }
    aotx_remove_tree(dir);
}

/* The Models screen lists without a system, loads through one, and runs local children. */
static void models_screen(void)
{
    static const char manifest[] =
        "{\"name\":\"embedding\",\"role\":\"embedding\","
        "\"path\":\"Qwen3-Embedding-0.6B-Q8_0.gguf\","
        "\"source\":\"Qwen/Qwen3-Embedding-0.6B-GGUF\","
        "\"revision\":\"370f27d7550e0def9b39c1f16d3fbaa13aa67728\","
        "\"license\":\"Apache-2.0\",\"bytes\":639150592,"
        "\"sha256\":\"06507c7b42688469c4e7298b0a1e16deff06caf291cf0a5b278c308249c3e439\"}\n";
    aotx_tui *tui = &aotx_test_tui;
    char dir[128];
    char source[256];
    char path[256];
    char rows[8][AOTX_TUI_LINE_BYTES];
    char line[AOTX_TUI_LINE_BYTES];
    int feeder = -1;
    int fd;
    int pipes[2];
    int writer;
    int tries;
    CHECK(aotx_temp_dir(dir, sizeof(dir)) == 0, "the model screen store does not open");
    snprintf(source, sizeof(source), "%s/Qwen3-Embedding-0.6B-Q8_0.gguf",
             AOTX_MODELS_DIRECTORY);
    snprintf(path, sizeof(path), "%s/Qwen3-Embedding-0.6B-Q8_0.gguf", dir);
    CHECK(symlink(source, path) == 0, "the model screen file does not link");
    snprintf(path, sizeof(path), "%s/manifest.jsonl", dir);
    fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0600);
    CHECK(fd >= 0 && write(fd, manifest, sizeof(manifest) - 1u) ==
          (ssize_t)sizeof(manifest) - 1, "the model screen manifest does not write");
    if (fd >= 0) {
        close(fd);
    }
    snprintf(path, sizeof(path), "%s/qwen3-reranker-0.6b-q8_0.gguf", dir);
    fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0600);
    CHECK(fd >= 0 && ftruncate(fd, 639153184) == 0,
          "the not-active model fixture does not form");
    if (fd >= 0) {
        close(fd);
    }
    snprintf(path, sizeof(path), "%s/Qwen3-4B-Q8_0.gguf", dir);
    fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0600);
    CHECK(fd >= 0 && ftruncate(fd, 1) == 0,
          "the digest-difference model fixture does not form");
    if (fd >= 0) {
        close(fd);
    }
    snprintf(path, sizeof(path), "%s/Qwen3-4B-Q4_0.gguf.part", dir);
    fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0600);
    CHECK(fd >= 0 && ftruncate(fd, 4096) == 0,
          "the fetching model fixture does not form");
    if (fd >= 0) {
        close(fd);
    }

    CHECK(open_state(tui, "aotx.settings", &feeder) == 0, "the model screen state does not open");
    snprintf(tui->settings.text[AOTX_SET_MODELS_DIR], AOTX_SETTING_TEXT_BYTES, "%s", dir);
    CHECK(aotx_rows_models(tui, (char *)rows, 8u, AOTX_TUI_LINE_BYTES) == 8u,
          "the Models screen does not list eight catalog entries");
    CHECK(strstr(rows[0], "on disk | embedding | embedding | Q8_0") != NULL,
          "the first model row reads %s", rows[0]);
    CHECK(strstr(rows[1], "on disk, not in the manifest") != NULL,
          "the second model row reads %s", rows[1]);
    CHECK(strstr(rows[2], "digest differs") != NULL,
          "the third model row reads %s", rows[2]);
    CHECK(strstr(rows[3], "fetching") != NULL,
          "the fourth model row reads %s", rows[3]);
    CHECK(strstr(rows[4], "not fetched") != NULL,
          "the fifth model row reads %s", rows[4]);
    CHECK(aotx_models_action(tui, 0u) == 1, "the active model row took no action");
    CHECK(taken_line(feeder, line, sizeof(line)) == 1 &&
          strcmp(line, "model load embedding embedding") == 0,
          "the active model row sent %s", line);
    CHECK(aotx_models_action(tui, 2u) == 1 && taken_line(feeder, line, sizeof(line)) == 0,
          "the digest-difference model row sent a line");
    CHECK(strstr(tui->says, "digest") != NULL,
          "the digest-difference model row gave no reason");

    snprintf(path, sizeof(path), "%s/manifest.jsonl", dir);
    CHECK(unlink(path) == 0, "the active model line does not leave the fixture");
    CHECK(aotx_rows_models(tui, (char *)rows, 8u, AOTX_TUI_LINE_BYTES) == 8u
          && strstr(rows[0], "on disk, not in the manifest") != NULL,
          "the active file did not become not active");
    CHECK(aotx_models_action(tui, 0u) == 1 && tui->model_pid > 0,
          "the not-active model row did not start activate");
    for (tries = 0; tries < 600 && tui->model_pid > 0; tries++) {
        aotx_models_poll(tui);
        usleep(10000u);
    }
    CHECK(tui->model_pid < 0 && strstr(tui->says, "on disk") != NULL,
          "activate did not end cleanly: %s", tui->says);
    CHECK(aotx_rows_models(tui, (char *)rows, 8u, AOTX_TUI_LINE_BYTES) == 8u
          && strstr(rows[0], "on disk | embedding") != NULL,
          "activate did not make the exact manifest row active");

    close(feeder);
    aotx_session_detach(&tui->session);
    CHECK(aotx_models_action(tui, 0u) == 1
          && strstr(tui->says, "running system") != NULL,
          "an active model row without a system did not refuse");

    snprintf(path, sizeof(path), "%s/Qwen3-0.6B-Q8_0.gguf.part", dir);
    CHECK(symlink("/dev/full", path) == 0,
          "the local-only fetch refusal does not form");
    CHECK(aotx_models_action(tui, 4u) == 1 && tui->model_pid > 0 &&
          strstr(tui->says, "fetch qwen3-0.6b-q8-0 started") != NULL,
          "an absent model did not start a fetch: %s", tui->says);
    for (tries = 0; tries < 600 && tui->model_pid > 0; tries++) {
        aotx_models_poll(tui);
        usleep(10000u);
    }
    CHECK(tui->model_pid < 0 && strstr(tui->says, "failed") != NULL,
          "the local fetch guard did not refuse: %s", tui->says);

    CHECK(pipe2(pipes, O_CLOEXEC | O_NONBLOCK) == 0,
          "the model progress pipe does not open");
    writer = fork();
    if (writer == 0) {
        int flags = fcntl(pipes[1], F_GETFL, 0);
        ssize_t wrote;
        close(pipes[0]);
        fcntl(pipes[1], F_SETFL, flags & ~O_NONBLOCK);
        wrote = write(pipes[1], "bytes 7 total 19 rate 3\n", 24u);
        usleep(300000u);
        close(pipes[1]);
        _exit(wrote == 24 ? 0 : 1);
    }
    close(pipes[1]);
    tui->model_pid = writer;
    tui->model_fd = pipes[0];
    snprintf(tui->model_name, sizeof(tui->model_name), "progress-model");
    usleep(50000u);
    aotx_models_poll(tui);
    CHECK(strcmp(tui->says, "note fetch progress-model 7 of 19") == 0,
          "the Models screen progress reads %s", tui->says);
    aotx_models_close(tui);
    aotx_remove_tree(dir);
}

/* The keys y and n of the Agents screen send the two lines the table names, and only on a
 * request that waits. */
static void agents(void)
{
    aotx_tui *tui = &aotx_test_tui;
    char settings[192];
    char dir[128];
    char line[AOTX_TUI_LINE_BYTES];
    int feeder = -1;
    CHECK(aotx_temp_dir(dir, sizeof(dir)) == 0, "the temporary directory does not open");
    snprintf(settings, sizeof(settings), "%s/aotx.settings", dir);
    CHECK(open_state(tui, settings, &feeder) == 0, "the state does not open");
    tui->screen = AOTX_SCREEN_AGENTS;
    tui->have_shot = 1;
    memset(&tui->shot, 0, sizeof(tui->shot));
    tui->shot.tables.request[0].request = 42u;
    tui->shot.tables.request[0].agent = 3u;
    snprintf(tui->shot.tables.request[0].tool_name,
             sizeof(tui->shot.tables.request[0].tool_name), "fs_read");
    aotx_screen_draw(tui, 1u, 22u);
    press(tui, 0, (unsigned int)'y');
    CHECK(taken_line(feeder, line, sizeof(line)) == 1, "the key y sent no line");
    CHECK(strcmp(line, "authorize 42") == 0, "the key y sent %s", line);
    press(tui, 0, (unsigned int)'n');
    CHECK(taken_line(feeder, line, sizeof(line)) == 1, "the key n sent no line");
    CHECK(strcmp(line, "refuse 42") == 0, "the key n sent %s", line);

    /* A row that is not a request sends nothing and says why. */
    tui->cursor = 1u;
    aotx_screen_draw(tui, 1u, 22u);
    tui->says[0] = '\0';
    press(tui, 0, (unsigned int)'y');
    CHECK(taken_line(feeder, line, sizeof(line)) == 0,
          "the key y on a row that is not a request sent a line");
    CHECK(tui->says[0] != '\0', "the key y on a row that is not a request said nothing");
    close(feeder);
    aotx_session_detach(&tui->session);
    aotx_remove_tree(dir);
}

/* The Session screen sends every action through the selected system. */
static void session_actions(void)
{
    aotx_tui *tui = &aotx_test_tui;
    char dir[128];
    char path[256];
    char line[AOTX_TUI_EDIT_BYTES];
    FILE *file;
    int feeder = -1;
    int pair[2];
    unsigned int i;
    aotx_tui_key send = key_of(AOTX_TUI_KEY_ENTER, 0u);
    CHECK(open_state(tui, "aotx.settings", &feeder) == 0, "the Session state does not open");
    tui->screen = AOTX_SCREEN_SESSION;
    tui->session_agent = 3u;
    tui->have_shot = 1;
    CHECK(aotx_temp_dir(dir, sizeof(dir)) == 0, "the transcript directory does not open");
    snprintf(tui->session.journal, sizeof(tui->session.journal), "%s", dir);
    snprintf(path, sizeof(path), "%s/0000000000000001", dir);
    CHECK(mkdir(path, 0700) == 0, "the boot transcript directory does not open");
    snprintf(path, sizeof(path), "%s/0000000000000001/transcript", dir);
    CHECK(mkdir(path, 0700) == 0, "the agent transcript directory does not open");
    snprintf(path, sizeof(path), "%s/0000000000000001/transcript/3.jsonl", dir);
    file = fopen(path, "w");
    CHECK(file != NULL, "the agent transcript does not open");
    if (file != NULL) {
        fputs("{\"tick\":1,\"kind\":\"result\",\"text\":\"first\\nsecond\","
              "\"tool\":\"fs_read\",\"request\":42,\"status\":\"ok\",\"turn\":2}\n",
              file);
        fclose(file);
    }
    tui->shot.head.boot_id = 1u;
    snprintf(tui->shot.head.panel[0].name, sizeof(tui->shot.head.panel[0].name), "console");
    tui->shot.head.panel[0].rows = 1u;
    tui->shot.head.panel[0].cols = 9u;
    for (i = 0u; i < 9u; i++) {
        tui->shot.cell[i].glyph = (unsigned char)("live tail"[i] - 32);
    }
    aotx_screen_draw(tui, 1u, 22u);
    CHECK(tui->paint.want[16u * tui->paint.cols + 2u].code == (unsigned int)'l'
          && tui->paint.want[16u * tui->paint.cols + 3u].code == (unsigned int)'i',
          "the Session screen does not show the live console tail");
    press(tui, AOTX_TUI_KEY_PAGE_UP, 0u);
    press(tui, AOTX_TUI_KEY_ENTER, 0u);
    CHECK(tui->session_result == 1, "Enter did not open the selected result");
    press(tui, AOTX_TUI_KEY_ENTER, 0u);
    CHECK(tui->session_result == 0, "Enter did not close the open result");
    tui->shot.tables.request[0].request = 42u;
    tui->shot.tables.request[0].agent = 3u;
    press(tui, 0u, (unsigned int)'y');
    CHECK(taken_line(feeder, line, sizeof(line)) == 1 && strcmp(line, "authorize 42") == 0,
          "the Session y key did not authorize request 42");
    press(tui, 0u, (unsigned int)'n');
    CHECK(taken_line(feeder, line, sizeof(line)) == 1 && strcmp(line, "refuse 42") == 0,
          "the Session n key did not refuse request 42");
    memset(&tui->shot.tables.request[0], 0, sizeof(tui->shot.tables.request[0]));
    press(tui, 0u, (unsigned int)'c');
    CHECK(taken_line(feeder, line, sizeof(line)) == 1
          && strcmp(line, "agent 3 compact") == 0,
          "the Session compact key sent %s", line);
    press(tui, 0u, (unsigned int)'p');
    press(tui, 0u, (unsigned int)'a');
    press(tui, 0u, (unsigned int)'u');
    press(tui, 0u, (unsigned int)'t');
    press(tui, 0u, (unsigned int)'o');
    send.mods = AOTX_TUI_MOD_CONTROL;
    aotx_screen_key(tui, &send);
    CHECK(taken_line(feeder, line, sizeof(line)) == 1
          && strcmp(line, "agent 3 pages auto") == 0,
          "the Session pages field sent %s", line);
    press(tui, 0u, (unsigned int)'s');
    press(tui, 0u, (unsigned int)'w');
    press(tui, 0u, (unsigned int)'o');
    press(tui, 0u, (unsigned int)'r');
    press(tui, 0u, (unsigned int)'k');
    press(tui, 0u, (unsigned int)'e');
    press(tui, 0u, (unsigned int)'r');
    aotx_screen_key(tui, &send);
    CHECK(taken_line(feeder, line, sizeof(line)) == 1 && strcmp(line, "spawn worker") == 0,
          "the Session spawn field sent %s", line);
    press(tui, 0u, (unsigned int)'o');
    press(tui, 0u, (unsigned int)'n');
    press(tui, 0u, (unsigned int)'e');
    press(tui, AOTX_TUI_KEY_ENTER, 0u);
    press(tui, 0u, (unsigned int)'t');
    press(tui, 0u, (unsigned int)'w');
    press(tui, 0u, (unsigned int)'o');
    send.mods = AOTX_TUI_MOD_ALT;
    aotx_screen_key(tui, &send);
    CHECK(taken_line(feeder, line, sizeof(line)) == 1
          && strcmp(line, "task 3 one\ntwo") == 0,
          "the Session task editor sent %s", line);

    CHECK(socketpair(AF_UNIX, SOCK_STREAM, 0, pair) == 0,
          "the second system socket does not open");
    tui->other_count = 1u;
    tui->other[0].card = 1u;
    tui->other[0].session.fd = pair[0];
    press(tui, AOTX_TUI_KEY_RIGHT, 0u);
    CHECK(tui->card == 1u, "the Session chooser did not select card 1");
    tui->session_agent = 0u;
    press(tui, 0u, (unsigned int)'h');
    press(tui, 0u, (unsigned int)'i');
    send.mods = AOTX_TUI_MOD_CONTROL;
    aotx_screen_key(tui, &send);
    CHECK(taken_line(pair[1], line, sizeof(line)) == 1 && strcmp(line, "say hi") == 0,
          "the selected system did not take the Session line");
    CHECK(taken_line(feeder, line, sizeof(line)) == 0,
          "the unselected system took the Session line");
    close(pair[1]);
    close(feeder);
    aotx_session_detach(&tui->session);
    aotx_session_detach(&tui->other[0].session);
    aotx_remove_tree(dir);
}

/* The all row of the Bus screen sends the command with no argument. Every other row is
 * one word of the static argument table. */
static void bus_rows(void)
{
    aotx_tui *tui = &aotx_test_tui;
    char line[AOTX_TUI_LINE_BYTES];
    unsigned int i;
    int feeder = -1;
    CHECK(open_state(tui, "aotx.settings", &feeder) == 0, "the state does not open");
    tui->screen = AOTX_SCREEN_BUS;
    for (i = 0u; i < 8u; i++) {
        char want[64];
        tui->cursor = i;
        aotx_screen_draw(tui, 1u, 22u);
        press(tui, AOTX_TUI_KEY_ENTER, 0u);
        CHECK(taken_line(feeder, line, sizeof(line)) == 1, "Bus row %u sent no line", i);
        if (i == 0u) {
            snprintf(want, sizeof(want), "bus");
        } else {
            snprintf(want, sizeof(want), "bus %s", aotx_tui_bus_kinds[i - 1u].word);
        }
        CHECK(strcmp(line, want) == 0, "Bus row %u sent %s, not %s", i, line, want);
    }
    close(feeder);
    aotx_session_detach(&tui->session);
}

/* With no system running the Settings screen writes the file and sends nothing. */
static void settings_file(int n)
{
    aotx_tui *tui = &aotx_test_tui;
    char settings[192];
    char dir[128];
    char line[AOTX_TUI_LINE_BYTES];
    FILE *file;
    int found = 0;
    int i;
    CHECK(aotx_temp_dir(dir, sizeof(dir)) == 0, "the temporary directory does not open");
    snprintf(settings, sizeof(settings), "%s/aotx.settings", dir);
    CHECK(open_state(tui, settings, NULL) == 0, "the state does not open");
    tui->screen = AOTX_SCREEN_SETTINGS;
    for (i = 0; i < n; i++) {
        aotx_tui_key enter = key_of(AOTX_TUI_KEY_ENTER, 0);
        tui->cursor = (unsigned int)i % AOTX_SETTING_NUMBER_COUNT;
        aotx_screen_draw(tui, 1u, 22u);
        aotx_screen_key(tui, &enter);
        CHECK(tui->editing == 1, "the field did not open at element %d", i);
        press(tui, 0, (unsigned int)'1');
        aotx_screen_key(tui, &enter);
        CHECK(tui->editing == 0, "the field did not close at element %d", i);
        CHECK(tui->says[0] != '\0', "the screen said nothing at element %d", i);
    }
    /* One key of a known range is written and read back, so the file holds the value and
     * the screen says where it went. */
    {
        aotx_tui_key enter = key_of(AOTX_TUI_KEY_ENTER, 0);
        unsigned int index = 0;
        CHECK(aotx_settings_find("tick.period_ms", 14u, &index) == 0,
              "the key of the period is not a number key");
        tui->cursor = index;
        aotx_screen_draw(tui, 1u, 22u);
        aotx_screen_key(tui, &enter);
        press(tui, 0, (unsigned int)'2');
        press(tui, 0, (unsigned int)'0');
        aotx_screen_key(tui, &enter);
        CHECK(strstr(tui->says, settings) != NULL,
              "the screen did not say where the value went: %s", tui->says);
    }
    file = fopen(settings, "r");
    CHECK(file != NULL, "the settings file was not written");
    if (file != NULL) {
        while (fgets(line, (int)sizeof(line), file) != NULL) {
            if (strstr(line, "= 1") != NULL) {
                found++;
            }
        }
        fclose(file);
    }
    CHECK(found > 0, "the settings file holds none of the %d values written", n);
    file = fopen(settings, "r");
    found = 0;
    if (file != NULL) {
        while (fgets(line, (int)sizeof(line), file) != NULL) {
            if (strstr(line, "tick.period_ms") != NULL && strstr(line, "20") != NULL) {
                found++;
            }
        }
        fclose(file);
    }
    CHECK(found == 1, "the file holds the period %d times, not one time", found);
    aotx_remove_tree(dir);
}

/* Every key of the key bar opens the screen its row names, and the same key closes it. */
static void keybar(void)
{
    aotx_tui *tui = &aotx_test_tui;
    unsigned int i;
    CHECK(open_state(tui, "aotx.settings", NULL) == 0, "the state does not open");
    for (i = 0; aotx_tui_keys[i].key != NULL; i++) {
        aotx_tui_key key = key_of(AOTX_TUI_KEY_F1 + i, 0);
        unsigned int screen = aotx_screen_of_key(&key);
        CHECK(screen != AOTX_TUI_SCREEN_NONE, "the key %s opens no screen",
              aotx_tui_keys[i].key);
        CHECK(screen == i, "the key %s opens the screen %u, not %u", aotx_tui_keys[i].key,
              screen, i);
        CHECK(strcmp(aotx_tui_screens[screen].key, aotx_tui_keys[i].key) == 0,
              "the screen of the key %s names another key", aotx_tui_keys[i].key);
    }
    /* Every screen the action table names is a screen of the screen table. */
    for (i = 0; aotx_tui_actions[i].screen != NULL; i++) {
        unsigned int at;
        int held = 0;
        for (at = 0; aotx_tui_screens[at].name != NULL; at++) {
            if (strcmp(aotx_tui_screens[at].name, aotx_tui_actions[i].screen) == 0) {
                held = 1;
            }
        }
        CHECK(held == 1, "the action row %u names the screen %s, which is not a screen", i,
              aotx_tui_actions[i].screen);
    }
}

/* The Menu screen holds the six local rows. A row sends no line. The console row moves
 * the live picture, and each other row opens the screen that it names. */
static void menu_rows(void)
{
    static const char *labels[] = { "console", "agents", "bus", "models", "tools",
                                    "settings" };
    aotx_tui *tui = &aotx_test_tui;
    unsigned int i;
    int feeder = -1;
    CHECK(open_state(tui, "aotx.settings", &feeder) == 0, "the state does not open");
    tui->have_shot = 1;
    snprintf(tui->shot.head.panel[0].name, sizeof(tui->shot.head.panel[0].name), "console");
    tui->shot.head.panel[0].row = 12u;
    for (i = 0u; i < sizeof(labels) / sizeof(labels[0]); i++) {
        char line[AOTX_TUI_LINE_BYTES];
        tui->screen = AOTX_SCREEN_MENU;
        tui->cursor = i;
        aotx_screen_draw(tui, 1u, 22u);
        CHECK(strcmp(aotx_tui_menu[i].label, labels[i]) == 0,
              "Menu row %u is %s, not %s", i, aotx_tui_menu[i].label, labels[i]);
        press(tui, AOTX_TUI_KEY_ENTER, 0u);
        CHECK(taken_line(feeder, line, sizeof(line)) == 0, "Menu row %s sent a line",
              labels[i]);
        if (i == 0u) {
            CHECK(tui->screen == AOTX_TUI_SCREEN_NONE, "the console row left a screen open");
            CHECK(tui->paint.pan_row == 12u, "the console row did not move the live picture");
        } else {
            CHECK(strcmp(aotx_screen_name(tui->screen), labels[i]) == 0,
                  "Menu row %s opened %s", labels[i], aotx_screen_name(tui->screen));
        }
    }
    close(feeder);
    aotx_session_detach(&tui->session);
}

/* The picker opens from both module screens and starts at the allowed root. */
static void picker_keys(void)
{
    static const unsigned int screens[] = { AOTX_SCREEN_TOOLS, AOTX_SCREEN_SKILLS };
    aotx_tui *tui = &aotx_test_tui;
    unsigned int i;
    CHECK(open_state(tui, "aotx.settings", NULL) == 0, "the state does not open");
    snprintf(tui->settings.text[AOTX_SET_TOOLS_ROOT], AOTX_SETTING_TEXT_BYTES, "/tmp");
    for (i = 0u; i < 2u; i++) {
        tui->screen = screens[i];
        press(tui, 0u, (unsigned int)'p');
        CHECK(tui->screen == AOTX_SCREEN_PICKER, "screen %u did not open the picker",
              screens[i]);
        CHECK(strcmp(tui->picker_dir, "/tmp") == 0, "the picker started at %s", tui->picker_dir);
    }
}

static void put_phase(const char *dir, const char *line)
{
    char path[256];
    int fd;
    snprintf(path, sizeof(path), "%s/phase", dir);
    fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0600);
    CHECK(fd >= 0, "the state file does not open");
    if (fd >= 0) {
        CHECK(write(fd, line, strlen(line)) == (ssize_t)strlen(line),
              "the state file does not take its line");
        close(fd);
    }
}

/* The detached state gives elapsed seconds for every active word. Only an absent or
 * closed file reports no running system. */
static void phase_states(void)
{
    static const char *lines[] = { "placing 100\n", "replaying 100\n", "running 100\n" };
    static const char *words[] = { "placing models, 23 seconds",
                                   "replaying the journal, 23 seconds",
                                   "running, 23 seconds" };
    char dir[128];
    char state[192];
    unsigned int i;
    CHECK(aotx_temp_dir(dir, sizeof(dir)) == 0, "the temporary directory does not open");
    for (i = 0u; i < 3u; i++) {
        put_phase(dir, lines[i]);
        CHECK(aotx_session_phase(dir, 123u, state, sizeof(state)) == 1,
              "the active state %u does not read", i);
        CHECK(strcmp(state, words[i]) == 0, "the active state is %s, not %s", state, words[i]);
    }
    put_phase(dir, "closed 100\n");
    CHECK(aotx_session_phase(dir, 123u, state, sizeof(state)) == 0,
          "the closed state is active");
    aotx_remove_tree(dir);
    CHECK(aotx_session_phase(dir, 123u, state, sizeof(state)) == 0,
          "the absent state is active");
}

/* A socket-close report wins over later detached-state text until a key or an attach clears
 * its flag. This arm changes the ordinary notice before the second draw. */
static void closed_notice(void)
{
    aotx_tui *tui = &aotx_test_tui;
    const char *closed = "the system closed the socket";
    unsigned int row;
    unsigned int i;
    CHECK(open_state(tui, "aotx.settings", NULL) == 0, "the terminal state does not open");
    tui->socket_closed = 1;
    snprintf(tui->says, sizeof(tui->says), "a later detached report");
    aotx_frame_draw(tui);
    row = aotx_paint_view_rows(&tui->paint);
    for (i = 0u; closed[i] != '\0'; i++) {
        CHECK(tui->paint.want[(size_t)row * tui->paint.cols + i].code
              == (unsigned int)(unsigned char)closed[i],
              "the socket-close report changed at byte %u", i);
    }
    tui->socket_closed = 0;
    aotx_frame_draw(tui);
    CHECK(tui->paint.want[(size_t)row * tui->paint.cols].code == (unsigned int)'a',
          "the ordinary report did not return after the close report cleared");
}

int main(void)
{
    keybar();
    menu_rows();
    picker_keys();
    phase_states();
    closed_notice();
    enter_of_each();
    models_screen();
    agents();
    session_actions();
    bus_rows();
    settings_file(1);
    settings_file(64);
    return aotx_report("screens_test", 150);
}
