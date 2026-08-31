/* Purpose: Drive the released terminal commands through a PTY and the attach socket.
 * Owns: One system process, one terminal process and one temporary journal.
 * Threading: The two child programs own their normal threads.
 * Lifetime: One terminal-path check. */

#include <glob.h>
#include <sys/ioctl.h>
#include <sys/wait.h>
#include <termios.h>

static int terminal_file_has(const char *pattern, const char *text)
{
    glob_t files;
    size_t i;
    int found = 0;
    memset(&files, 0, sizeof(files));
    if (glob(pattern, 0, NULL, &files) != 0) {
        globfree(&files);
        return 0;
    }
    for (i = 0; i < files.gl_pathc && !found; i++) {
        char bytes[16384];
        ssize_t got;
        int fd = open(files.gl_pathv[i], O_RDONLY | O_CLOEXEC);
        if (fd < 0) {
            continue;
        }
        got = read(fd, bytes, sizeof(bytes) - 1u);
        close(fd);
        if (got > 0) {
            bytes[got] = '\0';
            found = strstr(bytes, text) != NULL;
        }
    }
    globfree(&files);
    return found;
}

static unsigned int terminal_file_count(const char *pattern, const char *text)
{
    glob_t files;
    unsigned int count = 0u;
    memset(&files, 0, sizeof(files));
    if (glob(pattern, 0, NULL, &files) != 0) {
        globfree(&files);
        return 0u;
    }
    for (size_t i = 0u; i < files.gl_pathc; ++i) {
        char bytes[32768];
        ssize_t got;
        int fd = open(files.gl_pathv[i], O_RDONLY | O_CLOEXEC);
        if (fd < 0) {
            continue;
        }
        got = read(fd, bytes, sizeof(bytes) - 1u);
        close(fd);
        if (got <= 0) {
            continue;
        }
        bytes[got] = '\0';
        for (char *at = bytes; (at = strstr(at, text)) != NULL; at += strlen(text)) {
            count++;
        }
    }
    globfree(&files);
    return count;
}

static int terminal_wait_path(const char *path, unsigned int seconds)
{
    unsigned int turn;
    for (turn = 0u; turn < seconds * 10u; turn++) {
        if (access(path, F_OK) == 0) {
            return 0;
        }
        usleep(100000u);
    }
    return 1;
}

/* Drains terminal paint bytes so a PTY with no human reader cannot stop the key path. An
 * evidence run may name a capture file; the normal test only discards the paint stream. */
static void terminal_drain_pty(int master)
{
    const char *capture = getenv("AOTX_PTY_CAPTURE");
    unsigned char bytes[4096];
    int save = -1;
    if (capture != NULL && capture[0] != '\0') {
        save = open(capture, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0600);
    }
    for (;;) {
        struct pollfd fd = { master, POLLIN, 0 };
        if (poll(&fd, 1, 0) <= 0) {
            break;
        }
        ssize_t got = read(master, bytes, sizeof(bytes));
        if (got <= 0) {
            break;
        }
        if (save >= 0) {
            ssize_t wrote = write(save, bytes, (size_t)got);
            if (wrote != got) {
                close(save);
                save = -1;
            }
        }
    }
    if (save >= 0) {
        close(save);
    }
}

static int terminal_wait_pty_file(const char *pattern, const char *text,
                                  unsigned int seconds, int master)
{
    for (unsigned int turn = 0u; turn < seconds * 10u; ++turn) {
        terminal_drain_pty(master);
        if (terminal_file_has(pattern, text)) {
            return 0;
        }
        usleep(100000u);
    }
    terminal_drain_pty(master);
    return 1;
}

static int terminal_child_wait(pid_t child, unsigned int seconds)
{
    unsigned int turn;
    int status = 0;
    for (turn = 0u; turn < seconds * 10u; turn++) {
        pid_t got = waitpid(child, &status, WNOHANG);
        if (got == child) {
            return WIFEXITED(status) ? WEXITSTATUS(status) : 128;
        }
        usleep(100000u);
    }
    return -1;
}

static pid_t terminal_start_boot(const char *program, const char *journal,
                                 const char *models, const char *modules,
                                 const char *root, const char *settings, int *input)
{
    char log[512];
    int pipe_fd[2];
    int out;
    pid_t child;
    if (pipe(pipe_fd) != 0) {
        return -1;
    }
    snprintf(log, sizeof(log), "%s/boot.log", journal);
    out = open(log, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
    if (out < 0) {
        close(pipe_fd[0]);
        close(pipe_fd[1]);
        return -1;
    }
    child = fork();
    if (child == 0) {
        dup2(pipe_fd[0], STDIN_FILENO);
        dup2(out, STDOUT_FILENO);
        dup2(out, STDERR_FILENO);
        close(pipe_fd[0]);
        close(pipe_fd[1]);
        close(out);
        execl(program, program, "--journal", journal, "--models", models,
              "--roles", "language", "--root", root, "--modules", modules,
              "--settings", settings,
              (char *)NULL);
        _exit(127);
    }
    close(pipe_fd[0]);
    close(out);
    *input = pipe_fd[1];
    return child;
}

static pid_t terminal_start_tui(const char *program, const char *journal, int *master)
{
    char *slave_name;
    int slave;
    pid_t child;
    *master = posix_openpt(O_RDWR | O_NOCTTY | O_CLOEXEC);
    if (*master < 0 || grantpt(*master) != 0 || unlockpt(*master) != 0) {
        return -1;
    }
    slave_name = ptsname(*master);
    if (slave_name == NULL) {
        return -1;
    }
    child = fork();
    if (child == 0) {
        setsid();
        slave = open(slave_name, O_RDWR);
        if (slave < 0) {
            _exit(126);
        }
        ioctl(slave, TIOCSCTTY, 0);
        dup2(slave, STDIN_FILENO);
        dup2(slave, STDOUT_FILENO);
        dup2(slave, STDERR_FILENO);
        if (slave > STDERR_FILENO) {
            close(slave);
        }
        setenv("TERM", "xterm", 1);
        execl(program, program, "--attach", journal, "--no-splash", (char *)NULL);
        _exit(127);
    }
    return child;
}

static void terminal_write_file(const char *path, const char *text)
{
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
    CHECK(fd >= 0 && write(fd, text, strlen(text)) == (ssize_t)strlen(text),
          "the terminal skill file does not write.");
    if (fd >= 0) {
        close(fd);
    }
}

static void terminal_digest(const void *data, size_t bytes,
                            char out[AOTX_SHA256_DIGEST * 2u + 1u])
{
    aotx_sha256 state;
    unsigned char digest[AOTX_SHA256_DIGEST];
    aotx_sha256_init(&state);
    aotx_sha256_update(&state, data, bytes);
    aotx_sha256_final(&state, digest);
    aotx_sha256_text(digest, out);
}

static int terminal_path(int argc, char **argv)
{
    static const char skill_text[] =
        "---\nname: arithmetic\ndescription: Gives arithmetic rules.\n---\n"
        "Check each operation.\n";
    char dir[128];
    char journal[256];
    char skill[256];
    char file[320];
    char settings[320];
    char catalog[320];
    char fetched[320];
    char source[320];
    char revision[384];
    char model_source[448];
    char model_file[384];
    char socket[320];
    char consoles[320];
    char line[384];
    int boot_input = -1;
    int master = -1;
    int failed = 0;
    pid_t boot;
    pid_t tui;
    static const char model_bytes[] = "local terminal model\n";
    char digest[AOTX_SHA256_DIGEST * 2u + 1u];
    if (argc != 6) {
        printf("usage: attach_test --terminal-path <boot> <tui> <models> <modules>\n");
        return 2;
    }
    CHECK(aotx_temp_dir(dir, sizeof(dir)) == 0, "the terminal path directory does not open.");
    snprintf(journal, sizeof(journal), "%s/journal", dir);
    snprintf(skill, sizeof(skill), "%s/arithmetic", dir);
    mkdir(journal, 0700);
    mkdir(skill, 0700);
    snprintf(file, sizeof(file), "%s/SKILL.md", skill);
    terminal_write_file(file, skill_text);
    snprintf(fetched, sizeof(fetched), "%s/fetched", dir);
    snprintf(source, sizeof(source), "%s/source", dir);
    snprintf(revision, sizeof(revision), "%s/0000000000000000000000000000000000000001",
             source);
    mkdir(fetched, 0700);
    mkdir(source, 0700);
    mkdir(revision, 0700);
    snprintf(model_source, sizeof(model_source), "%s/terminal-local.gguf", revision);
    terminal_write_file(model_source, model_bytes);
    terminal_digest(model_bytes, sizeof(model_bytes) - 1u, digest);
    snprintf(catalog, sizeof(catalog), "%s/catalog.jsonl", dir);
    {
        char row[1536];
        snprintf(row, sizeof(row),
                 "{\"name\":\"terminal-local\",\"role\":\"language\","
                 "\"repository\":\"file://localhost%s\","
                 "\"file\":\"terminal-local.gguf\","
                 "\"revision\":\"0000000000000000000000000000000000000001\","
                 "\"bytes\":%zu,\"sha256\":\"%s\",\"license\":\"Apache-2.0\","
                 "\"quant\":\"Q8_0\",\"profiles\":\"12g,8g\",\"verified\":false,"
                 "\"source\":\"local/terminal-local\",\"note\":\"terminal check\"}\n",
                 source, sizeof(model_bytes) - 1u, digest);
        terminal_write_file(catalog, row);
    }
    snprintf(settings, sizeof(settings), "%s/aotx.settings", dir);
    {
        char setting_text[512];
        snprintf(setting_text, sizeof(setting_text), "models.dir = %s\n", fetched);
        terminal_write_file(settings, setting_text);
    }
    setenv("AOTX_MODEL_CATALOG", catalog, 1);
    boot = terminal_start_boot(argv[2], journal, argv[4], argv[5], dir, settings,
                               &boot_input);
    unsetenv("AOTX_MODEL_CATALOG");
    CHECK(boot > 0, "the terminal path system does not start.");
    snprintf(socket, sizeof(socket), "%s/%s", journal, AOTX_ATTACH_NAME);
    CHECK(terminal_wait_path(socket, 120u) == 0, "the terminal attach socket does not appear.");
    tui = terminal_start_tui(argv[3], journal, &master);
    CHECK(tui > 0, "the terminal program does not start in the PTY.");
    usleep(500000u);
    terminal_drain_pty(master);

    snprintf(line, sizeof(line), "import %s\n", skill);
    CHECK(write(master, line, strlen(line)) == (ssize_t)strlen(line),
          "the terminal import line does not write.");
    snprintf(consoles, sizeof(consoles), "%s/%c/console.log", journal, '*');
    CHECK(terminal_wait_pty_file(consoles, "import: the skill arithmetic is installed",
                                 120u, master) == 0,
          "the imported skill does not install.");
    terminal_drain_pty(master);
    CHECK(terminal_file_count(consoles, "import: the skill arithmetic is installed") == 1u,
          "one terminal import did not make exactly one installed result.");

    CHECK(write(master, "skills\n", 7u) == 7, "the terminal skills line does not write.");
    CHECK(terminal_wait_pty_file(consoles, "  arithmetic skill installed", 30u, master)
              == 0,
          "the skills output does not list arithmetic.");
    terminal_drain_pty(master);

    snprintf(line, sizeof(line), "import %s\n", file);
    CHECK(write(master, line, strlen(line)) == (ssize_t)strlen(line),
          "the terminal file import line does not write.");
    CHECK(terminal_wait_pty_file(consoles,
              "import: the directory does not read: the name is not one to sixty-three bytes",
              30u, master) == 0, "the file import refusal does not reach the console.");
    CHECK(terminal_file_count(consoles,
              "import: the directory does not read: the name is not one to sixty-three bytes")
              == 1u, "one terminal file import did not make exactly one refusal.");
    CHECK(terminal_file_count(consoles, "> import ") == 0u,
          "the feeder import refusal was also shown as a command.");
    terminal_drain_pty(master);

    CHECK(write(master, "model fetch terminal-local\n", 27u) == 27,
          "the terminal fetch line does not write.");
    CHECK(terminal_wait_pty_file(consoles, "> note fetch terminal-local started", 30u,
                                 master) == 0,
          "the fetch start note does not reach the console.");
    CHECK(terminal_wait_pty_file(consoles, "> note fetch terminal-local host local file",
                                 30u, master) == 0,
          "the local fetch source does not reach the console.");
    CHECK(terminal_wait_pty_file(consoles, "> note fetch terminal-local on disk", 30u,
                                 master) == 0,
          "the fetch completion note does not reach the console.");
    terminal_drain_pty(master);
    snprintf(model_file, sizeof(model_file), "%s/terminal-local.gguf", fetched);
    CHECK(access(model_file, R_OK) == 0, "the local model fetch does not install its file.");

    CHECK(write(master, "quit\n", 5u) == 5, "the terminal quit line does not write.");
    close(boot_input);
    if (terminal_child_wait(boot, 60u) != 0) {
        failed = 1;
        kill(boot, SIGTERM);
        waitpid(boot, NULL, 0);
    }
    kill(tui, SIGTERM);
    waitpid(tui, NULL, 0);
    close(master);
    printf("terminal path: directory import, file refusal and local fetch completed\n");
    aotx_remove_tree(dir);
    return failed ? 1 : aotx_report("terminal_path", 15);
}
