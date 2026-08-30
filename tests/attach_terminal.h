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

static int terminal_wait_file(const char *pattern, const char *text, unsigned int seconds)
{
    unsigned int turn;
    for (turn = 0u; turn < seconds * 10u; turn++) {
        if (terminal_file_has(pattern, text)) {
            return 0;
        }
        usleep(100000u);
    }
    return 1;
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
                                 const char *root, int *input)
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

static int terminal_path(int argc, char **argv)
{
    static const char skill_text[] =
        "---\nname: arithmetic\ndescription: Gives arithmetic rules.\n---\n"
        "Check each operation.\n";
    char dir[128];
    char journal[256];
    char skill[256];
    char file[320];
    char socket[320];
    char requests[320];
    char consoles[320];
    char manifests[320];
    char line[384];
    int boot_input = -1;
    int master = -1;
    int failed = 0;
    pid_t boot;
    pid_t tui;
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
    boot = terminal_start_boot(argv[2], journal, argv[4], argv[5], dir, &boot_input);
    CHECK(boot > 0, "the terminal path system does not start.");
    snprintf(socket, sizeof(socket), "%s/%s", journal, AOTX_ATTACH_NAME);
    CHECK(terminal_wait_path(socket, 120u) == 0, "the terminal attach socket does not appear.");
    tui = terminal_start_tui(argv[3], journal, &master);
    CHECK(tui > 0, "the terminal program does not start in the PTY.");
    usleep(500000u);

    snprintf(line, sizeof(line), "import %s\n", skill);
    CHECK(write(master, line, strlen(line)) == (ssize_t)strlen(line),
          "the terminal import line does not write.");
    snprintf(requests, sizeof(requests), "%s/requests.jsonl", journal);
    CHECK(terminal_wait_file(requests, "\"tool\":\"import\"", 120u) == 0,
          "the import request does not reach the drain requests line.");
    snprintf(consoles, sizeof(consoles), "%s/%c/console.log", journal, '*');
    CHECK(terminal_wait_file(consoles, "arithmetic skill installed", 120u) == 0,
          "the imported skill does not install.");

    CHECK(write(master, "skills\n", 7u) == 7, "the terminal skills line does not write.");
    CHECK(terminal_wait_file(consoles, "  arithmetic skill installed", 30u) == 0,
          "the skills output does not list arithmetic.");
    CHECK(write(master, "say what is a tick\n", 19u) == 19,
          "the terminal say line does not write.");
    snprintf(manifests, sizeof(manifests), "%s/manifest/%c.jsonl", journal, '*');
    CHECK(terminal_wait_file(manifests, "\"output_hash\"", 240u) == 0,
          "the conductor turn does not end.");
    CHECK(terminal_file_has(consoles, "conductor:"),
          "the conductor reply does not reach the console.");
    CHECK(!terminal_file_has(consoles, "no answer before its deadline"),
          "a ready device answer remained to its deadline.");
    CHECK(terminal_file_has(requests, "\"tool\":\"import\""),
          "the requests line lost the terminal import.");

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
    printf("terminal path: import request, skill list and conductor reply completed\n");
    aotx_remove_tree(dir);
    return failed ? 1 : aotx_report("terminal_path", 7);
}
