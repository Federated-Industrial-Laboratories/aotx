/* Purpose: Attach to a running system, send its keys and lines, and start one when asked.
 * Owns: The socket to the feeder, the mapped mirror, and the child that a start makes.
 * Threading: One thread; the socket does not block and the child is never waited for.
 * Lifetime: From the attach to the detach; a system that runs is left running. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "disk/tui/tui.h"

#include "cuda/seam/wire.h"
#include "disk/feed/attach.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>

/* The bytes one read takes from the socket. A reason is the only thing the feeder sends. */
#define AOTX_SESSION_READ 512

/* The name of the file that holds what a boot printed. */
#define AOTX_SESSION_LOG  "boot.log"
#define AOTX_SESSION_PHASE "phase"

/* Takes the mirror descriptor off the socket. Returns the descriptor, or -1. */
static int take_mirror(int fd)
{
    struct msghdr message;
    struct iovec io;
    struct cmsghdr *control;
    union {
        char bytes[CMSG_SPACE(sizeof(int))];
        struct cmsghdr align;
    } room;
    char payload = 0;
    int mirror = -1;

    memset(&message, 0, sizeof(message));
    memset(&room, 0, sizeof(room));
    io.iov_base = &payload;
    io.iov_len = 1;
    message.msg_iov = &io;
    message.msg_iovlen = 1;
    message.msg_control = room.bytes;
    message.msg_controllen = sizeof(room.bytes);
    if (recvmsg(fd, &message, 0) != 1) {
        return -1;
    }
    control = CMSG_FIRSTHDR(&message);
    if (control == NULL || control->cmsg_level != SOL_SOCKET
        || control->cmsg_type != SCM_RIGHTS
        || control->cmsg_len != CMSG_LEN(sizeof(int))) {
        return -1;
    }
    memcpy(&mirror, CMSG_DATA(control), sizeof(int));
    return mirror;
}

int aotx_session_attach(aotx_session *s, const char *journal)
{
    struct sockaddr_un address;
    struct stat state;
    char path[sizeof(address.sun_path)];
    int dir_fd;
    int fd;
    int mirror;
    void *base;

    dir_fd = open(journal, O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (dir_fd < 0) {
        snprintf(s->reason, sizeof(s->reason), "the journal directory does not open");
        return -1;
    }
    snprintf(path, sizeof(path), "/proc/self/fd/%d/%s", dir_fd, AOTX_ATTACH_NAME);
    fd = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (fd < 0) {
        snprintf(s->reason, sizeof(s->reason), "the socket does not open");
        close(dir_fd);
        return -1;
    }
    memset(&address, 0, sizeof(address));
    address.sun_family = AF_UNIX;
    memcpy(address.sun_path, path, strlen(path));
    if (connect(fd, (struct sockaddr *)&address, sizeof(address)) != 0) {
        snprintf(s->reason, sizeof(s->reason), "no system answers at this journal");
        close(dir_fd);
        close(fd);
        return -1;
    }
    close(dir_fd);
    mirror = take_mirror(fd);
    if (mirror < 0) {
        /* The feeder sends a reason before it closes, so the read of that reason states
         * why the attach did not happen. */
        s->fd = fd;
        if (aotx_session_take(s) != 1) {
            snprintf(s->reason, sizeof(s->reason), "the system sent no mirror");
        }
        s->fd = -1;
        close(fd);
        return -1;
    }
    if (fstat(mirror, &state) != 0 || (size_t)state.st_size < sizeof(aotx_mirror_preamble)) {
        snprintf(s->reason, sizeof(s->reason), "the mirror has no size this build reads");
        close(mirror);
        close(fd);
        return -1;
    }
    base = mmap(NULL, (size_t)state.st_size, PROT_READ, MAP_SHARED, mirror, 0);
    if (base == MAP_FAILED) {
        snprintf(s->reason, sizeof(s->reason), "the mirror does not map");
        close(mirror);
        close(fd);
        return -1;
    }
    s->fd = fd;
    s->mirror_fd = mirror;
    s->mirror = (unsigned char *)base;
    s->mirror_bytes = (size_t)state.st_size;
    snprintf(s->journal, sizeof(s->journal), "%s", journal);
    s->reason[0] = '\0';
    return 0;
}

void aotx_session_detach(aotx_session *s)
{
    if (s->mirror != NULL) {
        munmap(s->mirror, s->mirror_bytes);
        s->mirror = NULL;
        s->mirror_bytes = 0;
    }
    if (s->mirror_fd >= 0) {
        close(s->mirror_fd);
        s->mirror_fd = -1;
    }
    if (s->fd >= 0) {
        close(s->fd);
        s->fd = -1;
    }
}

/* Writes every byte of one frame. Returns 0 or -1. */
static int send_all(int fd, const unsigned char *bytes, size_t count)
{
    size_t at = 0;
    while (at < count) {
        ssize_t sent = write(fd, bytes + at, count - at);
        if (sent > 0) {
            at += (size_t)sent;
            continue;
        }
        if (sent < 0 && errno == EINTR) {
            continue;
        }
        return -1;
    }
    return 0;
}

int aotx_session_key(aotx_session *s, const aotx_tui_key *key)
{
    unsigned char frame[1u + sizeof(aotx_key_body)];
    aotx_key_body body;
    if (s->fd < 0) {
        return -1;
    }
    memset(&body, 0, sizeof(body));
    body.key = key->code;
    body.codepoint = key->codepoint;
    body.action = 1u;  /* a press; the terminal sends no release */
    body.mods = key->mods;
    frame[0] = (unsigned char)AOTX_ATTACH_KEY;
    memcpy(frame + 1, &body, sizeof(body));
    if (send_all(s->fd, frame, sizeof(frame)) != 0) {
        return -1;
    }
    s->keys++;
    return 0;
}

int aotx_session_line(aotx_session *s, const char *line)
{
    unsigned char frame[5u + AOTX_BODY_BYTES];
    size_t bytes = strlen(line);
    if (s->fd < 0) {
        return -1;
    }
    if (bytes > AOTX_BODY_BYTES) {
        snprintf(s->reason, sizeof(s->reason), "the line is longer than one record body");
        return -1;
    }
    frame[0] = (unsigned char)AOTX_ATTACH_LINE;
    frame[1] = (unsigned char)(bytes & 0xffu);
    frame[2] = (unsigned char)((bytes >> 8) & 0xffu);
    frame[3] = 0;
    frame[4] = 0;
    memcpy(frame + 5, line, bytes);
    if (send_all(s->fd, frame, bytes + 5u) != 0) {
        return -1;
    }
    s->lines++;
    return 0;
}

int aotx_session_take(aotx_session *s)
{
    unsigned char buffer[AOTX_SESSION_READ];
    ssize_t got;
    unsigned int length;
    if (s->fd < 0) {
        return -1;
    }
    got = recv(s->fd, buffer, sizeof(buffer), MSG_DONTWAIT);
    if (got == 0) {
        return -1;
    }
    if (got < 0) {
        return (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) ? 0 : -1;
    }
    if (got < 5 || buffer[0] != (unsigned char)AOTX_ATTACH_REASON) {
        return 0;
    }
    length = (unsigned int)buffer[1] | ((unsigned int)buffer[2] << 8);
    if (length > (unsigned int)got - 5u) {
        length = (unsigned int)got - 5u;
    }
    if (length >= sizeof(s->reason)) {
        length = (unsigned int)sizeof(s->reason) - 1u;
    }
    memcpy(s->reason, buffer + 5, length);
    s->reason[length] = '\0';
    return 1;
}

int aotx_session_start(aotx_session *s, const char *program, const char *settings,
                       const char *journal, int restore)
{
    char log[AOTX_PATH_BYTES];
    const char *argv[10];
    unsigned int at = 0;
    int pid;
    int fd;

    if (s->boot_pid > 0) {
        snprintf(s->reason, sizeof(s->reason), "a system of this terminal runs already");
        return -1;
    }
    if (mkdir(journal, 0700) != 0 && errno != EEXIST) {
        snprintf(s->reason, sizeof(s->reason), "the journal directory does not open");
        return -1;
    }
    if (aotx_tui_join(log, sizeof(log), journal, AOTX_SESSION_LOG) != 0) {
        snprintf(s->reason, sizeof(s->reason), "the path of the boot log is too long");
        return -1;
    }
    fd = open(log, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
    if (fd < 0) {
        snprintf(s->reason, sizeof(s->reason), "the boot log does not open");
        return -1;
    }
    argv[at++] = program;
    argv[at++] = "--journal";
    argv[at++] = journal;
    if (settings != NULL && settings[0] != '\0') {
        argv[at++] = "--settings";
        argv[at++] = settings;
    }
    if (restore != 0) {
        argv[at++] = "--restore";
    }
    argv[at++] = "--tui-attached";
    argv[at] = NULL;
    pid = fork();
    if (pid < 0) {
        snprintf(s->reason, sizeof(s->reason), "the boot does not start");
        close(fd);
        return -1;
    }
    if (pid == 0) {
        /* The child writes what it prints to the log, so the screen shows the reason a
         * start did not work where the key was pressed. */
        int null = open("/dev/null", O_RDONLY);
        if (null >= 0) {
            dup2(null, 0);
            close(null);
        }
        dup2(fd, 1);
        dup2(fd, 2);
        close(fd);
        execv(program, (char *const *)argv);
        _exit(127);
    }
    close(fd);
    s->boot_pid = pid;
    snprintf(s->journal, sizeof(s->journal), "%s", journal);
    s->reason[0] = '\0';
    return 0;
}

int aotx_session_boot_state(aotx_session *s, int *status)
{
    int state = 0;
    int got;
    *status = 0;
    if (s->boot_pid <= 0) {
        return -1;
    }
    got = (int)waitpid(s->boot_pid, &state, WNOHANG);
    if (got == 0) {
        return 1;
    }
    if (got < 0) {
        s->boot_pid = -1;
        return 0;
    }
    if (WIFEXITED(state)) {
        *status = WEXITSTATUS(state);
    } else if (WIFSIGNALED(state)) {
        *status = 128 + WTERMSIG(state);
    }
    s->boot_pid = -1;
    return 0;
}

int aotx_session_phase(const char *journal, uint64_t now_seconds, char *out, size_t bytes)
{
    char line[96];
    char word[32];
    unsigned long long started = 0ull;
    unsigned long long elapsed;
    int dir_fd = open(journal, O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    int fd;
    ssize_t got;
    if (dir_fd < 0) {
        return (errno == ENOENT) ? 0 : -1;
    }
    fd = openat(dir_fd, AOTX_SESSION_PHASE, O_RDONLY | O_CLOEXEC);
    close(dir_fd);
    if (fd < 0) {
        return (errno == ENOENT) ? 0 : -1;
    }
    got = read(fd, line, sizeof(line) - 1u);
    close(fd);
    if (got <= 0) {
        return -1;
    }
    line[got] = '\0';
    if (sscanf(line, "%31s %llu", word, &started) != 2) {
        return -1;
    }
    if (strcmp(word, "closed") == 0) {
        return 0;
    }
    elapsed = (now_seconds > started) ? now_seconds - started : 0ull;
    if (strcmp(word, "placing") == 0) {
        snprintf(out, bytes, "placing models, %llu seconds", elapsed);
    } else if (strcmp(word, "replaying") == 0) {
        snprintf(out, bytes, "replaying the journal, %llu seconds", elapsed);
    } else if (strcmp(word, "running") == 0) {
        snprintf(out, bytes, "running, %llu seconds", elapsed);
    } else {
        return -1;
    }
    return 1;
}

unsigned int aotx_session_boot_log(const aotx_session *s, char *out, unsigned int rows,
                                   unsigned int cols)
{
    char path[AOTX_PATH_BYTES];
    char line[AOTX_TUI_LINE_BYTES];
    FILE *file;
    unsigned int count = 0;
    unsigned int at = 0;
    if (s->journal[0] == '\0' || rows == 0) {
        return 0;
    }
    if (aotx_tui_join(path, sizeof(path), s->journal, AOTX_SESSION_LOG) != 0) {
        return 0;
    }
    file = fopen(path, "r");
    if (file == NULL) {
        return 0;
    }
    /* The rows hold the last lines of the file. The write goes around the rows, so the
     * file is read one time and the memory is the rows and nothing more. */
    while (fgets(line, (int)sizeof(line), file) != NULL) {
        size_t bytes = strlen(line);
        while (bytes > 0 && (line[bytes - 1u] == '\n' || line[bytes - 1u] == '\r')) {
            line[--bytes] = '\0';
        }
        snprintf(out + (size_t)at * cols, cols, "%s", line);
        at = (at + 1u) % rows;
        count++;
    }
    fclose(file);
    if (count < rows) {
        return count;
    }
    /* The rows are in a ring; the read moves them so the oldest line is first. */
    {
        size_t head = (size_t)at * cols;
        char *hold;
        if (head == 0u) {
            return rows;
        }
        hold = (char *)malloc(head);
        if (hold == NULL) {
            return rows;
        }
        memcpy(hold, out, head);
        memmove(out, out + head, ((size_t)rows - at) * cols);
        memcpy(out + ((size_t)rows - at) * cols, hold, head);
        free(hold);
    }
    return rows;
}

int aotx_session_version(const char *program, char *out, size_t bytes)
{
    int pipes[2];
    int pid;
    ssize_t got = 0;
    int state = 0;
    out[0] = '\0';
    if (pipe2(pipes, O_CLOEXEC) != 0) {
        return -1;
    }
    pid = fork();
    if (pid < 0) {
        close(pipes[0]);
        close(pipes[1]);
        return -1;
    }
    if (pid == 0) {
        int null = open("/dev/null", O_RDWR);
        if (null >= 0) {
            dup2(null, 0);
            dup2(null, 2);
            close(null);
        }
        dup2(pipes[1], 1);
        close(pipes[1]);
        close(pipes[0]);
        execl(program, program, "--version", (char *)NULL);
        _exit(127);
    }
    close(pipes[1]);
    got = read(pipes[0], out, bytes - 1u);
    close(pipes[0]);
    waitpid(pid, &state, 0);
    if (got <= 0) {
        return -1;
    }
    out[got] = '\0';
    /* The first line is the one the status line shows. */
    {
        char *end = strchr(out, '\n');
        if (end != NULL) {
            *end = '\0';
        }
    }
    return 0;
}
