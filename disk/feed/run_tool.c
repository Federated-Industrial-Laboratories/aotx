/* Purpose: Run a host tool as a child program and publish what the program wrote.
 * Owns: The children of the feeder, their pipes and the bytes they wrote.
 * Threading: One thread; the feeder starts and reaps every child from its poll loop.
 * Lifetime: From the first program to the close of the feeder. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "disk/feed/run_tool.h"

#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

extern char **environ;

/* Gives the monotonic time in nanoseconds. The timeout counts on this clock, because the
 * wall clock can step and a step must not end a program early or late. */
static uint64_t mono_ns(void)
{
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        return 0;
    }
    return (uint64_t)ts.tv_sec * 1000000000u + (uint64_t)ts.tv_nsec;
}

/* Makes a file in memory that holds the requests line and starts at its first byte. The
 * child reads its standard input from it, so no write of the feeder can wait on a child
 * that reads nothing. Returns the descriptor, or -1. */
static int line_input(const char *line)
{
    size_t len = strlen(line);
    int fd = memfd_create("aotx_request", 0);
    if (fd < 0) {
        return -1;
    }
    if (len > 0 && write(fd, line, len) != (ssize_t)len) {
        close(fd);
        return -1;
    }
    if (write(fd, "\n", 1) != 1 || lseek(fd, 0, SEEK_SET) != 0) {
        close(fd);
        return -1;
    }
    return fd;
}

/* Gives a free place of the table, or null. */
static aotx_child *free_place(aotx_children *c)
{
    uint32_t i;
    for (i = 0; i < AOTX_TOOL_PROGRAMS_MAX; i++) {
        if (c->at[i].used == 0) {
            return &c->at[i];
        }
    }
    return NULL;
}

/* Runs the program in the child process. The function returns only when the program does
 * not start, and the caller then ends the child. */
static void child_run(int dir_fd, const char *file, char *const argv[], int input_fd,
                      int out_fd, int err_fd, uint32_t agent, uint32_t request,
                      const char *tool)
{
    char text[32];
    /* The kernel ends the child when the feeder ends, so no program of the run stays
     * behind the feeder that started it. */
    aotx_die_with_parent();
    /* The child leads a process group of its own. A program that starts programs of its
     * own thus ends with them at the timeout. No program of it then holds the pipe that
     * the reply reads. */
    setpgid(0, 0);
    if (fchdir(dir_fd) != 0) {
        return;
    }
    if (dup2(input_fd, 0) < 0 || dup2(out_fd, 1) < 0 || dup2(err_fd, 2) < 0) {
        return;
    }
    snprintf(text, sizeof(text), "%u", request);
    setenv("AOTX_REQUEST", text, 1);
    snprintf(text, sizeof(text), "%u", agent);
    setenv("AOTX_AGENT", text, 1);
    setenv("AOTX_TOOL", tool, 1);
    execve(file, argv, environ);
}

int aotx_run_start(aotx_children *c, uint32_t agent, uint32_t request, int dir_fd,
                   const char *file, char *const argv[], const char *tool, const char *line,
                   uint32_t timeout, const char **reason)
{
    aotx_child *k = free_place(c);
    int out_fds[2];
    int err_fds[2];
    int input_fd;
    int pid;
    if (k == NULL) {
        c->refused++;
        *reason = "the feeder runs as many programs as it can at one time";
        return 1;
    }
    input_fd = line_input(line);
    if (input_fd < 0) {
        *reason = "the standard input of the program does not open";
        return 1;
    }
    if (pipe(out_fds) != 0) {
        close(input_fd);
        *reason = "the output pipe of the program does not open";
        return 1;
    }
    if (pipe(err_fds) != 0) {
        close(input_fd);
        close(out_fds[0]);
        close(out_fds[1]);
        *reason = "the error pipe of the program does not open";
        return 1;
    }
    /* The read end alone is not blocking. The write end keeps the shape a program expects,
     * so a program that fills the pipe waits and loses no byte. */
    fcntl(out_fds[0], F_SETFL, O_NONBLOCK);
    fcntl(err_fds[0], F_SETFL, O_NONBLOCK);
    pid = (int)fork();
    if (pid < 0) {
        close(input_fd);
        close(out_fds[0]);
        close(out_fds[1]);
        close(err_fds[0]);
        close(err_fds[1]);
        *reason = "the program does not start";
        return 1;
    }
    if (pid == 0) {
        child_run(dir_fd, file, argv, input_fd, out_fds[1], err_fds[1], agent, request, tool);
        _exit(127);
    }
    /* The group is made in the parent as well, so a signal that the timeout sends cannot
     * come before the child made it. */
    setpgid((pid_t)pid, (pid_t)pid);
    close(input_fd);
    close(out_fds[1]);
    close(err_fds[1]);
    memset(k, 0, sizeof(*k));
    k->used = 1;
    k->pid = pid;
    k->out_fd = out_fds[0];
    k->err_fd = err_fds[0];
    k->agent = agent;
    k->request = request;
    k->timeout = timeout;
    k->deadline_ns = mono_ns() + (uint64_t)timeout * 1000000000u;
    c->started++;
    return 0;
}

/* Takes what one pipe holds now. The read never waits, so one slow program cannot hold the
 * poll loop. The descriptor goes to -1 at the end of the pipe. */
static void take_pipe(int *fd, unsigned char *out, uint32_t *len, uint32_t cap, int *over)
{
    unsigned char drop[4096];
    for (;;) {
        ssize_t n;
        if (*len < cap) {
            n = read(*fd, out + *len, (size_t)(cap - *len));
            if (n > 0) {
                *len += (uint32_t)n;
                continue;
            }
        } else {
            /* The bytes over the cap are read and dropped. A pipe that fills thus does
             * not hold the program, and the reply states the cut. */
            n = read(*fd, drop, sizeof(drop));
            if (n > 0) {
                if (over != NULL) {
                    *over = 1;
                }
                continue;
            }
        }
        if (n == 0) {
            close(*fd);
            *fd = -1;
        }
        return;
    }
}

/* Appends text to the reason of a reply. The reason must fit one part, so a text that does
 * not fit is cut and no byte goes past the buffer. */
static void add_reason(char *out, size_t bytes, size_t *used, const char *text)
{
    size_t left = (bytes > *used + 1u) ? bytes - *used - 1u : 0u;
    size_t take = strlen(text);
    if (take > left) {
        take = left;
    }
    memcpy(out + *used, text, take);
    *used += take;
    out[*used] = '\0';
}

/* Writes the reason of one program that ended. Returns the reason, or null when the
 * program ended well and wrote every byte it had. */
static const char *end_reason(const aotx_child *k, char *out, size_t bytes)
{
    size_t used = 0;
    char text[96];
    out[0] = '\0';
    if (k->killed) {
        snprintf(text, sizeof(text),
                 "the program ran longer than the timeout of %u seconds", k->timeout);
        add_reason(out, bytes, &used, text);
        return out;
    }
    if (k->status < 0) {
        snprintf(text, sizeof(text), "a signal of the number %d ended the program", -k->status);
        add_reason(out, bytes, &used, text);
    } else if (k->status != 0) {
        snprintf(text, sizeof(text), "the program ended with the status %d", k->status);
        add_reason(out, bytes, &used, text);
    }
    if (k->status != 0) {
        if (k->err_len > 0) {
            add_reason(out, bytes, &used, ": ");
            add_reason(out, bytes, &used, k->err);
        }
    }
    if (k->out_over) {
        snprintf(text, sizeof(text),
                 "the output is longer than the cap and the reply holds the first %u bytes",
                 (unsigned)AOTX_FS_CAP);
        add_reason(out, bytes, &used, (used > 0u) ? "; " : "");
        add_reason(out, bytes, &used, text);
    }
    return (used > 0u) ? out : NULL;
}

/* Publishes the reply of one program and frees its place. Returns 0 or -1. */
static int answer(aotx_fs_tool *t, const aotx_inbound_ring *ring,
                  const volatile sig_atomic_t *stop, aotx_child *k)
{
    char text[AOTX_TOOL_REPLY_BYTES + 1];
    const char *reason = end_reason(k, text, sizeof(text));
    int rc;
    if (k->killed) {
        /* A program the timeout ended gives no content: the bytes it wrote are a part of
         * work that did not finish. */
        rc = aotx_fs_put_reason(t, ring, stop, k->agent, k->request, AOTX_TOOL_ERROR, reason);
    } else {
        rc = aotx_fs_put_bytes(t, ring, stop, k->agent, k->request, k->out, k->out_len,
                               reason);
    }
    k->used = 0;
    return rc;
}

int aotx_run_poll(aotx_fs_tool *t, const aotx_inbound_ring *ring,
                  const volatile sig_atomic_t *stop)
{
    aotx_children *c = t->kids;
    uint64_t now;
    uint32_t i;
    if (c == NULL) {
        return 0;
    }
    now = mono_ns();
    for (i = 0; i < AOTX_TOOL_PROGRAMS_MAX; i++) {
        aotx_child *k = &c->at[i];
        int state = 0;
        if (k->used == 0) {
            continue;
        }
        if (k->out_fd >= 0) {
            take_pipe(&k->out_fd, k->out, &k->out_len, AOTX_FS_CAP, &k->out_over);
        }
        if (k->err_fd >= 0) {
            take_pipe(&k->err_fd, (unsigned char *)k->err, &k->err_len,
                      AOTX_TOOL_ERR_BYTES - 1u, NULL);
            k->err[k->err_len] = '\0';
        }
        if (!k->killed && now >= k->deadline_ns) {
            /* The program passed its timeout. The signal goes to the group of the child,
             * so a program that the program started ends with it. The signal cannot be
             * caught, and every pipe of the group closes. */
            kill((pid_t)-k->pid, SIGKILL);
            k->killed = 1;
            c->killed++;
        }
        if (!k->reaped && waitpid((pid_t)k->pid, &state, WNOHANG) == (pid_t)k->pid) {
            k->reaped = 1;
            k->status = WIFEXITED(state) ? WEXITSTATUS(state) : -WTERMSIG(state);
        }
        if (k->reaped && k->out_fd < 0 && k->err_fd < 0) {
            c->ended++;
            if (answer(t, ring, stop, k) != 0) {
                return -1;
            }
        }
    }
    return 0;
}

void aotx_run_close(aotx_children *c)
{
    uint32_t i;
    for (i = 0; i < AOTX_TOOL_PROGRAMS_MAX; i++) {
        aotx_child *k = &c->at[i];
        int state = 0;
        if (k->used == 0) {
            continue;
        }
        if (!k->reaped) {
            kill((pid_t)-k->pid, SIGKILL);
            waitpid((pid_t)k->pid, &state, 0);
        }
        if (k->out_fd >= 0) {
            close(k->out_fd);
        }
        if (k->err_fd >= 0) {
            close(k->err_fd);
        }
        k->used = 0;
    }
}
