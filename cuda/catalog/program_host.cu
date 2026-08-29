/* Purpose: Run the program of a host tool over its example line and check the framing.
 * Owns: The child of the check and the bytes it wrote.
 * Launch shape: Host glue only; no kernel runs in this arm of the check.
 * Lifetime: One run of the check program.
 *
 * The feeder starts a host tool with the module directory as the working directory and the
 * requests line on the standard input. The check starts the program the same way, through
 * the same starter. It then waits for the program itself, because the poll of the feeder
 * writes to a ring that the check does not hold. */
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <stdio.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

#include "catalog/check.cuh"

extern "C" {
#include "disk/feed/modules.h"
#include "disk/feed/run_tool.h"
}

/* The milliseconds a wait step takes while the child runs. */
#define AOTX_CHECK_STEP_MS 20

/* Build the requests line the feeder gives a program: the shape the drain writes. */
static void aotx_check_line(char *out, size_t bytes, const char *name,
                            const aotx_check_entry *entry)
{
    size_t at = (size_t)snprintf(out, bytes,
                                 "{\"agent\":0,\"request\":1,\"tool\":\"%s\",\"arg\":\"",
                                 name);
    for (unsigned int i = 0u; i < entry->example_len && at + 8u < bytes; ++i) {
        unsigned char byte = (unsigned char)entry->example[i];
        if (byte < 0x20u || byte == '"' || byte == '\\') {
            at += (size_t)snprintf(out + at, bytes - at, "\\u%04x", (unsigned int)byte);
        } else {
            out[at] = (char)byte;
            at += 1u;
            out[at] = '\0';
        }
    }
    snprintf(out + at, bytes - at, "\"}");
}

/* Take one step of a pipe that does not block. The return is 1 while the pipe is open. */
static int aotx_check_step(int fd, char *out, unsigned int max, unsigned int *at)
{
    if (fd < 0 || *at + 1u >= max) {
        return 0;
    }
    ssize_t got = read(fd, out + *at, (size_t)(max - *at - 1u));
    if (got > 0) {
        *at += (unsigned int)got;
        out[*at] = '\0';
        return 1;
    }
    return (got < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) ? 1 : 0;
}

/* Read both pipes of the child until they end, or until the timeout passes. The feeder
 * makes the pipes of a child not block, so the read comes back at once and the wait
 * stands here. The return is the exit status, or the negative of a signal number. */
static int aotx_check_wait(aotx_child *kid, char *out, unsigned int out_max,
                           unsigned int *out_len, char *err, unsigned int err_max,
                           unsigned int *err_len, unsigned int timeout)
{
    struct pollfd fds[2];
    unsigned int steps = 0u;
    unsigned int limit = (timeout != 0u) ? (timeout * 1000u / AOTX_CHECK_STEP_MS) : 1000u;
    int alive_out = 1;
    int alive_err = 1;
    int status = 0;
    out[0] = '\0';
    err[0] = '\0';
    while ((alive_out != 0 || alive_err != 0) && steps < limit) {
        fds[0].fd = kid->out_fd;
        fds[0].events = POLLIN;
        fds[0].revents = 0;
        fds[1].fd = kid->err_fd;
        fds[1].events = POLLIN;
        fds[1].revents = 0;
        poll(fds, 2u, AOTX_CHECK_STEP_MS);
        if (alive_out != 0) {
            alive_out = aotx_check_step(kid->out_fd, out, out_max, out_len);
        }
        if (alive_err != 0) {
            alive_err = aotx_check_step(kid->err_fd, err, err_max, err_len);
        }
        steps += 1u;
    }
    if (waitpid(kid->pid, &status, 0) != kid->pid) {
        return -1;
    }
    kid->reaped = 1;
    return WIFEXITED(status) ? WEXITSTATUS(status) : -WTERMSIG(status);
}

int aotx_check_program(const char *dir, const aotx_check_entry *entry, const char *name,
                       unsigned int *applied, unsigned int *failed)
{
    aotx_children children;
    char line[1024];
    char out[AOTX_FS_CAP];
    char err[AOTX_TOOL_ERR_BYTES];
    char program[AOTX_CHECK_TEXT_BYTES + 2];
    const char *reason = "";
    unsigned int timeout = (entry->timeout != 0u) ? entry->timeout : AOTX_MODULE_TIMEOUT;

    memset(&children, 0, sizeof children);
    if (entry->program[0] == '\0') {
        printf("check: FAIL the manifest of a host tool names no program 0\n");
        *applied += 1u;
        *failed += 1u;
        return 1;
    }
    int dir_fd = open(dir, O_RDONLY | O_DIRECTORY);
    if (dir_fd < 0) {
        printf("check: FAIL the module directory opens 0\n");
        *applied += 1u;
        *failed += 1u;
        return 1;
    }
    snprintf(program, sizeof program, "./%s", entry->program);
    aotx_check_line(line, sizeof line, name, entry);
    char *argv[2];
    argv[0] = program;
    argv[1] = NULL;
    int rc = aotx_run_start(&children, 0u, 1u, dir_fd, program, argv, name, line, timeout,
                            &reason);
    *applied += 1u;
    if (rc != 0) {
        *failed += 1u;
        printf("check: FAIL the program starts: %s\n", reason);
        close(dir_fd);
        return 1;
    }
    printf("check: ok   the program starts under the timeout of seconds %u\n", timeout);

    /* The child writes its answer and ends. The check reads both pipes and then waits,
     * so a program that fills a pipe does not stop. */
    aotx_child *kid = &children.at[0];
    unsigned int bytes = 0u;
    unsigned int reason_bytes = 0u;
    int verdict = aotx_check_wait(kid, out, (unsigned int)sizeof out, &bytes, err,
                                  (unsigned int)sizeof err, &reason_bytes, timeout);
    close(kid->out_fd);
    close(kid->err_fd);
    kid->out_fd = -1;
    kid->err_fd = -1;
    kid->used = 0;
    aotx_run_close(&children);

    *applied += 1u;
    if (verdict != 0) {
        *failed += 1u;
        printf("check: FAIL the exit status of the program %d\n", verdict);
    } else {
        printf("check: ok   the exit status of the program %d\n", verdict);
    }
    *applied += 1u;
    if (bytes == 0u) {
        *failed += 1u;
        printf("check: FAIL the program wrote bytes 0\n");
    } else {
        printf("check: ok   the program wrote bytes %u\n", bytes);
    }
    *applied += 1u;
    if (bytes >= AOTX_FS_CAP) {
        *failed += 1u;
        printf("check: FAIL the answer is under the cap of bytes %u\n",
               (unsigned int)AOTX_FS_CAP);
    } else {
        printf("check: ok   the answer is under the cap of bytes %u\n",
               (unsigned int)AOTX_FS_CAP);
    }
    if (reason_bytes != 0u) {
        printf("check: the standard error of the program holds: %.120s\n", err);
    }
    close(dir_fd);
    return 0;
}
