/* Purpose: Run a model fetch child and publish its progress as console input notes.
 * Owns: One child and its nonblocking output pipe.
 * Threading: One feeder loop; the child is a separate process.
 * Lifetime: From initialization to close. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "disk/feed/model_fetch.h"

#include "disk/feed/line.h"

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

/* The parser echoes this fixed progress note to the console. */
#define AOTX_FETCH_NOTE "note fetch"

static int publish(aotx_fetch_child *child, const char *tail,
                   const aotx_inbound_ring *ring,
                   const volatile sig_atomic_t *stop)
{
    char line[AOTX_INPUT_LINE_BYTES];
    int wrote = snprintf(line, sizeof(line), "%s %s %s", AOTX_FETCH_NOTE,
                         child->name, tail);
    if (wrote < 0 || (size_t)wrote >= sizeof(line)) {
        return 0;
    }
    return aotx_line_publish(ring, stop, (const unsigned char *)line, (uint32_t)wrote);
}

static int program_path(char *out, size_t bytes)
{
    char path[AOTX_PATH_BYTES];
    ssize_t got = readlink("/proc/self/exe", path, sizeof(path) - 1u);
    char *slash;
    int wrote;
    if (got <= 0 || (size_t)got >= sizeof(path)) {
        return -1;
    }
    path[got] = '\0';
    slash = strrchr(path, '/');
    if (slash == NULL) {
        return -1;
    }
    *slash = '\0';
    wrote = snprintf(out, bytes, "%s/aotx_models", path);
    return (wrote < 0 || (size_t)wrote >= bytes) ? -1 : 0;
}

void aotx_fetch_child_init(aotx_fetch_child *child, const char *directory)
{
    memset(child, 0, sizeof(*child));
    child->pid = -1;
    child->fd = -1;
    snprintf(child->directory, sizeof(child->directory), "%s",
             (directory != NULL && directory[0] != '\0') ? directory : "models");
    if (program_path(child->program, sizeof(child->program)) != 0) {
        snprintf(child->program, sizeof(child->program), "aotx_models");
    }
}

static int name_ok(const unsigned char *name, uint32_t bytes)
{
    uint32_t i;
    if (bytes == 0u || bytes >= 96u) {
        return 0;
    }
    for (i = 0u; i < bytes; i++) {
        unsigned char c = name[i];
        if (!(isalnum(c) || c == '-' || c == '_' || c == '.')) {
            return 0;
        }
    }
    return 1;
}

static int start(aotx_fetch_child *child)
{
    int pipes[2];
    int pid;
    if (pipe2(pipes, O_CLOEXEC | O_NONBLOCK) != 0) {
        return -1;
    }
    pid = fork();
    if (pid < 0) {
        close(pipes[0]);
        close(pipes[1]);
        return -1;
    }
    if (pid == 0) {
        const char *catalog = getenv("AOTX_MODEL_CATALOG");
        char *args[9];
        int at = 0;
        args[at++] = child->program;
        args[at++] = (char *)"--dir";
        args[at++] = child->directory;
        if (catalog != NULL && catalog[0] != '\0') {
            args[at++] = (char *)"--catalog";
            args[at++] = (char *)catalog;
        }
        args[at++] = (char *)"fetch";
        args[at++] = child->name;
        args[at] = NULL;
        int flags;
        dup2(pipes[1], 1);
        dup2(pipes[1], 2);
        close(pipes[0]);
        close(pipes[1]);
        /* The child may block on its network pipe. Its standard output must block as well,
         * because a short progress line must not be lost on EAGAIN. */
        flags = fcntl(1, F_GETFL, 0);
        if (flags >= 0) {
            fcntl(1, F_SETFL, flags & ~O_NONBLOCK);
            fcntl(2, F_SETFL, flags & ~O_NONBLOCK);
        }
        execv(child->program, args);
        _exit(127);
    }
    close(pipes[1]);
    child->pid = pid;
    child->fd = pipes[0];
    child->fill = 0u;
    child->started++;
    return 0;
}

int aotx_fetch_child_line(aotx_fetch_child *child, const unsigned char *line, uint32_t bytes,
                          const aotx_inbound_ring *ring,
                          const volatile sig_atomic_t *stop)
{
    static const char head[] = "model fetch ";
    uint32_t name_bytes;
    char held[96];
    if (bytes < sizeof(head) || memcmp(line, head, sizeof(head) - 1u) != 0) {
        return 0;
    }
    name_bytes = bytes - (uint32_t)sizeof(head) + 1u;
    if (!name_ok(line + sizeof(head) - 1u, name_bytes)) {
        snprintf(held, sizeof(held), "%.*s", (int)name_bytes,
                 line + sizeof(head) - 1u);
        snprintf(child->name, sizeof(child->name), "%s", held);
        return publish(child, "refused because the model name is not valid", ring, stop) != 0
             ? -1 : 1;
    }
    if (child->pid > 0) {
        char tail[192];
        snprintf(held, sizeof(held), "%.*s", (int)name_bytes,
                 line + sizeof(head) - 1u);
        snprintf(tail, sizeof(tail), "refused because fetch %s runs", child->name);
        child->refused++;
        {
            char current[96];
            snprintf(current, sizeof(current), "%s", child->name);
            snprintf(child->name, sizeof(child->name), "%s", held);
            if (publish(child, tail, ring, stop) != 0) {
                snprintf(child->name, sizeof(child->name), "%s", current);
                return -1;
            }
            snprintf(child->name, sizeof(child->name), "%s", current);
        }
        return 1;
    }
    snprintf(child->name, sizeof(child->name), "%.*s", (int)name_bytes,
             line + sizeof(head) - 1u);
    if (start(child) != 0) {
        return publish(child, "failed because the child did not start", ring, stop) != 0
             ? -1 : 1;
    }
    return publish(child, "started", ring, stop) != 0 ? -1 : 1;
}

static int take_output_line(aotx_fetch_child *child, const char *line,
                            const aotx_inbound_ring *ring,
                            const volatile sig_atomic_t *stop)
{
    unsigned long long bytes;
    unsigned long long total;
    unsigned long long rate;
    char tail[256];
    if (sscanf(line, "bytes %llu total %llu rate %llu", &bytes, &total, &rate) == 3) {
        snprintf(tail, sizeof(tail), "%llu of %llu", bytes, total);
        (void)rate;
        return publish(child, tail, ring, stop);
    }
    if (strncmp(line, "host ", 5u) == 0 || strncmp(line, "restart ", 8u) == 0) {
        return publish(child, line, ring, stop);
    }
    return 0;
}

static int take_output(aotx_fetch_child *child, const aotx_inbound_ring *ring,
                       const volatile sig_atomic_t *stop)
{
    char bytes[512];
    for (;;) {
        ssize_t got = read(child->fd, bytes, sizeof(bytes));
        ssize_t i;
        if (got < 0 && (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR)) {
            return 0;
        }
        if (got <= 0) {
            return 0;
        }
        for (i = 0; i < got; i++) {
            if (bytes[i] == '\n') {
                child->line[child->fill] = '\0';
                if (take_output_line(child, child->line, ring, stop) != 0) {
                    return -1;
                }
                child->fill = 0u;
            } else if (child->fill + 1u < sizeof(child->line)) {
                child->line[child->fill++] = bytes[i];
            }
        }
    }
}

int aotx_fetch_child_poll(aotx_fetch_child *child, const aotx_inbound_ring *ring,
                          const volatile sig_atomic_t *stop)
{
    int state = 0;
    int got;
    if (child->pid <= 0) {
        return 0;
    }
    if (take_output(child, ring, stop) != 0) {
        return -1;
    }
    got = waitpid(child->pid, &state, WNOHANG);
    if (got <= 0) {
        return 0;
    }
    if (child->fill != 0u) {
        child->line[child->fill] = '\0';
        take_output_line(child, child->line, ring, stop);
    }
    close(child->fd);
    child->fd = -1;
    child->pid = -1;
    child->fill = 0u;
    child->ended++;
    return publish(child, (WIFEXITED(state) && WEXITSTATUS(state) == 0)
                          ? "on disk" : "failed", ring, stop);
}

void aotx_fetch_child_close(aotx_fetch_child *child)
{
    if (child->pid > 0) {
        kill(child->pid, SIGTERM);
        waitpid(child->pid, NULL, 0);
    }
    if (child->fd >= 0) {
        close(child->fd);
    }
    child->pid = -1;
    child->fd = -1;
}
