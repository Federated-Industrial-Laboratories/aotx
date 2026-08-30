/* Purpose: Fill the Models screen and run its fetch or activate child.
 * Owns: The store rows of the last fill; the terminal state owns the child.
 * Threading: One terminal loop and one child process.
 * Lifetime: Store rows last to the next fill; a child lasts to its exit. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "disk/tui/tui.h"

#include "disk/models/models.h"

#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

static aotx_model_view model_rows[AOTX_TUI_ROWS_LIST];
static unsigned int model_count;

static void size_text(uint64_t bytes, char *out, size_t out_bytes)
{
    if (bytes >= 1024ull * 1024ull * 1024ull) {
        uint64_t gib = 1024ull * 1024ull * 1024ull;
        snprintf(out, out_bytes, "%llu.%llu GB",
                 (unsigned long long)(bytes / gib),
                 (unsigned long long)(((bytes % gib) * 10ull) / gib));
    } else if (bytes >= 1024ull * 1024ull) {
        snprintf(out, out_bytes, "%llu MB",
                 (unsigned long long)(bytes / (1024ull * 1024ull)));
    } else {
        snprintf(out, out_bytes, "%llu B", (unsigned long long)bytes);
    }
}

unsigned int aotx_rows_models(aotx_tui *tui, char *out, unsigned int rows,
                              unsigned int cols)
{
    aotx_model_catalog catalog;
    char reason[192];
    int count;
    int i;
    model_count = 0u;
    if (rows == 0u) {
        return 0u;
    }
    if (aotx_model_catalog_read(AOTX_MODELS_CATALOG, &catalog,
                                reason, sizeof(reason)) < 0) {
        snprintf(out, cols, "- %s", reason);
        return 1u;
    }
    count = aotx_model_store_scan(tui->settings.text[AOTX_SET_MODELS_DIR], &catalog,
                                  model_rows, rows, reason, sizeof(reason));
    if (count < 0) {
        snprintf(out, cols, "- %s", reason);
        return 1u;
    }
    model_count = (unsigned int)count;
    for (i = 0; i < count; i++) {
        const aotx_model_view *view = &model_rows[i];
        char size[32];
        uint64_t bytes = (view->bytes_on_disk != 0u)
                       ? view->bytes_on_disk : view->catalog.bytes;
        size_text(bytes, size, sizeof(size));
        snprintf(out + (size_t)i * cols, cols, "%s | %s | %s | %s | %s | %s | %s",
                 aotx_model_state_text(view->state), view->catalog.name,
                 view->catalog.role, view->catalog.quant, size,
                 view->catalog.source, view->verified ? "verified" : "not verified");
    }
    return model_count;
}

static int model_program(char *out, size_t bytes)
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

static int start_child(aotx_tui *tui, const aotx_model_view *view, int activate)
{
    char program[AOTX_PATH_BYTES];
    int pipes[2];
    int pid;
    if (tui->model_pid > 0) {
        snprintf(tui->says, sizeof(tui->says), "fetch %s runs", tui->model_name);
        return 1;
    }
    if (model_program(program, sizeof(program)) != 0 ||
        pipe2(pipes, O_CLOEXEC | O_NONBLOCK) != 0) {
        snprintf(tui->says, sizeof(tui->says), "the model child pipe does not open");
        return 1;
    }
    pid = fork();
    if (pid < 0) {
        close(pipes[0]);
        close(pipes[1]);
        snprintf(tui->says, sizeof(tui->says), "the model child does not start");
        return 1;
    }
    if (pid == 0) {
        char *fetch_args[] = { program, (char *)"--dir",
                               tui->settings.text[AOTX_SET_MODELS_DIR],
                               (char *)"fetch", (char *)view->catalog.name, NULL };
        char *active_args[] = { program, (char *)"--dir",
                                tui->settings.text[AOTX_SET_MODELS_DIR],
                                (char *)"activate", (char *)view->catalog.role,
                                (char *)view->catalog.name, NULL };
        int flags;
        dup2(pipes[1], 1);
        dup2(pipes[1], 2);
        close(pipes[0]);
        close(pipes[1]);
        flags = fcntl(1, F_GETFL, 0);
        if (flags >= 0) {
            fcntl(1, F_SETFL, flags & ~O_NONBLOCK);
            fcntl(2, F_SETFL, flags & ~O_NONBLOCK);
        }
        execv(program, activate ? active_args : fetch_args);
        _exit(127);
    }
    close(pipes[1]);
    tui->model_pid = pid;
    tui->model_fd = pipes[0];
    tui->model_fill = 0u;
    snprintf(tui->model_name, sizeof(tui->model_name), "%s", view->catalog.name);
    snprintf(tui->says, sizeof(tui->says), "%s %s started",
             activate ? "activate" : "fetch", view->catalog.name);
    return 1;
}

int aotx_models_action(aotx_tui *tui, unsigned int row)
{
    const aotx_model_view *view;
    char line[AOTX_TUI_LINE_BYTES];
    if (row >= model_count || model_rows[row].catalog.role[0] == '-' ||
        model_rows[row].catalog.name[0] == '\0') {
        snprintf(tui->says, sizeof(tui->says), "this row names no catalog entry");
        return 1;
    }
    view = &model_rows[row];
    if (view->state == AOTX_MODEL_ON_DISK) {
        if (tui->session.fd >= 0) {
            snprintf(line, sizeof(line), "model load %s %s",
                     view->catalog.role, view->catalog.name);
            aotx_screen_send(tui, line);
            return 1;
        }
        snprintf(tui->says, sizeof(tui->says), "a running system is needed to load %s",
                 view->catalog.name);
        return 1;
    }
    if (view->state == AOTX_MODEL_NOT_ACTIVE) {
        return start_child(tui, view, 1);
    }
    if (view->state == AOTX_MODEL_NOT_FETCHED || view->state == AOTX_MODEL_FETCHING) {
        return start_child(tui, view, 0);
    }
    snprintf(tui->says, sizeof(tui->says), "the digest of %s differs",
             view->catalog.name);
    return 1;
}

static void take_line(aotx_tui *tui)
{
    unsigned long long bytes;
    unsigned long long total;
    unsigned long long rate;
    tui->model_line[tui->model_fill] = '\0';
    if (sscanf(tui->model_line, "bytes %llu total %llu rate %llu",
               &bytes, &total, &rate) == 3) {
        snprintf(tui->says, sizeof(tui->says), "note fetch %s %llu of %llu",
                 tui->model_name, bytes, total);
        (void)rate;
    } else {
        snprintf(tui->says, sizeof(tui->says), "%.250s", tui->model_line);
    }
    tui->model_fill = 0u;
}

void aotx_models_poll(aotx_tui *tui)
{
    char data[512];
    int status = 0;
    int got;
    if (tui->model_pid <= 0) {
        return;
    }
    for (;;) {
        ssize_t count = read(tui->model_fd, data, sizeof(data));
        ssize_t i;
        if (count <= 0) {
            break;
        }
        for (i = 0; i < count; i++) {
            if (data[i] == '\n') {
                take_line(tui);
            } else if (tui->model_fill + 1u < sizeof(tui->model_line)) {
                tui->model_line[tui->model_fill++] = data[i];
            }
        }
    }
    got = waitpid(tui->model_pid, &status, WNOHANG);
    if (got <= 0) {
        return;
    }
    if (tui->model_fill != 0u) {
        take_line(tui);
    } else {
        snprintf(tui->says, sizeof(tui->says), "%s %s",
                 tui->model_name,
                 (WIFEXITED(status) && WEXITSTATUS(status) == 0) ? "is on disk" : "failed");
    }
    close(tui->model_fd);
    tui->model_fd = -1;
    tui->model_pid = -1;
    tui->rows_ns = 0u;
}

void aotx_models_close(aotx_tui *tui)
{
    if (tui->model_pid > 0) {
        kill(tui->model_pid, SIGTERM);
        waitpid(tui->model_pid, NULL, 0);
    }
    if (tui->model_fd >= 0) {
        close(tui->model_fd);
    }
    tui->model_pid = -1;
    tui->model_fd = -1;
}
