/* Purpose: Check essential child exits and bounded cleanup with real processes.
 * Owns: Independent sets of child programs and their status observations.
 * Launch shape: Host process batches at N=1 and N=64; no device kernels.
 * Lifetime: Each batch reaps every child before it ends. */
#include "boot/boot.cuh"
#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

static unsigned checks, failures;
static void check(bool value, const char *message) {
    ++checks;
    if (!value) { ++failures; fprintf(stderr, "FAIL: %s\n", message); }
}
static void exit_zero(int signal) { (void)signal; _exit(0); }
static int child(bool ignore_term = false) {
    int ready[2];
    if (pipe(ready)) exit(2);
    pid_t pid = fork();
    if (pid < 0) exit(2);
    if (!pid) {
        close(ready[0]);
        signal(SIGUSR1, exit_zero);
        if (ignore_term) signal(SIGTERM, SIG_IGN);
        if (write(ready[1], "R", 1) != 1) _exit(2);
        close(ready[1]);
        for (;;) pause();
    }
    close(ready[1]); char mark = 0;
    if (read(ready[0], &mark, 1) != 1 || mark != 'R') exit(2);
    close(ready[0]); return pid;
}
static int *slot(aotx_boot_children *value, unsigned role) {
    if (!role) return &value->drain;
    if (role == 1) return &value->feed;
    if (role == 2) return &value->service;
    return &value->tui;
}
static void batch(unsigned n) {
    for (unsigned role = 0; role < 4; ++role) {
        aotx_boot_children values[64] = {};
        for (unsigned i = 0; i < n; ++i) *slot(values + i, role) = child();
        for (unsigned i = 0; i < n; ++i) {
            check(!aotx_boot_children_check(values + i), "a live child keeps the run active");
            kill(*slot(values + i, role), i % 2 ? SIGKILL : SIGUSR1);
        }
        for (unsigned i = 0; i < n; ++i) {
            int rc = 0;
            for (unsigned k = 0; *slot(values + i, role) && k < 200; ++k) {
                rc |= aotx_boot_children_check(values + i); usleep(1000);
            }
            check(rc == (role < 3), "only an essential child exit fails the run");
            check(!*slot(values + i, role), "the completed child is reaped and cleared");
            check(!aotx_boot_children_check(values + i), "a completed child is not reported twice");
            aotx_boot_children_abort(values + i);
        }
    }
}
static void shutdown_batch(unsigned n) {
    for (unsigned role = 0; role < 3; ++role) {
        aotx_boot_children values[64] = {};
        for (unsigned i = 0; i < n; ++i) *slot(values + i, role) = child();
        for (unsigned i = 0; i < n; ++i)
            kill(*slot(values + i, role), i % 2 ? SIGKILL : SIGUSR1);
        for (unsigned i = 0; i < n; ++i) {
            int pid = *slot(values + i, role), status = 0;
            check(aotx_boot_stop(values + i) == (int)(i % 2),
                  "shutdown accepts zero exits and reports essential failures");
            check(!*slot(values + i, role), "shutdown clears each completed child");
            check(waitpid(pid, &status, WNOHANG) == -1 && errno == ECHILD,
                  "shutdown leaves no owned child or zombie");
        }
    }
}
static void abort_run(void) {
    aotx_boot_children value = {};
    value.drain = child(true); value.feed = child(); value.service = child();
    value.tui = child(); value.restore = child();
    int pids[] = {value.drain, value.feed, value.service, value.tui, value.restore};
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    aotx_boot_children_abort(&value);
    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed = end.tv_sec - start.tv_sec + (end.tv_nsec - start.tv_nsec) / 1e9;
    check(elapsed >= 0.9 && elapsed < 5, "a child that ignores termination receives a bounded forced stop");
    check(!value.drain && !value.feed && !value.service && !value.tui && !value.restore,
          "failed-run cleanup clears every owned child");
    for (unsigned i = 0; i < 5; ++i) {
        int status = 0;
        check(waitpid(pids[i], &status, WNOHANG) == -1 && errno == ECHILD,
              "cleanup leaves no owned child or zombie");
    }
}
int main(void) {
    batch(1); batch(64); shutdown_batch(1); shutdown_batch(64); abort_run();
    printf("boot children: %u checks, %u failures\n", checks, failures);
    return failures || checks != 1632 ? 1 : 0;
}
