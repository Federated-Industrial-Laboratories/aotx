/* Purpose: Stop the runtime when an essential child program ends.
 * Owns: Child status reads and bounded cleanup after a failed run.
 * Launch shape: Host process control only; no device state changes.
 * Lifetime: From input readiness through child cleanup. */
#include "boot/boot.cuh"
#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <unistd.h>

static int aotx_boot_child_check(int *pid, const char *name, int closing)
{
    if (*pid <= 0) return 0;
    int ended = 0, status = 0;
    if (aotx_seam_poll(*pid, &ended, &status)) {
        if (errno == EINTR) return 0;
        if (errno == ECHILD) *pid = 0;
        fprintf(stderr, "boot: the %s status did not read\n", name);
        return 1;
    }
    if (!ended) return 0;
    *pid = 0;
    if (closing && !status) return 0;
    fprintf(stderr, "boot: the %s ended during %s (code %d)\n", name,
            closing ? "shutdown" : "the run", status);
    return 1;
}

int aotx_boot_children_check(aotx_boot_children *children)
{
    int bad = aotx_boot_child_check(&children->drain, "disk writer", 0);
    bad |= aotx_boot_child_check(&children->feed, "feeder", 0);
    bad |= aotx_boot_child_check(&children->service, "service", 0);
    aotx_boot_reap_tui(children);
    return bad;
}

/* Ring closure permits zero exits. A failed child stops the remaining children. */
int aotx_boot_stop(aotx_boot_children *children)
{
    if (children->tui > 0) kill(children->tui, SIGTERM);
    for (;;) {
        int bad = aotx_boot_child_check(&children->drain, "disk writer", 1);
        bad |= aotx_boot_child_check(&children->feed, "feeder", 1);
        bad |= aotx_boot_child_check(&children->service, "service", 1);
        bad |= aotx_boot_child_check(&children->restore, "restore", 1);
        aotx_boot_reap_tui(children);
        if (bad || (!children->drain && !children->feed &&
                    !children->service && !children->restore)) {
            aotx_boot_children_abort(children);
            return bad;
        }
        usleep(10000);
    }
}

void aotx_boot_children_abort(aotx_boot_children *children)
{
    int *pids[] = { &children->service, &children->tui, &children->feed,
                   &children->drain, &children->restore };
    for (unsigned i = 0; i < 5; ++i) if (*pids[i] > 0) kill(*pids[i], SIGTERM);
    for (unsigned attempt = 0; attempt < 100; ++attempt) {
        unsigned pending = 0;
        for (unsigned i = 0; i < 5; ++i) if (*pids[i] > 0) {
            int ended = 0, status = 0;
            int rc = aotx_seam_poll(*pids[i], &ended, &status);
            if ((!rc && ended) || (rc && errno == ECHILD)) *pids[i] = 0;
            else ++pending;
        }
        if (!pending) return;
        usleep(10000);
    }
    for (unsigned i = 0; i < 5; ++i) if (*pids[i] > 0) kill(*pids[i], SIGKILL);
    for (unsigned i = 0; i < 5; ++i) if (*pids[i] > 0) {
        while (aotx_seam_wait(*pids[i]) < 0 && errno == EINTR) {}
        *pids[i] = 0;
    }
}
