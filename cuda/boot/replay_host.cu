/* Purpose: Restore packaged or ordinary state before the input feeder starts.
 * Owns: The temporary restore child and the replay admission boundary.
 * Launch shape: Host glue drives complete tick batches until the child ends.
 * Lifetime: One boot recovery or initial runtime import. */
#include "boot/boot.cuh"
#include "boot/check.h"
#include "catalog/catalog.cuh"
#include <cuda_runtime.h>
#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <unistd.h>

int aotx_boot_replay(aotx_boot_children *children, const aotx_seam_rings *rings,
                     const aotx_boot_options *options, aotx_pump *pump, int (*cancelled)(void))
{
    const char *journal = options->ccir ? options->ccir : options->journal;
    int seed = options->runtime_seed != NULL;
    char fd[32];
    aotx_pump_report report;
    snprintf(fd, sizeof fd, "%d", rings->inbound_fd);
    char *restore[] = { (char *)"aotx_restore", (char *)"--inbound-fd", fd,
        (char *)(options->ccir ? "--ccir" : "--journal"), (char *)journal, NULL };
    char *create[] = { (char *)"aotx_feed", (char *)"--inbound-fd", fd,
        (char *)"--seed-only", (char *)"--runtime-seed", (char *)journal,
        (char *)"--settings", (char *)options->settings,
        (char *)"--modules", (char *)options->modules, NULL };
    const int keep[] = { rings->inbound_fd };
    if (aotx_boot_start(seed ? "aotx_feed" : "aotx_restore", seed ? create : restore,
                         keep, 1u, &children->restore) != 0) return 1;
    /* A replay applies the keys of the journal again. The flag makes the command layer
     * refuse to close the run while those keys go through it. */
    aotx_seam_set_replaying(seed ? 0 : 1);

    /* The tick load stays off while the journal is replayed, so the replay records are the
     * only records of these ticks. */
    aotx_pump_set(pump, 0ull, 1u);

    /* Wait for the restore program to select its source journal before the first tick.
     * Otherwise, a new block can make this boot the newest complete journal. */
    const volatile aotx_inbound_preamble *inbound =
        (const volatile aotx_inbound_preamble *)rings->inbound_map;
    int stopped = 0;
    int status = 0;
    int ended = 0;
    /* File validation has no fixed duration. A runtime waits for its reader or cancellation. */
    unsigned long long idle = 0ull;
    unsigned long long seen_head = 0ull;
    unsigned long long seen_consumed = 0ull;
    while ((options->ccir || idle < 1000000ull) && (!cancelled || !cancelled())) {
        if (stopped == 0) {
            if (aotx_seam_poll(children->restore, &stopped, &status)) {
                if (errno == EINTR) continue;
                stopped = 1; status = 1; break;
            }
        }
        if (stopped == 0 && inbound->head == 0ull) {
            usleep(200);
            idle += 1ull;
            continue;
        }
        aotx_pump_tick(pump);
        if (pump->model_refused != 0u) {
            fprintf(stderr, "restore: a model file was refused\n");
            break;
        }
        if (inbound->head != seen_head || inbound->consumed != seen_consumed) {
            seen_head = inbound->head;
            seen_consumed = inbound->consumed;
            idle = 0ull;
        } else {
            idle += 1ull;
        }
        if (stopped != 0 && inbound->consumed >= inbound->head) {
            ended = 1;
            break;
        }
    }
    aotx_seam_set_replaying(0);
    if (!stopped && children->restore > 0) {
        kill((pid_t)children->restore, SIGKILL);
        aotx_seam_wait(children->restore);
    }
    children->restore = 0;
    aotx_pump_read(&report);
    printf("restore: applied %llu hash %llx decode_refused %u pages %u paced %llu rejected %llu\n",
           report.applied, report.state_hash, report.refused, report.pages, report.paced, report.rejected);
    if (ended == 0) {
        fprintf(stderr, "restore: the replay did not finish; the run stops\n");
        return 1;
    }
    if (report.rejected != 0ull) {
        fprintf(stderr, "restore: a journal record was refused; the run stops\n");
        return 1;
    }
    if (seed) {
        unsigned refused = 0;
        aotx_check_runtime(cudaMemcpyFromSymbol(&refused, aotx_catalog, sizeof(refused),
            offsetof(aotx_catalog_state, count) + offsetof(aotx_catalog_counts, refused)), "cudaMemcpyFromSymbol");
        if (refused || !report.console_agent) {
            fprintf(stderr, "runtime: the initial module set was refused\n");
            return 1;
        }
    }
    return (status == 0) ? 0 : 1;
}
