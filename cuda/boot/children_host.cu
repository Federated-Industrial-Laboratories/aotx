/* Purpose: Start the disk side programs and run the replay that a restore needs.
 * Owns: The process id of each disk side program.
 * Launch shape: Host glue only; the pump supplies the kernels.
 * Lifetime: From the first start to the last wait. */
#include <limits.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "boot/boot.cuh"

/* The disk side programs sit beside this one, so the path of this program gives them. */
int aotx_boot_sibling(const char *name, char *path, unsigned int bytes)
{
    char self[PATH_MAX];
    ssize_t length = readlink("/proc/self/exe", self, sizeof self - 1u);
    if (length <= 0) {
        return 1;
    }
    self[length] = '\0';
    char *slash = strrchr(self, '/');
    if (slash == NULL) {
        return 1;
    }
    *slash = '\0';
    if (snprintf(path, bytes, "%s/%s", self, name) >= (int)bytes) {
        return 1;
    }
    return (access(path, X_OK) == 0) ? 0 : 1;
}

static int aotx_boot_start(const char *name, char *const argv[], int *pid)
{
    char path[PATH_MAX];
    if (aotx_boot_sibling(name, path, (unsigned int)sizeof path) != 0) {
        fprintf(stderr, "cannot find %s beside this program\n", name);
        return 1;
    }
    if (aotx_seam_spawn(path, argv, pid) != 0) {
        fprintf(stderr, "cannot start %s\n", name);
        return 1;
    }
    return 0;
}

int aotx_boot_start_drain(aotx_boot_children *children, const aotx_seam_rings *rings,
                          const char *journal)
{
    char fd[32];
    snprintf(fd, sizeof fd, "%d", rings->host_fd);
    char *argv[] = { (char *)"aotx_drain", (char *)"--ring-fd", fd,
                     (char *)"--journal", (char *)journal, NULL };
    return aotx_boot_start("aotx_drain", argv, &children->drain);
}

int aotx_boot_start_feed(aotx_boot_children *children, const aotx_seam_rings *rings)
{
    char fd[32];
    snprintf(fd, sizeof fd, "%d", rings->inbound_fd);
    char *argv[] = { (char *)"aotx_feed", (char *)"--inbound-fd", fd, NULL };
    return aotx_boot_start("aotx_feed", argv, &children->feed);
}

/* The replay puts its records in the inbound ring, and the last of them is the restore
 * report. The device puts its own hash in that report, so no text crosses the seam. */
int aotx_boot_replay(aotx_boot_children *children, const aotx_seam_rings *rings,
                     const char *journal, aotx_pump *pump)
{
    char fd[32];
    aotx_pump_report report;
    snprintf(fd, sizeof fd, "%d", rings->inbound_fd);
    char *argv[] = { (char *)"aotx_restore", (char *)"--inbound-fd", fd,
                     (char *)"--journal", (char *)journal, NULL };
    if (aotx_boot_start("aotx_restore", argv, &children->restore) != 0) {
        return 1;
    }

    /* The tick load stays off while the journal is replayed, so the replay records are the
     * only records of these ticks. */
    aotx_pump_set(pump, 0ull, 1u);

    /* The first tick waits until the replay has chosen its journal. A tick makes a block,
     * and a block gives the new journal a complete tick, which the replay could take for
     * the journal to read. The replay chooses before it opens the ring, so its first record
     * states that the choice is made. */
    const volatile aotx_inbound_preamble *inbound =
        (const volatile aotx_inbound_preamble *)rings->inbound_map;
    int stopped = 0;
    int status = 0;
    for (unsigned long long guard = 0ull; guard < 1000000ull; ++guard) {
        if (stopped == 0) {
            aotx_seam_poll(children->restore, &stopped, &status);
        }
        if (stopped == 0 && inbound->head == 0ull) {
            usleep(200);
            continue;
        }
        aotx_pump_tick(pump);
        if (stopped != 0 && inbound->consumed >= inbound->head) {
            break;
        }
    }
    children->restore = 0;
    aotx_pump_read(&report);
    printf("restore: applied %llu hash %llx\n", report.applied, report.state_hash);
    return (status == 0) ? 0 : 1;
}

void aotx_boot_stop(aotx_boot_children *children)
{
    if (children->feed != 0) {
        aotx_seam_wait(children->feed);
        children->feed = 0;
    }
    if (children->drain != 0) {
        aotx_seam_wait(children->drain);
        children->drain = 0;
    }
    if (children->restore != 0) {
        aotx_seam_wait(children->restore);
        children->restore = 0;
    }
}
