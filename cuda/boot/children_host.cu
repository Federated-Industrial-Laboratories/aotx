/* Purpose: Start the disk side programs and run the replay that a restore needs.
 * Owns: The process id of each disk side program.
 * Launch shape: Host glue only; the pump supplies the kernels.
 * Lifetime: From the first start to the last wait. */
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>

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

/* A program receives the descriptors it must map and no others. The keep list names them,
 * so a key pipe or a ring that belongs to another program does not reach this one. */
static int aotx_boot_start(const char *name, char *const argv[], const int *keep,
                           unsigned int keep_count, int *pid)
{
    char path[PATH_MAX];
    if (aotx_boot_sibling(name, path, (unsigned int)sizeof path) != 0) {
        fprintf(stderr, "cannot find %s beside this program\n", name);
        return 1;
    }
    if (aotx_seam_spawn(path, argv, keep, keep_count, pid) != 0) {
        fprintf(stderr, "cannot start %s\n", name);
        return 1;
    }
    return 0;
}

int aotx_boot_start_drain(aotx_boot_children *children, const aotx_seam_rings *rings,
                          const char *journal, const char *derive)
{
    char fd[32];
    char bulk[32];
    snprintf(fd, sizeof fd, "%d", rings->host_fd);
    snprintf(bulk, sizeof bulk, "%d", rings->bulk_fd);
    char *with[] = { (char *)"aotx_drain", (char *)"--ring-fd", fd,
                     (char *)"--bulk-fd", bulk,
                     (char *)"--journal", (char *)journal,
                     (char *)"--derive", (char *)derive, NULL };
    char *without[] = { (char *)"aotx_drain", (char *)"--ring-fd", fd,
                        (char *)"--bulk-fd", bulk,
                        (char *)"--journal", (char *)journal, NULL };
    const int keep[] = { rings->host_fd, rings->bulk_fd };
    /* A run that gives no list leaves the drain with its own default. */
    return aotx_boot_start("aotx_drain", (derive != NULL) ? with : without, keep, 2u,
                           &children->drain);
}

int aotx_boot_start_feed(aotx_boot_children *children, const aotx_seam_rings *rings,
                         int keys_fd, const char *root, const char *journal,
                         const char *settings, const char *modules, int no_stdin)
{
    char fd[32];
    char keys[32];
    char mirror[32];
    char ready[32];
    char requests[512];
    int ready_pipe[2];
    snprintf(fd, sizeof fd, "%d", rings->inbound_fd);
    snprintf(keys, sizeof keys, "%d", keys_fd);
    snprintf(mirror, sizeof mirror, "%d", rings->mirror_fd);
    snprintf(requests, sizeof requests, "%s/requests.jsonl",
             (journal != NULL) ? journal : ".");

    /* The argument list takes the key pipe, the root and the settings file. Each one is
     * left out when the run does not give it. The file read tool reaches no file without a
     * root. The feeder publishes the device keys of the settings file as records before
     * the first line of the operator. */
    if (pipe(ready_pipe) != 0) {
        fprintf(stderr, "the feeder status pipe does not open\n");
        return 1;
    }
    snprintf(ready, sizeof ready, "%d", ready_pipe[1]);
    char *argv[20];
    unsigned int at = 0u;
    argv[at++] = (char *)"aotx_feed";
    argv[at++] = (char *)"--inbound-fd";
    argv[at++] = fd;
    argv[at++] = (char *)"--ready-fd";
    argv[at++] = ready;
    if (keys_fd >= 0) {
        argv[at++] = (char *)"--keys-fd";
        argv[at++] = keys;
    }
    /* The mirror descriptor goes to the feeder, which gives it to a terminal that
     * attaches. The feeder counts the terminals in the preamble of the mirror. */
    if (rings->mirror_fd >= 0) {
        argv[at++] = (char *)"--mirror-fd";
        argv[at++] = mirror;
    }
    if (root != NULL) {
        argv[at++] = (char *)"--root";
        argv[at++] = (char *)root;
        argv[at++] = (char *)"--requests";
        argv[at++] = requests;
    }
    if (settings != NULL && settings[0] != '\0') {
        argv[at++] = (char *)"--settings";
        argv[at++] = (char *)settings;
    }
    /* The feeder listens for a terminal in the journal directory. A terminal that attaches
     * there receives the mirror and sends keys and lines. */
    if (journal != NULL && journal[0] != '\0') {
        argv[at++] = (char *)"--attach";
        argv[at++] = (char *)journal;
    }
    /* The feeder reads each module directory below this one and publishes the import
     * records before the first line. A restored run gives none, because the journal holds
     * the import records of the run it restores. */
    if (modules != NULL && modules[0] != '\0') {
        argv[at++] = (char *)"--modules";
        argv[at++] = (char *)modules;
    }
    /* A terminal owns the keyboard; the feeder then reads no line of standard input. */
    if (no_stdin != 0) {
        argv[at++] = (char *)"--no-stdin";
    }
    argv[at] = NULL;
    const int keep[] = { rings->inbound_fd, rings->mirror_fd, keys_fd, ready_pipe[1] };
    if (aotx_boot_start("aotx_feed", argv, keep, 4u, &children->feed) != 0) {
        close(ready_pipe[0]);
        close(ready_pipe[1]);
        return 1;
    }
    close(ready_pipe[1]);
    char mark = 0;
    ssize_t got;
    do {
        got = read(ready_pipe[0], &mark, 1u);
    } while (got < 0 && errno == EINTR);
    close(ready_pipe[0]);
    if (got != 1 || mark != 'R') {
        int status = aotx_seam_wait(children->feed);
        children->feed = 0;
        fprintf(stderr, "the feeder did not start, status %d\n", status);
        return 1;
    }
    return 0;
}

/* The terminal program takes the journal directory and finds the socket of the feeder in
 * it. It keeps the standard descriptors, because it draws on the terminal of the run. */
int aotx_boot_start_tui(aotx_boot_children *children, const char *journal,
                        const char *settings)
{
    if (journal == NULL || journal[0] == '\0') {
        fprintf(stderr, "the terminal program needs a journal directory\n");
        return 1;
    }
    char *argv[6];
    unsigned int at = 0u;
    argv[at++] = (char *)"aotx_tui";
    argv[at++] = (char *)"--attach";
    argv[at++] = (char *)journal;
    if (settings != NULL && settings[0] != '\0') {
        argv[at++] = (char *)"--settings";
        argv[at++] = (char *)settings;
    }
    argv[at] = NULL;
    /* The terminal keeps the standard descriptors and no ring. It reads the mirror over
     * the socket of the feeder, which sends the descriptor to it. */
    return aotx_boot_start("aotx_tui", argv, NULL, 0u, &children->tui);
}

/* The boot reaps a terminal that ends and says so once; the run goes on. */
void aotx_boot_reap_tui(aotx_boot_children *children)
{
    int done = 0;
    int status = 0;
    if (children->tui == 0) {
        return;
    }
    if (aotx_seam_poll(children->tui, &done, &status) == 0 && done != 0) {
        printf("boot: the terminal ended (code %d)\n", status);
        children->tui = 0;
    }
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
    const int keep[] = { rings->inbound_fd };
    if (aotx_boot_start("aotx_restore", argv, keep, 1u, &children->restore) != 0) {
        return 1;
    }
    /* A replay applies the keys of the journal again. The flag makes the command layer
     * refuse to close the run while those keys go through it. */
    aotx_seam_set_replaying(1);

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
    int ended = 0;
    /* The replay takes as many ticks as the journal holds, so the loop has no bound on its
     * ticks. The bound is on turns that make no progress. A replay whose ring does not
     * move for this many turns has lost its restore program. The run must not go on from
     * a part of the journal as if it were the whole. */
    unsigned long long idle = 0ull;
    unsigned long long seen_head = 0ull;
    unsigned long long seen_consumed = 0ull;
    while (idle < 1000000ull) {
        if (stopped == 0) {
            aotx_seam_poll(children->restore, &stopped, &status);
        }
        if (stopped == 0 && inbound->head == 0ull) {
            usleep(200);
            idle += 1ull;
            continue;
        }
        aotx_pump_tick(pump);
        if (pump->model_refused != 0u) {
            fprintf(stderr, "restore: a model file was refused\n");
            return 1;
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
    children->restore = 0;
    aotx_pump_read(&report);
    printf("restore: applied %llu hash %llx refused %u pages %u paced %llu\n",
           report.applied, report.state_hash, report.refused, report.pages, report.paced);
    if (ended == 0) {
        fprintf(stderr, "restore: the replay made no progress in %llu turns and did not"
                        " end; the run stops\n", idle);
        return 1;
    }
    return (status == 0) ? 0 : 1;
}

void aotx_boot_stop(aotx_boot_children *children)
{
    if (children->tui != 0) {
        /* The boot tells its terminal to end, then waits for it. */
        kill((pid_t)children->tui, SIGTERM);
        aotx_seam_wait(children->tui);
        children->tui = 0;
    }
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
