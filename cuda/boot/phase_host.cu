/* Purpose: Write the state of a boot beside its journal.
 * Owns: The descriptor of the journal directory while a boot runs.
 * Launch shape: Host glue only; no kernel.
 * Lifetime: From model placement to the closed state. */
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "boot/boot.cuh"

#define AOTX_BOOT_PHASE_FILE "phase"

static int aotx_boot_phase_fd = -1;
static int aotx_boot_phase_armed;

static int aotx_boot_phase_write(const char *word)
{
    char line[64];
    int fd;
    int used;
    if (aotx_boot_phase_fd < 0) {
        return 0;
    }
    used = snprintf(line, sizeof(line), "%s %lld\n", word, (long long)time(NULL));
    fd = openat(aotx_boot_phase_fd, AOTX_BOOT_PHASE_FILE,
                O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
    if (fd < 0 || write(fd, line, (size_t)used) != used) {
        if (fd >= 0) {
            close(fd);
        }
        fprintf(stderr, "the boot state does not write\n");
        return 1;
    }
    close(fd);
    return 0;
}

static void aotx_boot_phase_at_exit(void)
{
    if (aotx_boot_phase_armed != 0) {
        aotx_boot_phase_write("closed");
        close(aotx_boot_phase_fd);
        aotx_boot_phase_fd = -1;
        aotx_boot_phase_armed = 0;
    }
}

int aotx_boot_phase_open(const char *journal)
{
    if (journal == NULL || journal[0] == '\0') {
        return 0;
    }
    if (mkdir(journal, 0700) != 0 && errno != EEXIST) {
        fprintf(stderr, "the journal directory does not open for the boot state\n");
        return 1;
    }
    aotx_boot_phase_fd = open(journal, O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (aotx_boot_phase_fd < 0) {
        fprintf(stderr, "the journal directory does not open for the boot state\n");
        return 1;
    }
    aotx_boot_phase_armed = 1;
    atexit(aotx_boot_phase_at_exit);
    return aotx_boot_phase_write("placing");
}

int aotx_boot_phase_set(const char *word)
{
    return aotx_boot_phase_write(word);
}

void aotx_boot_phase_close(void)
{
    if (aotx_boot_phase_armed == 0) {
        return;
    }
    aotx_boot_phase_write("closed");
    close(aotx_boot_phase_fd);
    aotx_boot_phase_fd = -1;
    aotx_boot_phase_armed = 0;
}
