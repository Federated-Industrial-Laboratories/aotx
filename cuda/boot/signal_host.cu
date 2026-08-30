/* Purpose: Take the stop signals and expose the first one to the boot loop.
 * Owns: The stop signal number and the two process signal handlers.
 * Launch shape: Host glue only; this file launches no kernel.
 * Lifetime: The program. */
#include <signal.h>
#include <string.h>

#include "boot/boot.cuh"

/* A handler may set a flag of this type and do nothing else. */
static volatile sig_atomic_t aotx_boot_signal_number;

static void aotx_boot_on_signal(int number)
{
    /* The first signal stops the run and the report names it. A later one changes nothing. */
    if (aotx_boot_signal_number == 0) {
        aotx_boot_signal_number = (sig_atomic_t)number;
    }
}

int aotx_boot_signal(void)
{
    return (int)aotx_boot_signal_number;
}

/* Take the stop signals. The run then ends through the last flush, the closed rings, the
 * wait for the disk side programs and the reports. The handler stays in place, so a second
 * signal changes nothing. A run with a drawing context must not stop during a frame. */
void aotx_boot_signals_open(void)
{
    struct sigaction action;
    memset(&action, 0, sizeof action);
    action.sa_handler = aotx_boot_on_signal;
    sigemptyset(&action.sa_mask);
    sigaction(SIGTERM, &action, NULL);
    sigaction(SIGINT, &action, NULL);
}
