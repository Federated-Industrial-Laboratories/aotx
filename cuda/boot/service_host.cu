/* Purpose: Start the separately scoped local service broker.
 * Owns: No request state; the broker receives its mapped transport only.
 * Launch shape: Host glue only.
 * Lifetime: Runtime startup through broker exit. */
#include "boot/boot.cuh"
#include <stdio.h>
int aotx_boot_start_service(aotx_boot_children *children, const aotx_seam_rings *rings,
                            const char *journal, const char *grants)
{
    if (!grants) return 0;
    if (!journal || rings->service_fd < 0) return 1;
    char fd[32]; snprintf(fd, sizeof fd, "%d", rings->service_fd);
    char *argv[] = { (char *)"aotx_service", (char *)"--ring-fd", fd,
        (char *)"--journal", (char *)journal, (char *)"--grants", (char *)grants, NULL };
    const int keep[] = { rings->service_fd };
    return aotx_boot_start("aotx_service", argv, keep, 1, &children->service);
}
