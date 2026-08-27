/* Purpose: Start the system: context, modules, reservation, rings.
 * Owns: The boot record and the module table.
 * Launch shape: Host glue only; no kernels.
 * Lifetime: From start to the first tick. */
#ifndef AOTX_BOOT_CUH
#define AOTX_BOOT_CUH

#include "sched/sched.cuh"
#include "seam/seam.cuh"

/* The clock module proves the raw module path: load, graph node, launch, sample. */
int aotx_boot_clock_check(unsigned long long *sample);

/* The disk side programs of this run. A value of zero means the program does not run. */
typedef struct aotx_boot_children {
    int drain;
    int feed;
    int restore;
} aotx_boot_children;

/* Find a program that sits beside this one. The return is zero when the path is found. */
int aotx_boot_sibling(const char *name, char *path, unsigned int bytes);

/* Start the program that writes the journal. */
int aotx_boot_start_drain(aotx_boot_children *children, const aotx_seam_rings *rings,
                          const char *journal);

/* Start the program that writes the inbound ring. */
int aotx_boot_start_feed(aotx_boot_children *children, const aotx_seam_rings *rings);

/* Replay the journal: start the restore program, run ticks until it ends and the inbound
 * ring is empty, then write the restore record. */
int aotx_boot_replay(aotx_boot_children *children, const aotx_seam_rings *rings,
                     const char *journal, aotx_pump *pump);

/* Wait for every program that still runs. */
void aotx_boot_stop(aotx_boot_children *children);

#endif
