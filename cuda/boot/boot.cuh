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

/* Start the program that writes the inbound ring. A key descriptor of zero or more gives
 * the feeder the read end of the key pipe. The feeder makes a key record of each frame. */
int aotx_boot_start_feed(aotx_boot_children *children, const aotx_seam_rings *rings,
                         int keys_fd);

/* Replay the journal: start the restore program, run ticks until it ends and the inbound
 * ring is empty, then write the restore record. */
int aotx_boot_replay(aotx_boot_children *children, const aotx_seam_rings *rings,
                     const char *journal, aotx_pump *pump);

/* Wait for every program that still runs. */
void aotx_boot_stop(aotx_boot_children *children);

/* Read the flag that the quit command sets. A value above zero stops the run. */
unsigned int aotx_boot_quit(void);

/* Run the tick pump on its own thread and the window on the thread that calls this. The
 * return is zero when the window closed or the quit command stopped the run. */
int aotx_boot_window_run(aotx_pump *pump, int keys_fd);

#endif
