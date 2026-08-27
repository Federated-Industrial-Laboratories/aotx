/* Purpose: Hold the seam state and write the record that reports a restore.
 * Owns: The device ring state, the host ring state and the inbound cursor.
 * Launch shape: One thread for the restore record.
 * Lifetime: The whole run. */
#include "seam/seam.cuh"

/* The host glue writes this once, after it maps the ring region and the two host rings. */
__device__ aotx_seam_state aotx_seam;

/* The record that opens the journal of this run. */
__global__ void aotx_seam_note_boot(unsigned long long previous_boot_id,
                                    unsigned long long wall_ns)
{
    aotx_boot_body body;
    body.boot_id = aotx_seam.boot_id;
    body.previous_boot_id = previous_boot_id;
    body.wall_ns = wall_ns;
    aotx_seam_write(AOTX_WRITER_SYSTEM, AOTX_CLASS_A, AOTX_REC_BOOT, 0u,
                    &body, (unsigned int)sizeof body);
}
