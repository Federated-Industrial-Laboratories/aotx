/* Purpose: Hold a host call, so the seam gate refuses this module at the build.
 * Owns: Nothing; the batch and the output belong to the system.
 * Launch shape: One block for each request row; the row is blockIdx.x.
 * Lifetime: This module never becomes a module file. */
#include "aotx_tool.h"

extern "C" __global__ void aotx_tool_host_call(aotx_tool_batch *batch,
                                               aotx_tool_output *out)
{
    unsigned int row = blockIdx.x;
    if (batch == 0 || out == 0 || row >= batch->rows || batch->row[row].take == 0u) {
        return;
    }
    /* A device file may not hold this call. The gate refuses the module for it. */
    printf("row %u\n", row);
    out->head[row].status = AOTX_TOOL_STATUS_OK;
    out->head[row].length = 0u;
    __threadfence();
    out->head[row].done = 1u;
}
