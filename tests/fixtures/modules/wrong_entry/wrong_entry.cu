/* Purpose: Hold a kernel of another name, so the check program finds no entry.
 * Owns: Nothing; the batch and the output belong to the system.
 * Launch shape: One block for each request row; the row is blockIdx.x.
 * Lifetime: One launch of the check program. */
#include "aotx_tool.h"

extern "C" __global__ void aotx_tool_another_name(aotx_tool_batch *batch,
                                                  aotx_tool_output *out)
{
    unsigned int row = blockIdx.x;
    if (batch == 0 || out == 0 || row >= batch->rows || batch->row[row].take == 0u) {
        return;
    }
    if (threadIdx.x == 0u) {
        out->head[row].status = AOTX_TOOL_STATUS_OK;
        out->head[row].length = 0u;
        __threadfence();
        out->head[row].done = 1u;
    }
}
