/* Purpose: Give a result longer than the bound, so the check program refuses this module.
 * Owns: Nothing; the batch and the output belong to the system.
 * Launch shape: One block for each request row; the row is blockIdx.x.
 * Lifetime: One launch of the check program. */
#include "aotx_tool.h"

extern "C" __global__ void aotx_tool_over_bound(aotx_tool_batch *batch,
                                                aotx_tool_output *out)
{
    unsigned int row = blockIdx.x;
    if (batch == 0 || out == 0 || row >= batch->rows || batch->row[row].take == 0u) {
        return;
    }
    if (threadIdx.x != 0u) {
        return;
    }
    char *text = out->text + (unsigned long long)row * batch->out_bytes;
    text[0] = 'x';
    out->head[row].status = AOTX_TOOL_STATUS_OK;
    /* One byte more than the bound the batch gives. */
    out->head[row].length = (unsigned int)batch->out_bytes + 1u;
    __threadfence();
    out->head[row].done = 1u;
}
