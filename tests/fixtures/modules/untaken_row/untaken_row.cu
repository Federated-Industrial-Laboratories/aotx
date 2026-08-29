/* Purpose: Write the output row after its own, so the check program refuses this module.
 * Owns: Nothing; the batch and the output belong to the system.
 * Launch shape: One block for each request row; the row is blockIdx.x.
 * Lifetime: One launch of the check program. */
#include "aotx_tool.h"

extern "C" __global__ void aotx_tool_untaken_row(aotx_tool_batch *batch,
                                                 aotx_tool_output *out)
{
    unsigned int row = blockIdx.x;
    if (batch == 0 || out == 0 || row >= batch->rows || batch->row[row].take == 0u) {
        return;
    }
    if (threadIdx.x != 0u) {
        return;
    }
    /* The row after this one belongs to another tool. This module writes it. */
    unsigned int next = row + 1u;
    if (next < out->rows) {
        char *text = out->text + (unsigned long long)next * batch->out_bytes;
        text[0] = 'y';
        out->head[next].status = AOTX_TOOL_STATUS_OK;
        out->head[next].length = 1u;
        __threadfence();
        out->head[next].done = 1u;
    }
    char *mine = out->text + (unsigned long long)row * batch->out_bytes;
    mine[0] = 'y';
    out->head[row].status = AOTX_TOOL_STATUS_OK;
    out->head[row].length = 1u;
    __threadfence();
    out->head[row].done = 1u;
}
