/* Purpose: Hold local memory on purpose, so the check program refuses this module.
 * Owns: Nothing; the batch and the output belong to the system.
 * Launch shape: One block for each request row; the row is blockIdx.x.
 * Lifetime: One launch of the check program. */
#include "aotx_tool.h"

/* Words of the block this kernel keeps. The index comes from the data, so the compiler
 * cannot hold the block in registers and puts it in local memory. */
#define LOCAL_MEMORY_WORDS 256u

extern "C" __global__ void aotx_tool_local_memory(aotx_tool_batch *batch,
                                                  aotx_tool_output *out)
{
    unsigned int row = blockIdx.x;
    if (batch == 0 || out == 0 || row >= batch->rows || batch->row[row].take == 0u) {
        return;
    }
    if (threadIdx.x != 0u) {
        return;
    }
    unsigned int store[LOCAL_MEMORY_WORDS];
    unsigned int length = batch->row[row].argument[0].length;
    for (unsigned int i = 0u; i < LOCAL_MEMORY_WORDS; ++i) {
        store[i] = i + length;
    }
    unsigned int pick = store[length % LOCAL_MEMORY_WORDS];
    char *text = out->text + (unsigned long long)row * batch->out_bytes;
    text[0] = (char)('0' + (char)(pick % 10u));
    out->head[row].status = AOTX_TOOL_STATUS_OK;
    out->head[row].length = 1u;
    __threadfence();
    out->head[row].done = 1u;
}
