/* Purpose: Count the words of the text argument of every row a call gives this module.
 * Owns: Nothing; the batch and the output belong to the system.
 * Launch shape: One block for each request row; the row is blockIdx.x.
 * Lifetime: One node of one tick. */
#include "aotx_tool.h"

/* Threads of one block. The system launches a module node with this shape. */
#define WORD_COUNT_THREADS 256u

/* Report whether a byte parts two words. */
__device__ __forceinline__ static int word_count_space(char byte)
{
    return (byte == ' ' || byte == '\t' || byte == '\n' || byte == '\r') ? 1 : 0;
}

/* Write a whole number as text and give the bytes it took. The kernel keeps no block of
 * its own. A block that an index reads would stand in local memory, and the check program
 * refuses local memory. */
__device__ __forceinline__ static unsigned int word_count_number(unsigned int value,
                                                                 char *out,
                                                                 unsigned int max)
{
    unsigned int scale = 1u;
    while (value / scale >= 10u && scale <= 100000000u) {
        scale *= 10u;
    }
    unsigned int at = 0u;
    while (scale > 0u && at < max) {
        out[at] = (char)('0' + (char)((value / scale) % 10u));
        at += 1u;
        scale /= 10u;
    }
    return at;
}

/* Write a text that ends with a zero byte and give the position after it. */
__device__ __forceinline__ static unsigned int word_count_put(char *out, unsigned int at,
                                                              unsigned int max,
                                                              const char *text)
{
    for (unsigned int i = 0u; text[i] != '\0' && at < max; ++i) {
        out[at] = text[i];
        at += 1u;
    }
    return at;
}

extern "C" __global__ void aotx_tool_word_count(aotx_tool_batch *batch,
                                                aotx_tool_output *out)
{
    __shared__ unsigned int cell[WORD_COUNT_THREADS];
    unsigned int row = blockIdx.x;
    if (batch == 0 || out == 0 || row >= batch->rows || row >= out->rows) {
        return;
    }
    /* A row the batch did not give this module belongs to another tool. The module reads
     * no such row and writes no output row for it. */
    const aotx_tool_row *in = &batch->row[row];
    if (in->take == 0u) {
        return;
    }

    /* The text argument of the call. A call that names no text counts no word. */
    const aotx_tool_argument *arg = &in->argument[0];
    unsigned int length = (arg->length <= AOTX_TOOL_VALUE_BYTES) ? arg->length : 0u;

    /* Every thread counts the word starts of its own bytes. A word starts at a byte that
     * is not a space and follows a space or the front of the text. */
    unsigned int held = 0u;
    for (unsigned int i = threadIdx.x; i < length; i += WORD_COUNT_THREADS) {
        if (word_count_space(arg->value[i]) != 0) {
            continue;
        }
        if (i == 0u || word_count_space(arg->value[i - 1u]) != 0) {
            held += 1u;
        }
    }
    cell[threadIdx.x] = held;
    __syncthreads();
    for (unsigned int step = WORD_COUNT_THREADS / 2u; step > 0u; step >>= 1) {
        if (threadIdx.x < step) {
            cell[threadIdx.x] += cell[threadIdx.x + step];
        }
        __syncthreads();
    }
    if (threadIdx.x != 0u) {
        return;
    }

    /* The bytes of one output row come from the batch, so one module serves every build. */
    char *text = out->text + (unsigned long long)row * batch->out_bytes;
    unsigned int max = (unsigned int)batch->out_bytes;
    unsigned int at = word_count_number(cell[0], text, max);
    at = word_count_put(text, at, max, (cell[0] == 1u) ? " word" : " words");
    out->head[row].status = AOTX_TOOL_STATUS_OK;
    out->head[row].length = at;
    /* The done word goes last, after a fence, so a reader that sees it sees the text. */
    __threadfence();
    out->head[row].done = 1u;
}
