/* Purpose: Select the vocabulary of one text batch.
 * Owns: The saved language and embedding vocabulary descriptors.
 * Launch shape: One block; threads copy descriptor bytes before the batch starts.
 * Lifetime: From model load to the end of the run. */
#include "text/text.cuh"

__device__ aotx_text_vocab aotx_text_vocab_saved[2];

__global__ void aotx_text_vocab_select(unsigned int embedding)
{
    if (embedding > 1u) return;
    unsigned char *out = (unsigned char *)&aotx_text_vocab_table;
    const unsigned char *in = (const unsigned char *)&aotx_text_vocab_saved[embedding];
    for (unsigned int at = threadIdx.x; at < sizeof(aotx_text_vocab); at += blockDim.x) {
        out[at] = in[at];
    }
}
