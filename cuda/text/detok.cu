/* Purpose: Write the bytes of a batch of token lists.
 * Owns: Nothing; the buffers come from the caller.
 * Launch shape: One thread for each sequence.
 * Lifetime: One launch. */
#include "text/text.cuh"

/* A token string of this family is byte level. Each code point of the string stands for
 * one byte of the text. A control token holds plain text which the map does not cover. The
 * code points of that token go out as they are. */
__global__ void aotx_text_detok(const unsigned int *id, const unsigned int *count,
                                unsigned int sequences, unsigned int stride,
                                unsigned char *bytes, unsigned int *length,
                                unsigned int limit)
{
    unsigned int sequence = blockIdx.x * blockDim.x + threadIdx.x;
    if (sequence >= sequences) {
        return;
    }
    const aotx_text_vocab *vocab = &aotx_text_vocab_table;
    unsigned char *out = bytes + (unsigned long long)sequence * limit;
    unsigned int at = 0u;
    for (unsigned int i = 0u; i < count[sequence]; ++i) {
        unsigned int token = id[sequence * stride + i];
        if (token >= vocab->tokens) {
            continue;
        }
        unsigned long long from = vocab->token_at[token];
        unsigned int span = (unsigned int)(vocab->token_at[token + 1u] - from);
        const unsigned char *text = vocab->token_bytes + from;
        unsigned int walk = 0u;
        while (walk < span && at + 4u <= limit) {
            unsigned int point = 0u;
            walk += aotx_text_decode(text, span, walk, &point);
            unsigned int byte = aotx_text_point_byte(point);
            if (byte < 0x100u) {
                out[at] = (unsigned char)byte;
                at += 1u;
            } else {
                at += aotx_text_encode(point, out + at);
            }
        }
    }
    length[sequence] = at;
}
