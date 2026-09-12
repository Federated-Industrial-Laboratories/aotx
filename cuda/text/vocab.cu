/* Purpose: Build the vocabulary tables of the tokenizer from the model file arrays.
 * Owns: The vocabulary table that every text kernel reads.
 * Launch shape: One thread for each token, and one thread for each merge.
 * Lifetime: From model load to the end of the run. */
#include "text/text.cuh"

/* The table is empty until the host glue gives the arrays and these kernels fill it. */
__device__ aotx_text_vocab aotx_text_vocab_table;

/* Counters that the build writes so the host glue can state what the tables hold. The
 * entries are: tokens put in, merges put in, merges the table refused, special tokens. */
#define AOTX_TEXT_COUNTS   4u

__global__ void aotx_text_build_tokens(unsigned int *report)
{
    aotx_text_vocab *vocab = &aotx_text_vocab_table;
    unsigned int step = blockDim.x * gridDim.x;
    unsigned int mask = vocab->slots - 1u;
    unsigned int *slot = (unsigned int *)vocab->slot;
    for (unsigned int token = blockIdx.x * blockDim.x + threadIdx.x; token < vocab->tokens;
         token += step) {
        unsigned long long from = vocab->token_at[token];
        unsigned int length = (unsigned int)(vocab->token_at[token + 1u] - from);
        const unsigned char *text = vocab->token_bytes + from;
        unsigned int at = (unsigned int)aotx_text_hash(text, length) & mask;
        for (unsigned int probe = 0u; probe <= mask; ++probe) {
            unsigned int got = atomicCAS(&slot[at], AOTX_TEXT_NONE, token);
            if (got == AOTX_TEXT_NONE) {
                atomicAdd(&report[0], 1u);
                break;
            }
            /* Two tokens with the same string keep the lower token, so the result of a
             * search does not change from one run to the next. */
            if (aotx_text_same(vocab, got, text, length)) {
                atomicMin(&slot[at], token);
                break;
            }
            at = (at + 1u) & mask;
        }
    }
}

__global__ void aotx_text_build_specials(const int *type, unsigned int *report)
{
    aotx_text_vocab *vocab = &aotx_text_vocab_table;
    unsigned int step = blockDim.x * gridDim.x;
    for (unsigned int token = blockIdx.x * blockDim.x + threadIdx.x; token < vocab->tokens;
         token += step) {
        int kind = type[token];
        /* A token of the control type gets its bit. The detokenizer of a reply reads that
         * bit and gives no byte for such a token. */
        if (kind == AOTX_TEXT_TYPE_CONTROL && vocab->control != 0
            && (token >> 5) < vocab->control_words) {
            atomicOr((unsigned int *)&vocab->control[token >> 5], 1u << (token & 31u));
        }
        if (kind != AOTX_TEXT_TYPE_CONTROL && kind != AOTX_TEXT_TYPE_USER
            && kind != AOTX_TEXT_TYPE_UNKNOWN) {
            continue;
        }
        unsigned long long from = vocab->token_at[token];
        if (vocab->token_at[token + 1u] == from) {
            continue;
        }
        unsigned int at = atomicAdd(&vocab->specials, 1u);
        if (at >= vocab->tokens) {
            continue;
        }
        vocab->special[at] = token;
        atomicAdd(&report[3], 1u);
        /* The first byte of every special token goes in a mask of 256 bits. The search
         * over a sequence reads that mask and not the list. */
        unsigned int byte = vocab->token_bytes[from];
        atomicOr((unsigned long long *)&vocab->first[byte >> 6], 1ull << (byte & 63u));
    }
}

__global__ void aotx_text_build_pairs(const unsigned char *bytes,
                                      const unsigned long long *at_table, unsigned int merges,
                                      unsigned int *report)
{
    aotx_text_vocab *vocab = &aotx_text_vocab_table;
    unsigned int step = blockDim.x * gridDim.x;
    unsigned int mask = vocab->pairs - 1u;
    unsigned long long *key = (unsigned long long *)vocab->pair_key;
    unsigned int *rank = (unsigned int *)vocab->pair_rank;
    for (unsigned int merge = blockIdx.x * blockDim.x + threadIdx.x; merge < merges;
         merge += step) {
        unsigned long long from = at_table[merge];
        unsigned int length = (unsigned int)(at_table[merge + 1u] - from);
        const unsigned char *text = bytes + from;
        /* A merge string holds the left token, one space, and the right token. A token
         * string of this family holds no space, so the first space is the cut. */
        unsigned int cut = 0u;
        while (cut < length && text[cut] != ' ') {
            cut += 1u;
        }
        if (cut == 0u || cut + 1u >= length) {
            atomicAdd(&report[2], 1u);
            continue;
        }
        unsigned int left = aotx_text_find_token(vocab, text, cut);
        unsigned int right = aotx_text_find_token(vocab, text + cut + 1u,
                                                  length - cut - 1u);
        if (left == AOTX_TEXT_NONE || right == AOTX_TEXT_NONE) {
            atomicAdd(&report[2], 1u);
            continue;
        }
        unsigned long long pair = ((unsigned long long)left << 32) | (unsigned long long)right;
        unsigned int slot = (unsigned int)aotx_text_pair_hash(pair) & mask;
        for (unsigned int probe = 0u; probe <= mask; ++probe) {
            unsigned long long got = atomicCAS(&key[slot], 0xFFFFFFFFFFFFFFFFull, pair);
            if (got == 0xFFFFFFFFFFFFFFFFull || got == pair) {
                /* The rank comes in with the highest value, and every writer of the slot
                 * takes the lowest rank. A merge which comes first has the lower rank and
                 * joins the pair first. The order of the writers therefore does not
                 * change the table. */
                atomicMin(&rank[slot], merge);
                if (got == 0xFFFFFFFFFFFFFFFFull) {
                    atomicAdd(&report[1], 1u);
                }
                break;
            }
            slot = (slot + 1u) & mask;
        }
    }
}

__global__ void aotx_text_check_prefix(const unsigned char *bytes,
                                       const unsigned long long *at_table,
                                       unsigned int tokens, unsigned int *report)
{
    const aotx_text_vocab *vocab = &aotx_text_vocab_table;
    unsigned int step = blockDim.x * gridDim.x;
    for (unsigned int token = blockIdx.x * blockDim.x + threadIdx.x; token < tokens;
         token += step) {
        if (token >= vocab->tokens) {
            atomicAdd(&report[0], 1u);
            continue;
        }
        unsigned long long from = at_table[token];
        unsigned int length = (unsigned int)(at_table[token + 1u] - from);
        if (!aotx_text_same(vocab, token, bytes + from, length)) {
            atomicAdd(&report[0], 1u);
        }
    }
}
