/* Purpose: Put the vocabulary arrays of a model file on the device and start the build.
 * Owns: The device memory of the vocabulary tables.
 * Launch shape: Host glue only; the build kernels do the work.
 * Lifetime: From model load to the release at exit. */
#include <cuda.h>
#include <cuda_runtime.h>
#include <string.h>

#include "boot/check.h"
#include "text/text.cuh"

/* Blocks of the build. A block of 256 threads keeps the whole device busy at 151,000
 * tokens. The three build kernels take the same shape. */
#define AOTX_TEXT_BLOCK    256u
#define AOTX_TEXT_GRID     512u

/* Slots of a table are twice the entries at least, so a search stops after a few steps. */
static unsigned int aotx_text_size(unsigned long long entries)
{
    unsigned int slots = 1024u;
    while ((unsigned long long)slots < entries * 2ull) {
        slots <<= 1;
    }
    return slots;
}

/* Take device memory and hold it in the store, so the release gives every block back. */
static void *aotx_text_take(aotx_text_store *store, unsigned long long bytes)
{
    void *block = 0;
    aotx_check_runtime(cudaMalloc(&block, (size_t)bytes), "cudaMalloc");
    if (store != 0 && store->blocks < 8u) {
        store->block[store->blocks] = block;
        store->blocks += 1u;
        store->bytes += bytes;
    }
    return block;
}

int aotx_text_vocab_build(const aotx_text_source *source, aotx_text_store *store)
{
    memset(store, 0, sizeof *store);
    unsigned long long tokens = source->tokens;
    unsigned long long merges = source->merges;
    if (tokens == 0ull || merges == 0ull) {
        return 1;
    }
    unsigned long long token_bytes = source->token_at[tokens];
    unsigned long long merge_bytes = source->merge_at[merges];
    unsigned int slots = aotx_text_size(tokens);
    unsigned int pairs = aotx_text_size(merges);

    /* The tables that stay: the token strings, the offsets, and the two hash tables. */
    void *text = aotx_text_take(store, token_bytes);
    void *at = aotx_text_take(store, (tokens + 1ull) * sizeof(unsigned long long));
    void *slot = aotx_text_take(store, (unsigned long long)slots * sizeof(unsigned int));
    void *key = aotx_text_take(store, (unsigned long long)pairs * sizeof(unsigned long long));
    void *rank = aotx_text_take(store, (unsigned long long)pairs * sizeof(unsigned int));
    aotx_check_runtime(cudaMemcpy(text, source->token_bytes, (size_t)token_bytes,
                                  cudaMemcpyHostToDevice), "cudaMemcpy");
    aotx_check_runtime(cudaMemcpy(at, source->token_at,
                                  (size_t)((tokens + 1ull) * sizeof(unsigned long long)),
                                  cudaMemcpyHostToDevice), "cudaMemcpy");
    aotx_check_runtime(cudaMemset(slot, 0xFF, (size_t)slots * sizeof(unsigned int)),
                       "cudaMemset");
    aotx_check_runtime(cudaMemset(key, 0xFF, (size_t)pairs * sizeof(unsigned long long)),
                       "cudaMemset");
    aotx_check_runtime(cudaMemset(rank, 0xFF, (size_t)pairs * sizeof(unsigned int)),
                       "cudaMemset");

    /* The merge strings and the token types are read once, so they go in blocks which the
     * build gives back at the end. */
    void *merge_text = 0;
    void *merge_at = 0;
    void *type = 0;
    void *report = 0;
    aotx_check_runtime(cudaMalloc(&merge_text, (size_t)merge_bytes), "cudaMalloc");
    aotx_check_runtime(cudaMalloc(&merge_at,
                                  (size_t)((merges + 1ull) * sizeof(unsigned long long))),
                       "cudaMalloc");
    aotx_check_runtime(cudaMalloc(&type, (size_t)(tokens * sizeof(int))), "cudaMalloc");
    aotx_check_runtime(cudaMalloc(&report, 4u * sizeof(unsigned int)), "cudaMalloc");
    aotx_check_runtime(cudaMemcpy(merge_text, source->merge_bytes, (size_t)merge_bytes,
                                  cudaMemcpyHostToDevice), "cudaMemcpy");
    aotx_check_runtime(cudaMemcpy(merge_at, source->merge_at,
                                  (size_t)((merges + 1ull) * sizeof(unsigned long long)),
                                  cudaMemcpyHostToDevice), "cudaMemcpy");
    aotx_check_runtime(cudaMemcpy(type, source->token_type, (size_t)(tokens * sizeof(int)),
                                  cudaMemcpyHostToDevice), "cudaMemcpy");
    aotx_check_runtime(cudaMemset(report, 0, 4u * sizeof(unsigned int)), "cudaMemset");

    aotx_text_vocab table;
    memset(&table, 0, sizeof table);
    table.token_bytes = (const unsigned char *)text;
    table.token_at = (const unsigned long long *)at;
    table.tokens = (unsigned int)tokens;
    table.slots = slots;
    table.slot = (const unsigned int *)slot;
    table.pairs = pairs;
    table.pair_key = (const unsigned long long *)key;
    table.pair_rank = (const unsigned int *)rank;
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_text_vocab_table, &table, sizeof table),
                       "cudaMemcpyToSymbol");

    /* The token table comes first, because the pair build reads it. */
    aotx_text_build_tokens<<<AOTX_TEXT_GRID, AOTX_TEXT_BLOCK>>>((unsigned int *)report);
    aotx_text_build_specials<<<AOTX_TEXT_GRID, AOTX_TEXT_BLOCK>>>((const int *)type,
                                                                  (unsigned int *)report);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_text_build_pairs<<<AOTX_TEXT_GRID, AOTX_TEXT_BLOCK>>>(
        (const unsigned char *)merge_text, (const unsigned long long *)merge_at,
        (unsigned int)merges, (unsigned int *)report);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    unsigned int counts[4] = { 0u, 0u, 0u, 0u };
    aotx_check_runtime(cudaMemcpy(counts, report, sizeof counts, cudaMemcpyDeviceToHost),
                       "cudaMemcpy");
    cudaFree(merge_text);
    cudaFree(merge_at);
    cudaFree(type);
    cudaFree(report);

    /* Every token must be in the table, and every merge must name two tokens. */
    if (counts[0] != (unsigned int)tokens || counts[1] + counts[2] != (unsigned int)merges
        || counts[2] != 0u) {
        return 2;
    }
    if (counts[3] == 0u || counts[3] > AOTX_TEXT_SPECIAL_MAX) {
        return 3;
    }
    return 0;
}

void aotx_text_vocab_release(aotx_text_store *store)
{
    for (unsigned int i = 0u; i < store->blocks; ++i) {
        cudaFree(store->block[i]);
    }
    store->blocks = 0u;
    store->bytes = 0ull;
}

int aotx_text_vocab_prefix(const unsigned char *bytes, const unsigned long long *at,
                           unsigned long long tokens, unsigned int *wrong)
{
    if (tokens == 0ull) {
        return 1;
    }
    unsigned long long run = at[tokens];
    void *text = 0;
    void *table = 0;
    void *report = 0;
    aotx_check_runtime(cudaMalloc(&text, (size_t)run), "cudaMalloc");
    aotx_check_runtime(cudaMalloc(&table, (size_t)((tokens + 1ull) * sizeof(unsigned long long))),
                       "cudaMalloc");
    aotx_check_runtime(cudaMalloc(&report, sizeof(unsigned int)), "cudaMalloc");
    aotx_check_runtime(cudaMemcpy(text, bytes, (size_t)run, cudaMemcpyHostToDevice),
                       "cudaMemcpy");
    aotx_check_runtime(cudaMemcpy(table, at,
                                  (size_t)((tokens + 1ull) * sizeof(unsigned long long)),
                                  cudaMemcpyHostToDevice), "cudaMemcpy");
    aotx_check_runtime(cudaMemset(report, 0, sizeof(unsigned int)), "cudaMemset");
    aotx_text_check_prefix<<<AOTX_TEXT_GRID, AOTX_TEXT_BLOCK>>>(
        (const unsigned char *)text, (const unsigned long long *)table,
        (unsigned int)tokens, (unsigned int *)report);
    aotx_check_runtime(cudaMemcpy(wrong, report, sizeof(unsigned int),
                                  cudaMemcpyDeviceToHost), "cudaMemcpy");
    cudaFree(text);
    cudaFree(table);
    cudaFree(report);
    return 0;
}
