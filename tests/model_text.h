/* Purpose: Give the model gate the tokenizer buffers and the fixture readers it shares.
 * Owns: The tokenizer buffers of one gate run.
 * Threading: One thread; the gate calls these one at a time.
 * Lifetime: The program. */
#ifndef AOTX_TEST_MODEL_TEXT_H
#define AOTX_TEST_MODEL_TEXT_H

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "boot/check.h"
#include "text/text.cuh"

#define AOTX_GATE_MAX      80u
#define AOTX_GATE_STRIDE   1024u
#define AOTX_GATE_RUN      (256u * 1024u)
#define AOTX_GATE_CLEAN    8192u
#define AOTX_GATE_BLOCKS   64u

static void *aotx_gate_take(unsigned long long bytes)
{
    void *block = 0;
    aotx_check_runtime(cudaMalloc(&block, (size_t)bytes), "cudaMalloc");
    aotx_check_runtime(cudaMemset(block, 0, (size_t)bytes), "cudaMemset");
    return block;
}

/* Read a whole file into memory. */
static char *aotx_gate_slurp(const char *path, unsigned long long *bytes)
{
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        return 0;
    }
    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    fseek(file, 0, SEEK_SET);
    char *block = (char *)malloc((size_t)size + 1);
    size_t got = fread(block, 1u, (size_t)size, file);
    fclose(file);
    block[got] = '\0';
    *bytes = (unsigned long long)got;
    return block;
}

/* Cut a byte run into lines. Each line goes in its own buffer, and it keeps the newline
 * byte that ends it when the caller asks for it. */
static unsigned int aotx_gate_lines(const char *run, unsigned long long bytes, char **line,
                                    unsigned int max, int keep)
{
    unsigned int count = 0u;
    unsigned long long at = 0u;
    while (at < bytes && count < max) {
        unsigned long long end = at;
        while (end < bytes && run[end] != '\n') {
            ++end;
        }
        unsigned long long size = end - at + ((keep != 0 && end < bytes) ? 1u : 0u);
        line[count] = (char *)malloc((size_t)size + 1u);
        memcpy(line[count], run + at, (size_t)size);
        line[count][size] = '\0';
        count += 1u;
        at = end + 1u;
    }
    return count;
}

/* The tokenizer buffers. The gate tokenizes on the device, as the system does. */
typedef struct aotx_gate_text {
    unsigned char *bytes;
    unsigned int *start;
    unsigned int *length;
    unsigned char *clean;
    unsigned int *clean_start;
    unsigned int *clean_length;
    aotx_text_pieces pieces;
    aotx_text_tokens tokens;
} aotx_gate_text;

static void aotx_gate_text_open(aotx_gate_text *gear)
{
    unsigned long long slots = (unsigned long long)AOTX_GATE_MAX * AOTX_GATE_STRIDE;
    unsigned long long clean = (unsigned long long)AOTX_GATE_MAX * AOTX_GATE_CLEAN;
    unsigned int warps = AOTX_GATE_BLOCKS * AOTX_TEXT_WARPS;
    memset(gear, 0, sizeof *gear);
    gear->bytes = (unsigned char *)aotx_gate_take(AOTX_GATE_RUN);
    gear->start = (unsigned int *)aotx_gate_take(AOTX_GATE_MAX * sizeof(unsigned int));
    gear->length = (unsigned int *)aotx_gate_take(AOTX_GATE_MAX * sizeof(unsigned int));
    gear->clean = (unsigned char *)aotx_gate_take(clean);
    gear->clean_start = (unsigned int *)aotx_gate_take(AOTX_GATE_MAX * sizeof(unsigned int));
    gear->clean_length = (unsigned int *)aotx_gate_take(AOTX_GATE_MAX * sizeof(unsigned int));
    gear->pieces.start = (unsigned int *)aotx_gate_take(slots * sizeof(unsigned int));
    gear->pieces.length = (unsigned int *)aotx_gate_take(slots * sizeof(unsigned int));
    gear->pieces.token = (unsigned int *)aotx_gate_take(slots * sizeof(unsigned int));
    gear->pieces.count = (unsigned int *)aotx_gate_take(AOTX_GATE_MAX * sizeof(unsigned int));
    gear->pieces.work = (unsigned int *)aotx_gate_take(slots * sizeof(unsigned int));
    gear->pieces.works = (unsigned int *)aotx_gate_take(sizeof(unsigned int));
    gear->pieces.stride = AOTX_GATE_STRIDE;
    gear->tokens.id = (unsigned int *)aotx_gate_take(slots * sizeof(unsigned int));
    gear->tokens.count = (unsigned int *)aotx_gate_take(AOTX_GATE_MAX * sizeof(unsigned int));
    gear->tokens.chunk = (unsigned int *)aotx_gate_take(slots * sizeof(unsigned int));
    gear->tokens.scratch = (unsigned int *)aotx_gate_take(clean * sizeof(unsigned int));
    gear->tokens.merge = (unsigned char *)aotx_gate_take((unsigned long long)warps
                                                         * AOTX_TEXT_WARP_BYTES);
    gear->tokens.warps = warps;
    gear->tokens.stride = AOTX_GATE_STRIDE;
}

/* Tokenize a set of byte runs on the device and give the tokens of each run back. */
static void aotx_gate_tokenize(aotx_gate_text *gear, const char **text, unsigned int count,
                               unsigned int *ids, unsigned int *counts)
{
    unsigned char *run = (unsigned char *)malloc(AOTX_GATE_RUN);
    unsigned int *start = (unsigned int *)malloc(count * sizeof(unsigned int));
    unsigned int *length = (unsigned int *)malloc(count * sizeof(unsigned int));
    unsigned int at = 0u;
    for (unsigned int i = 0u; i < count; ++i) {
        unsigned int bytes = (unsigned int)strlen(text[i]);
        start[i] = at;
        length[i] = bytes;
        memcpy(run + at, text[i], bytes);
        at += bytes;
    }
    aotx_check_runtime(cudaMemcpy(gear->bytes, run, at, cudaMemcpyHostToDevice),
                       "cudaMemcpy");
    aotx_check_runtime(cudaMemcpy(gear->start, start, count * sizeof(unsigned int),
                                  cudaMemcpyHostToDevice), "cudaMemcpy");
    aotx_check_runtime(cudaMemcpy(gear->length, length, count * sizeof(unsigned int),
                                  cudaMemcpyHostToDevice), "cudaMemcpy");
    aotx_text_batch raw;
    raw.bytes = gear->bytes;
    raw.start = gear->start;
    raw.length = gear->length;
    raw.count = count;
    unsigned int blocks = (count + 63u) / 64u;
    aotx_text_clean<<<blocks, 64u>>>(raw, gear->clean, gear->clean_start,
                                     gear->clean_length, AOTX_GATE_CLEAN);
    aotx_text_batch batch;
    batch.bytes = gear->clean;
    batch.start = gear->clean_start;
    batch.length = gear->clean_length;
    batch.count = count;
    unsigned int zero = 0u;
    aotx_check_runtime(cudaMemcpy(gear->pieces.works, &zero, sizeof zero,
                                  cudaMemcpyHostToDevice), "cudaMemcpy");
    aotx_text_pretok<<<blocks, 64u>>>(batch, gear->pieces);
    aotx_text_merge<<<AOTX_GATE_BLOCKS, 32u * AOTX_TEXT_WARPS>>>(batch, gear->pieces,
                                                                 gear->tokens);
    aotx_text_gather<<<blocks, 64u>>>(batch, gear->pieces, gear->tokens);
    aotx_check_runtime(cudaMemcpy(counts, gear->tokens.count, count * sizeof(unsigned int),
                                  cudaMemcpyDeviceToHost), "cudaMemcpy");
    aotx_check_runtime(cudaMemcpy(ids, gear->tokens.id,
                                  (size_t)count * AOTX_GATE_STRIDE * sizeof(unsigned int),
                                  cudaMemcpyDeviceToHost), "cudaMemcpy");
    free(run);
    free(start);
    free(length);
}

#endif
