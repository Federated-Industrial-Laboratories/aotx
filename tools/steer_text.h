/* Purpose: Give the steer tool one bounded tokenizer batch.
 * Owns: Device buffers used while the tool runs.
 * Launch shape: Host glue launches the normal batch tokenizer kernels.
 * Lifetime: One derivation program. */
#ifndef AOTX_TOOLS_STEER_TEXT_H
#define AOTX_TOOLS_STEER_TEXT_H

#include <cuda_runtime.h>
#include <stdlib.h>
#include <string.h>

#include "boot/check.h"
#include "text/text.cuh"

#define AOTX_STEER_TEXTS 64u
#define AOTX_STEER_STRIDE 1024u
#define AOTX_STEER_BYTES (256u * 1024u)
#define AOTX_STEER_CLEAN 8192u
#define AOTX_STEER_BLOCKS 64u

typedef struct aotx_steer_text {
    unsigned char *bytes, *clean;
    unsigned int *start, *length, *clean_start, *clean_length;
    aotx_text_pieces pieces;
    aotx_text_tokens tokens;
    void *piece[16];
    unsigned int count;
} aotx_steer_text;

static void *aotx_steer_take(aotx_steer_text *s, size_t bytes)
{
    void *at = 0;
    aotx_check_runtime(cudaMalloc(&at, bytes), "cudaMalloc");
    aotx_check_runtime(cudaMemset(at, 0, bytes), "cudaMemset");
    s->piece[s->count++] = at;
    return at;
}

static void aotx_steer_text_open(aotx_steer_text *s)
{
    size_t slots = (size_t)AOTX_STEER_TEXTS * AOTX_STEER_STRIDE;
    size_t clean = (size_t)AOTX_STEER_TEXTS * AOTX_STEER_CLEAN;
    memset(s, 0, sizeof *s);
    s->bytes = (unsigned char *)aotx_steer_take(s, AOTX_STEER_BYTES);
    s->start = (unsigned int *)aotx_steer_take(s, AOTX_STEER_TEXTS * sizeof(unsigned int));
    s->length = (unsigned int *)aotx_steer_take(s, AOTX_STEER_TEXTS * sizeof(unsigned int));
    s->clean = (unsigned char *)aotx_steer_take(s, clean);
    s->clean_start = (unsigned int *)aotx_steer_take(s, AOTX_STEER_TEXTS * sizeof(unsigned int));
    s->clean_length = (unsigned int *)aotx_steer_take(s, AOTX_STEER_TEXTS * sizeof(unsigned int));
    s->pieces.start = (unsigned int *)aotx_steer_take(s, slots * sizeof(unsigned int));
    s->pieces.length = (unsigned int *)aotx_steer_take(s, slots * sizeof(unsigned int));
    s->pieces.token = (unsigned int *)aotx_steer_take(s, slots * sizeof(unsigned int));
    s->pieces.count = (unsigned int *)aotx_steer_take(s, AOTX_STEER_TEXTS * sizeof(unsigned int));
    s->pieces.work = (unsigned int *)aotx_steer_take(s, slots * sizeof(unsigned int));
    s->pieces.works = (unsigned int *)aotx_steer_take(s, sizeof(unsigned int));
    s->pieces.stride = AOTX_STEER_STRIDE;
    s->tokens.id = (unsigned int *)aotx_steer_take(s, slots * sizeof(unsigned int));
    s->tokens.count = (unsigned int *)aotx_steer_take(s, AOTX_STEER_TEXTS * sizeof(unsigned int));
    s->tokens.chunk = (unsigned int *)aotx_steer_take(s, slots * sizeof(unsigned int));
    s->tokens.scratch = (unsigned int *)aotx_steer_take(s, clean * sizeof(unsigned int));
    unsigned int warps = AOTX_STEER_BLOCKS * AOTX_TEXT_WARPS;
    s->tokens.merge = (unsigned char *)aotx_steer_take(s,
        (size_t)warps * AOTX_TEXT_WARP_BYTES);
    s->tokens.warps = warps;
    s->tokens.stride = AOTX_STEER_STRIDE;
}

static int aotx_steer_tokenize(aotx_steer_text *s, char **text, unsigned int count,
                               unsigned int *ids, unsigned int *counts)
{
    unsigned char *run = (unsigned char *)malloc(AOTX_STEER_BYTES);
    unsigned int start[AOTX_STEER_TEXTS], length[AOTX_STEER_TEXTS], at = 0u;
    if (run == 0 || count == 0u || count > AOTX_STEER_TEXTS) { free(run); return 1; }
    for (unsigned int i = 0u; i < count; ++i) {
        size_t n = strlen(text[i]);
        if (n > AOTX_STEER_CLEAN || at + n > AOTX_STEER_BYTES) { free(run); return 1; }
        start[i] = at; length[i] = (unsigned int)n; memcpy(run + at, text[i], n); at += n;
    }
    aotx_check_runtime(cudaMemcpy(s->bytes, run, at, cudaMemcpyHostToDevice), "cudaMemcpy");
    aotx_check_runtime(cudaMemcpy(s->start, start, count * sizeof(unsigned int), cudaMemcpyHostToDevice), "cudaMemcpy");
    aotx_check_runtime(cudaMemcpy(s->length, length, count * sizeof(unsigned int), cudaMemcpyHostToDevice), "cudaMemcpy");
    aotx_text_batch raw = { s->bytes, s->start, s->length, count };
    unsigned int blocks = (count + 63u) / 64u, zero = 0u;
    aotx_text_clean<<<blocks, 64u>>>(raw, s->clean, s->clean_start, s->clean_length, AOTX_STEER_CLEAN);
    aotx_text_batch batch = { s->clean, s->clean_start, s->clean_length, count };
    aotx_check_runtime(cudaMemcpy(s->pieces.works, &zero, sizeof zero, cudaMemcpyHostToDevice), "cudaMemcpy");
    aotx_text_pretok<<<blocks, 64u>>>(batch, s->pieces);
    aotx_text_merge<<<AOTX_STEER_BLOCKS, 32u * AOTX_TEXT_WARPS>>>(batch, s->pieces, s->tokens);
    aotx_text_gather<<<blocks, 64u>>>(batch, s->pieces, s->tokens);
    aotx_check_runtime(cudaMemcpy(counts, s->tokens.count, count * sizeof(unsigned int), cudaMemcpyDeviceToHost), "cudaMemcpy");
    aotx_check_runtime(cudaMemcpy(ids, s->tokens.id, (size_t)count * AOTX_STEER_STRIDE * sizeof(unsigned int), cudaMemcpyDeviceToHost), "cudaMemcpy");
    free(run); return 0;
}

static void aotx_steer_text_close(aotx_steer_text *s)
{
    for (unsigned int i = 0u; i < s->count; ++i) cudaFree(s->piece[i]);
    memset(s, 0, sizeof *s);
}

#endif
