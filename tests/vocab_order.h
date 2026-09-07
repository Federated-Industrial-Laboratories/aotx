/* Purpose: Check both language vocabulary orders after an embedding split.
 * Owns: Three temporary metadata files and their loaded vocabulary stores.
 * Launch shape: Host file setup and device vocabulary checks.
 * Lifetime: One vocabulary test process. */
#ifndef AOTX_TEST_VOCAB_ORDER_H
#define AOTX_TEST_VOCAB_ORDER_H
#include <stdlib.h>
#include <unistd.h>
#include "boot/vocab_host.h"

static int aotx_vocab_u32(FILE *file, uint32_t value)
{
    return fwrite(&value, sizeof value, 1u, file) != 1u;
}

static int aotx_vocab_u64(FILE *file, uint64_t value)
{
    return fwrite(&value, sizeof value, 1u, file) != 1u;
}

static int aotx_vocab_string(FILE *file, const char *value)
{
    size_t bytes = strlen(value);
    return aotx_vocab_u64(file, bytes) || fwrite(value, 1u, bytes, file) != bytes;
}

static int aotx_vocab_file(char *path, unsigned int keys, char prefix)
{
    int fd = mkstemp(path);
    if (fd < 0) return 1;
    FILE *file = fdopen(fd, "wb");
    if (file == NULL) { close(fd); return 1; }
    int bad = aotx_vocab_u32(file, 0x46554747u) || aotx_vocab_u32(file, 3u)
           || aotx_vocab_u64(file, 0u) || aotx_vocab_u64(file, 4u);
    bad |= aotx_vocab_string(file, "tokenizer.ggml.pre") || aotx_vocab_u32(file, 8u)
        || aotx_vocab_string(file, "qwen2");
    bad |= aotx_vocab_string(file, "tokenizer.ggml.tokens") || aotx_vocab_u32(file, 9u)
        || aotx_vocab_u32(file, 8u) || aotx_vocab_u64(file, keys + 4u);
    const char *base[] = {"a", "b", "ab", "<|endoftext|>"};
    for (const char *token : base) bad |= aotx_vocab_string(file, token);
    for (unsigned int key = 0u; key < keys; ++key) {
        char token[16];
        int bytes = snprintf(token, sizeof token, "%c%03u", prefix, key);
        if (bytes < 0 || (size_t)bytes >= sizeof token) { bad = 1; break; }
        bad |= aotx_vocab_string(file, token);
    }
    bad |= aotx_vocab_string(file, "tokenizer.ggml.merges") || aotx_vocab_u32(file, 9u)
        || aotx_vocab_u32(file, 8u) || aotx_vocab_u64(file, 1u)
        || aotx_vocab_string(file, "a b");
    bad |= aotx_vocab_string(file, "tokenizer.ggml.token_type") || aotx_vocab_u32(file, 9u)
        || aotx_vocab_u32(file, 5u) || aotx_vocab_u64(file, keys + 4u);
    for (unsigned int key = 0u; key < keys + 4u; ++key)
        bad |= aotx_vocab_u32(file, key == 3u ? AOTX_TEXT_TYPE_CONTROL : 1u);
    long bytes = ftell(file);
    if (bytes < 0) bad = 1;
    while (bytes >= 0 && (bytes++ % 32) != 0) bad |= fputc(0, file) == EOF;
    bad |= fclose(file) != 0;
    return bad;
}

static unsigned int aotx_vocab_order_cases(unsigned int *checks)
{
    char paths[3][64] = {"/tmp/aotx-vocab-embed-XXXXXX", "/tmp/aotx-vocab-small-XXXXXX",
                         "/tmp/aotx-vocab-large-XXXXXX"};
    unsigned int failed = 0u;
    aotx_modelfile *files[3] = {};
    const unsigned int keys[] = {8u, 2u, 4u};
    for (unsigned int i = 0u; i < 3u; ++i) {
        if (aotx_vocab_file(paths[i], keys[i], i == 0u ? 'e' : 'n')
            || aotx_modelfile_open(paths[i], &files[i]) != 0) failed = 1u;
    }
    if (failed == 0u) {
        unsigned int *device = NULL;
        aotx_check_runtime(cudaMalloc(&device, 4u * sizeof(unsigned int)), "cudaMalloc");
        for (unsigned int order = 0u; order < 2u; ++order) {
            unsigned int first = order == 0u ? 1u : 2u;
            unsigned int second = order == 0u ? 2u : 1u;
            int bad = aotx_boot_vocab_take(files[0], "embedding", 1, 1)
                   || aotx_boot_vocab_take(files[first], "first", 0, 0)
                   || aotx_boot_vocab_take(files[second], "second", 0, 0);
            if (!bad) {
                unsigned int out[4];
                aotx_text_vocab_select<<<1, 128>>>(0u);
                aotx_vocab_lookup_test<<<1, AOTX_SLOTS>>>(device, 4u);
                aotx_check_runtime(cudaMemcpy(out, device, sizeof out,
                                              cudaMemcpyDeviceToHost), "cudaMemcpy");
                for (unsigned int i = 0u; i < 4u; ++i) bad |= out[i] != i + 4u;
            }
            ++*checks;
            failed += bad != 0;
            printf("vocabulary order %u: %s\n", order, bad ? "failed" : "passed");
            aotx_boot_vocab_release();
        }
        cudaFree(device);
    } else {
        ++*checks;
        printf("vocabulary: the metadata fixtures did not open\n");
    }
    for (unsigned int i = 0u; i < 3u; ++i) {
        if (files[i]) aotx_modelfile_close(files[i]);
        unlink(paths[i]);
    }
    return failed;
}
#endif
