/* Purpose: Build a shared vocabulary or a separate embedding vocabulary.
 * Owns: The vocabulary allocation handles and their descriptors.
 * Launch shape: Host glue only; vocabulary checks and builds run on the device.
 * Lifetime: From model load to the end of the run. */
#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>
#include <stddef.h>
#include "boot/check.h"
#include "boot/vocab_host.h"
#include "text/text.cuh"

static aotx_text_store aotx_vocab_store;
static aotx_text_store aotx_vocab_embedding;
static aotx_text_store aotx_vocab_audio;
static int aotx_vocab_base_ready, aotx_vocab_audio_ready;
static unsigned int aotx_vocab_family;
static int aotx_vocab_first_embedding;
static int aotx_vocab_separate;

int aotx_boot_vocab_family(const aotx_modelfile *file, const char *name, unsigned int *row)
{
    const char *value = NULL;
    size_t length = 0u;
    if (aotx_modelfile_string(file, "tokenizer.ggml.pre", &value, &length) != 0) {
        fprintf(stderr, "%s does not name a pre-tokenizer\n", name);
        return 1;
    }
    if (aotx_text_family_find(value, length, row) != 0) {
        fprintf(stderr, "%s names the pre-tokenizer %.*s, which is not a family of this "
                "system\n", name, (int)length, value);
        return 1;
    }
    return 0;
}

static int aotx_vocab_arrays(const aotx_modelfile *file, const char *name,
                             aotx_text_source *source)
{
    aotx_string_array tokens;
    aotx_string_array merges;
    const int32_t *types = NULL;
    uint64_t type_count = 0ull;
    if (aotx_modelfile_strings(file, "tokenizer.ggml.tokens", &tokens) != 0
        || aotx_modelfile_strings(file, "tokenizer.ggml.merges", &merges) != 0
        || aotx_modelfile_i32s(file, "tokenizer.ggml.token_type", &types, &type_count) != 0) {
        fprintf(stderr, "a file does not hold the tokenizer arrays\n");
        return 1;
    }
    if (type_count != tokens.count) {
        fprintf(stderr, "the token count and the type count differ\n");
        return 1;
    }
    memset(source, 0, sizeof *source);
    source->token_bytes = tokens.bytes;
    source->token_at = (const unsigned long long *)tokens.offsets;
    source->tokens = tokens.count;
    source->merge_bytes = merges.bytes;
    source->merge_at = (const unsigned long long *)merges.offsets;
    source->merges = merges.count;
    source->token_type = (const int *)types;
    return aotx_boot_vocab_family(file, name, &source->family);
}

static int aotx_vocab_build(const aotx_text_source *source, aotx_text_store *store,
                            const char *name)
{
    int state = aotx_text_vocab_build(source, store);
    if (state != 0) {
        fprintf(stderr, "the vocabulary build of %s gave %d\n", name, state);
        return 1;
    }
    printf("vocabulary: %llu tokens %llu merges %llu KB from %s\n",
           source->tokens, source->merges, store->bytes >> 10, name);
    return 0;
}

static void aotx_vocab_save(unsigned int embedding)
{
    aotx_text_vocab table;
    aotx_check_runtime(cudaMemcpyFromSymbol(&table, aotx_text_vocab_table, sizeof table),
                       "cudaMemcpyFromSymbol");
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_text_vocab_saved, &table, sizeof table,
                                         embedding * sizeof table), "cudaMemcpyToSymbol");
}

int aotx_boot_vocab_take(const aotx_modelfile *file, const char *name, int build, int embedding)
{
    aotx_text_source source;
    if (aotx_vocab_arrays(file, name, &source) != 0) return 1;
    if (embedding == 2) {
        if (aotx_vocab_audio_ready) return 1;
        if (aotx_vocab_base_ready) aotx_vocab_save(0u);
        if (aotx_vocab_build(&source, &aotx_vocab_audio, name)) return 1;
        aotx_vocab_save(2u); aotx_vocab_audio_ready=1;
        if (aotx_vocab_base_ready) {
            aotx_text_vocab_select<<<1,128>>>(0u);
            aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        }
        return 0;
    }
    if (!aotx_vocab_base_ready) build=1;
    if (build) {
        aotx_vocab_base_ready=1;
        aotx_vocab_first_embedding = embedding;
        aotx_vocab_separate = 0;
        aotx_vocab_family = source.family;
        return aotx_vocab_build(&source, &aotx_vocab_store, name);
    }
    unsigned int wrong = 0u;
    unsigned int current_tokens = 0u;
    unsigned long long compare_tokens = source.tokens;
    if (aotx_vocab_separate && !embedding) {
        aotx_check_runtime(cudaMemcpyFromSymbol(&current_tokens, aotx_text_vocab_table,
                           sizeof current_tokens, offsetof(aotx_text_vocab, tokens)),
                           "cudaMemcpyFromSymbol");
        if (compare_tokens > current_tokens) compare_tokens = current_tokens;
    }
    if (source.family == aotx_vocab_family
        && aotx_text_vocab_prefix(source.token_bytes, source.token_at, compare_tokens,
                                  &wrong) != 0) {
        fprintf(stderr, "the vocabulary of %s did not compare\n", name);
        return 1;
    }
    if (source.family == aotx_vocab_family && wrong == 0u) {
        if (aotx_vocab_separate && !embedding && source.tokens > current_tokens) {
            aotx_text_store next = {};
            if (aotx_vocab_build(&source, &next, name) != 0) {
                aotx_text_vocab_release(&next);
                aotx_text_vocab_select<<<1, 128>>>(0u);
                aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
                return 1;
            }
            aotx_text_vocab_release(&aotx_vocab_store);
            aotx_vocab_store = next;
            aotx_vocab_save(0u);
        }
        if (!embedding) aotx_vocab_first_embedding = 0;
        return 0;
    }
    /* The shared path stays unchanged. Only the embedding role may use another table. */
    if (aotx_vocab_separate || embedding == aotx_vocab_first_embedding) {
        fprintf(stderr, "%s does not share the language vocabulary\n", name);
        return 1;
    }
    if (embedding) {
        aotx_vocab_save(0u);
        if (aotx_vocab_build(&source, &aotx_vocab_embedding, name) != 0) return 1;
        aotx_vocab_save(1u);
        aotx_text_vocab_select<<<1, 128>>>(0u);
        aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    } else {
        aotx_vocab_save(1u);
        aotx_vocab_embedding = aotx_vocab_store;
        memset(&aotx_vocab_store, 0, sizeof aotx_vocab_store);
        if (aotx_vocab_build(&source, &aotx_vocab_store, name) != 0) return 1;
        aotx_vocab_save(0u);
        aotx_vocab_family = source.family;
        aotx_vocab_first_embedding = 0;
    }
    aotx_vocab_separate = 1;
    printf("vocabulary: the embedding role has a separate table\n");
    return 0;
}

void aotx_boot_vocab_finish(int audio_default)
{
    if (aotx_vocab_base_ready) aotx_vocab_save(0u);
    if (!aotx_vocab_separate && aotx_vocab_base_ready) aotx_vocab_save(1u);
    if (audio_default && aotx_vocab_audio_ready) {
        aotx_text_vocab_select<<<1,128>>>(2u);
        aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        aotx_vocab_save(0u);
        if (aotx_vocab_base_ready) aotx_vocab_separate=1;
    } else if (aotx_vocab_base_ready) {
        aotx_text_vocab_select<<<1,128>>>(0u);
        aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    }
}

int aotx_text_embedding_separate(void)
{
    return aotx_vocab_separate;
}

void aotx_boot_vocab_release(void)
{
    aotx_text_vocab_release(&aotx_vocab_store);
    aotx_text_vocab_release(&aotx_vocab_embedding);
    aotx_text_vocab_release(&aotx_vocab_audio);
    aotx_vocab_base_ready=aotx_vocab_audio_ready=0;
    aotx_text_vocab empty[3]={};
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_text_vocab_saved,empty,sizeof empty),"cudaMemcpyToSymbol");
    aotx_vocab_separate = 0;
    aotx_vocab_first_embedding = 0;
}
