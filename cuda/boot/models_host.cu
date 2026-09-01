/* Purpose: Check the model files, load them, and build the vocabulary of the set.
 * Owns: The vocabulary tables of the run.
 * Launch shape: Host glue only; the vocabulary build holds the kernels.
 * Lifetime: From the model load at start to the end of the run. */
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "boot/boot.cuh"
#include "boot/check.h"
#include "mem/mem.cuh"
#include "model/forward.cuh"
#include "model/load.cuh"
#include "model/roles.h"
#include "model/conduct.cuh"
#include "text/text.cuh"

extern "C" {
#include "disk/modelfile/manifest.h"
#include "disk/modelfile/modelfile.h"
}

#define AOTX_MODELS_MAX     8

/* The vocabulary of the run. The tables live from the load to the end of the run. The
 * release is there for a caller which loads a model set twice. */
static aotx_text_store aotx_models_store;

/* Open one file of the record. The path comes from the library. A name which starts at the
 * root gives that name, and a name which does not gives the name under the directory. The
 * check of the digest reads the same path, so both read one file. */
static int aotx_models_open(const char *dir, const aotx_manifest_entry *entry,
                            aotx_modelfile **file)
{
    char path[AOTX_MANIFEST_PATH];
    if (aotx_manifest_path(path, sizeof path, dir, entry->path) != 0) {
        fprintf(stderr, "the path of %s is too long\n", entry->name);
        return 1;
    }
    if (aotx_modelfile_open(path, file) != 0) {
        fprintf(stderr, "the file %s did not open\n", entry->path);
        return 1;
    }
    return 0;
}

/* The pre-tokenizer name of every file must be the one the device state machine holds. */
static int aotx_models_family(const aotx_modelfile *file, const char *name)
{
    const char *value = NULL;
    size_t length = 0u;
    if (aotx_modelfile_string(file, "tokenizer.ggml.pre", &value, &length) != 0) {
        fprintf(stderr, "%s does not name a pre-tokenizer\n", name);
        return 1;
    }
    if (length != 5u || memcmp(value, "qwen2", 5) != 0) {
        fprintf(stderr, "%s names another pre-tokenizer family\n", name);
        return 1;
    }
    return 0;
}

/* Read the three tokenizer arrays of a file. */
static int aotx_models_arrays(const aotx_modelfile *file, aotx_text_source *source)
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
    /* The reader states its counts and offsets in the whole number types of the file
     * format. Those types and the types of the device header are the same width, and the
     * cast states that the two names stand for one layout. */
    source->token_bytes = tokens.bytes;
    source->token_at = (const unsigned long long *)tokens.offsets;
    source->tokens = tokens.count;
    source->merge_bytes = merges.bytes;
    source->merge_at = (const unsigned long long *)merges.offsets;
    source->merges = merges.count;
    source->token_type = (const int *)types;
    return 0;
}

/* One table serves the set. The table comes from the file with the most tokens. The tokens
 * of every other file must be the tokens of the same ids in that table. */
static int aotx_models_vocab(const aotx_modelfile *file, const char *name, int build)
{
    aotx_text_source source;
    if (aotx_models_arrays(file, &source) != 0) {
        return 1;
    }
    if (build) {
        int state = aotx_text_vocab_build(&source, &aotx_models_store);
        if (state != 0) {
            fprintf(stderr, "the vocabulary build of %s gave %d\n", name, state);
            return 1;
        }
        printf("vocabulary: %llu tokens %llu merges %llu KB from %s\n",
               (unsigned long long)source.tokens, (unsigned long long)source.merges,
               aotx_models_store.bytes >> 10, name);
        return 0;
    }
    unsigned int wrong = 0u;
    if (aotx_text_vocab_prefix(source.token_bytes, source.token_at, source.tokens,
                               &wrong) != 0) {
        fprintf(stderr, "the vocabulary of %s did not compare\n", name);
        return 1;
    }
    if (wrong != 0u) {
        fprintf(stderr, "%u tokens of %s are not the tokens of the table\n", wrong, name);
        return 1;
    }
    return 0;
}

static double aotx_models_now(void)
{
    struct timespec at;
    clock_gettime(CLOCK_MONOTONIC, &at);
    return (double)at.tv_sec + (double)at.tv_nsec * 1e-9;
}

void aotx_boot_models_release(void)
{
    aotx_conduct_release();
    aotx_text_vocab_release(&aotx_models_store);
}

int aotx_boot_models(const char *dir, const char *roles, int (*stopped)(void))
{
    char unknown[64];
    if (aotx_role_unknown(roles, unknown, sizeof unknown) != 0) {
        fprintf(stderr, "the role %s is not a role of this system\n", unknown);
        return 2;
    }
    aotx_manifest_entry entries[AOTX_MODELS_MAX];
    int count = aotx_manifest_read(dir, entries, AOTX_MODELS_MAX);
    if (count <= 0) {
        fprintf(stderr, "the model record in %s did not read\n", dir);
        return 2;
    }

    /* Only the entries the run asks for are read. An entry the list does not name costs
     * nothing: no digest, no bytes of the device, no place in the vocabulary. */
    int keep[AOTX_MODELS_MAX];
    int held = 0;
    for (int i = 0; i < count; ++i) {
        if (aotx_role_wanted(roles, entries[i].role) != 0) {
            keep[held++] = i;
        }
    }
    if (held == 0) {
        fprintf(stderr, "the model record in %s names none of the roles asked for\n", dir);
        return 2;
    }

    /* The first pass reads the digest of each file and the size of its vocabulary. The
     * digest of a large set takes tens of seconds, so a stop signal ends it. */
    int largest = 0;
    unsigned long long most = 0ull;
    unsigned long long digest = 0ull;
    double checked = aotx_models_now();
    for (int k = 0; k < held; ++k) {
        int i = keep[k];
        if (stopped != 0 && stopped() != 0) {
            fprintf(stderr, "a signal stopped the model load\n");
            return 1;
        }
        int state = aotx_manifest_check(dir, &entries[i]);
        if (state == 1) {
            fprintf(stderr, "the file %s does not match its record\n", entries[i].path);
            return 2;
        }
        if (state != 0) {
            fprintf(stderr, "the file %s did not read\n", entries[i].path);
            return 2;
        }
        aotx_modelfile *file = NULL;
        if (aotx_models_open(dir, &entries[i], &file) != 0) {
            return 2;
        }
        aotx_string_array tokens;
        if (aotx_models_family(file, entries[i].name) != 0
            || aotx_modelfile_strings(file, "tokenizer.ggml.tokens", &tokens) != 0) {
            aotx_modelfile_close(file);
            return 2;
        }
        if (tokens.count > most) {
            most = tokens.count;
            largest = k;
        }
        digest += entries[i].bytes;
        aotx_modelfile_close(file);
    }
    printf("digest: %d files %llu MB %.1f s\n", held, digest >> 20,
           aotx_models_now() - checked);

    /* The second pass places the tensors. The file with the most tokens comes first,
     * because it builds the table that every other file is compared with. */
    if (aotx_model_weights_open() != 0) {
        return 1;
    }
    double started = aotx_models_now();
    unsigned long long cursor = 0ull;
    unsigned int placed = 0u;
    unsigned int left = 0u;
    unsigned int loaded = 0u;
    int bad = 0;
    for (int k = 0; k < held && bad == 0; ++k) {
        int at = (k == 0) ? largest : ((k <= largest) ? k - 1 : k);
        int i = keep[at];
        if (stopped != 0 && stopped() != 0) {
            fprintf(stderr, "a signal stopped the model load\n");
            bad = 1;
            break;
        }
        aotx_modelfile *file = NULL;
        if (aotx_models_open(dir, &entries[i], &file) != 0) {
            bad = 1;
            break;
        }

        /* The number of a model in the tensor table is its place in the model record and
         * not its place in the run. A run of a subset therefore finds the same tensors. */
        bad = aotx_model_weights_place(file, (unsigned int)i, &cursor, &placed, &left);
        if (bad == 0) {
            bad = aotx_models_vocab(file, entries[i].name, k == 0);
        }
        aotx_modelfile_close(file);
        loaded += 1u;
    }
    aotx_model_weights_close();
    double spent = aotx_models_now() - started;
    double rate = (spent > 0.0) ? (double)cursor / spent / (1024.0 * 1024.0) : 0.0;
    printf("models: %u files %u tensors %u left %llu MB mapped %.2f s %.0f MB a second\n",
           loaded, placed, left, aotx_mem_weights_held() >> 20, spent, rate);

    /* The descriptor of each role comes from the same list, so a load and a descriptor
     * cannot fall out of step. */
    if (bad == 0) {
        bad = aotx_model_describe(dir, roles);
    }
    if (bad == 0) {
        bad = aotx_model_load_open(dir, roles, cursor);
    }
    if (bad == 0) {
        bad = aotx_conduct_load_store(dir);
    }
    return bad;
}
