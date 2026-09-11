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
#include "boot/vocab_host.h"
#include "mem/mem.cuh"
#include "model/forward.cuh"
#include "model/kinds.h"
#include "model/load.cuh"
#include "model/roles.h"
#include "model/conduct.cuh"
#include "text/text.cuh"
#ifdef AOTX_AFFECT
#include "affect/affect.cuh"
#endif

extern "C" {
#include "disk/modelfile/manifest.h"
#include "disk/runtime/assets.h"
#include "disk/modelfile/modelfile.h"
}

#define AOTX_MODELS_MAX     8

/* Open one file of the record. The path comes from the library. A name which starts at the
 * root gives that name, and a name which does not gives the name under the directory. The
 * check of the digest reads the same path, so both read one file. */
static int aotx_models_open(const char *dir, const aotx_manifest_entry *entry,
                            aotx_modelfile **file)
{
    if (aotx_modelfile_open_entry(dir, entry, file) != 0) {
        fprintf(stderr, "the file %s did not open\n", entry->path);
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
#ifdef AOTX_AFFECT
    aotx_affect_release();
#endif
    aotx_conduct_release();
    aotx_boot_vocab_release();
}

static int aotx_models_take(const char *dir, const char *roles, int (*stopped)(void))
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
        unsigned int row = 0u;
        if (aotx_boot_vocab_family(file, entries[i].name, &row) != 0
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
            bad = aotx_boot_vocab_take(file, entries[i].name, k == 0,
                                       strcmp(entries[i].role, "embedding") == 0);
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
        aotx_model_desc desc[AOTX_MODEL_ROLES];
        unsigned int want[AOTX_MODEL_ROLES];
        unsigned int wanted = aotx_role_list(roles, want);
        aotx_check_runtime(cudaMemcpyFromSymbol(desc, aotx_model, sizeof desc),
                           "cudaMemcpyFromSymbol");
        aotx_layer_print(desc, want, wanted);
    }
    if (bad == 0) {
        bad = aotx_model_load_open(dir, roles, cursor);
    }
    if (bad == 0) {
        bad = aotx_conduct_load_store(dir);
    }
#ifdef AOTX_AFFECT
    /* The probe rows come after the vectors, so the width check reads a placed model. */
    if (bad == 0) {
        bad = aotx_affect_load_store(dir);
    }
    if (bad == 0) {
        bad = aotx_quality_load_store(dir);
    }
#endif
    return bad;
}

int aotx_boot_models(const char *dir, const char *roles, int (*stopped)(void))
{
    int rc = aotx_asset_begin(dir);
    if (rc) {
        fprintf(stderr, "the model source is refused: %s\n", aotx_ccir_status_text(rc));
        return 2;
    }
    rc = aotx_models_take(dir, roles, stopped);
    aotx_asset_end();
    return rc;
}
