/* Purpose: Rebuild resident weights when one language allocation replaces another.
 * Owns: Nothing; the memory map owns the physical pieces and the model files own the bytes.
 * Launch shape: Host glue only; the weights loader builds tensor rows in batches.
 * Lifetime: One run-time language replacement. */
#include <cuda_runtime.h>
#include <stdio.h>

#include "boot/boot.cuh"
#include "boot/check.h"
#include "mem/mem.cuh"
#include "model/layout_host.h"
#include "model/roles.h"

extern "C" {
#include "disk/modelfile/manifest.h"
}

static int aotx_layout_open(const char *dir, const aotx_manifest_entry *entry,
                            aotx_modelfile **file)
{
    char path[AOTX_MANIFEST_PATH];
    if (aotx_manifest_path(path, sizeof path, dir, entry->path) != 0
        || aotx_modelfile_open(path, file) != 0) {
        fprintf(stderr, "the file %s did not open\n", entry->path);
        return 1;
    }
    return 0;
}

static unsigned int aotx_layout_sources(const aotx_model_load_state *state,
                                         unsigned int source,
                                         unsigned int out[AOTX_MODEL_FILES_MAX])
{
    unsigned int count = 0u;
    for (unsigned int role = 0u; role < AOTX_MODEL_ROLES; ++role) {
        const aotx_model_resident_row *row = &state->resident[role];
        if (row->active == 0u || row->slot == AOTX_MODEL_LANGUAGE
            || row->slot == AOTX_MODEL_LANGUAGE_Q4) {
            continue;
        }
        unsigned int seen = 0u;
        for (unsigned int i = 0u; i < count; ++i) {
            seen |= out[i] == row->source;
        }
        if (seen == 0u && count < AOTX_MODEL_FILES_MAX) {
            out[count++] = row->source;
        }
    }
    if (count < AOTX_MODEL_FILES_MAX) {
        out[count++] = source;
    }
    return count;
}

static int aotx_layout_preflight(const char *dir, const aotx_manifest_entry *entry,
                                 unsigned int entries, const unsigned int *source,
                                 unsigned int count, unsigned long long *end)
{
    unsigned long long cursor = 0ull;
    for (unsigned int i = 0u; i < count; ++i) {
        aotx_modelfile *file = NULL;
        if (source[i] >= entries || aotx_layout_open(dir, &entry[source[i]], &file) != 0) {
            return 1;
        }
        int bad = aotx_model_weights_fits(file, cursor, &cursor);
        aotx_modelfile_close(file);
        if (bad != 0) {
            return 1;
        }
    }
    *end = cursor;
    return 0;
}

int aotx_model_layout_replace(const char *dir, const aotx_manifest_entry *entry,
                              unsigned int entries, const aotx_model_load_state *state,
                              unsigned int source, unsigned long long *cursor,
                              unsigned int *placed, unsigned int *left,
                              unsigned long long *bytes)
{
    unsigned int sources[AOTX_MODEL_FILES_MAX];
    unsigned int count = aotx_layout_sources(state, source, sources);
    unsigned long long end = 0ull;
    if (count == 0u || sources[count - 1u] != source
        || aotx_layout_preflight(dir, entry, entries, sources, count, &end) != 0) {
        return 1;
    }

    /* The complete layout is known to fit before any resident allocation is released. */
    if (aotx_mem_weights_trim(0ull) != 0) {
        return 1;
    }
    void *table = NULL;
    aotx_check_runtime(cudaGetSymbolAddress(&table, aotx_mem_tensor_list),
                       "cudaGetSymbolAddress");
    aotx_check_runtime(cudaMemset(table, 0, sizeof(aotx_mem_tensor_table)), "cudaMemset");
    unsigned long long at = 0ull;
    int bad = aotx_model_weights_open();
    for (unsigned int i = 0u; i < count && bad == 0; ++i) {
        aotx_modelfile *file = NULL;
        bad = aotx_layout_open(dir, &entry[sources[i]], &file);
        unsigned int took = 0u;
        unsigned int omitted = 0u;
        unsigned long long before = at;
        if (bad == 0) {
            bad = aotx_model_weights_place(file, sources[i], &at, &took, &omitted);
            aotx_modelfile_close(file);
        }
        if (i + 1u == count) {
            *placed = took;
            *left = omitted;
            *bytes = at - before;
        }
    }
    aotx_model_weights_close();
    if (bad == 0 && at != end) {
        bad = 1;
    }
    if (bad == 0) {
        *cursor = at;
    }
    return bad;
}
