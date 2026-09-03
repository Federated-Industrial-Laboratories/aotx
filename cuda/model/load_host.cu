/* Purpose: Check and place one model file queued by the device.
 * Owns: The model directory, its host file list and the placement cursor.
 * Launch shape: Host glue only; the descriptor and table kernels do device work.
 * Lifetime: From the boot model load to the end of the run. */
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "boot/boot.cuh"
#include "boot/check.h"
#include "mem/mem.cuh"
#include "model/graph_host.h"
#include "model/kinds.h"
#include "model/layout_host.h"
#include "model/load.cuh"
#include "model/roles.h"
#include "sched/sched.cuh"

extern "C" {
#include "disk/modelfile/manifest.h"
#include "disk/modelfile/modelfile.h"
}

static char aotx_load_dir[AOTX_MANIFEST_PATH];
static aotx_manifest_entry aotx_load_entry[AOTX_MODEL_FILES_MAX];
static unsigned int aotx_load_entries;
static unsigned long long aotx_load_cursor;

static int aotx_load_hex(const char *text, unsigned char digest[32])
{
    for (unsigned int i = 0u; i < 32u; ++i) {
        unsigned int high = (text[2u * i] <= '9') ? (unsigned int)(text[2u * i] - '0')
                                                  : (unsigned int)(text[2u * i] - 'a') + 10u;
        unsigned int low = (text[2u * i + 1u] <= '9')
                         ? (unsigned int)(text[2u * i + 1u] - '0')
                         : (unsigned int)(text[2u * i + 1u] - 'a') + 10u;
        if (high > 15u || low > 15u) {
            return 1;
        }
        digest[i] = (unsigned char)((high << 4) | low);
    }
    return 0;
}

static void aotx_load_text(char *out, unsigned int max, const char *in)
{
    unsigned int i = 0u;
    while (i + 1u < max && in[i] != '\0') {
        out[i] = in[i];
        i += 1u;
    }
    while (i < max) {
        out[i++] = '\0';
    }
}

int aotx_model_load_open(const char *dir, const char *roles, unsigned long long cursor)
{
    int count = aotx_manifest_read(dir, aotx_load_entry, AOTX_MODEL_FILES_MAX);
    if (count <= 0 || (unsigned int)count > AOTX_MODEL_FILES_MAX
        || strlen(dir) >= sizeof aotx_load_dir) {
        fprintf(stderr, "the model file list cannot open for run-time loads\n");
        return 1;
    }
    strcpy(aotx_load_dir, dir);
    aotx_load_entries = (unsigned int)count;
    aotx_load_cursor = cursor;

    aotx_model_load_state state;
    aotx_mem_tensor_table *table =
        (aotx_mem_tensor_table *)malloc(sizeof(aotx_mem_tensor_table));
    if (table == NULL) {
        return 1;
    }
    aotx_check_runtime(cudaMemcpyFromSymbol(table, aotx_mem_tensor_list, sizeof *table),
                       "cudaMemcpyFromSymbol");
    unsigned int present[AOTX_MODEL_FILES_MAX] = { 0u };
    unsigned int tensors = (table->count < AOTX_MEM_TENSOR_MAX)
                         ? table->count : AOTX_MEM_TENSOR_MAX;
    for (unsigned int i = 0u; i < tensors; ++i) {
        if (table->tensor[i].model < AOTX_MODEL_FILES_MAX) {
            present[table->tensor[i].model] = 1u;
        }
    }
    free(table);
    memset(&state, 0, sizeof state);
    state.files = (unsigned int)count;
    for (int i = 0; i < count; ++i) {
        aotx_model_file_row *row = &state.file[i];
        unsigned int role = aotx_role_of(aotx_load_entry[i].role);
        if (role >= AOTX_MODEL_ROLES || strlen(aotx_load_entry[i].name) >= sizeof row->name
            || strlen(aotx_load_entry[i].path) >= sizeof row->file
            || aotx_load_hex(aotx_load_entry[i].sha256, row->digest) != 0) {
            fprintf(stderr, "the model entry %s does not fit the run-time table\n",
                    aotx_load_entry[i].name);
            return 1;
        }
        row->bytes = aotx_load_entry[i].bytes;
        row->model = (unsigned int)i;
        row->role = role;
        aotx_load_text(row->name, (unsigned int)sizeof row->name, aotx_load_entry[i].name);
        aotx_load_text(row->file, (unsigned int)sizeof row->file, aotx_load_entry[i].path);
        if (aotx_role_wanted(roles, aotx_load_entry[i].role) != 0
            && present[i] != 0u) {
            aotx_model_resident_row *resident = &state.resident[role];
            resident->source = (unsigned int)i;
            resident->slot = role;
            resident->active = 1u;
            resident->body.tick = 0ull;
            memcpy(resident->body.digest, row->digest, sizeof row->digest);
            aotx_load_text(resident->body.role, (unsigned int)sizeof resident->body.role,
                           aotx_load_entry[i].role);
            aotx_load_text(resident->body.file, (unsigned int)sizeof resident->body.file,
                           aotx_load_entry[i].path);
        }
    }
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_model_load, &state, sizeof state),
                       "cudaMemcpyToSymbol");
    return 0;
}

static int aotx_load_open_file(const aotx_manifest_entry *entry, aotx_modelfile **file)
{
    char path[AOTX_MANIFEST_PATH];
    if (aotx_manifest_path(path, sizeof path, aotx_load_dir, entry->path) != 0
        || aotx_modelfile_open(path, file) != 0) {
        fprintf(stderr, "the file %s did not open\n", entry->path);
        return 1;
    }
    return 0;
}

static void aotx_load_mark(aotx_pump *pump, unsigned int success, unsigned int reason,
                           unsigned long long bytes)
{
    aotx_model_load_finish<<<1, 1, 0, pump->stream>>>(success, reason, bytes);
    aotx_check_runtime(cudaEventRecord(pump->event, pump->stream), "cudaEventRecord");
    aotx_check_runtime(cudaEventSynchronize(pump->event), "cudaEventSynchronize");
    aotx_pump_flush(pump);
}

int aotx_model_load_step(aotx_pump *pump)
{
    aotx_model_load_state state;
    aotx_check_runtime(cudaMemcpyFromSymbol(&state, aotx_model_load, sizeof state),
                       "cudaMemcpyFromSymbol");
    if (state.replay_bad != 0u) {
        return 1;
    }
    if (state.pending_count == 0u) {
        return 0;
    }
    aotx_model_load_row load = state.pending[0];
    int replayed = load.replayed != 0u;
    if (load.source >= aotx_load_entries) {
        aotx_load_mark(pump, 0u, AOTX_MODEL_LOAD_FILE, 0ull);
        return replayed ? 1 : 0;
    }
    if (!replayed) {
        aotx_model_load_begin<<<1, 1, 0, pump->stream>>>();
        aotx_check_runtime(cudaEventRecord(pump->event, pump->stream), "cudaEventRecord");
        aotx_check_runtime(cudaEventSynchronize(pump->event), "cudaEventSynchronize");
        aotx_pump_flush(pump);
    }

    const aotx_manifest_entry *entry = &aotx_load_entry[load.source];
    int checked = aotx_manifest_check(aotx_load_dir, entry);
    if (checked != 0 || memcmp(load.body.digest, state.file[load.source].digest, 32u) != 0) {
        aotx_load_mark(pump, 0u, (checked == 1) ? AOTX_MODEL_LOAD_DIGEST
                                                : AOTX_MODEL_LOAD_FILE, 0ull);
        return replayed ? 1 : 0;
    }

    aotx_modelfile *file = NULL;
    if (aotx_load_open_file(entry, &file) != 0) {
        aotx_load_mark(pump, 0u, AOTX_MODEL_LOAD_FILE, 0ull);
        return replayed ? 1 : 0;
    }
    unsigned long long before = aotx_load_cursor;
    aotx_mem_tensor_table *old_table =
        (aotx_mem_tensor_table *)malloc(sizeof(aotx_mem_tensor_table));
    aotx_model_desc old_desc[AOTX_MODEL_ROLES];
    if (old_table == NULL) {
        aotx_modelfile_close(file);
        aotx_load_mark(pump, 0u, AOTX_MODEL_LOAD_FILE, 0ull);
        return replayed ? 1 : 0;
    }
    aotx_check_runtime(cudaMemcpyFromSymbol(old_table, aotx_mem_tensor_list,
                                            sizeof *old_table),
                       "cudaMemcpyFromSymbol");
    aotx_check_runtime(cudaMemcpyFromSymbol(old_desc, aotx_model, sizeof old_desc),
                       "cudaMemcpyFromSymbol");
    unsigned int placed = 0u;
    unsigned int left = 0u;
    unsigned long long bytes = 0ull;
    int reload = 0;
    for (unsigned int i = 0u; i < AOTX_MODEL_ROLES; ++i) {
        reload = reload || (state.resident[i].active != 0u
                            && state.resident[i].slot == load.slot
                            && state.resident[i].source == load.source);
    }
    int replace = 0;
    if (AOTX_MODELS_RESIDENT == 1u
        && (load.target == AOTX_MODEL_LANGUAGE
            || load.target == AOTX_MODEL_LANGUAGE_Q4)) {
        for (unsigned int i = 0u; i < AOTX_MODEL_ROLES; ++i) {
            replace |= state.resident[i].active != 0u
                    && state.resident[i].source != load.source
                    && (state.resident[i].slot == AOTX_MODEL_LANGUAGE
                        || state.resident[i].slot == AOTX_MODEL_LANGUAGE_Q4);
        }
    }
    unsigned long long placed_from = aotx_load_cursor;
    int bad = 0;
    if (replace != 0) {
        aotx_modelfile_close(file);
        file = NULL;
        bad = aotx_model_layout_replace(aotx_load_dir, aotx_load_entry,
                                         aotx_load_entries, &state, load.source,
                                         &aotx_load_cursor, &placed, &left, &bytes);
    } else {
        bad = aotx_model_weights_open();
    }
    if (bad == 0 && reload && replace == 0) {
        bad = aotx_model_weights_reload(file, load.source, &placed, &left, &bytes);
    } else if (bad == 0 && replace == 0) {
        bad = aotx_model_weights_place(file, load.source, &aotx_load_cursor,
                                       &placed, &left);
    }
    if (replace == 0) {
        aotx_model_weights_close();
        aotx_modelfile_close(file);
    }
    if (bad != 0) {
        aotx_load_cursor = before;
        aotx_check_runtime(cudaMemcpyToSymbol(aotx_mem_tensor_list, old_table,
                                              sizeof *old_table),
                           "cudaMemcpyToSymbol");
        free(old_table);
        aotx_load_mark(pump, 0u, AOTX_MODEL_LOAD_REGION, 0ull);
        return replayed ? 1 : 0;
    }
    if (aotx_model_describe_one(aotx_load_dir, entry->name, load.slot) != 0) {
        aotx_load_cursor = before;
        aotx_check_runtime(cudaMemcpyToSymbol(aotx_mem_tensor_list, old_table,
                                              sizeof *old_table),
                           "cudaMemcpyToSymbol");
        aotx_check_runtime(cudaMemcpyToSymbol(aotx_model, old_desc, sizeof old_desc),
                           "cudaMemcpyToSymbol");
        free(old_table);
        aotx_load_mark(pump, 0u, AOTX_MODEL_LOAD_DESC, 0ull);
        return replayed ? 1 : 0;
    }
    if (AOTX_MODELS_RESIDENT == 1u
        && (load.target == AOTX_MODEL_LANGUAGE
            || load.target == AOTX_MODEL_LANGUAGE_Q4)) {
        aotx_model_desc clear;
        memset(&clear, 0, sizeof clear);
        unsigned int other = (load.slot == AOTX_MODEL_LANGUAGE)
                           ? AOTX_MODEL_LANGUAGE_Q4 : AOTX_MODEL_LANGUAGE;
        aotx_check_runtime(cudaMemcpyToSymbol(aotx_model, &clear, sizeof clear,
                                              (size_t)other * sizeof clear),
                           "cudaMemcpyToSymbol");
    }
    if (replace != 0 && pump->graph != 0 && pump->exec != 0
        && (aotx_decode_replace(load.slot) != 0 || aotx_pump_recapture(pump) != 0)) {
        free(old_table);
        aotx_load_mark(pump, 0u, AOTX_MODEL_LOAD_DESC, 0ull);
        return 1;
    }
    if (!reload && replace == 0) {
        bytes = aotx_load_cursor - placed_from;
    }
    free(old_table);
    printf("model load: %s %u tensors %u left %llu MB placed\n", entry->path,
           placed, left, bytes >> 20);
    aotx_model_desc loaded;
    aotx_check_runtime(cudaMemcpyFromSymbol(&loaded, aotx_model, sizeof loaded,
                                            (size_t)load.slot * sizeof loaded),
                       "cudaMemcpyFromSymbol");
    aotx_layer_print_one(&loaded);
    aotx_mem_budget_read();
    aotx_load_mark(pump, 1u, AOTX_MODEL_LOAD_NONE, bytes);
    return 0;
}
