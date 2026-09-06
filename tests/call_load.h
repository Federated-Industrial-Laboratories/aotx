/* Purpose: Check real model call row uploads, selection, and metadata release.
 * Owns: Test counters and temporary metadata snapshots.
 * Launch shape: One thread for each selected row at N=1 and N=64.
 * Lifetime: One model load test. */
#ifndef AOTX_TEST_CALL_LOAD_H
#define AOTX_TEST_CALL_LOAD_H

#include <stddef.h>
#include "model/call_format.cuh"
#include "model/layout_host.h"

static unsigned int aotx_call_load_checks, aotx_call_load_failed;

static void aotx_call_load_check(int good, const char *text)
{
    ++aotx_call_load_checks;
    if (!good) ++aotx_call_load_failed;
    printf("call load: %s %s\n", good ? "pass" : "FAIL", text);
}

static void aotx_call_load_read(aotx_call_format *rows)
{
    aotx_check_runtime(cudaMemcpyFromSymbol(rows, aotx_model_call_format,
                        sizeof(aotx_call_format) * AOTX_MODEL_ROLES), "cudaMemcpyFromSymbol");
}

static void aotx_call_load_unchanged(const aotx_call_format *before)
{
    aotx_call_format after[AOTX_MODEL_ROLES];
    aotx_call_load_read(after);
    aotx_call_load_check(memcmp(before, after, sizeof after) == 0,
                         "unknown-wrap preflight keeps every call row");
}

static void aotx_call_load_uploaded(const char *models)
{
    aotx_manifest_entry entries[AOTX_MODEL_FILES_MAX];
    aotx_model_load_state state;
    aotx_call_format actual[AOTX_MODEL_ROLES];
    aotx_check_runtime(cudaMemcpyFromSymbol(&state, aotx_model_load, sizeof state),
                       "cudaMemcpyFromSymbol");
    aotx_call_load_read(actual);
    int count = aotx_manifest_read(models, entries, AOTX_MODEL_FILES_MAX);
    aotx_call_load_check(count > 0, "loaded manifest reads");
    if (count <= 0) return;
    unsigned int loaded = 0u;
    for (unsigned int role = 0u; role < AOTX_MODEL_ROLES; ++role) {
        const aotx_model_resident_row *resident = &state.resident[role];
        if (!resident->active) continue;
        ++loaded;
        if (resident->source >= (unsigned int)count || resident->slot >= AOTX_MODEL_ROLES) {
            aotx_call_load_check(0, "resident row names a file and a slot");
            continue;
        }
        char path[AOTX_MANIFEST_PATH];
        aotx_modelfile *file = NULL;
        aotx_call_format expected;
        int bad = aotx_manifest_path(path, sizeof path, models, entries[resident->source].path);
        if (!bad) bad = aotx_modelfile_open(path, &file);
        if (!bad) bad = aotx_call_format_read(file, &expected);
        aotx_call_load_check(!bad && aotx_call_format_valid(&expected) &&
            memcmp(&actual[resident->slot], &expected, sizeof expected) == 0,
            "real boot row matches the complete selected disk row");
        if (file != NULL) aotx_modelfile_close(file);
    }
    aotx_call_load_check(loaded != 0u, "boot selected at least one resident model");
}

__global__ void aotx_call_load_active(aotx_call_format *rows, unsigned int count)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) rows[i] = *aotx_call_format_active();
}

static void aotx_call_load_lifecycle(unsigned int loaded_role)
{
    aotx_model_desc saved_desc[AOTX_MODEL_ROLES], observed_desc[AOTX_MODEL_ROLES];
    aotx_wrap saved_wrap[AOTX_MODEL_ROLES], observed_wrap[AOTX_MODEL_ROLES];
    aotx_call_format saved_format[AOTX_MODEL_ROLES], observed_format[AOTX_MODEL_ROLES];
    aotx_check_runtime(cudaMemcpyFromSymbol(saved_desc, aotx_model, sizeof saved_desc),
                       "cudaMemcpyFromSymbol");
    aotx_check_runtime(cudaMemcpyFromSymbol(saved_wrap, aotx_model_wrap, sizeof saved_wrap),
                       "cudaMemcpyFromSymbol");
    aotx_call_load_read(saved_format);
    aotx_call_format primary = saved_format[loaded_role], alternate;
    int bad = aotx_call_format_make(primary.kind == AOTX_CALL_HERMES
                                    ? AOTX_CALL_LLAMA_JSON : AOTX_CALL_HERMES, &alternate);
    aotx_call_load_check(!bad && primary.kind != alternate.kind,
                         "selection fixtures have distinct complete rows");
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_model_call_format, &primary, sizeof primary,
                        AOTX_MODEL_LANGUAGE * sizeof primary), "cudaMemcpyToSymbol");
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_model_call_format, &alternate, sizeof alternate,
                        AOTX_MODEL_LANGUAGE_Q4 * sizeof alternate), "cudaMemcpyToSymbol");
    aotx_call_format *device = NULL;
    aotx_call_format rows[64];
    aotx_check_runtime(cudaMalloc(&device, sizeof rows), "cudaMalloc");
    for (unsigned int present = 0u; present <= 1u; ++present) {
        aotx_check_runtime(cudaMemcpyToSymbol(aotx_model, &present, sizeof present,
            AOTX_MODEL_LANGUAGE * sizeof(aotx_model_desc) + offsetof(aotx_model_desc, layers)),
            "cudaMemcpyToSymbol");
        const aotx_call_format *expected = present ? &primary : &alternate;
        for (unsigned int batch = 1u; batch <= 64u; batch *= 64u) {
            aotx_check_runtime(cudaMemset(device, 0xa5, sizeof rows), "cudaMemset");
            aotx_call_load_active<<<1, 64>>>(device, batch);
            aotx_check_runtime(cudaMemcpy(rows, device, sizeof rows, cudaMemcpyDeviceToHost),
                               "cudaMemcpy");
            int same = 1;
            for (unsigned int i = 0u; i < batch; ++i)
                same &= memcmp(&rows[i], expected, sizeof *expected) == 0;
            aotx_call_load_check(same, present ? "active row uses the primary language slot"
                                              : "active row uses the alternate language slot");
            printf("call load: active batch %u primary=%u\n", batch, present);
        }
    }
    cudaFree(device);
    const unsigned int language[] = { AOTX_MODEL_LANGUAGE, AOTX_MODEL_LANGUAGE_Q4 };
    aotx_model_desc zero_desc = {};
    aotx_wrap zero_wrap = {};
    aotx_call_format zero_format = {};
    for (unsigned int i = 0u; i < 2u; ++i) {
        unsigned int role = language[i];
        aotx_model_metadata_clear(role);
        aotx_check_runtime(cudaMemcpyFromSymbol(observed_desc, aotx_model, sizeof observed_desc),
                           "cudaMemcpyFromSymbol");
        aotx_check_runtime(cudaMemcpyFromSymbol(observed_wrap, aotx_model_wrap, sizeof observed_wrap),
                           "cudaMemcpyFromSymbol");
        aotx_call_load_read(observed_format);
        aotx_call_load_check(memcmp(&observed_desc[role], &zero_desc, sizeof zero_desc) == 0 &&
            memcmp(&observed_wrap[role], &zero_wrap, sizeof zero_wrap) == 0 &&
            memcmp(&observed_format[role], &zero_format, sizeof zero_format) == 0,
            "release clears descriptor, wrap, and call row together");
    }
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_model, saved_desc, sizeof saved_desc),
                       "cudaMemcpyToSymbol");
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_model_wrap, saved_wrap, sizeof saved_wrap),
                       "cudaMemcpyToSymbol");
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_model_call_format, saved_format, sizeof saved_format),
                       "cudaMemcpyToSymbol");
    aotx_check_runtime(cudaMemcpyFromSymbol(observed_desc, aotx_model, sizeof observed_desc),
                       "cudaMemcpyFromSymbol");
    aotx_check_runtime(cudaMemcpyFromSymbol(observed_wrap, aotx_model_wrap, sizeof observed_wrap),
                       "cudaMemcpyFromSymbol");
    aotx_call_load_read(observed_format);
    aotx_call_load_check(memcmp(saved_desc, observed_desc, sizeof saved_desc) == 0 &&
        memcmp(saved_wrap, observed_wrap, sizeof saved_wrap) == 0 &&
        memcmp(saved_format, observed_format, sizeof saved_format) == 0,
        "restore returns every metadata table byte");
}

#endif
