/* Purpose: Fill the descriptor of each model role from the file metadata and the tensors.
 * Owns: Nothing that lasts; the descriptor lives in device memory.
 * Launch shape: Host glue only; the bind kernel finds the tensors.
 * Lifetime: One model load.  */
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "boot/check.h"
#include "mem/mem.cuh"
#include "model/forward.cuh"
#include "model/roles.h"

extern "C" {
#include "disk/modelfile/manifest.h"
#include "disk/modelfile/modelfile.h"
}

#define AOTX_DESC_MAX_FILES     8
#define AOTX_DESC_MAX_BINDINGS (AOTX_DESC_WHOLE \
                                + AOTX_MODEL_MAX_LAYERS * AOTX_LAYER_TENSOR_SLOTS)

/* Let the device find every tensor in the host-built binding plan. */
static int aotx_desc_bind(const aotx_model_desc *desc,
                          const aotx_model_binding *binding, unsigned int count,
                          unsigned int role, unsigned int model)
{
    aotx_model_binding *device = NULL;
    unsigned int *missing = NULL;
    unsigned int report[2] = { 0u, ~0u };
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_model, desc, sizeof *desc,
                                          (size_t)role * sizeof *desc),
                       "cudaMemcpyToSymbol");
    aotx_check_runtime(cudaMalloc((void **)&device, count * sizeof *device), "cudaMalloc");
    aotx_check_runtime(cudaMalloc((void **)&missing, sizeof report), "cudaMalloc");
    aotx_check_runtime(cudaMemcpy(device, binding, count * sizeof *device,
                                  cudaMemcpyHostToDevice), "cudaMemcpy");
    aotx_check_runtime(cudaMemcpy(missing, report, sizeof report, cudaMemcpyHostToDevice),
                       "cudaMemcpy");
    unsigned int blocks = (count + 127u) / 128u;
    aotx_model_bind<<<blocks, 128>>>(role, model, device, count, missing);
    aotx_check_runtime(cudaMemcpy(report, missing, sizeof report, cudaMemcpyDeviceToHost),
                       "cudaMemcpy");
    cudaFree(missing);
    cudaFree(device);
    if (report[0] != 0u) {
        const char *name = (report[1] < count) ? binding[report[1]].name : "a binding";
        fprintf(stderr, "the model of role %u has no tensor %s, and %u more are absent\n",
                role, name, report[0] - 1u);
        return 1;
    }
    return 0;
}
/* Open one entry and bind its tensor rows to the descriptor role the caller names. */
static int aotx_desc_one(const char *dir, const aotx_manifest_entry *entry,
                         unsigned int model, unsigned int role)
{
    char path[AOTX_MANIFEST_PATH];
    aotx_modelfile *file = NULL;
    if (aotx_manifest_path(path, sizeof path, dir, entry->path) != 0
        || aotx_modelfile_open(path, &file) != 0) {
        fprintf(stderr, "the file %s did not open\n", entry->path);
        return 1;
    }
    aotx_model_binding *binding = (aotx_model_binding *)calloc(
        AOTX_DESC_MAX_BINDINGS, sizeof *binding);
    if (binding == NULL) {
        fprintf(stderr, "the tensor binding list did not open\n");
        aotx_modelfile_close(file);
        return 1;
    }
    aotx_model_desc desc;
    unsigned int count = 0u;
    char reason[192];
    int bad = aotx_model_desc_file(file, role, &desc, binding,
                                   AOTX_DESC_MAX_BINDINGS, &count,
                                   reason, sizeof reason);
    if (bad != 0) {
        fprintf(stderr, "%s\n", reason);
    } else {
        bad = aotx_desc_bind(&desc, binding, count, role, model);
    }
    free(binding);
    aotx_modelfile_close(file);
    return bad;
}

int aotx_model_describe_one(const char *dir, const char *name, unsigned int target)
{
    if (target >= AOTX_MODEL_ROLES) {
        fprintf(stderr, "the target model role %u is outside the table\n", target);
        return 1;
    }
    aotx_manifest_entry entries[AOTX_DESC_MAX_FILES];
    int count = aotx_manifest_read(dir, entries, AOTX_DESC_MAX_FILES);
    if (count <= 0) {
        fprintf(stderr, "the model file list in %s did not read\n", dir);
        return 1;
    }
    for (int i = 0; i < count; ++i) {
        if (strcmp(entries[i].name, name) == 0) {
            return aotx_desc_one(dir, &entries[i], (unsigned int)i, target);
        }
    }
    fprintf(stderr, "the model file list has no entry for %s\n", name);
    return 1;
}

int aotx_model_describe(const char *dir, const char *roles)
{
    char unknown[64];
    if (aotx_role_unknown(roles, unknown, sizeof unknown) != 0) {
        fprintf(stderr, "the role %s is not a role of this system\n", unknown);
        return 1;
    }
    aotx_manifest_entry entries[AOTX_DESC_MAX_FILES];
    int count = aotx_manifest_read(dir, entries, AOTX_DESC_MAX_FILES);
    if (count <= 0) {
        fprintf(stderr, "the model file list in %s did not read\n", dir);
        return 1;
    }
    unsigned int want[AOTX_MODEL_ROLES];
    unsigned int wanted = aotx_role_list(roles, want);
    unsigned int found = 0u;
    unsigned int done = 0u;
    for (int i = 0; i < count; ++i) {
        unsigned int role = aotx_role_of(entries[i].role);
        if (role >= AOTX_MODEL_ROLES || aotx_role_wanted(roles, entries[i].role) == 0) {
            continue;
        }
        if (aotx_desc_one(dir, &entries[i], (unsigned int)i, role) != 0) {
            return 1;
        }
        found |= 1u << role;
        done += 1u;
    }
    if (done != wanted) {
        for (unsigned int k = 0u; k < wanted; ++k) {
            if ((found & (1u << want[k])) == 0u) {
                fprintf(stderr, "the model file list has no entry for the role %s\n",
                        aotx_role_name[want[k]]);
            }
        }
        return 1;
    }
    return 0;
}
