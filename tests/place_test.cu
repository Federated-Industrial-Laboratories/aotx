/* Purpose: Check that every tensor of each model file in a store is placed in the region.
 * Owns: The memory map and the tensor table of one run.
 * Launch shape: Host glue calls the placement path; the table build holds the kernels.
 * Lifetime: One run of the check program.
 *
 * The check takes a store directory. It reads the manifest, opens each file, counts the
 * tensors of each block type, and places the file through the path the boot takes. A file
 * passes when the placed count is the tensor count and no tensor is left in the file. The
 * check does not build a descriptor or a vocabulary, so a store of another architecture or
 * tokenizer family still proves its placement. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <cuda.h>

#include "boot/boot.cuh"
#include "boot/check.h"
#include "mem/mem.cuh"

extern "C" {
#include "disk/modelfile/manifest.h"
#include "disk/modelfile/modelfile.h"
}

#define AOTX_PLACE_FILES  8
#define AOTX_PLACE_TYPES  64u

static const char *aotx_place_type_name(unsigned int type)
{
    switch (type) {
    case AOTX_TENSOR_F32: return "F32";
    case AOTX_TENSOR_F16: return "F16";
    case AOTX_TENSOR_Q4_0: return "Q4_0";
    case AOTX_TENSOR_Q8_0: return "Q8_0";
    case AOTX_TENSOR_Q4_K: return "Q4_K";
    case AOTX_TENSOR_Q5_K: return "Q5_K";
    case AOTX_TENSOR_Q6_K: return "Q6_K";
    default: return "other";
    }
}

/* The tally of one file by block type, and the byte count that every tensor gives. A
 * tensor whose byte count is zero has a type the reader cannot size. */
static int aotx_place_tally(aotx_modelfile *file, const char *name, unsigned int *total)
{
    unsigned int count[AOTX_PLACE_TYPES] = { 0u };
    unsigned int other = 0u;
    unsigned int unsized = 0u;
    unsigned long long tensors = aotx_modelfile_tensor_count(file);
    for (unsigned long long i = 0ull; i < tensors; ++i) {
        aotx_tensor_info info;
        if (aotx_modelfile_tensor(file, i, &info) != 0) {
            return 1;
        }
        if (info.type < AOTX_PLACE_TYPES) {
            count[info.type] += 1u;
        } else {
            other += 1u;
        }
        if (info.bytes == 0ull) {
            printf("place: %s tensor %s of type %u has no byte count\n", name, info.name,
                   info.type);
            unsized += 1u;
        }
    }
    printf("place: %s holds %llu tensors:", name, tensors);
    for (unsigned int t = 0u; t < AOTX_PLACE_TYPES; ++t) {
        if (count[t] != 0u) {
            printf(" %s %u", aotx_place_type_name(t), count[t]);
        }
    }
    if (other != 0u) {
        printf(" other %u", other);
    }
    printf("\n");
    *total = (unsigned int)tensors;
    return (unsized != 0u) ? 1 : 0;
}

int main(int argc, char **argv)
{
    const char *models = (argc > 1) ? argv[1] : "../models";
    aotx_manifest_entry entries[AOTX_PLACE_FILES];
    int count = aotx_manifest_read(models, entries, AOTX_PLACE_FILES);
    if (count <= 0) {
        printf("place: the manifest of %s did not read\n", models);
        return 1;
    }
    CUdevice device;
    CUcontext context;
    aotx_mem_map map;
    aotx_check_driver(cuInit(0), "cuInit");
    aotx_check_driver(cuDeviceGet(&device, 0), "cuDeviceGet");
    aotx_check_driver(cuDevicePrimaryCtxRetain(&context, device), "cuDevicePrimaryCtxRetain");
    aotx_check_driver(cuCtxSetCurrent(context), "cuCtxSetCurrent");
    if (aotx_mem_reserve(&map) != 0) {
        printf("place: the memory map did not open\n");
        return 1;
    }
    if (aotx_model_weights_open() != 0) {
        printf("place: the weights path did not open\n");
        return 1;
    }
    unsigned int checks = 0u;
    unsigned int failed = 0u;
    unsigned long long cursor = 0ull;
    for (int i = 0; i < count; ++i) {
        char path[AOTX_MANIFEST_PATH];
        aotx_modelfile *file = NULL;
        if (aotx_manifest_check(models, &entries[i]) != 0) {
            printf("place: FAILED %s does not match its manifest line\n", entries[i].path);
            checks += 1u;
            failed += 1u;
            continue;
        }
        if (aotx_manifest_path(path, sizeof path, models, entries[i].path) != 0
            || aotx_modelfile_open(path, &file) != 0) {
            printf("place: FAILED %s did not open\n", entries[i].path);
            checks += 1u;
            failed += 1u;
            continue;
        }
        unsigned int total = 0u;
        unsigned int placed = 0u;
        unsigned int left = 0u;
        int bad = aotx_place_tally(file, entries[i].path, &total);
        if (bad == 0) {
            bad = aotx_model_weights_place(file, (unsigned int)i, &cursor, &placed, &left);
        }
        aotx_modelfile_close(file);
        printf("place: %s placed %u of %u tensors, %u left, %llu MB in the region\n",
               entries[i].path, placed, total, left, cursor >> 20);
        checks += 1u;
        if (bad != 0 || placed != total || left != 0u) {
            printf("place: FAILED %s did not place every tensor\n", entries[i].path);
            failed += 1u;
        }
    }
    aotx_model_weights_close();
    aotx_mem_release(&map);
    cuDevicePrimaryCtxRelease(device);
    printf("place: %u files, %u failed\n", checks, failed);
    return (failed == 0u) ? 0 : 1;
}
