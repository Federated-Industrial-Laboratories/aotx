/* Purpose: Stop the program when a driver call or a runtime call fails, and name the call.
 * Owns: Nothing.
 * Launch shape: Not applicable; host glue support.
 * Lifetime: Each call site. */
#ifndef AOTX_BOOT_CHECK_H
#define AOTX_BOOT_CHECK_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

/* A driver call that fails stops the program and names the call. */
static inline void aotx_check_driver(CUresult result, const char *call)
{
    if (result == CUDA_SUCCESS) {
        return;
    }
    const char *name = "unknown";
    cuGetErrorName(result, &name);
    fprintf(stderr, "%s: %s\n", call, name);
    exit(1);
}

/* A runtime call that fails stops the program and names the call. */
static inline void aotx_check_runtime(cudaError_t result, const char *call)
{
    if (result == cudaSuccess) {
        return;
    }
    fprintf(stderr, "%s: %s\n", call, cudaGetErrorString(result));
    exit(1);
}

#endif
