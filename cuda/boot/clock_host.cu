/* Purpose: Load the clock module and launch it as a graph node through the driver.
 * Owns: The module handle and the graph of the check.
 * Launch shape: Host glue only; the module supplies the kernel.
 * Lifetime: One call at start. */
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#include "boot/boot.cuh"
#include "boot/check.h"

/* The module text is read whole; the driver compiles it at load. */
static char *aotx_clock_read(const char *path)
{
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        fprintf(stderr, "cannot open %s\n", path);
        return NULL;
    }
    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    fseek(file, 0, SEEK_SET);
    char *text = (char *)malloc((size_t)size + 1);
    if (text == NULL || fread(text, 1, (size_t)size, file) != (size_t)size) {
        fprintf(stderr, "cannot read %s\n", path);
        fclose(file);
        return NULL;
    }
    text[size] = '\0';
    fclose(file);
    return text;
}

int aotx_boot_clock_check(unsigned long long *sample)
{
    CUmodule module;
    CUfunction function;
    CUdeviceptr out;
    CUgraph graph;
    CUgraphExec exec;
    CUgraphNode node;

    char *text = aotx_clock_read(AOTX_PTX_DIR "/clock.ptx");
    if (text == NULL) {
        return 1;
    }
    aotx_check_driver(cuModuleLoadData(&module, text), "cuModuleLoadData");
    free(text);
    aotx_check_driver(cuModuleGetFunction(&function, module, "aotx_clock_sample"),
                      "cuModuleGetFunction");
    aotx_check_driver(cuMemAlloc(&out, sizeof *sample), "cuMemAlloc");

    /* The node takes its parameter through the driver, as every module node does. */
    void *params[] = { &out };
    CUDA_KERNEL_NODE_PARAMS node_params = {};
    node_params.func = function;
    node_params.gridDimX = 1;
    node_params.gridDimY = 1;
    node_params.gridDimZ = 1;
    node_params.blockDimX = 1;
    node_params.blockDimY = 1;
    node_params.blockDimZ = 1;
    node_params.kernelParams = params;

    aotx_check_driver(cuGraphCreate(&graph, 0), "cuGraphCreate");
    aotx_check_driver(cuGraphAddKernelNode(&node, graph, NULL, 0, &node_params),
                      "cuGraphAddKernelNode");
    aotx_check_driver(cuGraphInstantiate(&exec, graph, 0), "cuGraphInstantiate");
    aotx_check_driver(cuGraphLaunch(exec, 0), "cuGraphLaunch");
    aotx_check_driver(cuCtxSynchronize(), "cuCtxSynchronize");
    aotx_check_driver(cuMemcpyDtoH(sample, out, sizeof *sample), "cuMemcpyDtoH");

    aotx_check_driver(cuGraphExecDestroy(exec), "cuGraphExecDestroy");
    aotx_check_driver(cuGraphDestroy(graph), "cuGraphDestroy");
    aotx_check_driver(cuMemFree(out), "cuMemFree");
    aotx_check_driver(cuModuleUnload(module), "cuModuleUnload");
    return (*sample == 0ull) ? 1 : 0;
}
