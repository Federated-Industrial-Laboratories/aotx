/* Purpose: Add one node for each device tool module and launch one module on its own.
 * Owns: Nothing; the loader holds the modules and the device holds the batch.
 * Launch shape: Host glue only; the module supplies the kernel of each node.
 * Lifetime: From the first capture to the close at the end of the run. */
#include <cuda.h>
#include <cuda_runtime.h>
#include <stddef.h>

#include "boot/check.h"
#include "tool/module.cuh"
#include "tool/module_host.h"

/* The two parameters of a module node: the address of its batch and the address of the
 * output. Both stand in the module state, which never moves. */
static int aotx_tool_module_where(unsigned int at, CUdeviceptr *batch, CUdeviceptr *out)
{
    void *base = 0;
    if (cudaGetSymbolAddress(&base, aotx_tool_modules) != cudaSuccess) {
        return 1;
    }
    *out = (CUdeviceptr)((unsigned char *)base + offsetof(aotx_tool_module_state, out));
    *batch = (CUdeviceptr)((unsigned char *)base
                           + offsetof(aotx_tool_module_state, batch)
                           + (size_t)at * sizeof(aotx_tool_batch));
    return 0;
}

unsigned int aotx_tool_module_capture(void *stream)
{
    cudaStream_t on = (cudaStream_t)stream;
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    cudaGraph_t graph = 0;
    const cudaGraphNode_t *depends = 0;
    size_t held = 0;
    unsigned int made = 0u;

    for (unsigned int m = 0u; m < aotx_tool_module_held_count(); ++m) {
        CUdeviceptr batch = 0;
        CUdeviceptr out = 0;
        CUfunction function = aotx_tool_module_function(aotx_tool_module_entry_of(m));
        if (function == 0 || aotx_tool_module_where(m, &batch, &out) != 0) {
            continue;
        }
        aotx_check_runtime(cudaStreamGetCaptureInfo(on, &status, 0, &graph, &depends, 0,
                                                    &held),
                           "cudaStreamGetCaptureInfo");
        if (status != cudaStreamCaptureStatusActive) {
            return made;
        }
        void *params[] = { &batch, &out };
        CUDA_KERNEL_NODE_PARAMS node_params = {};
        node_params.func = function;
        node_params.gridDimX = AOTX_SLOTS;
        node_params.gridDimY = 1u;
        node_params.gridDimZ = 1u;
        node_params.blockDimX = AOTX_TOOL_MODULE_THREADS;
        node_params.blockDimY = 1u;
        node_params.blockDimZ = 1u;
        node_params.kernelParams = params;
        CUgraphNode node;
        if (cuGraphAddKernelNode(&node, (CUgraph)graph, (const CUgraphNode *)depends, held,
                                 &node_params) != CUDA_SUCCESS) {
            return made;
        }
        cudaGraphNode_t added = (cudaGraphNode_t)node;
        aotx_check_runtime(cudaStreamUpdateCaptureDependencies(on, &added, 0, 1u,
                                                    cudaStreamSetCaptureDependencies),
                           "cudaStreamUpdateCaptureDependencies");
        made += 1u;
    }
    return made;
}

int aotx_tool_module_place(unsigned int entry)
{
    for (unsigned int m = 0u; m < aotx_tool_module_held_count(); ++m) {
        if (aotx_tool_module_entry_of(m) == entry) {
            return (int)m;
        }
    }
    return -1;
}

int aotx_tool_module_figures(unsigned int entry, int *regs, int *local, int *threads,
                             int *ptx, int *arch)
{
    CUfunction function = aotx_tool_module_function(entry);
    if (function == 0) {
        return 1;
    }
    int rc = cuFuncGetAttribute(regs, CU_FUNC_ATTRIBUTE_NUM_REGS, function);
    rc |= cuFuncGetAttribute(local, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, function);
    rc |= cuFuncGetAttribute(threads, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, function);
    rc |= cuFuncGetAttribute(ptx, CU_FUNC_ATTRIBUTE_PTX_VERSION, function);
    rc |= cuFuncGetAttribute(arch, CU_FUNC_ATTRIBUTE_BINARY_VERSION, function);
    return (rc == CUDA_SUCCESS) ? 0 : 1;
}

int aotx_tool_module_launch(unsigned int entry)
{
    CUdeviceptr batch = 0;
    CUdeviceptr out = 0;
    int at = aotx_tool_module_place(entry);
    CUfunction function = aotx_tool_module_function(entry);
    if (at < 0 || function == 0
        || aotx_tool_module_where((unsigned int)at, &batch, &out) != 0) {
        return 1;
    }
    /* The launch takes the shape the node of the tick graph takes, so the check runs the
     * module the way a tick runs it. */
    void *params[] = { &batch, &out };
    return (cuLaunchKernel(function, AOTX_SLOTS, 1u, 1u, AOTX_TOOL_MODULE_THREADS, 1u, 1u,
                           0u, 0, params, 0) == CUDA_SUCCESS) ? 0 : 1;
}
