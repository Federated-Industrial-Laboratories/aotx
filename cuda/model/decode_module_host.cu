/* Purpose: Load the matrix PTX module and bind per-role batch counts.
 * Owns: The module handle, function and fixed device count addresses.
 * Launch shape: Host graph glue; product arithmetic remains on CUDA.
 * Lifetime: From first capture to the final decoder close. */
#include <cuda.h>
#include <stddef.h>
#include "boot/check.h"
#include "aotx_modules_ptx.h"
#include "model/decode_state.cuh"
#include "model/graph_host.h"
#include "model/decode_module_host.h"
static CUmodule aotx_decode_module;
static CUfunction aotx_decode_line;
static CUdeviceptr aotx_decode_batch[AOTX_MODEL_ROLES][2];
/* Keep the address of the two batch counts that a matrix node of a role reads. The module
 * node takes one of them as a pointer, because a module never reads a symbol of the
 * compiled code. Each role has its own call block, so each role has its own pair. The
 * addresses are read before the capture starts, so no query runs inside a capture. */
void aotx_model_batch_of(unsigned int role)
{
    if (role >= AOTX_MODEL_ROLES) {
        return;
    }
    void *call = 0;
    aotx_check_runtime(cudaGetSymbolAddress(&call, aotx_model_call), "cudaGetSymbolAddress");
    char *at = (char *)call + (size_t)role * sizeof(aotx_model_run);
    aotx_decode_batch[role][AOTX_MODEL_BATCH_TOKENS] =
        (CUdeviceptr)(at + offsetof(aotx_model_run, tokens));
    aotx_decode_batch[role][AOTX_MODEL_BATCH_ROWS] =
        (CUdeviceptr)(at + offsetof(aotx_model_run, rows));
}

const unsigned int *aotx_decode_batch_word(unsigned int role, unsigned int which)
{
    if (role >= AOTX_MODEL_ROLES) {
        return 0;
    }
    return (const unsigned int *)
        aotx_decode_batch[role][(which == AOTX_MODEL_BATCH_ROWS) ? 1u : 0u];
}

unsigned int aotx_model_module_node(aotx_model_hold *hold, const void *w, unsigned int type,
                                    unsigned int n, unsigned int k, const half *x, float *y,
                                    dim3 grid, unsigned int which)
{
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    cudaGraph_t graph = 0;
    const cudaGraphNode_t *depends = 0;
    size_t held = 0;
    if (aotx_decode_line == 0 || type != AOTX_WEIGHT_Q8_0) {
        return 0u;
    }
    if (cudaStreamGetCaptureInfo(hold->stream, &status, 0, &graph, &depends, 0, &held)
            != cudaSuccess
        || status != cudaStreamCaptureStatusActive) {
        return 0u;
    }
    CUdeviceptr pw = (CUdeviceptr)w;
    CUdeviceptr px = (CUdeviceptr)x;
    CUdeviceptr py = (CUdeviceptr)y;
    CUdeviceptr pm = aotx_decode_batch[hold->role][(which == AOTX_MODEL_BATCH_ROWS)
                                                   ? 1u : 0u];
    void *params[] = { &pw, &n, &k, &px, &py, &pm };
    CUDA_KERNEL_NODE_PARAMS node_params = {};
    node_params.func = aotx_decode_line;
    node_params.gridDimX = grid.x;
    node_params.gridDimY = grid.y;
    node_params.gridDimZ = 1u;
    node_params.blockDimX = AOTX_GEMV_THREADS;
    node_params.blockDimY = 1u;
    node_params.blockDimZ = 1u;
    node_params.kernelParams = params;

    CUgraphNode node;
    if (cuGraphAddKernelNode(&node, (CUgraph)graph, (const CUgraphNode *)depends, held,
                             &node_params) != CUDA_SUCCESS) {
        return 0u;
    }
    cudaGraphNode_t added = (cudaGraphNode_t)node;
    aotx_check_runtime(cudaStreamUpdateCaptureDependencies(hold->stream, &added, 0, 1u,
                                                           cudaStreamSetCaptureDependencies),
                       "cudaStreamUpdateCaptureDependencies");
    return 1u;
}

void aotx_decode_module_open(void)
{
    if (aotx_decode_module == 0) {
        if (cuModuleLoadData(&aotx_decode_module, aotx_matrix_ptx) != CUDA_SUCCESS
            || cuModuleGetFunction(&aotx_decode_line, aotx_decode_module,
                                   "aotx_gemv_q8") != CUDA_SUCCESS) {
            if (aotx_decode_module) cuModuleUnload(aotx_decode_module);
            aotx_decode_module = 0;
            aotx_decode_line = 0;
        }
    }
}
void aotx_decode_module_close(void)
{
    if(aotx_decode_module)cuModuleUnload(aotx_decode_module);
    aotx_decode_module=0;aotx_decode_line=0;
}
