/* Purpose: Capture the decode of one tick and load the module of the memory bound product.
 * Owns: The child graph of the forward pass, the module handle and the module function.
 * Launch shape: Host glue only; the graphs hold the kernels.
 * Lifetime: From the first capture to the close at the end of the run. */
#include <cuda.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "boot/check.h"
#include "model/decode_state.cuh"
#include "model/graph_host.h"

static cudaGraph_t aotx_decode_pass;
static CUmodule aotx_decode_module;
static CUfunction aotx_decode_line;
static unsigned int aotx_decode_role_now = AOTX_MODEL_ROLES;
static CUdeviceptr aotx_decode_batch[AOTX_MODEL_ROLES][2];

/* The module text is read whole; the driver compiles it at load. */
static char *aotx_decode_read(const char *path)
{
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        return NULL;
    }
    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    fseek(file, 0, SEEK_SET);
    char *text = (char *)malloc((size_t)size + 1);
    if (text == NULL || fread(text, 1, (size_t)size, file) != (size_t)size) {
        free(text);
        fclose(file);
        return NULL;
    }
    text[size] = '\0';
    fclose(file);
    return text;
}

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

/* Capture the forward pass of one role as a graph of its own. The graph holds no copy node,
 * because the plan writes the call block on the device. */
static int aotx_decode_build(unsigned int role)
{
    aotx_model_hold *hold = aotx_model_hold_of(role);
    if (hold == 0 || hold->ready == 0u) {
        return 1;
    }
    if (aotx_decode_module == 0) {
        char *text = aotx_decode_read(AOTX_PTX_DIR "/gemv_q8.ptx");
        if (text != NULL) {
            if (cuModuleLoadData(&aotx_decode_module, text) != CUDA_SUCCESS
                || cuModuleGetFunction(&aotx_decode_line, aotx_decode_module,
                                       "aotx_gemv_q8") != CUDA_SUCCESS) {
                aotx_decode_module = 0;
                aotx_decode_line = 0;
            }
            free(text);
        }
    }
    if (aotx_decode_pass != 0) {
        cudaGraphDestroy(aotx_decode_pass);
        aotx_decode_pass = 0;
    }
    hold->decode = 1u;
    hold->wave = AOTX_DECODE_WAVE;
    aotx_check_runtime(cudaStreamBeginCapture(hold->stream,
                                              cudaStreamCaptureModeThreadLocal),
                       "cudaStreamBeginCapture");
    aotx_model_open_rows<<<1, AOTX_MODEL_MAX_SEQS, 0, hold->stream>>>(role);
    aotx_model_gather<<<AOTX_DECODE_WAVE, AOTX_MODEL_ROW_THREADS, 0, hold->stream>>>(role);
    for (unsigned int l = 0u; l < hold->desc.layers; ++l) {
        aotx_model_capture_layer(hold, role, l);
    }
    aotx_model_capture_head(hold, role);
    aotx_model_shut_rows<<<1, AOTX_MODEL_MAX_SEQS, 0, hold->stream>>>(role);
    aotx_check_runtime(cudaStreamEndCapture(hold->stream, &aotx_decode_pass),
                       "cudaStreamEndCapture");
    hold->decode = 0u;
    hold->wave = 0u;
    return (aotx_decode_pass == 0) ? 1 : 0;
}

/* The language role of the run. The four bit role stands in when the eight bit role is
 * not loaded. A run of one language file therefore gives the decode its model. The pass of
 * a role that has no graph yet is captured here. */
static unsigned int aotx_decode_language(void)
{
    const unsigned int list[2] = { AOTX_MODEL_LANGUAGE, AOTX_MODEL_LANGUAGE_Q4 };
    for (unsigned int i = 0u; i < 2u; ++i) {
        aotx_model_hold *hold = aotx_model_hold_of(list[i]);
        if (hold != 0 && hold->ready != 0u) {
            return list[i];
        }
    }
    for (unsigned int i = 0u; i < 2u; ++i) {
        unsigned int layers = 0u;
        aotx_check_runtime(cudaMemcpyFromSymbol(&layers, aotx_model, sizeof layers,
                                                (size_t)list[i] * sizeof(aotx_model_desc)
                                                + offsetof(aotx_model_desc, layers)),
                           "cudaMemcpyFromSymbol");
        if (layers != 0u && aotx_model_open(list[i], AOTX_MODEL_MAX_TOKENS) == 0) {
            return list[i];
        }
    }
    return AOTX_MODEL_ROLES;
}

int aotx_decode_open(void)
{
    unsigned int ready = 1u;
    unsigned int role = aotx_decode_language();
    if (role >= AOTX_MODEL_ROLES) {
        return 1;
    }
    if (aotx_decode_pass == 0 || aotx_decode_role_now != role) {
        aotx_model_batch_of(role);
        if (aotx_decode_build(role) != 0) {
            return 1;
        }
        aotx_decode_role_now = role;
    }
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_decode, &role, sizeof role,
                                          offsetof(aotx_decode_state, role)),
                       "cudaMemcpyToSymbol");
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_decode, &ready, sizeof ready,
                                          offsetof(aotx_decode_state, ready)),
                       "cudaMemcpyToSymbol");
    return 0;
}

int aotx_decode_capture(void *stream)
{
    cudaStream_t s = (cudaStream_t)stream;
    unsigned int role = aotx_decode_role_now;
    if (aotx_decode_pass == 0 || role >= AOTX_MODEL_ROLES) {
        return 1;
    }

    /* The plan, the pass and the commit, in that order. The pass goes in as one child
     * node, because a graph that is captured cannot launch another graph. */
    aotx_decode_plan<<<1, AOTX_SEQ_SLOTS, 0, s>>>(0ull);
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    cudaGraph_t graph = 0;
    const cudaGraphNode_t *depends = 0;
    size_t held = 0;
    aotx_check_runtime(cudaStreamGetCaptureInfo(s, &status, 0, &graph, &depends, 0, &held),
                       "cudaStreamGetCaptureInfo");
    if (status != cudaStreamCaptureStatusActive) {
        return 1;
    }
    cudaGraphNode_t child = 0;
    aotx_check_runtime(cudaGraphAddChildGraphNode(&child, graph, depends, held,
                                                  aotx_decode_pass),
                       "cudaGraphAddChildGraphNode");
    aotx_check_runtime(cudaStreamUpdateCaptureDependencies(s, &child, 0, 1u,
                                                           cudaStreamSetCaptureDependencies),
                       "cudaStreamUpdateCaptureDependencies");
    aotx_decode_commit<<<1, AOTX_SEQ_SLOTS, 0, s>>>(0ull);
    return 0;
}

unsigned int aotx_decode_nodes(void)
{
    size_t count = 0;
    if (aotx_decode_pass == 0) {
        return 0u;
    }
    aotx_check_runtime(cudaGraphGetNodes(aotx_decode_pass, 0, &count), "cudaGraphGetNodes");
    return (unsigned int)count;
}

void aotx_decode_close(void)
{
    if (aotx_decode_pass != 0) {
        cudaGraphDestroy(aotx_decode_pass);
        aotx_decode_pass = 0;
    }
    if (aotx_decode_module != 0) {
        cuModuleUnload(aotx_decode_module);
        aotx_decode_module = 0;
        aotx_decode_line = 0;
    }
    aotx_decode_role_now = AOTX_MODEL_ROLES;
}
