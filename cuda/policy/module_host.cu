/* Purpose: Admit the exact selected native image before graph capture.
 * Owns: The driver module and checked entry point.
 * Launch shape: Host load and graph calls only; device rows own decisions.
 * Lifetime: One immutable policy revision. */
#include "policy/host.h"
#include "policy/state.cuh"
#include "disk/policy/file.h"
#include "boot/check.h"
#include <cuda.h>
#include <stddef.h>
#include <stdio.h>

static CUmodule aotx_policy_module;
static CUfunction aotx_policy_function;
static aotx_policy_config aotx_policy_loaded;
static aotx_policy_state *aotx_policy_address;

static bool aotx_policy_signature(CUfunction function) {
    size_t count = 0;
    if (cuFuncGetParamCount(function, &count) != CUDA_SUCCESS || count != 6) return false;
    const size_t offsets[] = {0, 8, 16, 24, 32, 36};
    const size_t sizes[] = {8, 8, 8, 8, 4, 4};
    for (size_t i = 0; i < count; ++i) {
        size_t offset = 0, size = 0;
        if (cuFuncGetParamInfo(function, i, &offset, &size) != CUDA_SUCCESS ||
            offset != offsets[i] || size != sizes[i]) return false;
    }
    return true;
}
static bool aotx_policy_resources(CUfunction f, const aotx_policy_config *c) {
    int threads = 0, registers = 0, shared = 0, local = 0, architecture = 0;
    return cuFuncGetAttribute(&threads, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, f) == CUDA_SUCCESS &&
        cuFuncGetAttribute(&registers, CU_FUNC_ATTRIBUTE_NUM_REGS, f) == CUDA_SUCCESS &&
        cuFuncGetAttribute(&shared, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, f) == CUDA_SUCCESS &&
        cuFuncGetAttribute(&local, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, f) == CUDA_SUCCESS &&
        cuFuncGetAttribute(&architecture, CU_FUNC_ATTRIBUTE_BINARY_VERSION, f) == CUDA_SUCCESS &&
        threads >= (int)c->threads && registers <= (int)c->registers &&
        shared <= (int)c->shared_bytes && local <= (int)c->local_bytes &&
        architecture == (int)c->architecture;
}
int aotx_policy_open(const char *path, const char *trust) {
    if (!path) {
        if (trust) { fprintf(stderr, "policy: trust requires a selected policy file\n"); return 1; }
        return 0;
    }
    if (aotx_policy_loaded.mode) return 1;
    aotx_policy_file file = {};
    int rc = aotx_policy_file_read(path, trust, 1, &file);
    if (rc) { fprintf(stderr, "policy refused: %s\n", aotx_policy_status_text(rc)); return 1; }
    if (file.config.mode == AOTX_POLICY_NATIVE) {
        CUdevice device; int major = 0, minor = 0;
        if (cuCtxGetDevice(&device) != CUDA_SUCCESS ||
            cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device) != CUDA_SUCCESS ||
            cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device) != CUDA_SUCCESS ||
            file.config.architecture != (unsigned)(10 * major + minor)) {
            fprintf(stderr, "policy refused: the native target does not match this device\n"); rc = 1;
        }
        if (!rc && cuModuleLoadData(&aotx_policy_module, file.image) != CUDA_SUCCESS) {
            fprintf(stderr, "policy refused: the native image did not load\n"); rc = 1;
        }
        if (!rc && (cuModuleGetFunction(&aotx_policy_function, aotx_policy_module, file.entry) != CUDA_SUCCESS ||
            !aotx_policy_signature(aotx_policy_function) || !aotx_policy_resources(aotx_policy_function, &file.config))) {
            fprintf(stderr, "policy refused: the native entry, ABI or resource declaration does not match\n"); rc = 1;
        }
    }
    if (!rc) {
        aotx_check_runtime(cudaGetSymbolAddress((void **)&aotx_policy_address, aotx_policy), "cudaGetSymbolAddress");
        aotx_check_runtime(cudaMemset(aotx_policy_address, 0, sizeof(aotx_policy_state)), "cudaMemset");
        aotx_check_runtime(cudaMemcpy(&aotx_policy_address->config, &file.config, sizeof(file.config),
            cudaMemcpyHostToDevice), "cudaMemcpy");
        aotx_check_runtime(cudaMemcpy(aotx_policy_address->digest, file.digest, 32, cudaMemcpyHostToDevice), "cudaMemcpy");
        unsigned enabled = 1;
        aotx_check_runtime(cudaMemcpy(&aotx_policy_address->enabled, &enabled, sizeof(enabled),
            cudaMemcpyHostToDevice), "cudaMemcpy");
        aotx_policy_loaded = file.config;
        printf("policy: admitted mode %u state schema %u bytes %u\n", file.config.mode,
            file.config.state_schema, file.config.state_bytes);
    }
    aotx_policy_file_close(&file);
    if (rc) aotx_policy_close();
    return rc;
}
static cudaGraphNode_t aotx_policy_add(cudaGraph_t graph, const cudaGraphNode_t *depends,
    size_t dependency_count, const aotx_policy_input *input, const unsigned char *prior,
    aotx_policy_output *output, unsigned char *next, unsigned count, unsigned stride) {
    unsigned threads = aotx_policy_loaded.threads, blocks = 1 + (count - 1) / threads;
    void *arguments[] = {&input, &prior, &output, &next, &count, &stride};
    cudaGraphNode_t result;
    if (!aotx_policy_function) {
        cudaKernelNodeParams params = {};
        params.func = (void *)aotx_policy_rules; params.gridDim = dim3(blocks);
        params.blockDim = dim3(threads); params.kernelParams = arguments;
        aotx_check_runtime(cudaGraphAddKernelNode(&result, graph, depends, dependency_count, &params),
            "cudaGraphAddKernelNode");
    } else {
        CUDA_KERNEL_NODE_PARAMS params = {};
        params.func = aotx_policy_function; params.gridDimX = blocks;
        params.gridDimY = params.gridDimZ = params.blockDimY = params.blockDimZ = 1;
        params.blockDimX = threads; params.kernelParams = arguments;
        CUgraphNode node;
        aotx_check_driver(cuGraphAddKernelNode(&node, (CUgraph)graph,
            (const CUgraphNode *)depends, dependency_count, &params), "cuGraphAddKernelNode");
        result = (cudaGraphNode_t)node;
    }
    return result;
}
unsigned int aotx_policy_rows_capture(void *stream, const aotx_policy_input *input,
    const unsigned char *prior, aotx_policy_output *output, unsigned char *next,
    unsigned int count, unsigned int stride) {
    if (!count || stride != aotx_policy_loaded.state_bytes || !aotx_policy_loaded.mode) return 0;
    cudaStream_t on = (cudaStream_t)stream;
    cudaStreamCaptureStatus status; cudaGraph_t graph = nullptr;
    const cudaGraphNode_t *dependencies = nullptr; size_t dependency_count = 0;
    aotx_check_runtime(cudaStreamGetCaptureInfo(on, &status, nullptr, &graph,
        &dependencies, nullptr, &dependency_count), "cudaStreamGetCaptureInfo");
    if (status != cudaStreamCaptureStatusActive) return 0;
    cudaGraphNode_t added = aotx_policy_add(graph, dependencies, dependency_count,
        input, prior, output, next, count, stride);
    aotx_check_runtime(cudaStreamUpdateCaptureDependencies(on, &added, nullptr, 1,
        cudaStreamSetCaptureDependencies), "cudaStreamUpdateCaptureDependencies");
    return 1;
}
unsigned int aotx_policy_capture(void *stream) {
    if (!aotx_policy_loaded.mode) return 0;
    cudaStream_t on = (cudaStream_t)stream;
    cudaStreamCaptureStatus status; cudaGraph_t graph = nullptr;
    const cudaGraphNode_t *dependencies = nullptr; size_t dependency_count = 0;
    aotx_check_runtime(cudaStreamGetCaptureInfo(on, &status, nullptr, &graph,
        &dependencies, nullptr, &dependency_count), "cudaStreamGetCaptureInfo");
    if (status != cudaStreamCaptureStatusActive) return 0;
    cudaGraphConditionalHandle condition;
    aotx_check_runtime(cudaGraphConditionalHandleCreate(&condition, graph, 0, cudaGraphCondAssignDefault),
        "cudaGraphConditionalHandleCreate");
    aotx_policy_prepare<<<1, 64, 0, on>>>(condition);
    aotx_check_runtime(cudaStreamGetCaptureInfo(on, &status, nullptr, &graph,
        &dependencies, nullptr, &dependency_count), "cudaStreamGetCaptureInfo");
    cudaGraphNodeParams params = {};
    params.type = cudaGraphNodeTypeConditional; params.conditional.handle = condition;
    params.conditional.type = cudaGraphCondTypeIf; params.conditional.size = 1;
    cudaGraphNode_t node;
    aotx_check_runtime(cudaGraphAddNode(&node, graph, dependencies, nullptr, dependency_count, &params), "cudaGraphAddNode");
    aotx_policy_add(params.conditional.phGraph_out[0], nullptr, 0, &aotx_policy_address->input,
        aotx_policy_address->current, &aotx_policy_address->output, aotx_policy_address->candidate,
        1, aotx_policy_loaded.state_bytes);
    aotx_check_runtime(cudaStreamUpdateCaptureDependencies(on, &node, nullptr, 1,
        cudaStreamSetCaptureDependencies), "cudaStreamUpdateCaptureDependencies");
    aotx_policy_publish<<<1, 64, 0, on>>>();
    return 3;
}
void aotx_policy_close(void) {
    if (aotx_policy_module) cuModuleUnload(aotx_policy_module);
    aotx_policy_module = nullptr; aotx_policy_function = nullptr; aotx_policy_loaded = {};
    if (aotx_policy_address) cudaMemset(aotx_policy_address, 0, sizeof(aotx_policy_state));
    aotx_policy_address = nullptr;
}
