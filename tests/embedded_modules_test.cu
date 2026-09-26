/* Purpose: Check runtime module loading without source-file access.
 * Owns: Small exact matrices and captured module nodes.
 * Launch shape: The driver clock node and matrix capture at N=1 and N=64.
 * Lifetime: Each case frees its graph and device buffers. */
#include "boot/boot.cuh"
#include "model/graph_host.h"
#include "model/decode_module_host.h"
#include "model/matrix.cuh"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

static unsigned checks, failures, file_reads;
extern "C" FILE *__wrap_fopen(const char *, const char *) {
    ++file_reads; errno = EACCES; return nullptr;
}
static void check(bool ok, const char *text) {
    ++checks; if (!ok) { ++failures; fprintf(stderr, "FAIL: %s\n", text); }
}
static void cu(cudaError_t rc) {
    if (rc != cudaSuccess) { fprintf(stderr, "%s\n", cudaGetErrorString(rc)); exit(2); }
}
static void matrix(unsigned n) {
    unsigned char *weights; half *x; float *y;
    cu(cudaMallocManaged(&weights, 32 * 34)); cu(cudaMallocManaged(&x, n * 32 * sizeof(*x)));
    cu(cudaMallocManaged(&y, n * 32 * sizeof(*y)));
    for (unsigned i = 0; i < 32; ++i) {
        weights[i * 34] = 0; weights[i * 34 + 1] = 0x3c;
        for (unsigned j = 0; j < 32; ++j) weights[i * 34 + 2 + j] = i + 1;
    }
    for (unsigned i = 0; i < n * 32; ++i) { x[i] = __float2half(i / 32 + 1); y[i] = -(float)(i + 1); }
    aotx_model_run run = {}; run.tokens = run.rows = n;
    cu(cudaMemcpyToSymbol(aotx_model_call, &run, sizeof(run)));
    aotx_model_batch_of(0); aotx_decode_module_open();
    aotx_model_hold hold = {}; hold.role = 0; cu(cudaStreamCreate(&hold.stream));
    cu(cudaStreamBeginCapture(hold.stream, cudaStreamCaptureModeGlobal));
    unsigned added = aotx_model_module_node(&hold, weights, AOTX_WEIGHT_Q8_0, 32, 32, x, y,
        dim3(2), AOTX_MODEL_BATCH_TOKENS);
    cu(cudaStreamEndCapture(hold.stream, &hold.graph));
    check(added == 1, "the embedded matrix function binds without a source path");
    size_t nodes = 0; cu(cudaGraphGetNodes(hold.graph, nullptr, &nodes));
    check(nodes == 1, "the capture contains the actual module node");
    if (added) {
        cu(cudaGraphInstantiate(&hold.exec, hold.graph, 0));
        cu(cudaGraphLaunch(hold.exec, hold.stream)); cu(cudaStreamSynchronize(hold.stream));
        for (unsigned i = 0; i < n * 32; ++i)
            check(y[i] == (n == 1 ? (float)(32 * (i + 1)) : -(float)(i + 1)),
                "the module computes one row and leaves larger batches to the compiled node");
        cu(cudaGraphExecDestroy(hold.exec));
    }
    cu(cudaGraphDestroy(hold.graph)); cu(cudaStreamDestroy(hold.stream)); aotx_decode_module_close();
    cu(cudaFree(weights)); cu(cudaFree(x)); cu(cudaFree(y));
}
int main(void) {
    cu(cudaFree(nullptr)); unsigned long long sample = 0;
    check(!aotx_boot_clock_check(&sample) && sample, "the embedded clock module returns a device sample");
    matrix(1); matrix(64);
    check(!file_reads, "runtime module loading opens no source files");
    printf("embedded modules: %u checks, %u failures\n", checks, failures);
    return failures || checks != 2086 ? 1 : 0;
}
