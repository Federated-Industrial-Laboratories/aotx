/* Purpose: Load typed state onto the GPU and export a new file checkpoint.
 * Owns: Device allocations, transfers, launches and completion waits.
 * Launch shape: Serialized 64-thread state batches; no base graph changes.
 * Lifetime: One offline restore, replay and checkpoint operation. */
#include <cuda_runtime.h>
#include "cognitive/state.cuh"
#include "cognitive/io.h"
#include <stdlib.h>

int main(int argc, char **argv) {
    int options = aotx_cognitive_file_options(argc, argv);
    if (options) return options < 0 ? 2 : 0;
    aotx_cognitive_file file;
    int status = aotx_cognitive_file_open(argv[1], &file);
    if (status) { aotx_cognitive_file_report(status, 0, 0, 0); return 1; }
    aotx_cognitive_store *live = nullptr, *stage = nullptr;
    unsigned char *image = nullptr, *output = (unsigned char *)malloc(AOTX_COG_IMAGE);
    aotx_cognitive_result *device_result = nullptr, result = {};
    if (!output) status = AOTX_CCIR_IO;
#define AOTX_COG_CUDA(call) do { if (!status && (call) != cudaSuccess) status = 100; } while (0)
    AOTX_COG_CUDA(cudaMalloc(&live, sizeof(*live)));
    AOTX_COG_CUDA(cudaMalloc(&stage, sizeof(*stage)));
    AOTX_COG_CUDA(cudaMalloc(&image, AOTX_COG_IMAGE));
    AOTX_COG_CUDA(cudaMalloc(&device_result, sizeof(*device_result)));
    AOTX_COG_CUDA(cudaMemset(live, 0, sizeof(*live)));
    AOTX_COG_CUDA(cudaMemcpy(image, file.checkpoint, file.checkpoint_bytes, cudaMemcpyHostToDevice));
    if (!status) {
        aotx_cognitive_restore<<<1, 64>>>(live, stage, image, file.checkpoint_bytes, device_result);
        AOTX_COG_CUDA(cudaGetLastError());
        AOTX_COG_CUDA(cudaMemcpy(&result, device_result, sizeof(result), cudaMemcpyDeviceToHost));
        if (!status && result.status) status = 200 + (int)result.status;
    }
    if (!status && file.tail_bytes) {
        AOTX_COG_CUDA(cudaMemcpy(image, file.tail, file.tail_bytes, cudaMemcpyHostToDevice));
        if (!status) {
            aotx_cognitive_apply<<<1, 64>>>(live, stage, image, file.tail_bytes, device_result);
            AOTX_COG_CUDA(cudaGetLastError());
            AOTX_COG_CUDA(cudaMemcpy(&result, device_result, sizeof(result), cudaMemcpyDeviceToHost));
            if (!status && result.status) status = 200 + (int)result.status;
        }
    }
    if (!status) {
        aotx_cognitive_checkpoint<<<1, 64>>>(live, image, AOTX_COG_IMAGE, device_result);
        AOTX_COG_CUDA(cudaGetLastError());
        AOTX_COG_CUDA(cudaMemcpy(&result, device_result, sizeof(result), cudaMemcpyDeviceToHost));
        if (!status && result.status) status = 200 + (int)result.status;
        AOTX_COG_CUDA(cudaMemcpy(output, image, result.bytes, cudaMemcpyDeviceToHost));
    }
    if (!status) status = aotx_cognitive_file_write(&file, argv[2], output, result.bytes);
    aotx_cognitive_file_report(status, result.sequence, result.bytes, file.view.fallback);
    cudaFree(device_result); cudaFree(image); cudaFree(stage); cudaFree(live);
    free(output); aotx_cognitive_file_close(&file);
    return status ? 1 : 0;
}
