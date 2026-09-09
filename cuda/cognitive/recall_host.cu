/* Purpose: Restore prepared GPU memory and select or replay recorded context.
 * Owns: Device allocation, transfer, launch and completion waits.
 * Launch shape: Serialized state batches and one block per recall request.
 * Lifetime: One offline command; the base graphs remain separate. */
#include <cuda_runtime.h>
#include "cognitive/recall.cuh"
#include "cognitive/recall_io.h"

int main(int argc, char **argv) {
    int mode = aotx_recall_file_options(argc, argv);
    if (mode < 0 || mode == 2) return mode < 0 ? 2 : 0;
    aotx_recall_file file;
    int status = aotx_recall_file_open(argv[2], mode ? nullptr : argv[3], &file);
    if (status) { aotx_recall_file_report(status); return 1; }
    aotx_cognitive_store *live = nullptr, *stage = nullptr;
    unsigned char *image = nullptr, *requests = nullptr;
    aotx_recall_result *rows = nullptr;
    aotx_cognitive_result *device_result = nullptr, result = {};
#define AOTX_RECALL_CUDA(call) do { if (!status && (call) != cudaSuccess) status = 100; } while (0)
#define AOTX_RECALL_WAIT() do { \
    AOTX_RECALL_CUDA(cudaGetLastError()); \
    AOTX_RECALL_CUDA(cudaMemcpy(&result, device_result, sizeof(result), cudaMemcpyDeviceToHost)); \
    if (!status && result.status) status = 200 + (int)result.status; \
} while (0)
    AOTX_RECALL_CUDA(cudaMalloc(&live, sizeof(*live)));
    AOTX_RECALL_CUDA(cudaMalloc(&stage, sizeof(*stage)));
    AOTX_RECALL_CUDA(cudaMalloc(&image, AOTX_COG_IMAGE));
    AOTX_RECALL_CUDA(cudaMalloc(&requests, AOTX_RECALL_REQUESTS));
    AOTX_RECALL_CUDA(cudaMalloc(&rows, AOTX_RECALL_BATCH * sizeof(*rows)));
    AOTX_RECALL_CUDA(cudaMalloc(&device_result, sizeof(*device_result)));
    AOTX_RECALL_CUDA(cudaMemset(live, 0, sizeof(*live)));
    AOTX_RECALL_CUDA(cudaMemcpy(image, file.source.checkpoint,
                              file.source.checkpoint_bytes, cudaMemcpyHostToDevice));
    if (!status) {
        aotx_cognitive_restore<<<1, 64>>>(live, stage, image, file.source.checkpoint_bytes, device_result);
        AOTX_RECALL_WAIT();
    }
    if (!status && file.source.tail_bytes) {
        AOTX_RECALL_CUDA(cudaMemcpy(image, file.source.tail, file.source.tail_bytes, cudaMemcpyHostToDevice));
        if (!status) {
            aotx_cognitive_apply<<<1, 64>>>(live, stage, image, file.source.tail_bytes, device_result);
            AOTX_RECALL_WAIT();
        }
    }
    if (!status && mode) {
        aotx_recall_requests<<<1, 64>>>(live, requests, device_result);
        AOTX_RECALL_WAIT();
        if (!status) {
            file.count = result.applied; file.request_bytes = result.bytes;
            if (!file.count || file.count > AOTX_RECALL_BATCH || file.request_bytes > AOTX_RECALL_REQUESTS)
                status = AOTX_CCIR_INVALID;
        }
        if (!status) {
            aotx_recall_replay<<<file.count, 64>>>(live, requests, file.request_bytes, rows, file.count);
            AOTX_RECALL_CUDA(cudaGetLastError());
            AOTX_RECALL_CUDA(cudaMemcpy(file.rows, rows, file.count * sizeof(*rows), cudaMemcpyDeviceToHost));
        }
    }
    if (!status && !mode) {
        AOTX_RECALL_CUDA(cudaMemcpy(requests, file.requests, file.request_bytes, cudaMemcpyHostToDevice));
        if (!status) {
            aotx_recall_search<<<file.count, 64>>>(live, requests, file.request_bytes, rows, file.count);
            AOTX_RECALL_CUDA(cudaGetLastError());
        }
        if (!status) {
            aotx_recall_record<<<1, 64>>>(live, requests, file.request_bytes, rows, file.count, image, device_result);
            AOTX_RECALL_WAIT();
        }
        if (!status) {
            aotx_cognitive_apply<<<1, 64>>>(live, stage, image, result.bytes, device_result);
            AOTX_RECALL_WAIT();
        }
        if (!status) {
            aotx_cognitive_checkpoint<<<1, 64>>>(live, image, AOTX_COG_IMAGE, device_result);
            AOTX_RECALL_WAIT();
            AOTX_RECALL_CUDA(cudaMemcpy(file.checkpoint, image, result.bytes, cudaMemcpyDeviceToHost));
            AOTX_RECALL_CUDA(cudaMemcpy(file.rows, rows, file.count * sizeof(*rows), cudaMemcpyDeviceToHost));
        }
        if (!status) status = aotx_recall_file_write(&file, argv[4], result.bytes);
    }
    if (!status) status = aotx_recall_file_rows(&file);
    aotx_recall_file_report(status);
    cudaFree(device_result); cudaFree(rows); cudaFree(requests);
    cudaFree(image); cudaFree(stage); cudaFree(live);
    aotx_recall_file_close(&file);
    return status ? 1 : 0;
}
