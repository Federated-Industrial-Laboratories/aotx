/* Purpose: Transfer candidate batches and launch measurement position checks.
 * Owns: Temporary device buffers for each bounded call.
 * Launch shape: Host glue for vector batches and tokenizer batches.
 * Lifetime: Every temporary buffer is freed before return. */
#include "tools/steer_measure.cuh"
#include "boot/check.h"
int aotx_steer_measure_vectors(aotx_steer_measure *rows, unsigned count) {
    if (!rows || !count || count > AOTX_STEER_TEXTS) return 1;
    aotx_steer_measure *device;
    aotx_check_runtime(cudaMalloc(&device, count * sizeof(*device)), "cudaMalloc");
    aotx_check_runtime(cudaMemcpy(device, rows, count * sizeof(*device), cudaMemcpyHostToDevice), "cudaMemcpy");
    aotx_steer_measure_admit<<<1, AOTX_CONDUCT_VECTORS>>>(device, count);
    aotx_steer_measure_read<<<1, AOTX_STEER_TEXTS>>>(device, count);
    aotx_check_runtime(cudaMemcpy(rows, device, count * sizeof(*device), cudaMemcpyDeviceToHost), "cudaMemcpy");
    aotx_check_runtime(cudaFree(device), "cudaFree");
    return 0;
}
int aotx_steer_positions_run(aotx_steer_text tokenizer, unsigned role, unsigned count, aotx_model_how *how) {
    if (!count || count > AOTX_STEER_TEXTS || !how) return 1;
    unsigned *bad, value = 0;
    aotx_check_runtime(cudaMalloc(&bad, sizeof(*bad)), "cudaMalloc");
    aotx_check_runtime(cudaMemset(bad, 0, sizeof(*bad)), "cudaMemset");
    aotx_steer_positions<<<1, AOTX_STEER_TEXTS>>>(tokenizer, role, count, how, bad);
    aotx_check_runtime(cudaMemcpy(&value, bad, sizeof(value), cudaMemcpyDeviceToHost), "cudaMemcpy");
    aotx_check_runtime(cudaFree(bad), "cudaFree");
    return value != 0;
}
