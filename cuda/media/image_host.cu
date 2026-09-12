/* Purpose: Capture the finite image decoder stages in one CUDA stream.
 * Owns: No data; the caller owns the batch and all buffers.
 * Launch shape: Host glue only; device stages inspect job state.
 * Lifetime: One capture or ordered launch batch. */
#include "media/image.cuh"

void aotx_image_capture(void *stream, aotx_image_job *jobs, uint32_t count, uint32_t quantum) {
    if (!count) return;
    cudaStream_t on = (cudaStream_t)stream;
    uint32_t groups = (count + 63u) / 64u;
    aotx_image_step<<<groups,64,0,on>>>(jobs, count, quantum);
    aotx_image_zero<<<dim3(32, count),256,0,on>>>(jobs, count);
    aotx_image_idct<<<dim3(128, count),64,0,on>>>(jobs, count);
    aotx_image_pixels<<<dim3(32, count),256,0,on>>>(jobs, count);
    aotx_image_finish<<<groups,64,0,on>>>(jobs, count);
}
