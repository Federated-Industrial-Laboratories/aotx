/* Purpose: Capture finite image encoder steps in the caller's stream.
 * Owns: No allocation or image state.
 * Launch shape: Batched image kernels and trained matrix tiles.
 * Lifetime: Graph construction; job progress remains on the device. */
#include "vision/vision.cuh"
#include "model/matrix.cuh"

void aotx_vision_capture(cudaStream_t on, aotx_vision_job *jobs, unsigned count,
    const unsigned char *weights, const aotx_vision_desc *desc,
    unsigned patch_capacity, unsigned query_quantum)
{
    if (!count || !patch_capacity || !query_quantum) return;
    unsigned rows = (patch_capacity + AOTX_GEMM_TILE_M - 1u) / AOTX_GEMM_TILE_M;
    dim3 pixels(128, count), norms(128, count);
    dim3 matrix(3072u / AOTX_GEMM_TILE_N, rows, count);
    unsigned groups = (count + 63u) / 64u;
    aotx_vision_step<<<groups,64,0,on>>>(jobs, count);
    aotx_vision_resize<<<pixels,256,0,on>>>(jobs, count, 0);
    aotx_vision_resize<<<pixels,256,0,on>>>(jobs, count, 1);
    aotx_vision_patch<<<pixels,256,0,on>>>(jobs, count);
    aotx_vision_product<<<matrix,256,0,on>>>(jobs, count, weights, desc, 0, 0);
    aotx_vision_product<<<matrix,256,0,on>>>(jobs, count, weights, desc, 0, 1);
    aotx_vision_position<<<pixels,256,0,on>>>(jobs, count, weights, desc, 0);
    aotx_vision_product<<<matrix,256,0,on>>>(jobs, count, weights, desc, 1, 0);
    aotx_vision_product<<<matrix,256,0,on>>>(jobs, count, weights, desc, 1, 1);
    aotx_vision_position<<<pixels,256,0,on>>>(jobs, count, weights, desc, 1);
    aotx_vision_norm<<<norms,128,0,on>>>(jobs, count, weights, desc, 0);
    aotx_vision_product<<<matrix,256,0,on>>>(jobs, count, weights, desc, 2, 0);
    aotx_vision_product<<<matrix,256,0,on>>>(jobs, count, weights, desc, 2, 1);
    aotx_vision_rope<<<norms,128,0,on>>>(jobs, count, weights, desc);
    aotx_vision_attention<<<dim3((query_quantum+3u)/4u,12,count),128,0,on>>>(jobs,count,query_quantum);
    aotx_vision_product<<<matrix,256,0,on>>>(jobs, count, weights, desc, 3, 0);
    aotx_vision_product<<<matrix,256,0,on>>>(jobs, count, weights, desc, 3, 1);
    aotx_vision_residual<<<pixels,256,0,on>>>(jobs, count, weights, desc, 0);
    aotx_vision_norm<<<norms,128,0,on>>>(jobs, count, weights, desc, 1);
    aotx_vision_product<<<matrix,256,0,on>>>(jobs, count, weights, desc, 4, 0);
    aotx_vision_product<<<matrix,256,0,on>>>(jobs, count, weights, desc, 4, 1);
    aotx_vision_activation<<<pixels,256,0,on>>>(jobs, count, weights, desc, 0);
    aotx_vision_product<<<matrix,256,0,on>>>(jobs, count, weights, desc, 5, 0);
    aotx_vision_product<<<matrix,256,0,on>>>(jobs, count, weights, desc, 5, 1);
    aotx_vision_residual<<<pixels,256,0,on>>>(jobs, count, weights, desc, 1);
    aotx_vision_norm<<<norms,128,0,on>>>(jobs, count, weights, desc, 2);
    aotx_vision_product<<<matrix,256,0,on>>>(jobs, count, weights, desc, 6, 0);
    aotx_vision_product<<<matrix,256,0,on>>>(jobs, count, weights, desc, 6, 1);
    aotx_vision_activation<<<pixels,256,0,on>>>(jobs, count, weights, desc, 1);
    aotx_vision_product<<<matrix,256,0,on>>>(jobs, count, weights, desc, 7, 0);
    aotx_vision_product<<<matrix,256,0,on>>>(jobs, count, weights, desc, 7, 1);
    aotx_vision_residual<<<pixels,256,0,on>>>(jobs, count, weights, desc, 2);
    aotx_vision_finish<<<groups,64,0,on>>>(jobs, count, query_quantum);
}
