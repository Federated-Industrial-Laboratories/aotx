/* Purpose: Publish and restore coherent live cognitive checkpoints.
 * Owns: Device checkpoint progress and the mapped transport binding.
 * Launch shape: One 64-thread block takes or imports a complete binding batch.
 * Lifetime: The optional memory mirror of one run. */
#ifndef AOTX_COGNITIVE_CHECKPOINT_CUH
#define AOTX_COGNITIVE_CHECKPOINT_CUH
#include "cognitive/checkpoint.h"
#include "cognitive/live.cuh"
typedef struct aotx_checkpoint_state {
    aotx_checkpoint_ring *ring;
    uint64_t head, captured, durable, generation, error;
    uint32_t copying, bytes, copied, bindings, capturing;
} aotx_checkpoint_state;
extern __device__ aotx_checkpoint_state aotx_checkpoint;
extern __device__ unsigned char aotx_checkpoint_image[AOTX_CP_BYTES];
__device__ bool aotx_checkpoint_pressure(void);
__device__ bool aotx_checkpoint_idle(void);
__device__ void aotx_checkpoint_encode(void);
__device__ void aotx_checkpoint_import(void);
__global__ void aotx_checkpoint_step(void);
__global__ void aotx_checkpoint_fill(void);
__global__ void aotx_checkpoint_copy(void);
__global__ void aotx_checkpoint_publish(void);
int aotx_checkpoint_capture(void *stream);
struct aotx_cli_out;
__device__ void aotx_checkpoint_status(struct aotx_cli_out *out);
#endif
