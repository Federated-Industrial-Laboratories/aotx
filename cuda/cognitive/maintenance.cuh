/* Purpose: Reclaim eligible GPU memory at a recorded completed-work boundary.
 * Owns: Store-sized mark and offset arrays; the existing candidate holds copied state.
 * Launch shape: Parallel scans, one closure block, parallel copies and one publication block.
 * Lifetime: An explicit lifecycle policy and its journal recovery. */
#ifndef AOTX_COGNITIVE_MAINTENANCE_CUH
#define AOTX_COGNITIVE_MAINTENANCE_CUH
#include "cognitive/checkpoint.cuh"
#define AOTX_LIVE_MAINTENANCE 8u
typedef struct aotx_memory_maintenance {
    uint32_t marks[AOTX_COG_OBJECTS], latest[AOTX_COG_OBJECTS];
    uint32_t indices[AOTX_COG_OBJECTS], offsets[AOTX_COG_OBJECTS];
    uint32_t running, status, removed, released;
    uint64_t last_attempt, passes;
} aotx_memory_maintenance;
extern __device__ aotx_memory_maintenance aotx_maintenance;
__device__ void aotx_memory_auto_request(void);
__device__ void aotx_memory_maintain_begin(void);
__device__ void aotx_memory_status(struct aotx_cli_out *out);
__global__ void aotx_memory_seed(void);
__global__ void aotx_memory_plan(void);
__global__ void aotx_memory_offsets(void);
__global__ void aotx_memory_copy(void);
__global__ void aotx_memory_install(void);
__global__ void aotx_memory_publish(void);
#endif
