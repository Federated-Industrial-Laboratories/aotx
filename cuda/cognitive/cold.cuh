/* Purpose: Connect explicit residency changes to live memory and exact replay.
 * Owns: Device batch state and bounded read progress.
 * Launch shape: One 64-thread block per tick; no kernel waits for disk IO.
 * Lifetime: One complete control and result pair. */
#ifndef AOTX_COGNITIVE_COLD_CUH
#define AOTX_COGNITIVE_COLD_CUH
#include "cognitive/checkpoint.cuh"
#define AOTX_COLD_WAIT 11u
#define AOTX_COLD_BUILD 12u
#define AOTX_COLD_TICKS 4096u
typedef struct aotx_cold_state {
    uint32_t active, mode, status, count, bytes, copied, ticks, recovery;
    uint64_t serial;
    uint32_t selected[AOTX_COG_OBJECTS], needed[AOTX_COG_OBJECTS], done[AOTX_COG_WORDS];
} aotx_cold_state;
extern __device__ aotx_cold_state aotx_cold;
__device__ void aotx_cold_begin(void);
__device__ void aotx_cold_replay(void);
__device__ void aotx_cold_publish(void);
__device__ void aotx_cold_cancel(void);
__device__ void aotx_cold_result_header(void);
__device__ void aotx_cold_candidate(const unsigned char *payload);
__global__ void aotx_cold_step(void);
struct aotx_cli_out;
__device__ void aotx_cold_status(struct aotx_cli_out *out);
#endif
