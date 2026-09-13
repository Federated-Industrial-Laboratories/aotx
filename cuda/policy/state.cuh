/* Purpose: Keep creator policy state and recorded decision publication on the GPU.
 * Owns: Accepted state, candidate bytes and bounded journal progress.
 * Launch shape: Batched evaluation between preparation and publication nodes.
 * Lifetime: One selected policy revision and its exact replay. */
#ifndef AOTX_POLICY_STATE_CUH
#define AOTX_POLICY_STATE_CUH
#include "policy/abi.h"
#include <cuda_runtime.h>
#define AOTX_POLICY_HEADER 256u
#define AOTX_POLICY_PART 32u
#define AOTX_POLICY_EMIT 8u
#define AOTX_POLICY_EVENT_BYTES (AOTX_POLICY_HEADER + AOTX_POLICY_STATE_BYTES)
typedef struct aotx_policy_state {
    aotx_policy_config config;
    unsigned char digest[32];
    uint64_t decision, source, root, calls, elapsed_ns, maximum_ns, started_ns, state_hash;
    uint32_t enabled, paused, stopped, status, fatal, pending, received, total;
    uint32_t emitted, launch, maintain, reserved;
    aotx_policy_input input;
    aotx_policy_output output;
    unsigned char current[AOTX_POLICY_STATE_BYTES];
    unsigned char candidate[AOTX_POLICY_STATE_BYTES];
    unsigned char event[AOTX_POLICY_EVENT_BYTES];
} aotx_policy_state;
extern __device__ aotx_policy_state aotx_policy;
__global__ void aotx_policy_prepare(cudaGraphConditionalHandle condition);
__global__ void aotx_policy_rules(const aotx_policy_input *, const unsigned char *,
    aotx_policy_output *, unsigned char *, uint32_t, uint32_t);
__global__ void aotx_policy_publish(void);
__device__ bool aotx_policy_part(const unsigned char *, uint32_t, uint32_t);
__device__ bool aotx_policy_restore_end(void);
__device__ bool aotx_policy_quiet(void);
__device__ bool aotx_policy_maintenance(void);
struct aotx_cli_out;
__device__ void aotx_policy_command(const unsigned char *, uint32_t, struct aotx_cli_out *);
#endif
