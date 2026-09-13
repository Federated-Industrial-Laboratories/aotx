/* Purpose: Connect persistent conversations to temporary memory and model slots.
 * Owns: Transient execution progress; exact results belong to shared records.
 * Launch shape: One ordered work batch and one ordered output batch.
 * Lifetime: One execution lease; replay restores no active model work. */
#ifndef AOTX_SHARED_BRIDGE_CUH
#define AOTX_SHARED_BRIDGE_CUH
#include "shared/state.cuh"
enum aotx_shared_stage { AOTX_SHARED_MEMORY = 1, AOTX_SHARED_PROMPT,
    AOTX_SHARED_TOKENIZE, AOTX_SHARED_DECODE, AOTX_SHARED_END };
struct aotx_shared_execution { unsigned stage, status, model_opened; unsigned long long opened; };
extern __device__ aotx_shared_execution aotx_shared_execution_slots[AOTX_SLOTS];
__device__ void aotx_shared_memory_choice(unsigned slot, unsigned status);
__device__ bool aotx_shared_sample(unsigned slot, aotx_model_how *sample);
__device__ unsigned aotx_shared_limit(unsigned slot, unsigned fallback);
__device__ void aotx_shared_start_result(unsigned slot, unsigned status);
__device__ bool aotx_shared_media_leased(unsigned object, unsigned long long generation);
__device__ unsigned aotx_shared_input_text(const aotx_shared_receipt *receipt, unsigned char *out, unsigned capacity);
__device__ unsigned aotx_shared_model_prompt(unsigned slot);
__device__ bool aotx_shared_memory_authorized(void);
__global__ void aotx_shared_work(void);
__global__ void aotx_shared_results(void);
#endif
