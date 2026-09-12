/* Purpose: Keep trained audio workspaces and published feature rows resident.
 * Owns: Audio component state; canonical sources use the common media registry.
 * Launch shape: Finite batched jobs selected by the device scheduler.
 * Lifetime: One runtime allocation and its exact parent model binding. */
#ifndef AOTX_AUDIO_RUNTIME_CUH
#define AOTX_AUDIO_RUNTIME_CUH
#include "audio/audio.cuh"
#include "audio/profile.h"
struct aotx_audio_runtime_state {
    aotx_audio_profile profile;
    aotx_audio_job *jobs;
    unsigned *owner;
    float *features;
    unsigned char *workspace;
    unsigned long long workspace_each, allocated;
    unsigned enabled, role;
    unsigned char parent_digest[32];
};
extern __device__ aotx_audio_runtime_state aotx_audio_runtime;
int aotx_audio_open(const char *,const char *,int (*)(void));
void aotx_audio_close(void);
int aotx_audio_allocate(const aotx_audio_profile *,const aotx_audio_desc *,unsigned,unsigned char **);
void aotx_audio_runtime_capture(cudaStream_t);
__global__ void aotx_audio_initialize(void);
__global__ void aotx_audio_schedule(void);
__global__ void aotx_audio_complete(void);
#endif
