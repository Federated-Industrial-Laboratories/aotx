/* Purpose: Admit scoped service requests and keep their results on the device.
 * Owns: Deployment grants, request identities, execution leases and result bytes.
 * Launch shape: Finite mailbox batches and one work thread for each execution slot.
 * Lifetime: One runtime epoch; ordinary request handles expire at restart. */
#ifndef AOTX_SERVICE_CUH
#define AOTX_SERVICE_CUH
#include "service/wire.h"
#include "service/profile.h"
#include "profile/profile.cuh"
#include "model/forward.cuh"
#include "media/prompt.cuh"

struct aotx_service_grant {
    unsigned char principal[16];
    unsigned long long revision, media_bytes;
    unsigned actions, models, pages, tokens, requests, media;
};
struct aotx_service_job {
    unsigned char id[16], principal[16], model_digest[32];
    unsigned long long revision, opened, changed;
    unsigned phase, status, role, limit, pages, slot, length, output, prompt, sampled, finish, cancel;
    aotx_model_how sample;
    unsigned media_count;
    aotx_media_reference media[AOTX_MEDIA_REFS];
    unsigned char text[AOTX_SAY_BYTES];
    unsigned char result[AOTX_SERVICE_OUTPUT_BYTES];
};
struct aotx_service_state {
    aotx_service_ring *ring;
    aotx_service_mailbox *mailbox;
    unsigned char *frames;
    unsigned *ready;
    aotx_service_grant *grants;
    aotx_service_job *jobs;
    struct aotx_service_upload *uploads;
    unsigned grant_count, enabled, cursor, media_count;
    unsigned slot[AOTX_SLOTS];
    unsigned long long epoch, revision, clock, bytes;
};
struct aotx_service_upload { unsigned char transfer[16]; unsigned long long deadline; };
extern __device__ aotx_service_state aotx_service;
__device__ bool aotx_service_owns(unsigned slot);
__device__ const unsigned char *aotx_service_principal(unsigned slot);
__device__ unsigned aotx_service_limit(unsigned slot, unsigned fallback);
__device__ bool aotx_service_sample(unsigned slot, aotx_model_how *sample);
__device__ bool aotx_service_media_leased(unsigned object, unsigned long long generation);
__device__ void aotx_service_start_result(unsigned slot, unsigned status);
__global__ void aotx_service_copy(void);
__global__ void aotx_service_admit(void);
__global__ void aotx_service_work(void);
__global__ void aotx_service_reply(void);

#endif
