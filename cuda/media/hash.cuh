/* Purpose: Verify immutable source identities with batched device SHA-256.
 * Owns: The caller supplies input bytes and resumable digest states.
 * Launch shape: One thread per byte stream, in blocks of 64 threads.
 * Lifetime: From complete source admission until digest completion. */
#ifndef AOTX_MEDIA_HASH_CUH
#define AOTX_MEDIA_HASH_CUH
#include <cuda_runtime.h>
struct aotx_media_hash {
    const unsigned char *source;
    unsigned long long bytes, cursor;
    unsigned h[8];
    unsigned char digest[32];
    unsigned active, done, status;
};
__global__ void aotx_media_hash_step(aotx_media_hash *jobs, unsigned count, unsigned blocks);
#endif
