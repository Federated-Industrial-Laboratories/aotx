/* Purpose: Bind affect to a conversation or its explicit shared space.
 * Owns: Scoped state transitions and temporary slot clearing.
 * Launch shape: One ordered lease or completion batch.
 * Lifetime: Recorded scope state survives execution slot reuse. */
#ifndef AOTX_SHARED_AFFECT_CUH
#define AOTX_SHARED_AFFECT_CUH
#include "shared/state.cuh"
#ifdef AOTX_AFFECT
__device__ aotx_shared_affect_state *aotx_shared_affect_scope(const aotx_shared_receipt *r);
__device__ unsigned aotx_shared_affect_managed(const aotx_shared_receipt *r);
__device__ bool aotx_shared_affect_conflict(const aotx_shared_receipt *a, const aotx_shared_receipt *b);
__device__ void aotx_shared_affect_clear(unsigned slot);
__device__ void aotx_shared_affect_lease(aotx_shared_receipt *r, unsigned slot, unsigned managed);
__device__ void aotx_shared_affect_read(unsigned conversation, unsigned char *out);
__device__ unsigned aotx_shared_affect_encode(const aotx_shared_receipt *r, unsigned status,
    unsigned finish, unsigned char *out);
__device__ bool aotx_shared_affect_apply(const aotx_shared_receipt *r, const unsigned char *p, unsigned n);
#endif
#endif
