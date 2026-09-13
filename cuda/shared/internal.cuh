/* Purpose: Share bounded table lookup and transaction checks.
 * Owns: No state outside the shared service tables.
 * Launch shape: One ordered device batch.
 * Lifetime: One complete command or read. */
#ifndef AOTX_SHARED_INTERNAL_CUH
#define AOTX_SHARED_INTERNAL_CUH
#include "shared/state.cuh"
#include "service/internal.cuh"
#define AOTX_SHARED_NONE (~0u)
extern __device__ aotx_shared_receipt aotx_shared_candidate;
#define AOTX_SHARED_ADMIT_HEAD (112u + AOTX_SHARED_MEDIA_REFS * 16u)
static __device__ __forceinline__ void aotx_shared_zero(void *p, unsigned n)
{ for (unsigned i = 0; i < n; ++i) ((unsigned char *)p)[i] = 0; }
static __device__ __forceinline__ unsigned aotx_shared_u32(const unsigned char *p)
{ return aotx_service_u32(p); }
static __device__ __forceinline__ unsigned long long aotx_shared_u64(const unsigned char *p)
{ return aotx_service_get(p, 8); }
static __device__ __forceinline__ bool aotx_shared_id(const unsigned char *a, const unsigned char *b)
{ return aotx_service_equal(a, b, 16); }
__device__ unsigned aotx_shared_participant_find(const unsigned char *id);
__device__ unsigned aotx_shared_space_find(const unsigned char *id);
__device__ unsigned aotx_shared_conversation_find(const unsigned char *id);
__device__ unsigned aotx_shared_receipt_find(unsigned participant, unsigned long long sequence);
__device__ unsigned aotx_shared_id_find(const unsigned char *id);
__device__ unsigned aotx_shared_key_find(unsigned participant, const unsigned char *key);
__device__ unsigned aotx_shared_rights(unsigned participant, unsigned space);
__device__ bool aotx_shared_visible(unsigned participant, unsigned space, unsigned rights);
__device__ unsigned aotx_shared_command_check(const unsigned char *p, unsigned n);
__device__ unsigned aotx_shared_admit(const aotx_service_grant *grant, const unsigned char *p,
                                      unsigned n, unsigned *receipt);
__device__ bool aotx_shared_apply(unsigned kind, const unsigned char *p, unsigned n,
                                  unsigned long long source, bool replay);
__device__ bool aotx_shared_admission_apply(const unsigned char *p, unsigned n,
                                            unsigned long long source, bool replay);
__device__ bool aotx_shared_begin(unsigned kind, unsigned bytes);
__device__ void aotx_shared_read(unsigned channel, const aotx_service_grant *grant,
                                const unsigned char *read, unsigned bytes);
__device__ unsigned char *aotx_shared_reply(unsigned channel, unsigned kind);
__device__ void aotx_shared_receipt_reply(unsigned channel, unsigned kind, unsigned receipt,
                                         unsigned long long cursor, unsigned http_status);
#endif
