/* Purpose: Check service wire values and publish bounded mailbox responses.
 * Owns: No state outside the service tables.
 * Launch shape: The ordered admission node processes a bounded mailbox batch.
 * Lifetime: One admission pass. */
#ifndef AOTX_SERVICE_INTERNAL_CUH
#define AOTX_SERVICE_INTERNAL_CUH
#include "service/service.cuh"
#include "seam/seam.cuh"
#include "sched/sched.cuh"
static __device__ __forceinline__ unsigned aotx_service_u32(const unsigned char *p)
{ return (unsigned)aotx_service_get(p, 4); }
static __device__ __forceinline__ bool aotx_service_equal(const unsigned char *a,
                                                          const unsigned char *b, unsigned n)
{
    unsigned difference = 0;
    for (unsigned i = 0; i < n; ++i) difference |= a[i] ^ b[i];
    return difference == 0;
}
static __device__ __forceinline__ bool aotx_service_nonzero(const unsigned char *p, unsigned n)
{
    unsigned value = 0;
    for (unsigned i = 0; i < n; ++i) value |= p[i];
    return value != 0;
}
static __device__ __forceinline__ void aotx_service_bytes(unsigned char *to,
                                                          const unsigned char *from, unsigned n)
{ for (unsigned i = 0; i < n; ++i) to[i] = from[i]; }
static __device__ __forceinline__ aotx_service_grant *aotx_service_granted(const unsigned char *id)
{
    for (unsigned i = 0; i < aotx_service.grant_count; ++i)
        if (aotx_service_equal(id, aotx_service.grants[i].principal, 16))
            return aotx_service.grants + i;
    return 0;
}
static __device__ __forceinline__ aotx_service_job *aotx_service_request(const unsigned char *id,
                                                                       const unsigned char *principal)
{
    for (unsigned i = 0; i < AOTX_SERVICE_REQUESTS; ++i) {
        aotx_service_job *j = aotx_service.jobs + i;
        if (j->phase && aotx_service_equal(id, j->id, 16) &&
            aotx_service_equal(principal, j->principal, 16)) return j;
    }
    return 0;
}
static __device__ __forceinline__ void aotx_service_answer(unsigned channel, unsigned status,
                                                           unsigned phase, unsigned bytes = 0)
{
    unsigned char *f = aotx_service.frames + (unsigned long long)channel * AOTX_SERVICE_FRAME;
    aotx_service_mailbox *m = aotx_service.mailbox + channel;
    for (unsigned i = 0; i < 8; ++i) f[i] = AOTX_SERVICE_MAGIC[i];
    aotx_service_put(f + 8, status, 4); aotx_service_put(f + 12, phase, 4);
    aotx_service_put(f + 40, aotx_service.epoch, 8);
    aotx_service_put(f + 88, bytes, 4);
    aotx_service_bytes(m->bytes, f, AOTX_SERVICE_HEAD + bytes);
    m->length = AOTX_SERVICE_HEAD + bytes;
    aotx_seam_release_sys((unsigned long long *)&m->state, 2);
    aotx_service.ready[channel] = 0;
}
__device__ unsigned aotx_service_install(const unsigned char *f, unsigned n);
__device__ unsigned aotx_service_submit(const aotx_service_grant *g, unsigned char *f);
__device__ unsigned aotx_service_render(aotx_service_job *job, const unsigned char *p, unsigned n);
__device__ void aotx_service_information(unsigned channel, const aotx_service_grant *g, bool telemetry);
__device__ void aotx_service_media(unsigned channel, const aotx_service_grant *g, bool read);
__device__ void aotx_service_media_expire(void);
__device__ void aotx_service_media_list(unsigned channel, const aotx_service_grant *g);
#endif
