/* Purpose: Encode upload metadata and preserve pending refusals before source reuse.
 * Owns: Inline access to the bounded device upload receipt table.
 * Launch shape: One ordered source or service admission thread.
 * Lifetime: Current-boot transport state only. */
#ifndef AOTX_SERVICE_MEDIA_RECEIPT_CUH
#define AOTX_SERVICE_MEDIA_RECEIPT_CUH
#include "service/internal.cuh"
#include "media/runtime.cuh"
static __device__ __forceinline__ void aotx_service_media_fields(unsigned char *p, const aotx_media_object &o)
{
    for (unsigned i = 0; i < 64; ++i) p[i] = 0;
    aotx_service_bytes(p, o.digest, 32); aotx_service_put(p + 32, o.bytes, 8);
    aotx_service_put(p + 40, o.phase, 4); aotx_service_put(p + 44, o.status, 4);
    aotx_service_put(p + 48, o.format, 4); aotx_service_put(p + 52, o.samples, 4);
    aotx_service_put(p + 56, o.rows, 4);
}
static __device__ __forceinline__ int aotx_service_upload_find(const unsigned char *id)
{
    for (unsigned i = 0; aotx_service.uploads && i < aotx_service.media_count; ++i)
        if (aotx_service.uploads[i].state && aotx_service_equal(aotx_service.uploads[i].transfer, id, 16))
            return (int)i;
    return -1;
}
static __device__ __forceinline__ void aotx_service_media_retain(unsigned object)
{
    if (!aotx_service.enabled || aotx_seam.replaying || object >= aotx_media.profile.objects) return;
    const auto &o = aotx_media.objects[object];
    if (o.phase != AOTX_MEDIA_REFUSED) return;
    int at = aotx_service_upload_find(o.transfer); if (at < 0) return;
    auto &u = aotx_service.uploads[at]; if (u.state != 1) return;
    aotx_service_media_fields(u.fields, o); u.state = 2;
    u.deadline = aotx_service.clock + AOTX_SERVICE_UPLOAD_SECONDS * 1000000000ull;
}
#endif
