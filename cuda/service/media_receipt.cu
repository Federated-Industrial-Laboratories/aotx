/* Purpose: Keep an upload refusal available after its source descriptor is reused.
 * Owns: Bounded device transport receipts and their expiry.
 * Launch shape: Ordered source mutations and finite service admission batches.
 * Lifetime: One upload through readiness, explicit removal or receipt expiry. */
#include "service/media_receipt.cuh"

static __device__ unsigned char aotx_service_media_cancel[24];

__device__ int aotx_service_upload_free(void)
{
    for (unsigned i = 0; aotx_service.uploads && i < aotx_service.media_count; ++i)
        if (!aotx_service.uploads[i].state) return (int)i;
    return -1;
}
__device__ void aotx_service_upload_bind(unsigned receipt, unsigned object)
{
    auto &u = aotx_service.uploads[receipt]; const auto &o = aotx_media.objects[object];
    u = {}; u.state = 1;
    aotx_service_bytes(u.transfer, o.transfer, 16); aotx_service_bytes(u.principal, o.principal, 16);
    u.deadline = aotx_service.clock + AOTX_SERVICE_UPLOAD_SECONDS * 1000000000ull;
}
__device__ void aotx_service_upload_result(unsigned channel, unsigned receipt, unsigned status)
{
    unsigned char *f = aotx_service.frames + (unsigned long long)channel * AOTX_SERVICE_FRAME;
    aotx_service_bytes(f + AOTX_SERVICE_HEAD, aotx_service.uploads[receipt].fields, 64);
    aotx_service_answer(channel, status, 0, 64);
}
__device__ void aotx_service_media_expire(void)
{
    if (!aotx_media.enabled || !aotx_service.uploads) return;
    for (unsigned i = 0; i < aotx_service.media_count; ++i) {
        auto &u = aotx_service.uploads[i];
        if (!u.state) continue;
        if (u.state == 2) {
            if (aotx_service.clock >= u.deadline) u = {};
            continue;
        }
        for (unsigned j = 0; j < aotx_media.profile.objects; ++j) {
            const auto &o = aotx_media.objects[j];
            if (!o.phase || !aotx_service_equal(o.transfer, u.transfer, 16)) continue;
            if (o.phase == AOTX_MEDIA_READY) u = {};
            else if (o.phase == AOTX_MEDIA_REFUSED) aotx_service_media_retain(j);
            else if (o.phase == AOTX_MEDIA_RECEIVE && aotx_service.clock >= u.deadline) {
                auto *p = aotx_service_media_cancel;
                aotx_service_put(p, AOTX_MEDIA_SCHEMA, 4); aotx_service_put(p + 4, AOTX_MEDIA_CANCEL, 4);
                aotx_service_bytes(p + 8, o.transfer, 16); aotx_media_publish(p, 24);
                aotx_service_media_retain(j);
            }
            break;
        }
    }
}
