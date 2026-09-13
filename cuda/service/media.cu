/* Purpose: Admit private service media through the canonical device source store.
 * Owns: Scoped transport checks and canonical publication for the service batch.
 * Launch shape: Ordered bounded frames; numerical preparation uses the media graph.
 * Lifetime: Source admission through removal or runtime recovery. */
#include "service/media_receipt.cuh"
static __device__ unsigned char aotx_service_media_body[AOTX_BODY_BYTES];
static __device__ unsigned aotx_service_media_index(const unsigned char *id)
{
    for (unsigned i = 0; aotx_media.enabled && i < aotx_media.profile.objects; ++i)
        if (aotx_media.objects[i].phase && aotx_service_equal(aotx_media.objects[i].transfer, id, 16)) return i;
    return aotx_media.profile.objects;
}
static __device__ void aotx_service_media_result(unsigned channel, unsigned index, unsigned status)
{
    unsigned char *f = aotx_service.frames + (unsigned long long)channel * AOTX_SERVICE_FRAME;
    if (index == aotx_media.profile.objects) { aotx_service_answer(channel, status, 0); return; }
    aotx_service_media_fields(f + AOTX_SERVICE_HEAD, aotx_media.objects[index]);
    aotx_service_answer(channel, status, 0, 64);
}
__device__ void aotx_service_media_list(unsigned channel, const aotx_service_grant *g)
{
    unsigned char *f = aotx_service.frames + (unsigned long long)channel * AOTX_SERVICE_FRAME;
    if (!(g->actions & AOTX_SERVICE_UPLOAD)) { aotx_service_answer(channel, 403, 0); return; }
    if (!aotx_media.enabled) { aotx_service_answer(channel, 503, 0); return; }
    unsigned long long from = aotx_service_get(f + 64, 8);
    if (from > aotx_media.profile.objects) { aotx_service_answer(channel, 409, 0); return; }
    unsigned at = (unsigned)from, bytes = 0;
    for (; at < aotx_media.profile.objects; ++at) {
        const aotx_media_object &o = aotx_media.objects[at];
        if (!o.phase || o.phase == AOTX_MEDIA_REFUSED || o.scope != AOTX_MEDIA_PRIVATE ||
            !aotx_service_equal(o.principal, g->principal, 16)) continue;
        if (AOTX_SERVICE_DATA - bytes < 80) break;
        unsigned char *p = f + AOTX_SERVICE_HEAD + bytes;
        aotx_service_bytes(p, o.transfer, 16); aotx_service_media_fields(p + 16, o); bytes += 80;
    }
    aotx_service_put(f + 64, at == aotx_media.profile.objects ? 0 : at, 8);
    aotx_service_answer(channel, 200, 0, bytes);
}
__device__ void aotx_service_media(unsigned channel, const aotx_service_grant *g, bool read)
{
    unsigned char *f = aotx_service.frames + (unsigned long long)channel * AOTX_SERVICE_FRAME;
    unsigned n = aotx_service_u32(f + 88), index = aotx_service_media_index(f + 48);
    if (!(g->actions & AOTX_SERVICE_UPLOAD)) { aotx_service_answer(channel, 403, 0); return; }
    if (!aotx_media.enabled) { aotx_service_answer(channel, 503, 0); return; }
    if (aotx_media.profile.objects != aotx_service.media_count) { aotx_service_answer(channel, 503, 0); return; }
    bool exists = index < aotx_media.profile.objects;
    bool owned = exists && aotx_media.objects[index].scope == AOTX_MEDIA_PRIVATE &&
        aotx_service_equal(aotx_media.objects[index].principal, g->principal, 16);
    int receipt = aotx_service_upload_find(f + 48), available = -1;
    bool cached = receipt >= 0 && aotx_service.uploads[receipt].state == 2 &&
        aotx_service_equal(aotx_service.uploads[receipt].principal, g->principal, 16);
    if (read) {
        if (n) aotx_service_answer(channel, 400, 0);
        else if (owned) aotx_service_media_result(channel, index, 200);
        else if (cached) aotx_service_upload_result(channel, (unsigned)receipt, 200);
        else aotx_service_answer(channel, 404, 0);
        return;
    }
    const unsigned char *m = f + AOTX_SERVICE_HEAD;
    if (n < AOTX_MEDIA_FRAME_HEAD || n > AOTX_MEDIA_FRAME_HEAD + 8192 ||
        aotx_service_u32(m) != AOTX_MEDIA_SCHEMA || !aotx_service_equal(m + 8, f + 48, 16) ||
        !aotx_service_nonzero(m + 8, 16) || aotx_service_u32(m + 40) != n - AOTX_MEDIA_FRAME_HEAD) {
        aotx_service_answer(channel, 400, 0); return;
    }
    for (unsigned i = 44; i < 64; ++i) if (m[i]) { aotx_service_answer(channel, 400, 0); return; }
    unsigned op = aotx_service_u32(m + 4), payload = n - AOTX_MEDIA_FRAME_HEAD;
    if (exists && !owned) { aotx_service_answer(channel, 404, 0); return; }
    if (receipt >= 0 && !aotx_service_equal(aotx_service.uploads[receipt].principal, g->principal, 16)) {
        aotx_service_answer(channel, 404, 0); return;
    }
    if (op == AOTX_MEDIA_CANCEL && !payload && !exists && cached) {
        aotx_service_upload_result(channel, (unsigned)receipt, 200);
        aotx_service.uploads[receipt] = {}; return;
    }
    if (op != AOTX_MEDIA_BEGIN && !owned) { aotx_service_answer(channel, 404, 0); return; }
    unsigned char *body = aotx_service_media_body;
    for (unsigned i = 0; i < AOTX_BODY_BYTES; ++i) body[i] = 0;
    aotx_service_bytes(body, m, AOTX_MEDIA_PART);
    if (op == AOTX_MEDIA_BEGIN) {
        unsigned long long bytes = aotx_service_get(m + 24, 8), used = 0;
        unsigned count = 0;
        for (unsigned i = 0; i < aotx_media.profile.objects; ++i) {
            const aotx_media_object &o = aotx_media.objects[i];
            if (o.phase && o.phase != AOTX_MEDIA_REFUSED && o.scope == AOTX_MEDIA_PRIVATE &&
                aotx_service_equal(o.principal, g->principal, 16)) { ++count; used += o.bytes; }
        }
        for (unsigned i = 0; i < aotx_service.media_count; ++i) {
            const auto &u = aotx_service.uploads[i];
            if (u.state == 2 && aotx_service_equal(u.principal, g->principal, 16)) ++count;
        }
        if (exists || receipt >= 0) { aotx_service_answer(channel, 409, 0); return; }
        if (payload != 48 || aotx_service_get(m + 32, 8) || !bytes || bytes > ~0ull / 8u ||
            aotx_service_get(m + 68, 8) || aotx_service_u32(m + 76) ||
            (aotx_service_u32(m + 64) != AOTX_IMAGE_JPEG && aotx_service_u32(m + 64) != AOTX_AUDIO_WAV)) {
            aotx_service_answer(channel, 400, 0); return;
        }
        if (count >= g->media || used > g->media_bytes || bytes > g->media_bytes - used) {
            aotx_service_answer(channel, 429, 0); return;
        }
        if (aotx_service_u32(m + 64) == AOTX_AUDIO_WAV ? !aotx_audio_runtime.enabled : !aotx_media.image_enabled) {
            aotx_service_answer(channel, 503, 0); return;
        }
        bool audio = aotx_service_u32(m + 64) == AOTX_AUDIO_WAV;
        unsigned rows = audio ? 1u : aotx_media.profile.patches / 4u;
        unsigned capacity = audio ? aotx_audio_runtime.profile.feature_rows : aotx_media.profile.feature_rows;
        if (rows > capacity || aotx_media_rows(rows, audio) == ~0u) {
            aotx_service_answer(channel, rows > capacity ? 413 : 429, 0); return;
        }
        available = aotx_service_upload_free();
        if (available < 0) { aotx_service_answer(channel, 429, 0); return; }
        aotx_service_put(body + 44, AOTX_MEDIA_PRIVATE, 4);
        aotx_service_bytes(body + 48, m + 64, 16);
        aotx_service_bytes(body + 80, g->principal, 16);
        aotx_service_bytes(body + 96, m + 80, 32);
        aotx_media_publish(body, AOTX_MEDIA_BEGIN_BYTES);
    } else if (op == AOTX_MEDIA_CHUNK && payload) {
        const aotx_media_object &o = aotx_media.objects[index];
        unsigned long long offset = aotx_service_get(m + 32, 8);
        if (o.phase != AOTX_MEDIA_RECEIVE || aotx_service_get(m + 24, 8) != o.bytes ||
            offset != o.received || payload > o.bytes - o.received) { aotx_service_answer(channel, 409, 0); return; }
        for (unsigned at = 0; at < payload;) {
            unsigned take = min(AOTX_MEDIA_DATA, payload - at);
            aotx_service_put(body + 32, offset + at, 8);
            aotx_service_bytes(body + AOTX_MEDIA_PART, m + AOTX_MEDIA_FRAME_HEAD + at, take);
            aotx_media_publish(body, AOTX_MEDIA_PART + take); at += take;
        }
    } else if ((op == AOTX_MEDIA_END || op == AOTX_MEDIA_CANCEL) && !payload) {
        aotx_media_publish(body, op == AOTX_MEDIA_CANCEL ? 24 : AOTX_MEDIA_PART);
    } else { aotx_service_answer(channel, 400, 0); return; }
    index = aotx_service_media_index(f + 48);
    if (op == AOTX_MEDIA_BEGIN && index < aotx_media.profile.objects &&
        aotx_media.objects[index].phase == AOTX_MEDIA_RECEIVE)
        aotx_service_upload_bind((unsigned)available, index);
    unsigned status = index == aotx_media.profile.objects ? 429 : 200;
    if (index < aotx_media.profile.objects && aotx_media.objects[index].phase == AOTX_MEDIA_REFUSED &&
        op != AOTX_MEDIA_CANCEL) {
        unsigned cause = aotx_media.objects[index].status;
        status = cause == AOTX_MEDIA_PRESSURE ? 429 : cause == AOTX_MEDIA_LIMIT ? 413 : 400;
    }
    if (op == AOTX_MEDIA_CANCEL && index < aotx_media.profile.objects &&
        aotx_media.objects[index].phase != AOTX_MEDIA_REFUSED) status = 409;
    if (op == AOTX_MEDIA_CANCEL && status == 200 && receipt >= 0) aotx_service.uploads[receipt] = {};
    aotx_service_media_result(channel, index, status);
}
