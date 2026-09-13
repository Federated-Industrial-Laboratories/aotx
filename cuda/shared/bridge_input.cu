/* Purpose: Validate shared input and freeze its model and media dependencies.
 * Owns: No host policy; device grants and current model placement bound admission.
 * Launch shape: One ordered command batch.
 * Lifetime: Admission through the recorded execution result. */
#include "shared/bridge.cuh"
#include "shared/internal.cuh"
#include "model/load.cuh"
#include "model/wrap.cuh"
#include "model/decode.cuh"
#include "media/runtime.cuh"
#include "media/prompt.cuh"
#include <math.h>
static __device__ bool aotx_shared_text(const unsigned char *p, unsigned n)
{
    for (unsigned i = 0; i < n;) {
        unsigned c = p[i++], follow = 0, point = c, least = 0;
        if (c >= 0xc2 && c <= 0xdf) { follow = 1; point = c & 31u; least = 0x80; }
        else if (c >= 0xe0 && c <= 0xef) { follow = 2; point = c & 15u; least = 0x800; }
        else if (c >= 0xf0 && c <= 0xf4) { follow = 3; point = c & 7u; least = 0x10000; }
        else if (c >= 0x80 || !c) return false;
        if (follow > n - i) return false;
        for (unsigned j = 0; j < follow; ++j) {
            c = p[i++]; if ((c & 0xc0u) != 0x80u) return false;
            point = (point << 6) | (c & 63u);
        }
        if (point < least || point > 0x10ffff || (point >= 0xd800 && point <= 0xdfff)) return false;
    }
    for (unsigned i = 0; i + 7 <= n; ++i)
        if (aotx_service_equal(p + i, (const unsigned char *)"[image:", 7) ||
            aotx_service_equal(p + i, (const unsigned char *)"[audio:", 7) ||
            aotx_media_reserved(p + i, n - i)) return false;
    return true;
}
__device__ unsigned aotx_shared_input_text(const aotx_shared_receipt *r, unsigned char *out, unsigned cap)
{
    unsigned text = aotx_shared_u32(r->command + 136), count = aotx_shared_u32(r->command + 140);
    if (text > cap || count > (cap - text) / 73u) return cap + 1u;
    if (out) aotx_service_bytes(out, r->command + AOTX_SHARED_COMMAND_HEAD, text);
    unsigned at = text;
    const unsigned char *media = r->command + AOTX_SHARED_COMMAND_HEAD + text;
    for (unsigned i = 0; i < count; ++i) {
        const unsigned char *row = media + i * AOTX_SHARED_MEDIA_ROW;
        if (out) {
            out[at] = '\n';
            const char *head = aotx_shared_u32(row) == 2 ? "[audio:" : "[image:";
            aotx_service_bytes(out + at + 1, (const unsigned char *)head, 7);
            const char *hex = "0123456789abcdef";
            for (unsigned j = 0; j < 32; ++j) {
                out[at + 8 + 2*j] = hex[row[8+j] >> 4];
                out[at + 9 + 2*j] = hex[row[8+j] & 15];
            }
            out[at + 72] = ']';
        }
        at += 73;
    }
    return at;
}
__device__ unsigned aotx_shared_input_check(aotx_shared_receipt *r, const aotx_service_grant *g)
{
    const unsigned char *p = r->command;
    r->role = aotx_shared_u32(p + 104); r->limit = aotx_shared_u32(p + 108);
    r->pages = aotx_shared_u32(p + 112) ? aotx_shared_u32(p + 112) : g->pages; r->media_count = aotx_shared_u32(p + 140);
    unsigned text = aotx_shared_u32(p + 136);
    if (r->role >= AOTX_MODEL_ROLES || !(g->models & (1u << r->role)) ||
        !aotx_model_is_language(r->role) || !aotx_model_load.resident[r->role].active ||
        !aotx_model_wrap[r->role].usable) return 404;
    if (!r->limit || r->limit > g->tokens || r->limit >= AOTX_SEQ_MAX_TOKENS ||
        !r->pages || r->pages > g->pages || r->media_count > AOTX_SHARED_MEDIA_REFS) return 400;
    if (!aotx_live.ready || aotx_live.fatal) return 503;
    if (aotx_shared_input_text(r, 0, AOTX_RECALL_TEXT) > AOTX_RECALL_TEXT) return 413;
    if ((!text && !r->media_count) || !aotx_shared_text(p + AOTX_SHARED_COMMAND_HEAD, text)) return 400;
    float temperature = __uint_as_float(aotx_shared_u32(p + 120));
    float top_p = __uint_as_float(aotx_shared_u32(p + 124));
    if (!isfinite(temperature) || temperature < 0 || temperature > 2 ||
        !isfinite(top_p) || top_p <= 0 || top_p > 1) return 400;
    r->sample = {}; r->sample.temperature = temperature; r->sample.top_p = top_p;
    r->sample.repeat_penalty = 1; r->sample.voice = ~0u;
    for (unsigned i = 0; i < AOTX_MODEL_STEERS; ++i) r->sample.steer[i] = ~0u;
    aotx_service_bytes(r->model_digest, aotx_model_load.resident[r->role].body.digest, 32);
    const unsigned char *media = p + AOTX_SHARED_COMMAND_HEAD + text;
    for (unsigned i = 0; i < r->media_count; ++i) {
        const unsigned char *row = media + i * AOTX_SHARED_MEDIA_ROW;
        unsigned kind = aotx_shared_u32(row);
        if ((kind != 1 && kind != 2) || aotx_shared_u32(row + 4)) return 400;
        bool audio = kind == 2;
        if (!aotx_media.enabled || (audio ? !aotx_audio_runtime.enabled || r->role != aotx_audio_runtime.role
            : !aotx_media.image_enabled || r->role != aotx_media.role)) return 400;
        unsigned index = aotx_media.profile.objects;
        for (unsigned j = 0; j < aotx_media.profile.objects; ++j) {
            const aotx_media_object &o = aotx_media.objects[j];
            if (o.phase == AOTX_MEDIA_READY && o.scope == AOTX_MEDIA_PRIVATE &&
                aotx_shared_id(o.principal, r->actor) && aotx_service_equal(o.digest, row + 8, 32) &&
                aotx_media_is_audio(o.format) == audio) { index = j; break; }
        }
        if (index == aotx_media.profile.objects) return 404;
        r->media[i] = {index, aotx_media.objects[index].generation};
    }
    return 200;
}
__device__ bool aotx_shared_media_leased(unsigned object, unsigned long long generation)
{
    if (!aotx_shared.enabled) return false;
    if (aotx_shared.kind == AOTX_SHARED_ADMIT_RECORD && aotx_shared_candidate.operation == AOTX_SHARED_INPUT)
        for (unsigned m = 0; m < aotx_shared_candidate.media_count; ++m)
            if (aotx_shared_candidate.media[m].object == object &&
                aotx_shared_candidate.media[m].generation == generation) return true;
    for (unsigned i = 0; i < aotx_shared.receipt_capacity; ++i) {
        const aotx_shared_receipt &r = aotx_shared.receipts[i];
        if (!r.phase || r.phase >= AOTX_SHARED_DONE || r.operation != AOTX_SHARED_INPUT) continue;
        for (unsigned m = 0; m < r.media_count; ++m)
            if (r.media[m].object == object && r.media[m].generation == generation) return true;
    }
    return false;
}
