/* Purpose: Render complete client message arrays with the loaded model wrap.
 * Owns: Bounded request text; no agent overlay or retained conversation is appended.
 * Launch shape: The admission batch renders each accepted request on the device.
 * Lifetime: Admission through execution. */
#include "service/internal.cuh"
#include "model/wrap.cuh"
#include "model/load.cuh"
#include "model/selection.cuh"
#include "media/runtime.cuh"
#include <math.h>
#include <stddef.h>
static __device__ aotx_service_job aotx_service_candidate;

static __device__ bool aotx_service_text(const unsigned char *p, unsigned n)
{
    for (unsigned i = 0; i < n;) {
        unsigned c = p[i++], follow = 0, point = c, least = 0;
        if (c >= 0xc2 && c <= 0xdf) { follow = 1; point = c & 31u; least = 0x80; }
        else if (c >= 0xe0 && c <= 0xef) { follow = 2; point = c & 15u; least = 0x800; }
        else if (c >= 0xf0 && c <= 0xf4) { follow = 3; point = c & 7u; least = 0x10000; }
        else if (c >= 0x80) return false;
        if (follow > n - i) return false;
        for (unsigned j = 0; j < follow; ++j) {
            c = p[i++]; if ((c & 0xc0u) != 0x80u) return false;
            point = (point << 6) | (c & 63u);
        }
        if (point < least || point > 0x10ffff || (point >= 0xd800 && point <= 0xdfff)) return false;
    }
    return true;
}
__device__ unsigned aotx_service_render(aotx_service_job *job, const unsigned char *p, unsigned n)
{
    if (n < 4) return 400;
    unsigned messages = aotx_service_u32(p), from = 4, at = 0;
    job->media_count = 0;
    if (!messages || messages > 64) return 400;
    const aotx_wrap *wrap = aotx_wrap_active(job->role);
    at = aotx_wrap_prefix(job->text, at, AOTX_SAY_BYTES, wrap);
    for (unsigned i = 0; i < messages; ++i) {
        if (from > n || n - from < 8) return 400;
        unsigned role = aotx_service_u32(p + from), parts = aotx_service_u32(p + from + 4);
        from += 8;
        if (role > 2 || !parts || parts > 32) return 400;
        unsigned head = role * 2, skip = role == 0 ? wrap->prefix_length : 0;
        at = aotx_wrap_run(job->text, at, AOTX_SAY_BYTES,
            wrap->bytes + wrap->offset[head] + skip, wrap->length[head] - skip);
        for (unsigned part = 0; part < parts; ++part) {
            if (from > n || n - from < 8) return 400;
            unsigned kind = aotx_service_u32(p + from), bytes = aotx_service_u32(p + from + 4);
            from += 8;
            if (bytes > n - from) return 400;
            if (!kind) {
                if (!aotx_service_text(p + from, bytes)) return 400;
                unsigned begin = at > 6 ? at - 6 : 0;
                at = aotx_wrap_run(job->text, at, AOTX_SAY_BYTES, p + from, bytes);
                if (at > AOTX_SAY_BYTES) return 413;
                for (unsigned j = begin; j + 7 <= at; ++j) {
                    if (aotx_service_equal(job->text + j, (const unsigned char *)"[image:", 7) ||
                        aotx_service_equal(job->text + j, (const unsigned char *)"[audio:", 7)) return 400;
                }
            } else if ((kind == 1 || kind == 2) && role == 1 && bytes == 32) {
                bool audio = kind == 2;
                if (!aotx_media.enabled || (audio ? !aotx_audio_runtime.enabled || job->role != aotx_audio_runtime.role
                    : !aotx_media.image_enabled || job->role != aotx_media.role)) return 400;
                unsigned index = aotx_media.profile.objects;
                for (unsigned j = 0; j < aotx_media.profile.objects; ++j) {
                    const aotx_media_object &o = aotx_media.objects[j];
                    if (o.phase == AOTX_MEDIA_READY && o.scope == AOTX_MEDIA_PRIVATE &&
                        aotx_service_equal(o.principal, job->principal, 16) &&
                        aotx_service_equal(o.digest, p + from, 32) && aotx_media_is_audio(o.format) == audio) { index = j; break; }
                }
                if (index == aotx_media.profile.objects) return 404;
                if (job->media_count >= AOTX_MEDIA_REFS) return 413;
                job->media[job->media_count++] = {index, aotx_media.objects[index].generation};
                const char *prefix = audio ? "[audio:" : "[image:";
                at = aotx_wrap_run(job->text, at, AOTX_SAY_BYTES, (const unsigned char *)prefix, 7);
                const char *hex = "0123456789abcdef";
                if (at > AOTX_SAY_BYTES || AOTX_SAY_BYTES - at < 65) return 413;
                for (unsigned j = 0; j < 32; ++j) {
                    job->text[at++] = hex[p[from + j] >> 4]; job->text[at++] = hex[p[from + j] & 15];
                }
                job->text[at++] = ']';
            } else return 400;
            from += bytes;
            if (at > AOTX_SAY_BYTES) return 413;
        }
        at = aotx_wrap_put(job->text, at, AOTX_SAY_BYTES, wrap, head + 1);
    }
    if (from != n) return 400;
    at = aotx_wrap_generation(job->text, at, AOTX_SAY_BYTES, wrap);
    if (at > AOTX_SAY_BYTES) return 413;
    job->length = at; return 200;
}
__device__ unsigned aotx_service_submit(const aotx_service_grant *g, unsigned char *f)
{
    unsigned role = aotx_service_u32(f + 72), limit = aotx_service_u32(f + 76);
    if (aotx_service_get(f + 40, 8) != aotx_service.epoch) return 410;
    if (!aotx_service_nonzero(f + 48, 16) || aotx_service_get(f + 64, 8)) return 400;
    if (role >= AOTX_MODEL_ROLES || !(g->models & (1u << role)) || !aotx_model_is_language(role) ||
        !aotx_model_load.resident[role].active || !aotx_model_wrap[role].usable) return 404;
    if (!limit || limit > g->tokens || limit >= AOTX_SEQ_MAX_TOKENS) return 400;
    float temperature = __uint_as_float(aotx_service_u32(f + 80));
    float top_p = __uint_as_float(aotx_service_u32(f + 84));
    if (!isfinite(temperature) || temperature < 0 || temperature > 2 || !isfinite(top_p) || top_p <= 0 || top_p > 1) return 400;
    if (aotx_service_request(f + 48, g->principal)) return 409;
    unsigned active = 0, index = AOTX_SERVICE_REQUESTS;
    unsigned long long oldest = ~0ull;
    for (unsigned i = 0; i < AOTX_SERVICE_REQUESTS; ++i) {
        const aotx_service_job &j = aotx_service.jobs[i];
        if (j.phase && j.phase < AOTX_SERVICE_DONE && aotx_service_equal(j.principal, g->principal, 16)) ++active;
        if (!j.phase) { if (oldest) { index = i; oldest = 0; } }
        else if (j.phase >= AOTX_SERVICE_DONE && j.slot == AOTX_SLOTS && j.changed < oldest) { index = i; oldest = j.changed; }
    }
    if (active >= g->requests || index == AOTX_SERVICE_REQUESTS) return 429;
    aotx_service_job *j = &aotx_service_candidate;
    j->phase = AOTX_SERVICE_FREE; j->slot = AOTX_SLOTS; j->role = role;
    aotx_service_bytes(j->principal, g->principal, 16);
    unsigned bytes = aotx_service_u32(f + 88), control = aotx_service_u32(f + 92);
    if ((control && control != AOTX_CONTROL_SELECTION_BYTES) || control > bytes) return 400;
    for (unsigned i = 0; i < AOTX_CONTROL_SELECTION_BYTES; ++i)
        j->control[i] = control ? f[AOTX_SERVICE_HEAD + bytes - control + i] : 0;
    unsigned status = aotx_service_render(j, f + AOTX_SERVICE_HEAD, bytes - control);
    if (status != 200) return status;
    aotx_service_bytes(j->id, f + 48, 16); j->revision = g->revision;
    aotx_service_bytes(j->model_digest, aotx_model_load.resident[role].body.digest, 32);
    j->limit = limit; j->pages = g->pages; j->opened = j->changed = aotx_service.clock;
    j->status = j->output = j->prompt = j->sampled = j->finish = j->cancel = 0;
    j->sample = {}; j->sample.temperature = temperature; j->sample.top_p = top_p;
    j->sample.repeat_penalty = 1; j->sample.think_limit = 0; j->sample.voice = ~0u;
    for (unsigned i = 0; i < AOTX_MODEL_STEERS; ++i) j->sample.steer[i] = ~0u;
    status = aotx_control_select(j->control, role, &j->sample);
    if (status != 200) return status;
    aotx_service_put(f + 92, 0, 4);
    j->phase = AOTX_SERVICE_QUEUED;
    aotx_service_bytes((unsigned char *)(aotx_service.jobs + index), (const unsigned char *)j,
        (unsigned)offsetof(aotx_service_job, result));
    return 202;
}
__device__ bool aotx_service_media_leased(unsigned object, unsigned long long generation)
{
    if (!aotx_service.enabled) return false;
    for (unsigned i = 0; i < AOTX_SERVICE_REQUESTS; ++i) {
        const aotx_service_job &j = aotx_service.jobs[i];
        if (!j.phase || j.phase >= AOTX_SERVICE_DONE) continue;
        for (unsigned m = 0; m < j.media_count; ++m)
            if (j.media[m].object == object && j.media[m].generation == generation) return true;
    }
    return false;
}
