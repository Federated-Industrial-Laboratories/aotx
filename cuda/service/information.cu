/* Purpose: Report granted model capabilities and explicitly scoped device counters.
 * Owns: No cognitive state or host sensor collection.
 * Launch shape: Bounded information requests in the admission batch.
 * Lifetime: One response; counters name the current runtime epoch. */
#include "service/internal.cuh"
#include "model/load.cuh"
#include "model/wrap.cuh"
#include "media/runtime.cuh"
#include "cognitive/live.cuh"
#include "shared/state.cuh"
#include "cognitive/intake_capability.cuh"
#include "model/conduct.cuh"
#include "model/control.cuh"

static __device__ unsigned controls(unsigned char *out, const aotx_service_grant *g) {
    unsigned role = aotx_model_default_language(), count = 0;
    if (!(g->models & (1u << role)) || !aotx_model_load.resident[role].active || !aotx_model_wrap[role].usable) return 0;
    for (unsigned i = 0; i < aotx_conduct.vectors; ++i) {
        const aotx_steer_vector *v = aotx_conduct.vector + i;
        unsigned char *p = out + 160 * count++;
        for (unsigned j = 0; j < 160; ++j) p[j] = 0;
        bool accepted = v->permit.status == AOTX_QUALIFICATION_ACCEPTED && aotx_control_matches(&v->identity, role);
        aotx_service_put(p, role, 4); aotx_service_put(p + 4, AOTX_CONTROL_VECTOR, 4);
        aotx_service_put(p + 8, accepted, 4); aotx_service_put(p + 12, v->positions, 4);
        aotx_service_put(p + 16, AOTX_CONTROL_HOOK, 4);
        aotx_service_put(p + 20, accepted ? v->permit.count : 0, 4);
        aotx_service_put(p + 24, v->layers, 8);
        aotx_service_bytes(p + 32, (const unsigned char *)v->name, AOTX_CONDUCT_NAME_BYTES);
        aotx_service_bytes(p + 64, v->permit.digest, 32);
        for (unsigned j = 0; accepted && j < v->permit.count; ++j)
            aotx_service_put(p + 96 + 4 * j, (unsigned)v->permit.dose[j], 4);
    }
    return count;
}
__device__ void aotx_service_information(unsigned channel, const aotx_service_grant *g, bool telemetry)
{
    unsigned char *f = aotx_service.frames + (unsigned long long)channel * AOTX_SERVICE_FRAME;
    if (aotx_service_u32(f + 88)) { aotx_service_answer(channel, 400, 0); return; }
    if (telemetry && !(g->actions & AOTX_SERVICE_TELEMETRY)) { aotx_service_answer(channel, 403, 0); return; }
    unsigned char *p = f + AOTX_SERVICE_HEAD;
    for (unsigned i = 0; i < 192 + AOTX_MODEL_ROLES * 40; ++i) p[i] = 0;
    const unsigned values[] = {2, AOTX_SLOTS, AOTX_SAY_BYTES, AOTX_SEQ_MAX_TOKENS,
        AOTX_SERVICE_REQUESTS, AOTX_SERVICE_OUTPUT_BYTES, AOTX_SERVICE_CHANNELS,
        g->tokens, g->pages, g->requests, g->actions, 0,
        aotx_media.enabled ? aotx_media.profile.objects : 0, aotx_media.image_enabled,
        aotx_audio_runtime.enabled, g->media};
    for (unsigned i = 0; i < 16; ++i) aotx_service_put(p + i * 4, values[i], 4);
    aotx_service_put(p + 64, min(g->media_bytes, aotx_media.enabled ? aotx_media.profile.bytes : 0ull), 8);
    if (telemetry) {
        aotx_service_put(p + 72, aotx_service.bytes, 8);
        aotx_service_put(p + 80, aotx_time_tick, 8);
        aotx_service_put(p + 88, aotx_service.clock, 8);
    }
    if (aotx_live.ready) aotx_service_bytes(p + 96, aotx_live_store.lineage, 16);
    if (aotx_media.enabled) {
        aotx_service_put(p + 112, aotx_media.profile.pixels, 4);
        aotx_service_put(p + 116, aotx_media.profile.dimension, 4);
        aotx_service_put(p + 120, aotx_media.profile.patches, 4);
        aotx_service_put(p + 124, aotx_media.profile.feature_rows, 4);
    }
    if (aotx_audio_runtime.enabled) {
        aotx_service_put(p + 128, aotx_audio_runtime.profile.source_frames, 4);
        aotx_service_put(p + 132, aotx_audio_runtime.profile.feature_rows, 4);
    }
    aotx_service_put(p + 136, AOTX_SERVICE_REQUEST_SECONDS, 4);
    aotx_service_put(p + 140, AOTX_SERVICE_UPLOAD_SECONDS, 4);
    aotx_service_put(p + 144, AOTX_SERVICE_PRINCIPALS, 4);
    aotx_service_put(p + 148, AOTX_MEDIA_REFS, 4);
    aotx_service_put(p + 152, aotx_shared.enabled && !aotx_shared.fatal &&
        (g->actions & (AOTX_SHARED_READ_ACTION | AOTX_SHARED_WRITE_ACTION)), 4);
    unsigned count = 0, memory = 0;
    for (unsigned role = 0; role < AOTX_MODEL_ROLES; ++role) {
        if (!aotx_model_is_language(role) || !(g->models & (1u << role)) ||
            !aotx_model_load.resident[role].active || !aotx_model_wrap[role].usable) continue;
        unsigned char *r = p + 192 + count++ * 40;
        unsigned modalities = 1u;
        if (aotx_media.image_enabled && aotx_media.role == role) modalities |= 2u;
        if (aotx_audio_runtime.enabled && aotx_audio_runtime.role == role) modalities |= 4u;
        aotx_service_put(r, role, 4); aotx_service_put(r + 4, modalities, 4);
        aotx_service_bytes(r + 8, aotx_model_load.resident[role].body.digest, 32);
        if (aotx_intake_qualified(role)) memory |= 1u << role;
    }
    aotx_service_put(p + 44, count, 4);
    aotx_service_put(p + 156, memory, 4);
    unsigned control_count = controls(p + 192 + count * 40, g);
    aotx_service_put(p + 160, control_count, 4); aotx_service_put(p + 164, 160, 4);
    aotx_service_put(p + 168, 1, 4);
    aotx_service_answer(channel, 200, 0, 192 + count * 40 + control_count * 160);
}
