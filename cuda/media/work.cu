/* Purpose: Schedule bounded codec and vision work over resident image sources.
 * Owns: Workspace assignments; immutable source and feature addresses do not move.
 * Launch shape: One initializer thread per workspace and one ordered scheduler thread.
 * Lifetime: One runtime allocation. */
#include "media/runtime.cuh"
#include "sched/sched.cuh"
#include "cli/cli.cuh"

static __device__ unsigned char *aotx_media_take(unsigned char *&at, unsigned long long n)
{
    unsigned char *out = at; at += (n + 255u) & ~255ull; return out;
}
__global__ void aotx_media_initialize(void)
{
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    for (unsigned j = i; j < aotx_media.profile.objects; j += gridDim.x * blockDim.x)
        aotx_media.objects[j].worker = ~0u;
    if (!aotx_media.image_enabled || i >= aotx_media.profile.workers) return;
    aotx_media.owner[i] = ~0u;
    aotx_image_job &d = aotx_media.image[i];
    aotx_vision_job &v = aotx_media.vision[i];
    unsigned char *p = aotx_media.workspace + i * aotx_media.workspace_each;
    unsigned long long coefficients = aotx_media.coefficient_count;
    d.coefficients = (int32_t *)aotx_media_take(p, coefficients * 4u);
    d.planes = aotx_media_take(p, coefficients);
    d.rgb = aotx_media_take(p, (unsigned long long)aotx_media.profile.pixels * 3u);
    d.coefficient_count = coefficients; d.plane_bytes = coefficients;
    d.rgb_bytes = (unsigned long long)aotx_media.profile.pixels * 3u;
    d.pixel_limit = aotx_media.profile.pixels; d.dimension_limit = aotx_media.profile.dimension;
    d.phase = AOTX_IMAGE_REFUSED;
    unsigned patches = aotx_media.profile.patches;
    v.horizontal = (float *)aotx_media_take(p, aotx_media.profile.horizontal * 4u);
    v.rgb = aotx_media_take(p, (unsigned long long)patches * 256u * 3u);
    v.residual = (float *)aotx_media_take(p, (unsigned long long)patches * 768u * 4u);
    v.product = (float *)aotx_media_take(p, (unsigned long long)patches * 3072u * 4u);
    v.input = (half *)aotx_media_take(p, (unsigned long long)patches * 3072u * 2u);
    v.input_low = (half *)aotx_media_take(p, (unsigned long long)patches * 3072u * 2u);
    v.qkv = (float *)aotx_media_take(p, (unsigned long long)patches * 2304u * 4u);
    v.horizontal_values = aotx_media.profile.horizontal;
    v.rgb_bytes = (unsigned long long)patches * 256u * 3u;
    v.max_pixels = min(AOTX_VISION_MAX_PIXELS, patches * 256u);
    v.patch_capacity = patches; v.phase = AOTX_VISION_REFUSED;
}
__device__ unsigned aotx_media_rows(unsigned count,bool audio)
{
    unsigned at = 0;
    unsigned capacity=audio?aotx_audio_runtime.profile.feature_rows:aotx_media.profile.feature_rows;
    if (count > capacity) return ~0u;
    for (unsigned pass = 0; pass <= aotx_media.profile.objects; ++pass) {
        unsigned next = at;
        for (unsigned i = 0; i < aotx_media.profile.objects; ++i) {
            const aotx_media_object &o = aotx_media.objects[i];
            if (aotx_media_is_audio(o.format)!=audio || !o.span || (o.phase == AOTX_MEDIA_REFUSED && o.worker == ~0u)) continue;
            if (at < o.feature + o.span && o.feature < at + count)
                next = max(next, o.feature + o.span);
        }
        if (next == at) return at;
        at = next;
        if (at > capacity - count) return ~0u;
    }
    return ~0u;
}
__global__ void aotx_media_schedule(void)
{
    if (!aotx_media.enabled || aotx_sched.held) return;
    for (unsigned i = 0; i < aotx_media.profile.objects; ++i) {
        aotx_media_object &o = aotx_media.objects[i];
        aotx_media_hash &h = aotx_media.hash[i];
        if (o.phase != AOTX_MEDIA_HASH || !h.done) continue;
        h.active = 0;
        bool equal = !h.status;
        for (unsigned k = 0; k < 32; ++k) equal &= o.digest[k] == h.digest[k];
        if (o.format == AOTX_IMAGE_RGB8) {
            const unsigned char *p = aotx_media.source + o.offset;
            const char *magic = "AOTXRGB1";
            for (unsigned k = 0; k < 8; ++k) equal &= p[k] == (unsigned char)magic[k];
            equal &= aotx_media_get(p + 8, 4) == o.width && aotx_media_get(p + 12, 4) == o.height &&
                aotx_media_get(p + 16, 8) == o.bytes - AOTX_MEDIA_RGB_HEAD;
        }
        o.phase = equal ? AOTX_MEDIA_WAIT : AOTX_MEDIA_REFUSED;
        o.status = equal ? 0 : AOTX_MEDIA_DIGEST;
        if (!equal) ++aotx_media.refused;
    }
    for (unsigned w = 0; aotx_media.image_enabled && w < aotx_media.profile.workers; ++w) {
        if (aotx_media.owner[w] != ~0u) continue;
        unsigned next = ~0u;
        for (unsigned i = 0; i < aotx_media.profile.objects; ++i) {
            const aotx_media_object &o = aotx_media.objects[i];
            if (o.phase == AOTX_MEDIA_WAIT && !aotx_media_is_audio(o.format) &&
                (next == ~0u || o.generation < aotx_media.objects[next].generation)) next = i;
        }
        if (next == ~0u) break;
        aotx_media_object &o = aotx_media.objects[next];
        unsigned rows = aotx_media.profile.patches / 4u, feature = aotx_media_rows(rows);
        if (feature == ~0u) {
            o.phase = AOTX_MEDIA_REFUSED;
            o.status = rows > aotx_media.profile.feature_rows ? AOTX_MEDIA_LIMIT : AOTX_MEDIA_PRESSURE;
            ++aotx_media.refused; continue;
        }
        o.feature = feature; o.span = rows; o.worker = w; o.phase = AOTX_MEDIA_DECODE;
        aotx_media.owner[w] = next;
        aotx_image_job &d = aotx_media.image[w];
        d.source = aotx_media.source + o.offset; d.bytes = o.bytes;
        if (o.format == AOTX_IMAGE_RGB8) { d.source += AOTX_MEDIA_RGB_HEAD; d.bytes -= AOTX_MEDIA_RGB_HEAD; }
        d.format = o.format; d.width = o.width; d.height = o.height;
        d.phase = AOTX_IMAGE_NEW; d.status = d.cancel = 0;
        aotx_vision_job &v = aotx_media.vision[w];
        v.phase = AOTX_VISION_REFUSED; v.status = v.cancel = 0;
        v.features = aotx_media.features + (unsigned long long)feature * AOTX_VISION_OUTPUT;
        v.feature_capacity = rows;
    }
}
__global__ void aotx_media_complete(void)
{
    if (!aotx_media.enabled) return;
    for (unsigned w = 0; aotx_media.image_enabled && w < aotx_media.profile.workers; ++w) {
        unsigned i = aotx_media.owner[w];
        if (i == ~0u) continue;
        aotx_media_object &o = aotx_media.objects[i];
        aotx_image_job &d = aotx_media.image[w];
        aotx_vision_job &v = aotx_media.vision[w];
        if (o.phase == AOTX_MEDIA_DECODE && d.phase == AOTX_IMAGE_READY) {
            o.width = d.width; o.height = d.height; o.phase = AOTX_MEDIA_ENCODE;
            v.source = d.rgb; v.source_bytes = (unsigned long long)d.width * d.height * 3u;
            v.width = d.width; v.height = d.height; v.phase = AOTX_VISION_NEW;
        } else if (o.phase == AOTX_MEDIA_DECODE && d.phase == AOTX_IMAGE_REFUSED) {
            o.phase = AOTX_MEDIA_REFUSED; o.status = AOTX_MEDIA_CODEC; ++aotx_media.refused;
        } else if (o.phase == AOTX_MEDIA_ENCODE && v.phase == AOTX_VISION_READY) {
            o.phase = AOTX_MEDIA_READY; o.rows = v.rows; o.span = v.rows;
            o.columns = v.resized_width / 32u; o.lines = v.resized_height / 32u;
        } else if (o.phase == AOTX_MEDIA_ENCODE && v.phase == AOTX_VISION_REFUSED) {
            o.phase = AOTX_MEDIA_REFUSED; o.status = AOTX_MEDIA_ENCODER; ++aotx_media.refused;
        }
        if (o.phase == AOTX_MEDIA_REFUSED || o.phase == AOTX_MEDIA_READY) {
            o.worker = ~0u; aotx_media.owner[w] = ~0u;
            d.phase = AOTX_IMAGE_REFUSED; v.phase = AOTX_VISION_REFUSED;
        }
    }
    if (!aotx_seam.replaying && !aotx_sched.held) {
        for (unsigned i=0;i<aotx_media.profile.objects;++i) {
            aotx_media_object &o=aotx_media.objects[i];
            if (o.notified || (o.phase!=AOTX_MEDIA_READY && o.phase!=AOTX_MEDIA_REFUSED)) continue;
            aotx_media_report(i,o.status,AOTX_MEDIA_END);o.notified=1;break;
        }
    }
}
