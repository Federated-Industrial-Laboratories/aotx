/* Purpose: Assemble immutable source records and enforce their storage and scope.
 * Owns: The source table and its byte and feature reservations.
 * Launch shape: The ordered input thread applies records; slot threads resolve sources.
 * Lifetime: One runtime; journal replay uses the same record application. */
#include "media/runtime.cuh"
#include "media/prompt.cuh"
#include "cognitive/live.cuh"

__device__ aotx_media_state aotx_media;
static __device__ bool aotx_media_equal(const unsigned char *a, const unsigned char *b, unsigned n)
{
    for (unsigned i = 0; i < n; ++i) if (a[i] != b[i]) return false;
    return true;
}
static __device__ void aotx_media_copy(unsigned char *a, const unsigned char *b, unsigned n)
{
    for (unsigned i = 0; i < n; ++i) a[i] = b[i];
}
static __device__ bool aotx_media_zero(const unsigned char *p, unsigned n)
{
    for (unsigned i = 0; i < n; ++i) if (p[i]) return false;
    return true;
}
static __device__ void aotx_media_fail(aotx_media_object &o, unsigned status)
{
    if(aotx_media_is_audio(o.format)) {
        if(o.worker!=~0u)aotx_audio_runtime.jobs[o.worker].cancel=1;
    } else {
        if (o.phase == AOTX_MEDIA_DECODE) aotx_media.image[o.worker].cancel = 1;
        if (o.phase == AOTX_MEDIA_ENCODE) aotx_media.vision[o.worker].cancel = 1;
    }
    o.phase = AOTX_MEDIA_REFUSED; o.status = status;
    ++aotx_media.refused;
}
/* A bounded first-fit scan preserves all live addresses. No source moves. */
static __device__ unsigned long long aotx_media_bytes(unsigned long long bytes)
{
    unsigned long long at = 0;
    if (bytes > aotx_media.profile.bytes) return ~0ull;
    for (unsigned pass = 0; pass <= aotx_media.profile.objects; ++pass) {
        unsigned long long next = at;
        for (unsigned i = 0; i < aotx_media.profile.objects; ++i) {
            const aotx_media_object &o = aotx_media.objects[i];
            if (!o.phase || (o.phase == AOTX_MEDIA_REFUSED && o.worker == ~0u)) continue;
            if (at < o.offset + o.bytes && o.offset < at + bytes)
                next = max(next, o.offset + o.bytes);
        }
        if (next == at) return at;
        at = next;
        if (at > aotx_media.profile.bytes - bytes) return ~0ull;
    }
    return ~0ull;
}
__device__ bool aotx_media_part(const unsigned char *p, unsigned n, unsigned long long sequence)
{
    if (n < 24 || aotx_media_get(p, 4) != AOTX_MEDIA_SCHEMA ||
        aotx_media_zero(p + 8, 16)) return false;
    unsigned op = (unsigned)aotx_media_get(p + 4, 4);
    if ((op == AOTX_MEDIA_BEGIN && n != AOTX_MEDIA_BEGIN_BYTES) ||
        (op == AOTX_MEDIA_CHUNK && (n <= AOTX_MEDIA_PART || n > AOTX_BODY_BYTES)) ||
        (op == AOTX_MEDIA_END && n != AOTX_MEDIA_PART) ||
        (op == AOTX_MEDIA_CANCEL && n != 24) || op < AOTX_MEDIA_BEGIN || op > AOTX_MEDIA_CANCEL)
        return false;
    if (!aotx_media.enabled) { ++aotx_media.refused; return true; }
    unsigned index = aotx_media.profile.objects, vacant = index;
    for (unsigned i = 0; i < aotx_media.profile.objects; ++i) {
        const aotx_media_object &o = aotx_media.objects[i];
        if (o.phase && aotx_media_equal(o.transfer, p + 8, 16)) index = i;
        if ((!o.phase || (o.phase == AOTX_MEDIA_REFUSED &&
            o.worker == ~0u)) && vacant == aotx_media.profile.objects) vacant = i;
    }
    if (op == AOTX_MEDIA_BEGIN) {
        if (!aotx_media_zero(p + 32, 8) || !aotx_media_zero(p + 60, 4)) return false;
        if (index != aotx_media.profile.objects || vacant == aotx_media.profile.objects) {
            ++aotx_media.refused; return true;
        }
        unsigned slot = (unsigned)aotx_media_get(p + 40, 4);
        unsigned scope = (unsigned)aotx_media_get(p + 44, 4);
        unsigned format = (unsigned)aotx_media_get(p + 48, 4);
        unsigned width = (unsigned)aotx_media_get(p + 52, 4);
        unsigned height = (unsigned)aotx_media_get(p + 56, 4);
        unsigned long long bytes = aotx_media_get(p + 24, 8);
        if (slot >= AOTX_SLOTS || scope > AOTX_MEDIA_LOCAL || !bytes ||
            (format != AOTX_IMAGE_JPEG && format != AOTX_IMAGE_RGB8 && !aotx_media_is_audio(format)) ||
            (aotx_media_is_audio(format) && (width || height)) ||
            (format == AOTX_IMAGE_JPEG && (width || height)) ||
            (format == AOTX_IMAGE_RGB8 && (!width || !height ||
                bytes < AOTX_MEDIA_RGB_HEAD ||
                (unsigned long long)width * height != (bytes - AOTX_MEDIA_RGB_HEAD) / 3u ||
                (bytes - AOTX_MEDIA_RGB_HEAD) % 3u)) ||
            ((scope == AOTX_MEDIA_LOCAL || scope == AOTX_MEDIA_SHARED) &&
                !aotx_media_zero(p + 64, 32)) ||
            (scope == AOTX_MEDIA_ROOM && !aotx_media_zero(p + 80, 16))) {
            ++aotx_media.refused; return false;
        }
        unsigned long long offset = aotx_media_bytes(bytes);
        aotx_media_object &o = aotx_media.objects[vacant];
        o = {}; o.phase = AOTX_MEDIA_RECEIVE; o.worker = ~0u;
        o.slot = slot; o.scope = scope; o.format = format; o.width = width; o.height = height;
        o.bytes = bytes; o.offset = offset; o.generation = sequence;
        aotx_media_copy(o.transfer, p + 8, 16); aotx_media_copy(o.digest, p + 96, 32);
        aotx_media_copy(o.room, p + 64, 16); aotx_media_copy(o.principal, p + 80, 16);
        aotx_media.hash[vacant] = {};
        if ((aotx_media_is_audio(format) && !aotx_audio_runtime.enabled) ||
            (!aotx_media_is_audio(format) && !aotx_media.image_enabled)) {
            aotx_media_fail(o,AOTX_MEDIA_UNAVAILABLE);return true;
        }
        if (offset == ~0ull) { aotx_media_fail(o, AOTX_MEDIA_LIMIT); return true; }
        ++aotx_media.accepted; return true;
    }
    if (index == aotx_media.profile.objects) { ++aotx_media.refused; return true; }
    aotx_media_object &o = aotx_media.objects[index];
    if (op == AOTX_MEDIA_CANCEL && n == 24) {
        if (o.phase == AOTX_MEDIA_READY && aotx_media_leased(index)) { ++aotx_media.refused; return true; }
        aotx_media.hash[index].active = 0;
        if (o.phase != AOTX_MEDIA_REFUSED) aotx_media_fail(o, AOTX_MEDIA_CANCELLED);
        return true;
    }
    if (o.phase == AOTX_MEDIA_REFUSED) return true;
    if (o.phase != AOTX_MEDIA_RECEIVE) { ++aotx_media.refused; return true; }
    if (aotx_media_get(p + 24, 8) != o.bytes || aotx_media_get(p + 32, 8) != o.received) {
        aotx_media_fail(o, AOTX_MEDIA_INVALID); return true;
    }
    if (op == AOTX_MEDIA_CHUNK && n > AOTX_MEDIA_PART && n <= AOTX_BODY_BYTES &&
        n - AOTX_MEDIA_PART <= o.bytes - o.received) {
        aotx_media_copy(aotx_media.source + o.offset + o.received, p + AOTX_MEDIA_PART,
            n - AOTX_MEDIA_PART);
        o.received += n - AOTX_MEDIA_PART;
        return true;
    }
    if (op == AOTX_MEDIA_END && n == AOTX_MEDIA_PART && o.received == o.bytes) {
        aotx_media_hash &h = aotx_media.hash[index];
        h = {}; h.source = aotx_media.source + o.offset; h.bytes = o.bytes; h.active = 1;
        o.phase = AOTX_MEDIA_HASH; return true;
    }
    aotx_media_fail(o, AOTX_MEDIA_INVALID); return true;
}
__device__ int aotx_media_find(const unsigned char *digest, unsigned slot)
{
    if (!aotx_media.enabled || slot >= AOTX_SLOTS) return -1;
    bool bound = aotx_live_bound(slot);
    const aotx_live_binding &b = aotx_live_bindings[slot];
    int pending = -1;
    for (unsigned i = 0; i < aotx_media.profile.objects; ++i) {
        const aotx_media_object &o = aotx_media.objects[i];
        if (!o.phase || o.phase == AOTX_MEDIA_REFUSED ||
            !aotx_media_equal(o.digest, digest, 32)) continue;
        bool allowed = o.scope == AOTX_MEDIA_SHARED ||
            (o.scope == AOTX_MEDIA_LOCAL && !bound && o.slot == slot) ||
            (bound && (o.scope == AOTX_MEDIA_ROOM || o.scope == AOTX_MEDIA_PRIVATE) &&
                aotx_media_equal(o.room, b.room, 16) &&
                (o.scope == AOTX_MEDIA_ROOM || aotx_media_equal(o.principal, b.principal, 16)));
        if (!allowed) continue;
        if (o.phase == AOTX_MEDIA_READY) return (int)i;
        pending = -2;
    }
    return pending;
}
__device__ bool aotx_media_quiet(void)
{
    if (aotx_media.ring && aotx_media_acquire(&aotx_media.ring->head) != aotx_media.consumed)
        return false;
    for (unsigned i = 0; aotx_media.enabled && i < aotx_media.profile.objects; ++i) {
        const aotx_media_object &o = aotx_media.objects[i];
        if ((o.phase && o.phase != AOTX_MEDIA_READY && o.phase != AOTX_MEDIA_REFUSED) ||
            o.worker != ~0u) return false;
    }
    return true;
}
__device__ void aotx_media_restore_end(void)
{
    for (unsigned i = 0; aotx_media.enabled && i < aotx_media.profile.objects; ++i)
        if (aotx_media.objects[i].phase == AOTX_MEDIA_RECEIVE)
            aotx_media_fail(aotx_media.objects[i], AOTX_MEDIA_CANCELLED);
}
__device__ unsigned aotx_media_window(unsigned long long base, unsigned count)
{
    if (!aotx_seam.replaying) return count;
    for (unsigned i = 0; aotx_media.enabled && i < aotx_media.profile.objects; ++i) {
        unsigned phase = aotx_media.objects[i].phase;
        if (phase >= AOTX_MEDIA_HASH && phase <= AOTX_MEDIA_ENCODE) return 0;
    }
    for (unsigned i = 0; i < count; ++i) {
        const volatile aotx_record_header *h = (const volatile aotx_record_header *)
            (aotx_seam.in.slots + ((base+i) & aotx_seam.in.mask) * AOTX_SLOT_BYTES);
        if (h->cls != AOTX_CLASS_A || h->type != AOTX_REC_MEDIA || h->body_len < 8) continue;
        const volatile unsigned char *p = (const volatile unsigned char *)h + AOTX_HEADER_BYTES;
        if (p[4] == AOTX_MEDIA_END && !p[5] && !p[6] && !p[7]) return i + 1u;
    }
    return count;
}
__device__ bool aotx_media_model_allowed(unsigned role, const unsigned char *digest)
{
    if(aotx_audio_runtime.enabled && role==aotx_audio_runtime.role &&
        !aotx_media_equal(aotx_audio_runtime.parent_digest,digest,32))return false;
    return !aotx_media.image_enabled || role != aotx_media.role ||
        aotx_media_equal(aotx_media.parent_digest, digest, 32);
}
