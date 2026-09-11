/* Purpose: Define bounded image producer frames and canonical source records.
 * Owns: Portable byte offsets and the single-producer transport preamble.
 * Threading: The producer releases complete frames; CUDA releases copied frames.
 * Lifetime: One runtime; canonical source bytes survive journal recovery. */
#ifndef AOTX_MEDIA_WIRE_H
#define AOTX_MEDIA_WIRE_H
#include <stdint.h>
#define AOTX_MEDIA_SCHEMA 1u
#define AOTX_MEDIA_BEGIN 1u
#define AOTX_MEDIA_CHUNK 2u
#define AOTX_MEDIA_END 3u
#define AOTX_MEDIA_CANCEL 4u
#define AOTX_MEDIA_PRIVATE 0u
#define AOTX_MEDIA_ROOM 1u
#define AOTX_MEDIA_SHARED 2u
#define AOTX_MEDIA_LOCAL 3u
#define AOTX_MEDIA_PART 40u
#define AOTX_MEDIA_DATA 152u
#define AOTX_MEDIA_BEGIN_BYTES 128u
#define AOTX_MEDIA_RGB_HEAD 24u
/* RGB8 source bytes start with AOTXRGB1, width/height at 8/12 and pixel bytes at 16.
 * The source digest covers this header and all pixel bytes, so dimensions bind the identity. */
#define AOTX_MEDIA_EMIT 64u
#define AOTX_MEDIA_FRAME_BYTES 65536u
#define AOTX_MEDIA_FRAME_HEAD 64u
#define AOTX_MEDIA_FRAME_DATA (AOTX_MEDIA_FRAME_BYTES - AOTX_MEDIA_FRAME_HEAD)
#define AOTX_MEDIA_RING_SLOTS 16u
#define AOTX_MEDIA_RING_MAGIC 0x31444d41u
/* Canonical record: schema/op at 0/4, transfer ID at 8, total bytes at 24,
 * source offset at 32. A chunk has source bytes at 40. End has no data.
 * Begin: slot/scope/format/width/height at 40/44/48/52/56, zero at 60,
 * room/principal at 64/80, complete source SHA-256 at 96.
 * Cancel has schema/op/transfer only, in 24 bytes.
 *
 * Transport frame: schema/op at 0/4, transfer ID at 8, source bytes/offset at 24/32.
 * Payload bytes are at 40, owner slot/scope at 44/48, zero at 52..63, payload at 64.
 * Begin payload: format/width/height/zero at 0/4/8/12 and SHA-256 at 16.
 * The device supplies canonical scope from the addressed slot's binding. */
typedef struct aotx_media_preamble {
    uint32_t magic, schema, slots, frame_bytes;
    uint64_t head, consumed, closed;
    uint64_t status, reserved[2];
} aotx_media_preamble;

#ifdef __CUDACC__
#define AOTX_MEDIA_INLINE __host__ __device__ __forceinline__
#else
#define AOTX_MEDIA_INLINE static inline
#endif
AOTX_MEDIA_INLINE uint64_t aotx_media_get(const unsigned char *p, unsigned n)
{
    uint64_t value = 0;
    for (unsigned i = 0; i < n; ++i) value |= (uint64_t)p[i] << (i*8u);
    return value;
}
AOTX_MEDIA_INLINE void aotx_media_put(unsigned char *p, uint64_t value, unsigned n)
{
    for (unsigned i = 0; i < n; ++i) p[i] = (unsigned char)(value >> (i*8u));
}
#undef AOTX_MEDIA_INLINE
#endif
