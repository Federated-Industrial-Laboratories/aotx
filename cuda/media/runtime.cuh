/* Purpose: Keep immutable scoped image sources and their trained feature rows.
 * Owns: Device source descriptors, byte spans, workspaces and feature leases.
 * Launch shape: Ordered admission and batched finite image work in the tick graph.
 * Lifetime: One runtime; canonical source records rebuild this state on restore. */
#ifndef AOTX_MEDIA_RUNTIME_CUH
#define AOTX_MEDIA_RUNTIME_CUH
#include "media/profile.h"
#include "media/wire.h"
#include "media/image.cuh"
#include "media/hash.cuh"
#include "vision/vision.cuh"
#include "audio/runtime.cuh"
#include "seam/seam.cuh"

enum aotx_media_phase {
    AOTX_MEDIA_FREE, AOTX_MEDIA_RECEIVE, AOTX_MEDIA_HASH,
    AOTX_MEDIA_WAIT, AOTX_MEDIA_DECODE, AOTX_MEDIA_ENCODE,
    AOTX_MEDIA_READY, AOTX_MEDIA_REFUSED
};
enum aotx_media_status {
    AOTX_MEDIA_INVALID = 1, AOTX_MEDIA_LIMIT, AOTX_MEDIA_DIGEST,
    AOTX_MEDIA_CANCELLED, AOTX_MEDIA_CODEC, AOTX_MEDIA_ENCODER,
    AOTX_MEDIA_UNAVAILABLE, AOTX_MEDIA_LEASED, AOTX_MEDIA_NO_SIGNAL,
    AOTX_MEDIA_AUDIO_FORMAT, AOTX_MEDIA_AUDIO_NUMERIC
};
struct aotx_media_object {
    unsigned char transfer[16], digest[32], room[16], principal[16];
    unsigned long long generation, offset, bytes, received;
    unsigned phase, status, slot, scope, format, width, height;
    unsigned feature, span, rows, columns, lines, worker, notified;
    unsigned rate, channels, source_frames, samples, encoding;
};
struct aotx_media_state {
    aotx_media_profile profile;
    aotx_media_object *objects;
    aotx_media_hash *hash;
    aotx_image_job *image;
    aotx_vision_job *vision;
    unsigned *owner;
    unsigned char *source;
    float *features;
    unsigned char *workspace;
    unsigned long long workspace_each, coefficient_count, allocated;
    aotx_media_preamble *ring;
    unsigned char *frames;
    unsigned long long consumed, accepted, refused;
    unsigned frame_at, enabled, role, fatal, image_enabled;
    unsigned char parent_digest[32];
};
extern __device__ aotx_media_state aotx_media;
__device__ __forceinline__ bool aotx_media_is_audio(unsigned format)
{
    return format==AOTX_AUDIO_WAV || format==AOTX_AUDIO_PCM;
}
__device__ unsigned aotx_media_rows(unsigned count,bool audio=false);
__device__ bool aotx_media_part(const unsigned char *, unsigned, unsigned long long);
__device__ void aotx_media_publish(const unsigned char *, unsigned);
__device__ bool aotx_media_quiet(void);
__device__ void aotx_media_report(unsigned index, unsigned status, unsigned op);
__device__ void aotx_media_restore_end(void);
__device__ unsigned aotx_media_window(unsigned long long base, unsigned count);
__device__ bool aotx_media_model_allowed(unsigned role, const unsigned char *digest);
__device__ int aotx_media_find(const unsigned char *digest, unsigned slot);
__global__ void aotx_media_initialize(void);
__global__ void aotx_media_ingest(void);
__global__ void aotx_media_schedule(void);
__global__ void aotx_media_complete(void);
int aotx_media_open(const char *store, const char *roles, int (*stopped)(void));
void aotx_media_close(void);
void aotx_media_capture(cudaStream_t);
int aotx_media_allocate(const aotx_media_profile *, const aotx_vision_desc *,
                         unsigned role, unsigned char **weights);
int aotx_media_ring_open(aotx_seam_rings *);
void aotx_media_ring_finish(const aotx_seam_rings *);
void aotx_media_ring_close(aotx_seam_rings *);
static __device__ __forceinline__ unsigned long long aotx_media_acquire(const uint64_t *p)
{
    return aotx_seam_acquire_sys((const unsigned long long *)p);
}
static __device__ __forceinline__ void aotx_media_release(uint64_t *p, unsigned long long value)
{
    aotx_seam_release_sys((unsigned long long *)p, value);
}
#endif
