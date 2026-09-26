/* Purpose: Define bounded local service frames and grant rows.
 * Owns: Portable byte offsets and mailbox ownership states.
 * Threading: The broker and device exchange complete mailbox contents.
 * Lifetime: One runtime epoch; no deployment credential enters this format. */
#ifndef AOTX_SERVICE_WIRE_H
#define AOTX_SERVICE_WIRE_H
#include <stdint.h>
#define AOTX_SERVICE_MAGIC "AOTXAPI1"
#define AOTX_SERVICE_FRAME 65536u
#define AOTX_SERVICE_HEAD 128u
#define AOTX_SERVICE_DATA (AOTX_SERVICE_FRAME - AOTX_SERVICE_HEAD)
#define AOTX_SERVICE_GRANT_BYTES 64u
#define AOTX_SERVICE_SCHEMA 1u

enum aotx_service_operation {
    AOTX_SERVICE_GRANTS = 1, AOTX_SERVICE_INFO, AOTX_SERVICE_SUBMIT,
    AOTX_SERVICE_READ, AOTX_SERVICE_CANCEL, AOTX_SERVICE_MEDIA,
    AOTX_SERVICE_MEDIA_READ, AOTX_SERVICE_METRICS, AOTX_SERVICE_MEDIA_LIST,
    AOTX_SERVICE_POLICY = 12
};
enum aotx_service_action {
    AOTX_SERVICE_INFER = 1u, AOTX_SERVICE_UPLOAD = 2u,
    AOTX_SERVICE_FETCH = 4u, AOTX_SERVICE_TELEMETRY = 8u, AOTX_SERVICE_POLICY_MANAGE = 128u
};
enum aotx_service_phase {
    AOTX_SERVICE_FREE, AOTX_SERVICE_QUEUED, AOTX_SERVICE_PREPARE,
    AOTX_SERVICE_RUNNING, AOTX_SERVICE_DONE, AOTX_SERVICE_FAILED,
    AOTX_SERVICE_CANCELLED
};

/* Header: magic at 0; operation/status and flags/phase at 8/12; principal at 16.
 * Grant revision and epoch are at 32/40; request ID at 48; byte cursor at 64.
 * Model and output limit are at 72/76; temperature and top_p at 80/84; payload length at 88.
 * Input bytes 92..127 are zero.
 * Reply: total output bytes at 76; prompt/output tokens at 80/84;
 * finish reason at 92; error code at 96; cancel request at 100. Integers are little endian.
 */
/* Grant row: principal at 0, revision at 16, actions/model mask at 24/28.
 * Page/output token limits are at 32/36; request/media counts at 40/44; media byte quota at 48 and zero at 56. */
typedef struct aotx_service_mailbox {
    uint64_t state, generation, length, reserved[5];
    unsigned char bytes[AOTX_SERVICE_FRAME];
} aotx_service_mailbox;
typedef struct aotx_service_ring {
    uint32_t schema, channels, frame_bytes, reserved0;
    uint64_t closed, epoch, reserved[4];
} aotx_service_ring;

#ifdef __CUDACC__
#define AOTX_SERVICE_INLINE __host__ __device__ __forceinline__
#else
#define AOTX_SERVICE_INLINE static inline
#endif
AOTX_SERVICE_INLINE uint64_t aotx_service_get(const unsigned char *p, unsigned n)
{
    uint64_t v = 0;
    for (unsigned i = 0; i < n; ++i) v |= (uint64_t)p[i] << (8u * i);
    return v;
}
AOTX_SERVICE_INLINE void aotx_service_put(unsigned char *p, uint64_t v, unsigned n)
{
    for (unsigned i = 0; i < n; ++i) p[i] = (unsigned char)(v >> (8u * i));
}
#undef AOTX_SERVICE_INLINE
#endif
