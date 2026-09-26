/* Purpose: Keep bounded task review work and its durable source frontier.
 * Owns: Eligibility, exact requests and publication counters on the device.
 * Launch shape: Serial admission; one block constructs each admitted batch.
 * Lifetime: The runtime and its complete or interrupted journal recovery. */
#ifndef AOTX_REFLECTION_STATE_CUH
#define AOTX_REFLECTION_STATE_CUH
#include "cognitive/live.cuh"
#define AOTX_REVIEW_BUILD 24u
typedef struct aotx_review_state {
    uint64_t frontier, after, wake, work_wake, control_revision, completed, interrupted, refused, calls;
    uint64_t observed, root, bytes, maintenance, started_ns, elapsed_ns, maximum_ns;
    uint64_t blocked_sequence, blocked_root, blocked_bytes;
    uint32_t enabled, active, recovery, pending, count, status, observed_count, blocked_count;
    uint32_t indices[AOTX_REVIEW_BATCH];
    unsigned char queries[AOTX_REVIEW_BATCH * AOTX_RECALL_QUERY];
} aotx_review_state;
extern __device__ aotx_review_state aotx_review;
__device__ uint32_t aotx_review_pending(void);
__device__ void aotx_review_auto_request(void);
__device__ void aotx_review_begin(void);
__device__ void aotx_review_decide(void);
__device__ void aotx_review_publish(void);
__device__ bool aotx_review_query(const aotx_cognitive_store *, uint32_t, unsigned char *);
#endif
