/* Purpose: Connect typed memory batches to the tick graph and agent prompts.
 * Owns: One device store, transfer workspace and bounded conversation bindings.
 * Launch shape: State changes use one block; recall uses one block per query.
 * Lifetime: The run and its replayed journal. */
#ifndef AOTX_COGNITIVE_LIVE_CUH
#define AOTX_COGNITIVE_LIVE_CUH
#include "cognitive/live.h"
#include "cognitive/recall.cuh"
#include "profile/profile.cuh"
#define AOTX_LIVE_IDLE 0u
#define AOTX_LIVE_READY 1u
#define AOTX_LIVE_SEARCH 2u
#define AOTX_LIVE_WRITE 3u
#define AOTX_LIVE_WAIT 4u
#define AOTX_LIVE_REPLAY 5u
#define AOTX_LIVE_ENCODING 6u
typedef struct aotx_live_binding {
    uint32_t active, pages, scope, context_bytes;
    uint64_t ordinal;
    unsigned char principal[16], room[16], conversation[16];
    unsigned char query[AOTX_RECALL_QUERY];
    aotx_recall_result choice;
    uint32_t focus_count;
    unsigned char focus[AOTX_RECALL_PINS][24];
} aotx_live_binding;
typedef struct aotx_live_state {
    uint32_t ready, phase, op, total, received, count, status, written, choice_bytes, fatal;
    uint64_t source_seq, request_seq, accepted, refused, searches, replays;
    unsigned char transfer_id[16], query_id[16];
    unsigned char input[AOTX_LIVE_BYTES];
    unsigned char prefixes[AOTX_RECALL_BATCH][64];
    unsigned char requests[AOTX_RECALL_REQUESTS];
    unsigned char choices[AOTX_LIVE_RESULTS];
    aotx_recall_result results[AOTX_RECALL_BATCH];
    aotx_cognitive_result result;
    unsigned char retain_rows[AOTX_RECALL_BATCH][AOTX_LIVE_RETAIN_ROW];
    uint32_t text_mode;
    unsigned long long encoded;
    uint32_t text_row[AOTX_SLOTS], text_status[AOTX_SLOTS];
} aotx_live_state;
extern __device__ aotx_live_state aotx_live;
extern __device__ aotx_cognitive_store aotx_live_store;
extern __device__ aotx_live_binding aotx_live_bindings[AOTX_SLOTS];
__device__ __forceinline__ bool aotx_live_bound(uint32_t slot) {
    return slot < AOTX_SLOTS && aotx_live_bindings[slot].active;
}
__device__ bool aotx_live_busy(uint32_t slot);
__device__ unsigned int aotx_live_window(uint64_t base, unsigned int count);
__device__ void aotx_live_part(const unsigned char *body, uint32_t bytes, uint64_t seq);
__device__ bool aotx_live_restore_end(void);
__device__ void aotx_live_note(uint32_t op, uint32_t status, uint32_t count);
__device__ uint32_t aotx_live_prompt_check(uint32_t slot);
__device__ uint32_t aotx_live_context(uint32_t slot, unsigned char *out, uint32_t at);
extern __device__ aotx_cognitive_store aotx_live_candidate, aotx_live_scratch;
__global__ void aotx_live_stage(void);
__global__ void aotx_live_prepare(void);
__device__ void aotx_live_text_begin(void);
__device__ __forceinline__ bool aotx_live_text_pending(uint32_t slot) {
    return slot < AOTX_SLOTS && aotx_live.phase == AOTX_LIVE_ENCODING &&
        aotx_live.text_row[slot] && !aotx_live.text_status[slot];
}
__device__ void aotx_live_text_fail(uint32_t slot, uint32_t status);
__device__ void aotx_live_text_done(uint32_t slot);
__global__ void aotx_live_search(void);
__global__ void aotx_live_decide(void);
__global__ void aotx_live_commit(void);
#endif
