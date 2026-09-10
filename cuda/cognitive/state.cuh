/* Purpose: Admit, checkpoint, restore and resolve batches of typed device objects.
 * Owns: Caller-allocated live and staging stores; no global state or model work.
 * Launch shape: One 64-thread block for state changes; batched threads for lookup.
 * Lifetime: Explicit state mode only; base conversation graphs do not call this module. */
#ifndef AOTX_COGNITIVE_STATE_CUH
#define AOTX_COGNITIVE_STATE_CUH
#include "cognitive/format.h"

typedef struct aotx_cognitive_store {
    uint64_t sequence, tick;
    uint32_t count, bytes;
    unsigned char lineage[16];
    unsigned char objects[AOTX_COG_OBJECTS][AOTX_COG_OBJECT];
    unsigned char payload[AOTX_COG_PAYLOAD];
} aotx_cognitive_store;

typedef struct aotx_cognitive_result {
    uint32_t status, applied;
    uint64_t bytes, sequence;
} aotx_cognitive_result;

typedef struct aotx_cognitive_query {
    unsigned char id[16], principal[16], room[16];
    uint64_t version;
} aotx_cognitive_query;
typedef struct aotx_cognitive_match {
    uint32_t status, index;
    uint64_t version;
} aotx_cognitive_match;

__device__ void aotx_cognitive_restore_block(aotx_cognitive_store *live, aotx_cognitive_store *stage,
    const unsigned char *image, uint64_t bytes, aotx_cognitive_result *result);
__device__ void aotx_cognitive_apply_block(aotx_cognitive_store *live, aotx_cognitive_store *stage,
    const unsigned char *tail, uint64_t bytes, aotx_cognitive_result *result);

/* Input/output images use the explicit byte layouts in format.h and docs/18. */
__global__ void aotx_cognitive_restore(aotx_cognitive_store *live, aotx_cognitive_store *stage,
    const unsigned char *image, uint64_t bytes, aotx_cognitive_result *result);
__global__ void aotx_cognitive_apply(aotx_cognitive_store *live, aotx_cognitive_store *stage,
    const unsigned char *tail, uint64_t bytes, aotx_cognitive_result *result);
__global__ void aotx_cognitive_checkpoint(const aotx_cognitive_store *live,
    unsigned char *image, uint64_t capacity, aotx_cognitive_result *result);
__device__ void aotx_cognitive_checkpoint_header_block(const aotx_cognitive_store *live,
    unsigned char *image, uint64_t capacity, aotx_cognitive_result *result);
__device__ void aotx_cognitive_checkpoint_block(const aotx_cognitive_store *live,
    unsigned char *image, uint64_t capacity, aotx_cognitive_result *result);
__global__ void aotx_cognitive_resolve(const aotx_cognitive_store *live,
    const aotx_cognitive_query *queries, aotx_cognitive_match *matches, uint32_t count);
#endif
