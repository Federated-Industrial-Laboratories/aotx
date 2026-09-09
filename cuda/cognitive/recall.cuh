/* Purpose: Select prepared memory, record choices and reconstruct saved context.
 * Owns: Caller-owned device buffers; no base conversation state.
 * Launch shape: Search/replay use one 64-thread block per request; other calls use one block.
 * Lifetime: Serialized operations over a quiescent admitted store. */
#ifndef AOTX_COGNITIVE_RECALL_CUH
#define AOTX_COGNITIVE_RECALL_CUH
#include "cognitive/recall.h"
#include "cognitive/state.cuh"
__global__ void aotx_recall_search(const aotx_cognitive_store *live,
    const unsigned char *requests, uint64_t bytes, aotx_recall_result *results, uint32_t count);
__global__ void aotx_recall_record(const aotx_cognitive_store *live,
    const unsigned char *requests, uint64_t bytes, const aotx_recall_result *results,
    uint32_t count, unsigned char *tail, aotx_cognitive_result *result);
/* Build a new request envelope from recorded selection sources, without search. */
__global__ void aotx_recall_requests(const aotx_cognitive_store *live,
    unsigned char *requests, aotx_cognitive_result *result);
__global__ void aotx_recall_replay(const aotx_cognitive_store *live,
    const unsigned char *requests, uint64_t bytes, aotx_recall_result *results, uint32_t count);
#endif
