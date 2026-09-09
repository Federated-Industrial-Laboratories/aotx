/* Purpose: Launch bounded prepared recall through the shared device search body.
 * Owns: No state; the caller supplies the store, requests and results.
 * Launch shape: One 64-thread block per query.
 * Lifetime: One admitted request batch. */
#include "cognitive/recall_search.cuh"

__global__ void aotx_recall_search(const aotx_cognitive_store *live,
    const unsigned char *requests, uint64_t bytes, aotx_recall_result *results,
    aotx_recall_scratch *scratch, uint32_t count) {
    aotx_recall_search_block(live, requests, bytes, results, scratch, count);
}
