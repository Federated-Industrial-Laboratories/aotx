/* Purpose: Resolve current object versions for a batch of scoped requests.
 * Owns: Reference freshness and visibility checks; no authentication.
 * Launch shape: One thread per request, with caller-supplied principal and room.
 * Lifetime: A quiescent admitted store and one request batch. */
#include "cognitive/lookup.cuh"

__global__ void aotx_cognitive_resolve(const aotx_cognitive_store *live,
    const aotx_cognitive_query *queries, aotx_cognitive_match *matches, uint32_t count) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) matches[i] = aotx_cog_resolve_one(live, queries + i);
}
