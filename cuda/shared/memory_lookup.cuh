/* Purpose: Supply explicit device workspace to shared memory scope checks.
 * Owns: No scope rule; the common cognitive resolver uses the supplied mark arrays.
 * Launch shape: One ordered service or shared record thread; graph nodes do not overlap.
 * Lifetime: Transient marks are cleared for each lookup and never saved. */
#ifndef AOTX_SHARED_MEMORY_LOOKUP_CUH
#define AOTX_SHARED_MEMORY_LOOKUP_CUH
#include "cognitive/lookup.cuh"
extern __device__ unsigned aotx_shared_memory_marks[2][AOTX_COG_WORDS];
__device__ inline aotx_cognitive_match aotx_shared_resolve(const aotx_cognitive_store *store,
    const aotx_cognitive_query *query, bool evidence, unsigned long long cut)
{
    return aotx_cog_resolve_scratch(store, query, evidence, cut,
        aotx_shared_memory_marks[0], aotx_shared_memory_marks[1]);
}
#endif
