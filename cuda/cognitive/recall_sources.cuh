/* Purpose: Resolve exact source groups and reported actors for memory records.
 * Owns: Read-only source references; no identity inference or state changes.
 * Launch shape: One helper call per candidate in each recall query.
 * Lifetime: One validated store cut and its recorded selection. */
#ifndef AOTX_COGNITIVE_RECALL_SOURCES_CUH
#define AOTX_COGNITIVE_RECALL_SOURCES_CUH
#include "cognitive/recall_format.cuh"

__device__ inline uint32_t aotx_recall_source_index(const aotx_cognitive_store *s, uint32_t index) {
    const unsigned char *r = s->objects[index];
    int source = aotx_cog_find(s, r + AOTX_CO_SOURCE, aotx_cog_u64(r + AOTX_CO_SOURCE_VERSION));
    return source >= 0 && aotx_cog_u16(s->objects[source] + AOTX_CO_KIND) == AOTX_COG_EVENT ?
        (uint32_t)source : index;
}
__device__ inline const unsigned char *aotx_recall_source_actor(const aotx_cognitive_store *s, uint32_t source) {
    const unsigned char *r = s->objects[source];
    if (aotx_cog_cold(r)) return 0;
    const unsigned char *p = s->payload + aotx_cog_u64(r + AOTX_CO_OFFSET);
    return aotx_cog_u16(r + AOTX_CO_KIND) == AOTX_COG_EVENT &&
        aotx_cog_u32(r + AOTX_CO_SOURCE_KIND) == AOTX_COG_REPORTED &&
        aotx_cog_u64(r + AOTX_CO_BYTES) >= 32 && aotx_recall_magic(p, "AOTXMEM1") &&
        !aotx_cog_zero(r + AOTX_CO_SUBJECT, 16) ? r + AOTX_CO_SUBJECT : 0;
}
#endif
