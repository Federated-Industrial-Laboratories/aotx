/* Purpose: Mark exact retained references and current access and correction guards.
 * Owns: Device mark bits; no state is discarded by these helpers.
 * Launch shape: One thread per object or binding within the planning block.
 * Lifetime: One finite dependency closure over the admitted store. */
#ifndef AOTX_COGNITIVE_MAINTENANCE_ROOTS_CUH
#define AOTX_COGNITIVE_MAINTENANCE_ROOTS_CUH
#include "cognitive/maintenance.cuh"
#include "cognitive/lookup.cuh"
#include "shared/state.cuh"

__device__ inline void aotx_memory_mark(int index) {
    if (index >= 0) atomicOr(aotx_maintenance.marks + index, 1u);
}
__device__ inline void aotx_memory_reference(const unsigned char *id, uint64_t version) {
    if (!aotx_cog_zero(id, 16)) {
        int index = aotx_cog_find(&aotx_live_store, id, version);
        if (index < 0) atomicCAS(&aotx_maintenance.status, 0u, AOTX_COG_REFERENCE);
        else aotx_memory_mark(index);
    }
}
__device__ inline bool aotx_memory_root(uint32_t i, uint64_t floor, uint32_t age) {
    const unsigned char *r = aotx_live_store.objects[i];
    uint64_t updated = aotx_cog_u64(r + AOTX_CO_UPDATED);
    if (updated > floor) return true;
    if (i != aotx_maintenance.latest[i]) return false;
    uint32_t flags = aotx_cog_u32(r + AOTX_CO_FLAGS);
    if ((flags & AOTX_COG_PROTECTED) || aotx_cog_u32(r + AOTX_CO_RETENTION)) return true;
    if (flags & AOTX_COG_TOMBSTONE) return false;
    if (aotx_cog_u16(r + AOTX_CO_KIND) == AOTX_COG_INTENTION) return true;
    uint64_t expiry = aotx_cog_u64(r + AOTX_CO_EXPIRY);
    if ((expiry && expiry <= aotx_live_store.sequence) || aotx_cog_superseded(&aotx_live_store, r)) return false;
    return !age || aotx_live_store.sequence - updated < age;
}
__device__ inline void aotx_memory_binding_roots(const aotx_live_binding *b) {
    if (!b->active) return;
    for (uint32_t i = 0; i < b->focus_count; ++i)
        aotx_memory_reference(b->focus[i], aotx_cog_u64(b->focus[i] + 16));
    for (uint32_t group = 0; group < 2; ++group)
    for (uint32_t i = 0; i < aotx_cog_u32(b->query + 140 + group * 4); ++i) {
        const unsigned char *p = b->query + 4256 + group * 192 + i * 24;
        aotx_memory_reference(p, aotx_cog_u64(p + 16));
    }
    for (uint32_t i = 0; i < b->choice.count; ++i) {
        const unsigned char *p = b->choice.selection + 16 + i * 32;
        aotx_memory_reference(p, aotx_cog_u64(p + 16));
    }
}
__device__ inline void aotx_memory_dependencies(uint32_t i) {
    const unsigned char *r = aotx_live_store.objects[i];
    aotx_memory_mark(aotx_maintenance.latest[i]);
    aotx_memory_reference(r + AOTX_CO_SOURCE, aotx_cog_u64(r + AOTX_CO_SOURCE_VERSION));
    aotx_memory_reference(r + AOTX_CO_SUPERSEDES, aotx_cog_u64(r + AOTX_CO_SUPER_VERSION));
    aotx_memory_reference(r + AOTX_CO_EMBEDDING, aotx_cog_u64(r + AOTX_CO_EMBED_VERSION));
    for (uint32_t k = 0; k < 2; ++k) {
        const unsigned char *extra = aotx_appraisal_reference(r,
            aotx_live_store.payload + aotx_cog_u64(r + AOTX_CO_OFFSET), aotx_cog_u64(r + AOTX_CO_BYTES), k);
        if (extra) aotx_memory_reference(extra, aotx_cog_u64(extra + 16));
    }
    if (aotx_cog_u16(r + AOTX_CO_KIND) == AOTX_COG_SELECTION &&
        !(aotx_cog_u32(r + AOTX_CO_FLAGS) & AOTX_COG_TOMBSTONE)) {
        const unsigned char *p = aotx_live_store.payload + aotx_cog_u64(r + AOTX_CO_OFFSET);
        for (uint32_t j = 0; j < aotx_cog_u32(p + 4); ++j)
            aotx_memory_reference(p + 16 + j * 32, aotx_cog_u64(p + 32 + j * 32));
    }
    if (aotx_cog_u16(r + AOTX_CO_KIND) == AOTX_COG_EVENT &&
        aotx_cog_u64(r + AOTX_CO_BYTES) == 16 + AOTX_RECALL_QUERY) {
        const unsigned char *p = aotx_live_store.payload + aotx_cog_u64(r + AOTX_CO_OFFSET);
        if (aotx_cog_equal(p, (const unsigned char *)"AOTXQUE1", 8)) {
            const unsigned char *q = p + 16;
            for (uint32_t group = 0; group < 2; ++group)
            for (uint32_t j = 0; j < aotx_cog_u32(q + 140 + group * 4) && j < AOTX_RECALL_PINS; ++j) {
                const unsigned char *ref = q + 4256 + group * 192 + j * 24;
                aotx_memory_reference(ref, aotx_cog_u64(ref + 16));
            }
        }
    }
    for (uint32_t j = 0; j < aotx_live_store.count; ++j) {
        const unsigned char *p = aotx_live_store.objects[j];
        if (aotx_cog_equal(p + AOTX_CO_SUPERSEDES, r + AOTX_CO_ID) &&
            aotx_cog_u64(p + AOTX_CO_SUPER_VERSION) == aotx_cog_u64(r + AOTX_CO_VERSION))
            aotx_memory_mark(j);
    }
}
#endif
