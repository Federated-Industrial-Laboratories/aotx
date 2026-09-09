/* Purpose: Validate retention metadata and select a bounded working set.
 * Owns: No live state; result rows hold exact chosen references.
 * Launch shape: One thread per row; serial checks enforce batch ID order.
 * Lifetime: Before canonical staging and recorded publication. */
#ifndef AOTX_COGNITIVE_RETAIN_VALIDATE_CUH
#define AOTX_COGNITIVE_RETAIN_VALIDATE_CUH
#include "cognitive/live_validate.cuh"

__device__ inline const unsigned char *aotx_retain_source(const unsigned char *r) {
    uint32_t slot = aotx_cog_u32(r);
    if (aotx_live.auto_mode) for (uint32_t i = 0; i < aotx_live.count; ++i)
        if (aotx_cog_u32(aotx_live.prefixes[i]) == slot) return aotx_live.requests + 64 + i * AOTX_RECALL_QUERY;
    return aotx_live_bindings[slot].query;
}

static __device__ __noinline__ uint32_t aotx_retain_row_check(const unsigned char *r, uint32_t count) {
    uint32_t slot = aotx_cog_u32(r);
    if (slot >= AOTX_SLOTS || aotx_cog_u32(r + 4) > 1 || !aotx_cog_zero(r + 144, 16) ||
        !aotx_cog_u64(r + 104) || !aotx_cog_scaled(aotx_cog_u32(r + 128)) ||
        aotx_cog_u32(r + 132) > 2) return AOTX_COG_FORMAT;
    const aotx_live_binding *b = aotx_live_bindings + slot;
    if (!b->active || !aotx_cog_equal(r + 8, b->conversation)) return AOTX_COG_DENIED;
    uint64_t ordinal = b->ordinal + (aotx_live.auto_mode ? 1 : 0);
    if (!ordinal || aotx_cog_u64(r + 24) != ordinal) return AOTX_COG_SEQUENCE;
    if (!aotx_cog_equal(r + 32, aotx_retain_source(r))) return AOTX_COG_SOURCE;
    for (uint32_t i = 0; i < 3; ++i) {
        const unsigned char *id = r + 32 + i * 16;
        if (aotx_cog_zero(id, 16)) return AOTX_COG_REFERENCE;
        if (aotx_cog_latest(&aotx_live_store, id) >= 0) return AOTX_COG_VERSION;
        for (uint32_t j = 0; j < i; ++j)
            if (aotx_cog_equal(id, r + 32 + j * 16)) return AOTX_COG_REFERENCE;
    }
    uint64_t expiry = aotx_cog_u64(r + 136);
    if (expiry && expiry <= aotx_live_store.sequence + count * 3) return AOTX_COG_STALE;
    uint64_t version = aotx_cog_u64(r + 96);
    if (aotx_cog_zero(r + 80, 16)) return version ? AOTX_COG_REFERENCE : AOTX_COG_OK;
    aotx_cognitive_match m = aotx_recall_match(&aotx_live_store, aotx_retain_source(r), r + 80, version,
        aotx_live_store.sequence + count * 3);
    if (m.status) return m.status;
    const unsigned char *old = aotx_live_store.objects[m.index];
    if (aotx_cog_u16(old + AOTX_CO_KIND) != AOTX_COG_WORKING ||
        !aotx_cog_equal(old + AOTX_CO_SUBJECT, r + 112)) return AOTX_COG_SOURCE;
    if (!aotx_cog_equal(old + AOTX_CO_OWNER, b->principal) ||
        aotx_cog_u32(old + AOTX_CO_FLAGS) & AOTX_COG_PROTECTED ||
        aotx_cog_u32(old + AOTX_CO_RETENTION) > aotx_cog_u32(r + 132)) return AOTX_COG_DENIED;
    return AOTX_COG_OK;
}

static __device__ __noinline__ uint32_t aotx_retain_focus(unsigned char *out) {
    const aotx_live_binding *b = aotx_live_bindings + aotx_cog_u32(out);
    uint32_t count = 0;
    bool add = aotx_cog_u32(out + 4) != 0;
    for (uint32_t i = 0; i < b->focus_count; ++i) {
        bool replaced = aotx_cog_equal(b->focus[i], out + 80, 24);
        if (replaced) { add = true; continue; }
        for (uint32_t j = 0; j < 24; ++j) out[192 + count * 24 + j] = b->focus[i][j];
        ++count;
    }
    if (add && count == AOTX_RECALL_PINS) {
        uint32_t drop = count;
        for (uint32_t i = 0; i < count; ++i) {
            const unsigned char *ref = out + 192 + i * 24;
            int k = aotx_cog_latest(&aotx_live_store, ref);
            if (k < 0) return AOTX_COG_REFERENCE;
            const unsigned char *r = aotx_live_store.objects[k];
            if (!(aotx_cog_u32(r + AOTX_CO_FLAGS) & AOTX_COG_PROTECTED) &&
                !aotx_cog_u32(r + AOTX_CO_RETENTION)) { drop = i; break; }
        }
        if (drop == count) return AOTX_COG_CAPACITY;
        for (uint32_t i = drop; i + 1 < count; ++i)
            for (uint32_t j = 0; j < 24; ++j) out[192 + i * 24 + j] = out[192 + (i + 1) * 24 + j];
        --count;
    }
    if (add) {
        for (uint32_t j = 0; j < 16; ++j) out[192 + count * 24 + j] = out[48 + j];
        aotx_cog_put(out + 192 + count * 24 + 16, 1, 8); ++count;
    }
    aotx_cog_put(out + 160, count, 4);
    return AOTX_COG_OK;
}
__device__ inline uint32_t aotx_retain_unique(uint32_t count) {
    for (uint32_t i = 0; i < count; ++i) {
        const unsigned char *r = aotx_live.retain_rows[i];
        for (uint32_t j = 0; j < i; ++j) {
            const unsigned char *old = aotx_live.retain_rows[j];
            if (aotx_cog_u32(old) == aotx_cog_u32(r) ||
                (!aotx_cog_zero(r + 80, 16) && aotx_cog_equal(r + 80, old + 80, 24))) return AOTX_COG_REFERENCE;
            for (uint32_t x = 0; x < 3; ++x) for (uint32_t y = 0; y < 3; ++y)
                if (aotx_cog_equal(r + 32 + x * 16, old + 32 + y * 16)) return AOTX_COG_REFERENCE;
        }
    }
    return AOTX_COG_OK;
}

#endif
