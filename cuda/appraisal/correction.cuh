/* Purpose: Resolve the current relationship paired with an appraisal correction.
 * Owns: One shared lookup for target admission and result encoding.
 * Launch shape: Each caller handles one source row in a device batch.
 * Lifetime: Stable live memory during an internal appraisal operation. */
#ifndef AOTX_APPRAISAL_CORRECTION_CUH
#define AOTX_APPRAISAL_CORRECTION_CUH
#include "appraisal/appraisal.cuh"
#include "appraisal/schema.cuh"
#include "cognitive/lookup.cuh"

__device__ inline uint32_t aotx_appraisal_old_relation(uint32_t prior) {
    const unsigned char *a = aotx_live_store.objects[prior];
    const unsigned char *p = aotx_live_store.payload + aotx_cog_u64(a + AOTX_CO_OFFSET);
    for (uint32_t j = 0; j < aotx_live_store.count; ++j) {
        const unsigned char *r = aotx_live_store.objects[j], *rp = aotx_live_store.payload + aotx_cog_u64(r + AOTX_CO_OFFSET);
        if (aotx_cog_u16(r + AOTX_CO_KIND) == AOTX_COG_RELATIONSHIP &&
            aotx_appraisal_magic(rp, aotx_cog_u64(r + AOTX_CO_BYTES), "AOTXREL1") &&
            aotx_cog_equal(rp + 136, p + 96, 24) &&
            aotx_cog_latest(&aotx_live_store, r + AOTX_CO_ID) == (int)j && !aotx_cog_superseded(&aotx_live_store, r)) return j;
    }
    return UINT32_MAX;
}
#endif
