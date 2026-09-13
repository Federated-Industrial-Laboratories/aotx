/* Purpose: Replace the complete operator grant table after validation.
 * Owns: Grant publication and revision checks.
 * Launch shape: One ordered control operation before the request batch.
 * Lifetime: One deployment epoch; no grant is restored from model data. */
#include "service/internal.cuh"
#include "kvcache/kvcache.cuh"
__device__ unsigned aotx_service_install(const unsigned char *f, unsigned n)
{
    unsigned count = aotx_service_u32(f + 76);
    unsigned long long revision = aotx_service_get(f + 32, 8);
    if (count > AOTX_SERVICE_PRINCIPALS || n != AOTX_SERVICE_HEAD + count * AOTX_SERVICE_GRANT_BYTES ||
        !revision || revision <= aotx_service.revision) return 400;
    for (unsigned i = 0; i < count; ++i) {
        const unsigned char *p = f + AOTX_SERVICE_HEAD + i * AOTX_SERVICE_GRANT_BYTES;
        unsigned actions = aotx_service_u32(p + 24), pages = aotx_service_u32(p + 32);
        unsigned tokens = aotx_service_u32(p + 36), requests = aotx_service_u32(p + 40);
        if (!aotx_service_nonzero(p, 16) || aotx_service_get(p + 16, 8) != revision ||
            !actions || (actions & ~127u) || pages > AOTX_KV_PAGES_EACH ||
            (aotx_service_u32(p + 28) & ~((1u << AOTX_MODEL_LANGUAGE) |
                (1u << AOTX_MODEL_LANGUAGE_Q4) | (1u << AOTX_MODEL_LANGUAGE_AUDIO))) ||
            !tokens || tokens >= AOTX_SEQ_MAX_TOKENS || !requests || requests > AOTX_SERVICE_REQUESTS ||
            aotx_service_get(p + 56, 8)) return 400;
        for (unsigned j = 0; j < i; ++j)
            if (aotx_service_equal(p, f + AOTX_SERVICE_HEAD + j * AOTX_SERVICE_GRANT_BYTES, 16)) return 400;
    }
    for (unsigned i = 0; i < count; ++i) {
        const unsigned char *p = f + AOTX_SERVICE_HEAD + i * AOTX_SERVICE_GRANT_BYTES;
        aotx_service_grant &g = aotx_service.grants[i];
        aotx_service_bytes(g.principal, p, 16); g.revision = revision;
        g.actions = aotx_service_u32(p + 24); g.models = aotx_service_u32(p + 28);
        g.pages = aotx_service_u32(p + 32); g.tokens = aotx_service_u32(p + 36);
        if (!g.pages) g.pages = (unsigned)min((unsigned long long)AOTX_KV_PAGES_EACH, AOTX_KV_RANGE_BYTES / 2097152ull);
        g.requests = aotx_service_u32(p + 40); g.media = aotx_service_u32(p + 44);
        g.media_bytes = aotx_service_get(p + 48, 8);
    }
    aotx_service.grant_count = count; aotx_service.revision = revision;
    return 200;
}
