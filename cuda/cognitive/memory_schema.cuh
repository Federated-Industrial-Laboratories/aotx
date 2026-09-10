/* Purpose: Validate contextual memory text before store admission.
 * Owns: UTF-8 and task-bound text payload rules; no persistent state.
 * Launch shape: One helper call for each object in a batch.
 * Lifetime: One staged store or prepared recall operation. */
#ifndef AOTX_COGNITIVE_MEMORY_SCHEMA_CUH
#define AOTX_COGNITIVE_MEMORY_SCHEMA_CUH
#include "cognitive/codec.cuh"

__device__ inline bool aotx_recall_utf8(const unsigned char *p, uint32_t n) {
    for (uint32_t i = 0; i < n;) {
        uint32_t c = p[i++], more = 0, minimum = 0;
        if (c < 128) { if ((!c || c < 32 || c == 127) && c != 9 && c != 10) return false; continue; }
        if (c >= 194 && c <= 223) { more = 1; c &= 31; minimum = 128; }
        else if (c >= 224 && c <= 239) { more = 2; c &= 15; minimum = 2048; }
        else if (c >= 240 && c <= 244) { more = 3; c &= 7; minimum = 65536; }
        else return false;
        if (more > n - i) return false;
        while (more--) { uint32_t b = p[i++]; if ((b & 192) != 128) return false; c = c * 64 + (b & 63); }
        if ((c >= 0x80 && c <= 0x9f) || c < minimum || c > 0x10ffff || (c >= 0xd800 && c <= 0xdfff)) return false;
    }
    return true;
}
__device__ inline uint32_t aotx_memory_schema(const unsigned char *r, const unsigned char *p, uint64_t bytes) {
    uint32_t kind = aotx_cog_u16(r + AOTX_CO_KIND);
    if (bytes < 64 || aotx_cog_u32(p + 8) != 2 || aotx_cog_u32(p + 32) > 1 ||
        aotx_cog_zero(p + 16, 16) || !aotx_cog_zero(p + 36, 28) ||
        aotx_cog_zero(r + AOTX_CO_SOURCE, 16) ||
        (kind != AOTX_COG_ASSERTION && kind != AOTX_COG_CUE && kind != AOTX_COG_INTENTION) ||
        (kind != AOTX_COG_CUE && aotx_cog_zero(r + AOTX_CO_SUBJECT, 16))) return AOTX_COG_FORMAT;
    uint32_t text = aotx_cog_u32(p + 12);
    return text && text <= 2048 && bytes == 64 + text && aotx_recall_utf8(p + 64, text)
        ? AOTX_COG_OK : AOTX_COG_FORMAT;
}
#endif
