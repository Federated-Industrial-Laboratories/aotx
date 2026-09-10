/* Purpose: Validate typed appraisal, media and recorded selection payloads.
 * Owns: Payload schema and exact-reference checks; no derived scores.
 * Launch shape: One thread per bounded object.
 * Lifetime: One staged state validation. */
#ifndef AOTX_COGNITIVE_PAYLOAD_CUH
#define AOTX_COGNITIVE_PAYLOAD_CUH
#include "cognitive/intake_schema.cuh"

__device__ inline uint32_t aotx_cog_media(const unsigned char *p, uint64_t bytes) {
    if (bytes < AOTX_COG_MEDIA_HEADER || aotx_cog_u32(p) != 1 ||
        !aotx_cog_zero(p + 172, 20) || aotx_cog_zero(p + 72, 32)) return AOTX_COG_LAYOUT;
    uint32_t modality = aotx_cog_u32(p + 4), representation = aotx_cog_u32(p + 8);
    uint32_t dtype = aotx_cog_u32(p + 12), rank = aotx_cog_u32(p + 68);
    uint32_t layout = aotx_cog_u32(p + 64);
    uint64_t data = aotx_cog_u64(p + 48), positions = aotx_cog_u64(p + 56);
    uint32_t rate = aotx_cog_u32(p + 168);
    if (representation == AOTX_COG_SOURCE_BYTES && modality == AOTX_COG_AUDIO_MEDIA) {
        if (!rate || rate > 384000) return AOTX_COG_LAYOUT;
    } else if (rate) return AOTX_COG_LAYOUT;
    if (dtype < AOTX_COG_U8 || dtype > AOTX_COG_F32 || !rank || rank > 4 ||
        data > bytes - AOTX_COG_MEDIA_HEADER ||
        positions > (bytes - AOTX_COG_MEDIA_HEADER - data) / 8 ||
        bytes != AOTX_COG_MEDIA_HEADER + data + positions * 8) return AOTX_COG_LAYOUT;
    uint64_t product = dtype == AOTX_COG_U8 ? 1 : (dtype == AOTX_COG_F32 ? 4 : 2);
    for (unsigned d = 0; d < 4; ++d) {
        uint64_t dimension = aotx_cog_u64(p + 16 + d * 8);
        if (d >= rank) { if (dimension) return AOTX_COG_LAYOUT; continue; }
        if (!dimension || product > data / dimension) return AOTX_COG_LAYOUT;
        product *= dimension;
    }
    if (product != data) return AOTX_COG_LAYOUT;
    if (representation == AOTX_COG_SOURCE_BYTES) {
        if (positions || !aotx_cog_zero(p + 104, 64)) return AOTX_COG_LAYOUT;
        if (modality == AOTX_COG_IMAGE_MEDIA) {
            if (rank != 3 || dtype != AOTX_COG_U8 || layout != AOTX_COG_SPATIAL ||
                aotx_cog_u64(p + 32) != 3) return AOTX_COG_LAYOUT;
        } else if (modality == AOTX_COG_AUDIO_MEDIA) {
            if (rank != 2 || (dtype != AOTX_COG_I16 && dtype != AOTX_COG_F32) ||
                layout != AOTX_COG_TEMPORAL) return AOTX_COG_LAYOUT;
        } else return AOTX_COG_LAYOUT;
    } else if (representation == AOTX_COG_EXACT_FEATURES) {
        if (modality != AOTX_COG_FEATURE_MEDIA || rank != 2 ||
            (dtype != AOTX_COG_F16 && dtype != AOTX_COG_F32) ||
            aotx_cog_zero(p + 104, 32) || aotx_cog_zero(p + 136, 32) ||
            layout < AOTX_COG_LINEAR || layout > AOTX_COG_TEMPORAL ||
            positions != aotx_cog_u64(p + 16) * (layout == AOTX_COG_SPATIAL ? 3 : 1))
            return AOTX_COG_LAYOUT;
    } else return AOTX_COG_LAYOUT;
    if (dtype == AOTX_COG_F16 || dtype == AOTX_COG_F32) {
        unsigned width = dtype == AOTX_COG_F16 ? 2 : 4;
        for (uint64_t i = 0; i < data; i += width) {
            const unsigned char *value = p + AOTX_COG_MEDIA_HEADER + i;
            if (width == 2 ? (aotx_cog_u16(value) & 0x7c00u) == 0x7c00u
                           : (aotx_cog_u32(value) & 0x7f800000u) == 0x7f800000u)
                return AOTX_COG_LAYOUT;
        }
    }
    return AOTX_COG_OK;
}

__device__ inline uint32_t aotx_cog_payload(const aotx_cognitive_store *s,
                                          const unsigned char *r) {
    uint64_t bytes = aotx_cog_u64(r + AOTX_CO_BYTES);
    if (aotx_cog_u32(r + AOTX_CO_FLAGS) & AOTX_COG_TOMBSTONE)
        return bytes ? AOTX_COG_FORMAT : AOTX_COG_OK;
    if (!bytes) return AOTX_COG_FORMAT;
    const unsigned char *p = s->payload + aotx_cog_u64(r + AOTX_CO_OFFSET);
    uint16_t kind = aotx_cog_u16(r + AOTX_CO_KIND);
    if (bytes >= 8 && aotx_cog_equal(p, (const unsigned char *)"AOTXMEM2", 8)) return aotx_memory_schema(r, p, bytes);
    if (bytes >= 8 && aotx_cog_equal(p, (const unsigned char *)"AOTXMEM3", 8)) return aotx_intake_schema(s, r, p, bytes);
    if (kind == AOTX_COG_MEDIA) return aotx_cog_media(p, bytes);
    if (kind == AOTX_COG_APPRAISAL) {
        if (bytes != AOTX_COG_APPRAISAL_BYTES || aotx_cog_u32(p) != 1 ||
            !aotx_cog_scaled(aotx_cog_u32(p + 4)) || !aotx_cog_scaled(aotx_cog_u32(p + 8)) ||
            !aotx_cog_scaled(aotx_cog_u32(p + 12)) || aotx_cog_u32(p + 16) > 4 ||
            !aotx_cog_scaled(aotx_cog_u32(p + 20)) || aotx_cog_u32(p + 24) != 1 ||
            aotx_cog_u32(p + 28) || aotx_cog_zero(r + AOTX_CO_SUBJECT, 16) ||
            aotx_cog_zero(r + AOTX_CO_SOURCE, 16)) return AOTX_COG_FORMAT;
    }
    if (kind == AOTX_COG_SELECTION) {
        if (bytes < 16 || aotx_cog_u32(p) != 1 || !aotx_cog_zero(p + 8, 8))
            return AOTX_COG_FORMAT;
        uint32_t count = aotx_cog_u32(p + 4);
        if (count > AOTX_COG_SELECTION_MAX || bytes != 16 + (uint64_t)count * 32)
            return AOTX_COG_FORMAT;
        for (uint32_t i = 0; i < count; ++i) {
            const unsigned char *entry = p + 16 + i * 32;
            if (aotx_cog_zero(entry, 16) || aotx_cog_u32(entry + 28)) return AOTX_COG_REFERENCE;
            uint32_t status = aotx_cog_reference(s, r, entry, aotx_cog_u64(entry + 16));
            if (status) return status;
            int j = aotx_cog_find(s, entry, aotx_cog_u64(entry + 16));
            uint32_t representation = aotx_cog_u16(s->objects[j] + AOTX_CO_KIND) == AOTX_COG_MEDIA ? 2 : 1;
            if (aotx_cog_u32(entry + 24) != representation) return AOTX_COG_LAYOUT;
        }
    }
    return AOTX_COG_OK;
}
#endif
