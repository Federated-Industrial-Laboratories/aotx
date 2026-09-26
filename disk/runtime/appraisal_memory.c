/* Purpose: Check appraisal processor and model identities in memory byte extents.
 * Owns: Bounded header reads; semantic memory admission remains on the device.
 * Threading: One disk reader walks the configured object batch.
 * Lifetime: One checkpoint inspection or complete file publication. */
#include "disk/runtime/appraisal.h"
#include "cognitive/format.h"

typedef struct aotx_appraisal_reader {
    const unsigned char *memory;
    int fd;
    uint64_t offset, bytes;
} aotx_appraisal_reader;
static int read_bytes(const aotx_appraisal_reader *r, uint64_t at, size_t bytes, unsigned char *out) {
    if (at > r->bytes || bytes > r->bytes - at) return AOTX_CCIR_INVALID;
    if (r->memory) { memcpy(out, r->memory + at, bytes); return 0; }
    if (at > UINT64_MAX - r->offset) return AOTX_CCIR_INVALID;
    return aotx_ccir_pread(r->fd, out, bytes, r->offset + at);
}
static int identity(const unsigned char *processor, const unsigned char *model,
    const aotx_runtime_appraisal_models *models, int required) {
    if (!aotx_runtime_appraisal_contract(processor)) return AOTX_CCIR_UNSUPPORTED;
    if (!model) return 0;
    if (aotx_ccir_zero(model, 32)) return required ? AOTX_CCIR_INVALID : 0;
    if (!models) return 0;
    for (uint32_t i = 0; i < models->count; ++i) if (!memcmp(model, models->digest[i], 32)) return 0;
    return AOTX_CCIR_UNSUPPORTED;
}
static int payload(const unsigned char *row, const unsigned char *p, uint64_t bytes,
    const aotx_runtime_appraisal_models *models, uint32_t *required) {
    unsigned kind = aotx_ccir_u16(row + AOTX_CO_KIND);
    if (kind == AOTX_COG_REVIEW) {
        *required |= AOTX_RUNTIME_REVIEW;
        if (aotx_ccir_u32(row + AOTX_CO_FLAGS) & AOTX_COG_TOMBSTONE)
            return bytes ? AOTX_CCIR_INVALID : 0;
        return bytes == AOTX_REVIEW_CUE_BYTES && !memcmp(p, "AOTXMEM4", 8) && aotx_ccir_u32(p + 8) == 4
            ? 0 : AOTX_CCIR_UNSUPPORTED;
    }
    if (bytes >= 7 && !memcmp(p, "AOTXAPC", 7)) {
        if (p[7] != '1' || bytes < 12 || aotx_ccir_u32(p + 8) != 1) return AOTX_CCIR_UNSUPPORTED;
        if (kind != AOTX_COG_POLICY || bytes != AOTX_APPRAISAL_CONFIG_BYTES) return AOTX_CCIR_INVALID;
        *required |= AOTX_RUNTIME_APPRAISAL;
        return identity(p + 40, NULL, models, 0);
    }
    if (bytes >= 7 && !memcmp(p, "AOTXAPQ", 7)) {
        if (p[7] != '1' || bytes < 12 || aotx_ccir_u32(p + 8) != 1) return AOTX_CCIR_UNSUPPORTED;
        if (kind != AOTX_COG_POLICY || bytes != AOTX_APPRAISAL_QUEUE_BYTES ||
            aotx_ccir_u32(p + 12) > AOTX_APPRAISAL_INTERRUPTED) return AOTX_CCIR_INVALID;
        *required |= AOTX_RUNTIME_APPRAISAL;
        return identity(p + 64, p + 96, models, aotx_ccir_u32(p + 12) == AOTX_APPRAISAL_COMPLETE);
    }
    if (bytes >= 7 && !memcmp(p, "AOTXREL", 7)) {
        if (p[7] != '1' || bytes < 12 || aotx_ccir_u32(p + 8) != 1) return AOTX_CCIR_UNSUPPORTED;
        if (kind != AOTX_COG_RELATIONSHIP || bytes != AOTX_APPRAISAL_RELATION_BYTES) return AOTX_CCIR_INVALID;
        *required |= AOTX_RUNTIME_APPRAISAL;
        return identity(p + 72, p + 104, models, 1);
    }
    if (kind == AOTX_COG_APPRAISAL && bytes) {
        if (bytes < 4) return AOTX_CCIR_INVALID;
        if (aotx_ccir_u32(p) == 1) return bytes == AOTX_COG_APPRAISAL_BYTES ? 0 : AOTX_CCIR_INVALID;
        if (aotx_ccir_u32(p) != 2) return AOTX_CCIR_UNSUPPORTED;
        if (bytes != AOTX_APPRAISAL_ASSESS_BYTES) return AOTX_CCIR_INVALID;
        *required |= AOTX_RUNTIME_APPRAISAL;
        return identity(p + 32, p + 64, models, 1);
    }
    return 0;
}
int aotx_runtime_appraisal_scan(const unsigned char *memory, int fd, uint64_t offset, uint64_t bytes,
    const aotx_runtime_appraisal_models *models, uint32_t *required) {
    *required = 0;
    aotx_appraisal_reader r = {memory, fd, offset, bytes};
    unsigned char h[AOTX_COG_HEADER];
    int rc = read_bytes(&r, 0, sizeof(h), h);
    if (rc) return rc;
    /* Generic checkpoint validity is checked by device admission. */
    if (memcmp(h, "AOTXOBJ1", 8)) return 0;
    if (aotx_ccir_u32(h + 8) == 3) *required |= AOTX_RUNTIME_COLD;
    uint32_t count = aotx_ccir_u32(h + 20);
    if (!count) return 0;
    uint64_t start = AOTX_COG_HEADER + (uint64_t)count * AOTX_COG_OBJECT, size = aotx_ccir_u64(h + 24);
    if (aotx_ccir_u32(h + 8) < 1 || aotx_ccir_u32(h + 8) > 3 ||
        aotx_ccir_u32(h + 12) != AOTX_COG_HEADER || aotx_ccir_u32(h + 16) != AOTX_COG_OBJECT ||
        start > bytes || size != bytes - start ||
        aotx_ccir_u64(h + 64) != AOTX_COG_HEADER || aotx_ccir_u64(h + 72) != start ||
        aotx_ccir_u64(h + 80) != bytes) return AOTX_CCIR_INVALID;
    for (uint32_t i = 0; i < count; ++i) {
        unsigned char row[AOTX_COG_OBJECT], p[AOTX_APPRAISAL_RELATION_BYTES] = {0};
        rc = read_bytes(&r, AOTX_COG_HEADER + (uint64_t)i * AOTX_COG_OBJECT, sizeof(row), row);
        if (rc) return rc;
        uint64_t at = aotx_ccir_u64(row + AOTX_CO_OFFSET), n = aotx_ccir_u64(row + AOTX_CO_BYTES);
        if (aotx_ccir_u32(row + AOTX_CO_FLAGS) & AOTX_COG_COLD) {
            unsigned kind = aotx_ccir_u16(row + AOTX_CO_KIND);
            if (kind == AOTX_COG_REVIEW) *required |= AOTX_RUNTIME_REVIEW;
            if (aotx_ccir_u32(h + 8) != 3 || at || !n || n > AOTX_COG_PAYLOAD ||
                (kind != AOTX_COG_EVENT && kind != AOTX_COG_ASSERTION && kind != AOTX_COG_CUE &&
                 kind != AOTX_COG_IDENTITY && kind != AOTX_COG_MEDIA && kind != AOTX_COG_REVIEW)) return AOTX_CCIR_INVALID;
            *required |= AOTX_RUNTIME_COLD;
            continue;
        }
        if (at > size || n > size - at) return AOTX_CCIR_INVALID;
        rc = read_bytes(&r, start + at, n < sizeof(p) ? (size_t)n : sizeof(p), p);
        if (!rc) rc = payload(row, p, n, models, required);
        if (rc) return rc;
    }
    return 0;
}
void aotx_runtime_appraisal_require(unsigned char *header, const aotx_runtime_appraisal_models *models) {
    static const unsigned char processor[32] = AOTX_APPRAISAL_PROCESSOR_BYTES;
    aotx_ccir_put(header + 20, aotx_ccir_u32(header + 20) | AOTX_RUNTIME_APPRAISAL, 4);
    aotx_ccir_put(header + 188, 1, 4); memcpy(header + 192, processor, 32);
    memcpy(header + 224, models->selected, 32);
}
