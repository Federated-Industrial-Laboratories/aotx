/* Purpose: Read a complete CCIR memory checkpoint for device admission.
 * Owns: Bounded file buffers and framing checks; no object interpretation.
 * Threading: One disk reader copies the two required state extents as a batch.
 * Lifetime: One file lease and one owned output image. */
#include "cognitive/checkpoint_io.h"
#include <stdlib.h>
#include <string.h>

uint64_t aotx_cp_get(const unsigned char *p, uint32_t bytes) {
    uint64_t value = 0;
    for (uint32_t j = 0; j < bytes; ++j) value |= (uint64_t)p[j] << (8 * j);
    return value;
}
int aotx_checkpoint_framing(const unsigned char *p, uint64_t bytes, uint32_t *base) {
    if (bytes < AOTX_CP_HEADER || bytes > AOTX_CP_BYTES || memcmp(p, "AOTXLCP1", 8) ||
        aotx_cp_get(p + 8, 4) != 1 || aotx_cp_get(p + 12, 4) != AOTX_CP_ROW ||
        aotx_cp_get(p + 16, 4) > AOTX_SLOTS || aotx_cp_get(p + 20, 4)) return AOTX_CCIR_INVALID;
    *base = AOTX_CP_HEADER + (uint32_t)aotx_cp_get(p + 16, 4) * AOTX_CP_ROW;
    uint64_t length = aotx_cp_get(p + 24, 8);
    if (length < AOTX_COG_HEADER || length > AOTX_COG_IMAGE || *base + length != bytes)
        return AOTX_CCIR_LIMIT;
    const unsigned char *object = p + *base;
    if (memcmp(object, "AOTXOBJ1", 8) || memcmp(object + 48, p + 32, 16) ||
        aotx_cp_get(object + 32, 8) != aotx_cp_get(p + 48, 8) ||
        aotx_cp_get(object + 40, 8) != aotx_cp_get(p + 56, 8) ||
        aotx_cp_get(object + 80, 8) != length) return AOTX_CCIR_INVALID;
    return AOTX_CCIR_OK;
}
int aotx_checkpoint_file_read(const char *path, unsigned char **image, uint32_t *bytes) {
    aotx_ccir_view view;
    int status = aotx_ccir_open(path, NULL, &view);
    if (status) return status;
    uint32_t live = UINT32_MAX, state = UINT32_MAX;
    for (uint32_t j = 0; j < view.count; ++j) {
        if (view.sections[j].type == AOTX_CCIR_LIVE && (view.sections[j].flags & AOTX_CCIR_REQUIRED)) live = j;
        if (view.sections[j].type == AOTX_CCIR_CHECKPOINT) state = j;
    }
    unsigned char *data = NULL;
    uint64_t total = 0;
    if (live == UINT32_MAX || state == UINT32_MAX) status = AOTX_CCIR_UNSUPPORTED;
    else {
        uint64_t a = view.sections[live].bytes, b = view.sections[state].bytes;
        if (a < AOTX_CP_HEADER || a > AOTX_CP_BINDINGS || b < AOTX_COG_HEADER || b > AOTX_COG_IMAGE)
            status = AOTX_CCIR_LIMIT;
        else {
            total = a + b; data = malloc((size_t)total);
            if (!data) status = AOTX_CCIR_IO;
            else {
                aotx_ccir_read reads[2] = {{live, 0, (size_t)a, data}, {state, 0, (size_t)b, data + a}};
                status = aotx_ccir_read_batch(&view, reads, 2);
                uint32_t base = 0;
                if (!status) status = aotx_checkpoint_framing(data, total, &base);
                if (!status && (base != a || memcmp(data + 32, view.lineage, 16) || aotx_cp_get(data + 48, 8) != view.meta.durable_sequence ||
                    aotx_cp_get(data + 56, 8) != view.meta.source_tick)) status = AOTX_CCIR_INVALID;
            }
        }
    }
    aotx_ccir_close(&view);
    if (status) free(data);
    else { *image = data; *bytes = (uint32_t)total; }
    return status;
}
