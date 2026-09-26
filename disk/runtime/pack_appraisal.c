/* Purpose: Declare required appraisal support when prepared memory contains it.
 * Owns: The runtime profile fields; model and memory bytes keep their source owners.
 * Threading: One packager checks the complete prepared object batch.
 * Lifetime: Before the first complete runtime publication. */
#include "disk/runtime/pack.h"
#include "disk/runtime/appraisal.h"

int aotx_runtime_pack_appraisal(aotx_runtime_pack *p) {
    uint32_t required = 0;
    int rc = aotx_runtime_appraisal_scan(p->memory, -1, 0, p->memory_bytes, NULL, &required);
    if (!rc) aotx_ccir_put(p->index.header + 20, aotx_ccir_u32(p->index.header + 20) |
        (required & (AOTX_RUNTIME_REVIEW | AOTX_RUNTIME_COLD)), 4);
    if (rc || !(required & AOTX_RUNTIME_APPRAISAL)) return rc;
    const aotx_ccir_input *manifest = NULL;
    for (uint32_t i = 0; i < p->index.count; ++i) {
        if (strcmp((const char *)p->index.rows[i] + 64, "manifest.jsonl")) continue;
        for (uint32_t j = 0; j < p->count; ++j)
            if (!memcmp(p->index.rows[i], p->inputs[j].section.id, 16)) manifest = p->inputs + j;
    }
    if (!manifest || manifest->source != AOTX_CCIR_FILE) return AOTX_CCIR_INVALID;
    aotx_runtime_appraisal_models models;
    rc = aotx_runtime_appraisal_model_read(manifest->fd, manifest->source_offset, manifest->section.bytes,
        (const char *)p->index.header + 64, &models);
    if (!rc) rc = aotx_runtime_appraisal_scan(p->memory, -1, 0, p->memory_bytes, &models, &required);
    if (!rc) aotx_runtime_appraisal_require(p->index.header, &models);
    return rc;
}
