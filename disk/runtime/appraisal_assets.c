/* Purpose: Bind appraisal runtime metadata to packaged language model assets.
 * Owns: Manifest buffers and exact processor/model dependency admission.
 * Threading: One disk caller checks all referenced memory rows and model assets.
 * Lifetime: One complete runtime inspection or checkpoint update. */
#include "disk/runtime/appraisal.h"
#include "disk/modelfile/manifest.h"
#include <stdlib.h>

static int selected(const char *roles, const char *role) {
    size_t n = strlen(role);
    for (const char *p = roles; *p;) {
        const char *end = strchr(p, ','); size_t bytes = end ? (size_t)(end - p) : strlen(p);
        if (bytes == n && !memcmp(p, role, n)) return 1;
        if (!end) break;
        p = end + 1;
    }
    return 0;
}
int aotx_runtime_appraisal_model_read(int fd, uint64_t offset, uint64_t bytes,
    const char *roles, aotx_runtime_appraisal_models *models) {
    memset(models, 0, sizeof(*models));
    if (!bytes || bytes > 8u * AOTX_MANIFEST_LINE) return AOTX_CCIR_LIMIT;
    char *text = malloc((size_t)bytes + 1);
    if (!text) return AOTX_CCIR_IO;
    int rc = aotx_ccir_pread(fd, text, (size_t)bytes, offset);
    if (!rc && memchr(text, 0, (size_t)bytes)) rc = AOTX_CCIR_INVALID;
    text[bytes] = 0;
    char *save = NULL; unsigned count = 0, active = 0;
    for (char *line = !rc ? strtok_r(text, "\n", &save) : NULL; line && !rc; line = strtok_r(NULL, "\n", &save)) {
        aotx_manifest_entry e;
        if (++count > 8 || strlen(line) >= AOTX_MANIFEST_LINE || aotx_manifest_line(line, &e)) {
            rc = AOTX_CCIR_INVALID; break;
        }
        if (strcmp(e.role, "language") && strcmp(e.role, "language-q4")) continue;
        unsigned char *digest = models->digest[models->count++];
        if (aotx_manifest_digest(e.sha256, digest)) { rc = AOTX_CCIR_INVALID; break; }
        if (selected(roles, e.role)) { memcpy(models->selected, digest, 32); ++active; }
    }
    free(text);
    return rc ? rc : active == 1 && !aotx_ccir_zero(models->selected, 32) ? 0 : AOTX_CCIR_UNSUPPORTED;
}
int aotx_runtime_appraisal_model_view(const aotx_ccir_view *view, const aotx_runtime_index *index,
    aotx_runtime_appraisal_models *models) {
    int at = -1;
    for (uint32_t i = 0; i < index->count; ++i)
        if (!strcmp((const char *)index->rows[i] + 64, "manifest.jsonl")) at = aotx_runtime_section(view, index->rows[i]);
    if (at < 0) return AOTX_CCIR_INVALID;
    int rc = aotx_runtime_appraisal_model_read(view->fd, view->sections[at].offset, view->sections[at].bytes,
        (const char *)index->header + 64, models);
    for (uint32_t i = 0; !rc && i < models->count; ++i) {
        unsigned found = 0;
        for (uint32_t j = 0; j < index->count; ++j)
            found += aotx_ccir_u32(index->rows[j] + 16) == 1 && !memcmp(models->digest[i], index->rows[j] + 32, 32);
        if (!found) rc = AOTX_CCIR_UNSUPPORTED;
    }
    return rc;
}
int aotx_runtime_appraisal_dependencies(const aotx_ccir_view *view, const aotx_runtime_index *index) {
    const aotx_ccir_section *memory = NULL;
    for (uint32_t i = 0; i < view->count; ++i)
        if (view->sections[i].type == AOTX_CCIR_CHECKPOINT) memory = view->sections + i;
    if (!memory) return AOTX_CCIR_INVALID;
    uint32_t required = 0;
    int rc = aotx_runtime_appraisal_scan(NULL, view->fd, memory->offset, memory->bytes, NULL, &required);
    unsigned declared = aotx_ccir_u32(index->header + 20);
    if (!rc && (required & ~declared)) rc = AOTX_CCIR_UNSUPPORTED;
    if (rc || !(declared & AOTX_RUNTIME_APPRAISAL)) return rc;
    aotx_runtime_appraisal_models models;
    rc = aotx_runtime_appraisal_model_view(view, index, &models);
    if (!rc && memcmp(models.selected, index->header + 224, 32)) rc = AOTX_CCIR_UNSUPPORTED;
    return rc ? rc : aotx_runtime_appraisal_scan(NULL, view->fd, memory->offset, memory->bytes, &models, &required);
}
int aotx_runtime_appraisal_checkpoint(const aotx_ccir_view *view, aotx_runtime_index *index,
    const unsigned char *memory, uint64_t bytes, uint32_t replay_features) {
    uint32_t required = 0;
    int rc = aotx_runtime_appraisal_scan(memory, -1, 0, bytes, NULL, &required);
    required |= replay_features | (aotx_ccir_u32(index->header + 20) & AOTX_RUNTIME_APPRAISAL);
    if (rc || !(required & AOTX_RUNTIME_APPRAISAL)) return rc;
    aotx_runtime_appraisal_models models;
    rc = aotx_runtime_appraisal_model_view(view, index, &models);
    if (!rc) rc = aotx_runtime_appraisal_scan(memory, -1, 0, bytes, &models, &required);
    if (!rc) aotx_runtime_appraisal_require(index->header, &models);
    return rc;
}
