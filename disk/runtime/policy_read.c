/* Purpose: Validate required selected and historical policy assets.
 * Owns: Complete bounded disk buffers and parsed revision descriptors.
 * Threading: One reader retains the source file lease.
 * Lifetime: The caller releases returned bundle and history buffers. */
#include "disk/runtime/policy.h"
#include "disk/ccir/internal.h"
#include <stdlib.h>
#include <string.h>

int aotx_runtime_policy_read(const aotx_ccir_view *view, const aotx_runtime_index *index,
    aotx_policy_file *selected, aotx_policy_history *history, unsigned char **raw, size_t *bytes) {
    memset(selected, 0, sizeof(*selected)); memset(history, 0, sizeof(*history));
    *raw = NULL; *bytes = 0;
    uint32_t features = aotx_ccir_u32(index->header + 20);
    if (!(features & AOTX_RUNTIME_POLICY)) return features & AOTX_RUNTIME_POLICY_HISTORY ? AOTX_CCIR_INVALID : 0;
    const aotx_ccir_section *policy = NULL, *prior = NULL;
    for (uint32_t i = 0; i < index->count; ++i) {
        const unsigned char *row = index->rows[i];
        int at = aotx_runtime_section(view, row);
        if (at < 0) return AOTX_CCIR_INVALID;
        if (!strcmp((const char *)row + 64, "policy.bin")) policy = view->sections + at;
        if (!strcmp((const char *)row + 64, "policy-history.bin")) prior = view->sections + at;
    }
    if (!policy || !!prior != !!(features & AOTX_RUNTIME_POLICY_HISTORY)) return AOTX_CCIR_INVALID;
    int rc = aotx_policy_file_extent(view->fd, policy->offset, policy->bytes, selected);
    if (!rc && selected->config.abi == AOTX_POLICY_REVIEW_ABI && !(features & AOTX_RUNTIME_REVIEW)) rc = AOTX_CCIR_UNSUPPORTED;
    if (!rc && memcmp(selected->digest, policy->digest, 32)) rc = AOTX_CCIR_INVALID;
    if (!rc && selected->config.mode == AOTX_POLICY_NATIVE &&
        selected->config.architecture != aotx_ccir_u32(index->header + 36)) rc = AOTX_CCIR_UNSUPPORTED;
    if (!rc && prior) {
        if (prior->bytes > AOTX_POLICY_HISTORY_CAPACITY) rc = AOTX_CCIR_LIMIT;
        else {
            *raw = malloc((size_t)prior->bytes);
            if (!*raw) rc = AOTX_CCIR_IO;
            else {
                *bytes = (size_t)prior->bytes;
                rc = aotx_ccir_pread(view->fd, *raw, *bytes, prior->offset);
                if (!rc) rc = aotx_policy_history_decode(*raw, *bytes, selected, history);
            }
        }
    }
    if (rc) {
        free(*raw); *raw = NULL; *bytes = 0; aotx_policy_file_close(selected);
    }
    return rc == AOTX_POLICY_FILE_DIGEST ? AOTX_CCIR_INVALID : rc;
}
