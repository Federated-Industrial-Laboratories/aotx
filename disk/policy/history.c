/* Purpose: Check exact revision bundles and compatible state declarations.
 * Owns: History framing and bounded descriptor extraction.
 * Threading: One caller validates the complete ordered revision batch.
 * Lifetime: Temporary bundle buffers are released before return. */
#include "disk/policy/history.h"
#include "disk/ccir/internal.h"
#include <string.h>

int aotx_policy_compatible(const aotx_policy_config *a, const aotx_policy_config *b) {
    return a->abi == b->abi && a->state_schema == b->state_schema && a->state_bytes == b->state_bytes;
}
int aotx_policy_history_decode(const void *data, size_t bytes, const aotx_policy_file *selected,
    aotx_policy_history *out) {
    memset(out, 0, sizeof(*out));
    if (!data || !selected || bytes < 64) return AOTX_CCIR_INVALID;
    if (bytes > AOTX_POLICY_HISTORY_CAPACITY) return AOTX_CCIR_LIMIT;
    const unsigned char *p = data;
    if (memcmp(p, "AOTXPH01", 8) || aotx_ccir_u32(p + 8) != 1 ||
        memcmp(p + 16, selected->digest, 32) || !aotx_ccir_zero(p + 48, 16)) return AOTX_CCIR_INVALID;
    uint32_t count = aotx_ccir_u32(p + 12);
    if (!count || count > AOTX_POLICY_REVISIONS) return AOTX_CCIR_LIMIT;
    size_t at = 64;
    for (uint32_t i = 0; i < count; ++i) {
        if (bytes - at < 64) return AOTX_CCIR_INVALID;
        uint64_t length = aotx_ccir_u64(p + at), last = aotx_ccir_u64(p + at + 8);
        if (!aotx_ccir_zero(p + at + 48, 16) || length > bytes - at - 64 ||
            (i && last < out->rows[i - 1].last_decision)) return AOTX_CCIR_INVALID;
        aotx_policy_file file = {0};
        int rc = aotx_policy_file_decode(p + at + 64, (size_t)length, &file);
        if (!rc && (memcmp(file.digest, p + at + 16, 32) ||
            !memcmp(file.digest, selected->digest, 32))) rc = AOTX_CCIR_INVALID;
        if (!rc && !aotx_policy_compatible(&file.config, &selected->config)) rc = AOTX_CCIR_UNSUPPORTED;
        for (uint32_t j = 0; !rc && j < i; ++j)
            if (!memcmp(file.digest, out->rows[j].digest, 32)) rc = AOTX_CCIR_INVALID;
        if (!rc) {
            out->rows[i].config = file.config;
            memcpy(out->rows[i].digest, file.digest, 32);
            out->rows[i].last_decision = last;
        }
        aotx_policy_file_close(&file);
        if (rc) return rc == AOTX_POLICY_FILE_DIGEST ? AOTX_CCIR_INVALID : rc;
        at += 64 + (size_t)length;
    }
    if (at != bytes) return AOTX_CCIR_INVALID;
    out->count = count;
    return 0;
}
