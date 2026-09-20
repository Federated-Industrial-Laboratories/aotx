/* Purpose: Validate complete policy history assets without code execution.
 * Owns: Portable history framing and declared identity conversion checks.
 * Threading: One leased disk reader checks the complete revision batch.
 * Lifetime: History descriptors remain valid independently of read buffers. */
#ifndef AOTX_POLICY_HISTORY_FILE_H
#define AOTX_POLICY_HISTORY_FILE_H
#include "disk/policy/file.h"
#include "cuda/policy/history.h"
#define AOTX_POLICY_HISTORY_HEADER 64u
#define AOTX_POLICY_HISTORY_CAPACITY (64ull + AOTX_POLICY_REVISIONS * \
    (64ull + AOTX_POLICY_FILE_HEADER + AOTX_POLICY_IMAGE_BYTES + 2ull * AOTX_POLICY_METADATA_BYTES))
#ifdef __cplusplus
extern "C" {
#endif
int aotx_policy_compatible(const aotx_policy_config *before, const aotx_policy_config *after);
int aotx_policy_history_decode(const void *data, size_t bytes, const aotx_policy_file *selected,
    aotx_policy_history *out);
#ifdef __cplusplus
}
#endif
#endif
