/* Purpose: Read and update declared compatible policy revisions in complete files.
 * Owns: Required asset lookup and exact saved decision boundaries.
 * Threading: One leased source reader; updates publish a separate complete file.
 * Lifetime: Source files remain unchanged throughout conversion. */
#ifndef AOTX_RUNTIME_POLICY_H
#define AOTX_RUNTIME_POLICY_H
#include "disk/runtime/runtime.h"
#include "disk/policy/history.h"
#ifdef __cplusplus
extern "C" {
#endif
int aotx_runtime_policy_read(const aotx_ccir_view *view, const aotx_runtime_index *index,
    aotx_policy_file *selected, aotx_policy_history *history, unsigned char **raw, size_t *bytes);
int aotx_runtime_policy_replay(const aotx_ccir_view *view, const aotx_policy_file *selected,
    const aotx_policy_history *history, uint64_t *last);
int aotx_runtime_policy_update(const char *source, const char *expected, const char *bundle,
    const char *destination);
#ifdef __cplusplus
}
#endif
#endif
