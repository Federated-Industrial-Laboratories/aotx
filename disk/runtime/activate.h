/* Purpose: Validate runtime compatibility and prepare data files for existing disk consumers.
 * Owns: A private temporary directory containing settings and data modules.
 * Threading: One boot process performs activation before accepting input.
 * Lifetime: The temporary files are removed when the boot process ends. */
#ifndef AOTX_RUNTIME_ACTIVATE_H
#define AOTX_RUNTIME_ACTIVATE_H
#include "disk/runtime/runtime.h"
#include "cuda/policy/history.h"
typedef struct aotx_runtime_boot {
    uint32_t mode, owned, features;
    aotx_runtime_shared_profile shared;
    aotx_policy_history policy_history;
    unsigned char revision[32];
    char roles[64], root[1024], modules[1024], settings[1024], policy[1024];
} aotx_runtime_boot;
#ifdef __cplusplus
extern "C" {
#endif
int aotx_runtime_prepare(const char *path, const char *journal, unsigned architecture,
                          aotx_runtime_boot *boot);
int aotx_runtime_promote_shared(const char *path);
void aotx_runtime_release(aotx_runtime_boot *boot);
#ifdef __cplusplus
}
#endif
#endif
