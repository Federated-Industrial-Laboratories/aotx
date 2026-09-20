/* Purpose: Bind compatible policy revisions to exact saved decision intervals.
 * Owns: Bounded immutable descriptors; state bytes keep their declared layout.
 * Launch shape: One descriptor lookup per replayed policy decision.
 * Lifetime: One complete runtime revision and its subsequent checkpoints. */
#ifndef AOTX_POLICY_HISTORY_H
#define AOTX_POLICY_HISTORY_H
#include "abi.h"
#define AOTX_POLICY_REVISIONS 8u
typedef struct aotx_policy_revision {
    aotx_policy_config config;
    unsigned char digest[32];
    uint32_t reserved;
    uint64_t last_decision;
} aotx_policy_revision;
typedef struct aotx_policy_history {
    uint32_t count, reserved;
    aotx_policy_revision rows[AOTX_POLICY_REVISIONS];
} aotx_policy_history;
#ifdef __cplusplus
static_assert(sizeof(aotx_policy_revision) == 96, "policy revision size");
#endif
#endif
