/* Purpose: Capture the tool selection of each agent at the start of a turn.
 * Owns: Conversation choices and immutable masks for active turns.
 * Launch shape: One thread for each agent in the prompt batch.
 * Lifetime: Choices survive turns; settings records rebuild them at restore. */
#ifndef AOTX_TOOL_POLICY_CUH
#define AOTX_TOOL_POLICY_CUH

#include "tool/policy.h"
#include "catalog/catalog.cuh"
#include "cli/cli.cuh"

typedef struct aotx_tool_policy {
    unsigned int choices;
    unsigned int selected;
    unsigned int ready;
    unsigned int effective[AOTX_CATALOG_MASK_WORDS];
} aotx_tool_policy;
extern __device__ aotx_tool_policy aotx_tool_policies[AOTX_SLOTS];

__device__ void aotx_tool_policy_reset(unsigned int agent);
__device__ void aotx_tool_policy_capture(unsigned int agent, unsigned int role);
__device__ unsigned int aotx_tool_policy_selected(unsigned int defaults, unsigned int choices);
__device__ unsigned int aotx_tool_policy_group(unsigned int entry);
__device__ int aotx_tool_policy_enabled(unsigned int agent, unsigned int entry);
__device__ int aotx_tool_policy_allows(unsigned int agent, unsigned int entry);
__device__ int aotx_tool_policy_any(unsigned int agent);
__device__ int aotx_tool_policy_key(const char *key, unsigned int length, unsigned int *agent);
__device__ int aotx_tool_policy_valid(long long choices);
__device__ void aotx_tool_policy_show(unsigned int agent);
__device__ void aotx_tool_policy_show_all(void);
__device__ void aotx_tool_policy_command(aotx_cli_out *out, unsigned int agent,
    const char *name, unsigned int name_len, const char *choice, unsigned int choice_len,
    unsigned int extra);

#endif
