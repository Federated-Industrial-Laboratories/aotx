/* Purpose: Define the batched creator policy input and output contract.
 * Owns: Fixed-width observations and proposals; no device addresses are saved.
 * Launch shape: Rows contain independent observations and state byte spans.
 * Lifetime: Policy ABI 1. */
#ifndef AOTX_POLICY_ABI_H
#define AOTX_POLICY_ABI_H
#include <stdint.h>
#define AOTX_POLICY_ABI 1u
#define AOTX_POLICY_SUPPLIED 1u
#define AOTX_POLICY_RULES 2u
#define AOTX_POLICY_NATIVE 3u
#define AOTX_POLICY_QUIET 0u
#define AOTX_POLICY_MAINTAIN 1u
#define AOTX_POLICY_REASON_PRESSURE 1ull
#define AOTX_POLICY_REASON_INTERVAL 2ull
#ifndef AOTX_POLICY_STATE_BYTES
#define AOTX_POLICY_STATE_BYTES 65536u
#endif
#ifndef AOTX_POLICY_IMAGE_BYTES
#define AOTX_POLICY_IMAGE_BYTES 16777216u
#endif
typedef struct aotx_policy_input {
    uint64_t source, root, objects, bytes, object_capacity, byte_capacity;
    uint64_t previous_source, previous_root, decision;
    uint32_t enabled, pressure, keep_recent, max_age, foreground, paused, valid, reserved0;
    uint32_t rule_pressure, minimum_move, backoff, reserved1[3];
} aotx_policy_input;
typedef struct aotx_policy_output {
    uint32_t action, status;
    uint64_t reason, reserved[6];
} aotx_policy_output;
typedef struct aotx_policy_config {
    uint32_t mode, state_schema, state_bytes, architecture;
    uint32_t threads, registers, shared_bytes, local_bytes;
    uint32_t pressure, minimum_move, backoff, format;
} aotx_policy_config;
/* Native entry parameters, in order:
 * const aotx_policy_input *, const unsigned char *, aotx_policy_output *,
 * unsigned char *, uint32_t rows, uint32_t state_stride.
 * Each active row reads its prior state and writes its own next-state span.
 * Invalid rows perform no work. State contains portable bytes, never pointers. */
#ifdef __cplusplus
static_assert(sizeof(aotx_policy_input) == 128, "policy input size");
static_assert(sizeof(aotx_policy_output) == 64, "policy output size");
static_assert(sizeof(aotx_policy_config) == 48, "policy configuration size");
#endif
#endif
