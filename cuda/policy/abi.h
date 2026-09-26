/* Purpose: Define the batched creator policy input and output contract.
 * Owns: Fixed-width observations and proposals; no device addresses are saved.
 * Launch shape: Rows contain independent observations and state byte spans.
 * Lifetime: Explicit policy ABI 1, 2 or 3. */
#ifndef AOTX_POLICY_ABI_H
#define AOTX_POLICY_ABI_H
#include <stdint.h>
#define AOTX_POLICY_ABI 1u
#define AOTX_POLICY_APPRAISAL_ABI 2u
#define AOTX_POLICY_REVIEW_ABI 3u
#define AOTX_POLICY_SUPPLIED 1u
#define AOTX_POLICY_RULES 2u
#define AOTX_POLICY_NATIVE 3u
#define AOTX_POLICY_QUIET 0u
#define AOTX_POLICY_MAINTAIN 1u
#define AOTX_POLICY_APPRAISE 2u
#define AOTX_POLICY_REVIEW 3u
#define AOTX_POLICY_REASON_PRESSURE 1ull
#define AOTX_POLICY_REASON_INTERVAL 2ull
#define AOTX_POLICY_REASON_EVIDENCE 4ull
#ifndef AOTX_POLICY_STATE_BYTES
#define AOTX_POLICY_STATE_BYTES 65536u
#endif
#ifndef AOTX_POLICY_IMAGE_BYTES
#define AOTX_POLICY_IMAGE_BYTES 16777216u
#endif
/* ABI 1 keeps all reserved input words zero.
 * ABI 2 sets reserved0 to 2. The enabled field selects maintenance.
 * reserved1 holds the admitted appraisal count, then the low and high work revision words.
 * Appraisal work does not require enabled maintenance.
 * ABI 3 sets reserved0 to 3. The enabled bits select maintenance (1) and eligible reviews (2).
 * reserved1 keeps the appraisal count and a control/work revision. */
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
    uint32_t pressure, minimum_move, backoff, format, abi;
} aotx_policy_config;
/* Native entry parameters, in order:
 * const aotx_policy_input *, const unsigned char *, aotx_policy_output *,
 * unsigned char *, uint32_t rows, uint32_t state_stride.
 * Each active row reads its prior state and writes its own next-state span.
 * Invalid rows perform no work. State contains portable bytes, never pointers. */
#ifdef __cplusplus
static_assert(sizeof(aotx_policy_input) == 128, "policy input size");
static_assert(sizeof(aotx_policy_output) == 64, "policy output size");
static_assert(sizeof(aotx_policy_config) == 52, "policy configuration size");
#endif
#endif
