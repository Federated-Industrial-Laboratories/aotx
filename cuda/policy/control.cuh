/* Purpose: Define recorded policy operator controls shared by local and native clients.
 * Owns: Revision checks and portable command constants.
 * Launch shape: Ordered command batches on the device.
 * Lifetime: Exact input replay and scoped native requests. */
#ifndef AOTX_POLICY_CONTROL_CUH
#define AOTX_POLICY_CONTROL_CUH
#include "policy/state.cuh"
#define AOTX_POLICY_PAUSE 1u
#define AOTX_POLICY_RESUME 2u
#define AOTX_POLICY_STOP 3u
#define AOTX_POLICY_REVIEW_ON 4u
#define AOTX_POLICY_REVIEW_OFF 5u
#define AOTX_POLICY_CONTROL_BYTES 64u
__device__ uint32_t aotx_policy_control_check(uint32_t, uint64_t);
__device__ void aotx_policy_control_apply(uint32_t);
__device__ bool aotx_policy_control_part(const unsigned char *, uint32_t, uint32_t);
#endif
