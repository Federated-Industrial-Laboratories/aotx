/* Purpose: Store the selected tool call form for each model role.
 * Owns: The bounded device call form table.
 * Launch shape: Read by one thread for each reply or prompt.
 * Lifetime: From model load to model release. */
#include "model/call_format.cuh"

__device__ aotx_call_format aotx_model_call_format[AOTX_MODEL_ROLES];
