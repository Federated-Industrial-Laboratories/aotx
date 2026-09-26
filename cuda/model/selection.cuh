/* Purpose: Select a qualified control through one versioned request field.
 * Owns: No state; requests retain the exact qualification digest and dose.
 * Launch shape: One caller for each request in the admission batch.
 * Lifetime: Selection is checked at admission and before execution. */
#ifndef AOTX_MODEL_SELECTION_CUH
#define AOTX_MODEL_SELECTION_CUH
#include "model/conduct.cuh"
#define AOTX_CONTROL_SELECTION_BYTES 48u
/* Zero means off. Version, kind, signed dose, zero, and digest occupy 48 bytes. */
__device__ bool aotx_control_selection_shape(const unsigned char *bytes);
__device__ unsigned aotx_control_select(const unsigned char *bytes, unsigned role,
    aotx_model_how *how);
#endif
