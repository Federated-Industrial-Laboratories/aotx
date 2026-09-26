/* Purpose: Declare bounded measurement selection and response-position launches.
 * Owns: No allocation; callers supply candidate names and output rows.
 * Launch shape: One thread per registered vector or requested candidate.
 * Lifetime: One isolated measurement program. */
#ifndef AOTX_TOOLS_STEER_MEASURE_CUH
#define AOTX_TOOLS_STEER_MEASURE_CUH
#include "tools/steer_text.cuh"
#include "model/conduct.cuh"
typedef struct aotx_steer_measure {
    char name[AOTX_CONDUCT_NAME_BYTES];
    unsigned id;
    aotx_steer_vector vector;
} aotx_steer_measure;
__global__ void aotx_steer_measure_admit(const aotx_steer_measure *, unsigned);
__global__ void aotx_steer_measure_read(aotx_steer_measure *, unsigned);
int aotx_steer_measure_vectors(aotx_steer_measure *, unsigned);
int aotx_steer_positions_run(aotx_steer_text, unsigned, unsigned, aotx_model_how *);
#endif
