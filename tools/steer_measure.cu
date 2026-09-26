/* Purpose: Select candidate vectors for isolated measurements on the device.
 * Owns: Measurement permits in the registered vector batch.
 * Launch shape: One thread per registered vector, then one per requested candidate.
 * Lifetime: Permits end with the calling measurement program. */
#include "tools/steer_measure.cuh"
static __device__ bool same(const char *a, const char *b) {
    for (unsigned i = 0; i < AOTX_CONDUCT_NAME_BYTES; ++i) {
        if (a[i] != b[i]) return false;
        if (!a[i]) return i != 0;
    }
    return false;
}
__global__ void aotx_steer_measure_admit(const aotx_steer_measure *rows, unsigned count) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= aotx_conduct.vectors) return;
    for (unsigned j = 0; j < count; ++j) if (same(rows[j].name, aotx_conduct.vector[i].name)) {
        aotx_conduct.vector[i].permit = {};
        aotx_conduct.vector[i].permit.status = AOTX_QUALIFICATION_MEASUREMENT;
        return;
    }
}
__global__ void aotx_steer_measure_read(aotx_steer_measure *rows, unsigned count) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    rows[i].id = AOTX_MODEL_CONDUCT_NONE; rows[i].vector = {};
    for (unsigned j = 0; j < aotx_conduct.vectors; ++j) if (same(rows[i].name, aotx_conduct.vector[j].name)) {
        rows[i].id = j; rows[i].vector = aotx_conduct.vector[j]; return;
    }
}
