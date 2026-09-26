/* Purpose: Place a distinct synthetic model identity for fitted control tests.
 * Owns: A resident row and its turn format.
 * Launch shape: Host transfers before the test kernels start.
 * Lifetime: One test process. */
#ifndef AOTX_TEST_CONTROL_FIXTURE_H
#define AOTX_TEST_CONTROL_FIXTURE_H
#include "model/control.cuh"
#include "boot/check.h"
static void aotx_control_test_model(unsigned role) {
    aotx_model_desc desc;
    aotx_check_runtime(cudaMemcpyFromSymbol(&desc, aotx_model, sizeof(desc), role * sizeof(desc)), "cudaMemcpyFromSymbol");
    if (!desc.layers) { desc.layers = 1; desc.role = role; }
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_model, &desc, sizeof(desc), role * sizeof(desc)), "cudaMemcpyToSymbol");
    aotx_model_resident_row resident = {};
    resident.active = 1; resident.slot = role;
    for (unsigned i = 0; i < 32; ++i) resident.body.digest[i] = (unsigned char)(i + 3 * role + 1);
    aotx_wrap wrap = {};
    wrap.end_count = 1; wrap.end_ids[0] = 7; wrap.usable = 1;
    wrap.think_open_id = wrap.think_close_id = UINT32_MAX;
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_model_load, &resident, sizeof(resident),
        offsetof(aotx_model_load_state, resident) + role * sizeof(resident)), "cudaMemcpyToSymbol");
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_model_wrap, &wrap, sizeof(wrap),
        role * sizeof(wrap)), "cudaMemcpyToSymbol");
}
#endif
