/* Purpose: Transfer the current model identity for fitted asset admission and output.
 * Owns: No persistent state or device allocations.
 * Launch shape: Host glue only; device hooks check the transferred identity.
 * Lifetime: Each call completes before the model graph starts. */
#include "model/control.cuh"
#include "boot/check.h"
#include <stddef.h>
#include <stdio.h>
#include <string.h>
unsigned aotx_control_current(aotx_control_identity *identity) {
    aotx_model_desc desc;
    unsigned role = aotx_model_default_desc(&desc);
    if (!desc.layers) { memset(identity, 0, sizeof(*identity)); return role; }
    aotx_check_runtime(cudaMemcpyFromSymbol(identity->model, aotx_model_load, 32,
        offsetof(aotx_model_load_state, resident) + role * sizeof(aotx_model_resident_row) +
        offsetof(aotx_model_resident_row, body) + offsetof(aotx_model_body, digest)), "cudaMemcpyFromSymbol");
    aotx_check_runtime(cudaMemcpyFromSymbol(&identity->wrap, aotx_model_wrap, sizeof(identity->wrap),
        role * sizeof(aotx_wrap)), "cudaMemcpyFromSymbol");
    identity->wrap.usable = 0;
    identity->wrap.think_open_id = identity->wrap.think_close_id = UINT32_MAX;
    return role;
}
int aotx_control_check(const char *store, const char *name, unsigned kind, FILE *asset, unsigned *positions) {
    aotx_control_identity identity; aotx_control_current(&identity);
    int bad = aotx_control_read(store, name, kind, &identity, asset, positions);
    if (bad) fprintf(stderr, "the control %s has no matching model, turn format, hook or file binding\n", name);
    return bad;
}
int aotx_control_save(const char *path, unsigned kind, unsigned positions) {
    aotx_control_identity identity; aotx_control_current(&identity);
    int bad = aotx_control_write(path, kind, &identity, positions);
    if (bad) fprintf(stderr, "the control binding for %s does not write\n", path);
    return bad;
}
int aotx_control_values(const float *values, unsigned long long count) {
    unsigned *device = NULL, bad = 0;
    if (!values || !count) return 1;
    aotx_check_runtime(cudaMalloc(&device, sizeof(*device)), "cudaMalloc");
    aotx_check_runtime(cudaMemset(device, 0, sizeof(*device)), "cudaMemset");
    aotx_control_values_check<<<256, 256>>>(values, count, device);
    aotx_check_runtime(cudaMemcpy(&bad, device, sizeof(bad), cudaMemcpyDeviceToHost), "cudaMemcpy");
    aotx_check_runtime(cudaFree(device), "cudaFree");
    return bad != 0;
}
