/* Purpose: Check absent control evidence and accepted continuous dose bounds.
 * Owns: Temporary qualification removal and distinct per-agent control states.
 * Launch shape: One block for each agent at N=1 and N=AOTX_SLOTS.
 * Lifetime: The fixture restores all files and table bounds before return. */
#ifndef AOTX_TEST_AFFECT_QUALIFICATION_H
#define AOTX_TEST_AFFECT_QUALIFICATION_H
static void aotx_affect_test_missing_evidence(aotx_affect_test_store *store) {
    const char *names[] = {"affect/valence.aotxprb", "affect/calibration.jsonl"};
    for (unsigned i = 0; i < 2; ++i) {
        char path[2048], saved[2056];
        snprintf(path, sizeof(path), "%s/%s.qualification", store->dir, names[i]);
        snprintf(saved, sizeof(saved), "%s.saved", path);
        int moved = rename(path, saved), loaded = aotx_affect_load_store(store->dir);
        aotx_affect_table probes; aotx_affect_composite_desc composite;
        aotx_check_runtime(cudaMemcpyFromSymbol(&probes, aotx_affect_rows, sizeof(probes)), "cudaMemcpyFromSymbol");
        aotx_check_runtime(cudaMemcpyFromSymbol(&composite, aotx_affect_composite_table, sizeof(composite)), "cudaMemcpyFromSymbol");
        bool unavailable = i ? !composite.trusted : probes.count && probes.row[0].monitor;
        int restored = rename(saved, path);
        aotx_affect_note(i ? "unqualified calibration does not load" : "unqualified probe cannot drive state",
            !moved && !loaded && !restored && unavailable, "unavailable", unavailable, 1);
    }
    aotx_affect_load_store(store->dir);
}
__global__ void aotx_affect_qualification_states(unsigned count) {
    unsigned i = threadIdx.x;
    if (i >= AOTX_SLOTS) return;
    aotx_say.slot[i].wanted = i < count; aotx_say_count[i] = i < count;
    aotx_affect_state[i] = {};
    aotx_affect_state[i].fast[0] = 8192 + 32 * i;
    aotx_affect_state[i].fast[1] = 16384 + 16 * i;
}
static void aotx_affect_test_qualification_bounds(unsigned count) {
    aotx_affect_composite_desc original;
    aotx_check_runtime(cudaMemcpyFromSymbol(&original, aotx_affect_composite_table, sizeof(original)), "cudaMemcpyFromSymbol");
    size_t cells = (size_t)original.layer_count * original.hidden;
    float *values = (float *)malloc(count * cells * sizeof(float)), *device;
    aotx_check_runtime(cudaMemcpyFromSymbol(&device, aotx_affect_steer, sizeof(device)), "cudaMemcpyFromSymbol");
    float basis[2][AOTX_AFFECT_TEST_HIDDEN];
    for (unsigned j = 0; j < 2; ++j) aotx_affect_test_direction(j, original.hidden, basis[j]);
    for (unsigned mode = 0; mode < 3; ++mode) {
        aotx_affect_composite_desc table = original;
        table.permit.dose[0] = mode ? 40000 : 1250;
        table.permit.dose[1] = mode ? 40000 : 2500;
        table.permit.dose[2] = mode ? 100 : 40000;
        if (mode == 2) table.permit.status = 0;
        aotx_check_runtime(cudaMemcpyToSymbol(aotx_affect_composite_table, &table, sizeof(table)), "cudaMemcpyToSymbol");
        aotx_affect_test_actuator_settings(1, 10000, 40000);
        aotx_affect_qualification_states<<<1, AOTX_SLOTS>>>(count);
        aotx_affect_build<<<AOTX_SLOTS, 256>>>();
        aotx_check_runtime(cudaMemcpy(values, device, count * cells * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy");
        aotx_affect_agent_state state[AOTX_SLOTS]; aotx_affect_law_state_get(state);
        unsigned right = 0;
        for (unsigned i = 0; i < count; ++i) {
            float a = (8192 + 32 * i) / 32768.0f, b = (16384 + 16 * i) / 32768.0f;
            float q = 4 * a * a + b * b;
            float scale = mode == 2 ? 0 : mode == 1 ? sqrtf(0.02f / q) : fminf(0.125f / a, 0.25f / b);
            bool good = fabsf(state[i].budget_spent - 0.5f * scale * scale * q) < 1e-6f;
            for (unsigned x = 0; x < original.hidden; ++x)
                good &= fabsf(values[i * cells + x] - scale * (a * basis[0][x] + b * basis[1][x])) < 1e-6f;
            right += good;
        }
        char label[96]; snprintf(label, sizeof(label), "accepted composite bounds %u at %u", mode, count);
        aotx_affect_note(label, right == count, "rows", right, count);
    }
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_affect_composite_table, &original, sizeof(original)), "cudaMemcpyToSymbol");
    free(values);
}
#endif
