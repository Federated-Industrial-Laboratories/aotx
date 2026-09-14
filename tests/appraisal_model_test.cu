/* Purpose: Verify appraisal source attribution and correction authority checks.
 * Owns: Independent wrong-actor, scope, version, quote and task refusal fixtures.
 * Launch shape: N=1 and N=64 distinct sources and prior assessment objects.
 * Lifetime: One maintained test process without model weights. */
#include "appraisal_model_fixture.h"

__global__ void aotx_appraisal_model_fault(unsigned n, unsigned mode) {
    unsigned i = threadIdx.x; if (i >= n) return;
    aotx_appraisal_row *r = aotx_appraisal.rows + i;
    unsigned char *old = aotx_live_store.objects[n + i], *source = aotx_live_store.objects[2 * n + i];
    unsigned char *prior = aotx_live_store.objects[i], *task = aotx_live_store.objects[3 * n + i];
    unsigned char *p = aotx_live_store.payload + aotx_cog_u64(old + AOTX_CO_OFFSET);
    switch (mode) {
    case 1: old[AOTX_CO_SUBJECT] ^= 1; break;
    case 2: old[AOTX_CO_OWNER] ^= 1; break;
    case 3: old[AOTX_CO_ROOM] ^= 1; break;
    case 4: old[AOTX_CO_SCOPE] = (old[AOTX_CO_SCOPE] + 1) % 3; break;
    case 5: old[AOTX_CO_FLAGS] = AOTX_COG_PROTECTED; break;
    case 6: old[AOTX_CO_SOURCE_KIND] = AOTX_COG_AUTHORED; break;
    case 7: old[AOTX_CO_FLAGS] = AOTX_COG_TOMBSTONE; break;
    case 8: old[AOTX_CO_EVIDENCE] = 3; break;
    case 9: old[AOTX_CO_EXPIRY] = 1; break;
    case 10:
        for (unsigned j = 0; j < 16; ++j) source[AOTX_CO_SUPERSEDES + j] = old[AOTX_CO_ID + j];
        source[AOTX_CO_SUPER_VERSION] = 1; break;
    case 11: r->prior[0] = UINT32_MAX; break;
    case 12: old[AOTX_CO_SOURCE_VERSION] = 2; break;
    case 13: prior[AOTX_CO_SUBJECT] ^= 1; break;
    case 14: prior[AOTX_CO_BYTES] = 1; break;
    case 15: for (unsigned j = 0; j < 16; ++j) source[AOTX_CO_SUBJECT + j] = 0; break;
    case 16: aotx_live.requests[64 + i * AOTX_RECALL_QUERY + 4640] = 'X'; break;
    case 17: for (unsigned j = 0; j < 16; ++j) r->task[j] = 0; break;
    case 18: for (unsigned j = 0; j < 4; ++j) p[120 + j] = 255; break;
    case 19: source[AOTX_CO_BYTES] = 1; break;
    case 20: r->prior_count = 0; break;
    case 21: p[0] = 1; break;
    case 22: prior[AOTX_CO_FLAGS] = AOTX_COG_TOMBSTONE; break;
    case 23: prior[AOTX_CO_EVIDENCE] = 3; break;
    case 24: prior[AOTX_CO_EXPIRY] = 1; break;
    case 25: source[AOTX_CO_SOURCE_KIND] = AOTX_COG_INFERRED; break;
    case 26: prior[AOTX_CO_SOURCE_KIND] = AOTX_COG_INFERRED; break;
    case 27: r->task_source = UINT32_MAX; break;
    case 28: task[AOTX_CO_OWNER] ^= 1; break;
    case 29: task[AOTX_CO_SCOPE] = (task[AOTX_CO_SCOPE] + 1) % 3; break;
    case 30: task[AOTX_CO_SOURCE_KIND] = AOTX_COG_INFERRED; break;
    case 31: task[AOTX_CO_FLAGS] = AOTX_COG_TOMBSTONE; break;
    case 32: task[AOTX_CO_EXPIRY] = 1; break;
    case 33: task[AOTX_CO_KIND] = AOTX_COG_ASSERTION; break;
    case 34: task[AOTX_CO_ID] ^= 1; break;
    case 35: task[AOTX_CO_BYTES] = 1; break;
    case 36: task[AOTX_CO_EVIDENCE] = 3; break;
    case 37:
        for (unsigned j = 0; j < 16; ++j) source[AOTX_CO_SUPERSEDES + j] = task[AOTX_CO_ID + j];
        source[AOTX_CO_SUPER_VERSION] = 1; break;
    }
}
static void aotx_appraisal_model_authority(unsigned n) {
    aotx_appraisal_model_device d(n);
    for (unsigned mode = 0; mode < 38; ++mode) {
        d.reset(); aotx_appraisal_model_fault<<<1,64>>>(n, mode); aotx_intake_index<<<n,64>>>();
        AOTX_CUDA(cudaDeviceSynchronize());
        std::vector<std::string> responses;
        for (unsigned i = 0; i < n; ++i) {
            auto task = "task " + std::to_string(i);
            responses.push_back(aotx_appraisal_response("I helped with " + task + ".",
                "700000,800000,4294967295,3,400000,900000,200000,600000,100000", task, "", "1"));
        }
        d.responses(responses);
        aotx_appraisal_model_parse<<<1,64>>>(d.reply, d.lengths, d.out, n, 1); AOTX_CUDA(cudaDeviceSynchronize());
        for (unsigned i = 0; i < n; ++i) {
            aotx_check((d.out[i * 16 + 1] == AOTX_COG_OK) == (mode == 0), "independent final admission rejects each invalid source or correction authority");
            if (mode != 15 && mode != 16 && mode != 19 && mode != 25)
                aotx_check(d.out[i * 16] == (mode == 0), "prefix admission excludes unavailable tasks and ineligible correction indexes");
        }
    }
}
static void aotx_appraisal_model_unknown(unsigned n) {
    aotx_appraisal_model_device d(n);
    aotx_appraisal_model_fault<<<1,64>>>(n, 17); AOTX_CUDA(cudaDeviceSynchronize());
    std::vector<std::string> responses;
    unsigned expected[64][7] = {};
    for (unsigned i = 0; i < n; ++i) {
        const unsigned values[7] = {700000 + i, 800000 - i, 300000 + i,
            1 + i % 4, 400000 + i, 900000 - i, 200000 + i};
        std::string numbers;
        for (unsigned j = 0; j < 7; ++j) {
            expected[i][j] = values[j];
            numbers += (j ? "," : "") + std::to_string(values[j]);
        }
        responses.push_back(aotx_appraisal_response("I helped with task " + std::to_string(i) + ".",
            numbers + ",4294967295,4294967295"));
    }
    d.responses(responses);
    aotx_appraisal_model_parse<<<1,64>>>(d.reply, d.lengths, d.out, n, 2); AOTX_CUDA(cudaDeviceSynchronize());
    for (unsigned i = 0; i < n; ++i) {
        aotx_check(d.out[i * 16] && !d.out[i * 16 + 1], "unknown task trust is valid when no task is admitted");
        for (unsigned j = 0; j < 7; ++j)
            aotx_check(d.out[i * 16 + 2 + j] == expected[i][j],
                "missing task identity preserves each supported non-trust dimension");
        aotx_check(d.out[i * 16 + 9] == AOTX_COG_UNKNOWN && d.out[i * 16 + 10] == AOTX_COG_UNKNOWN,
            "missing task identity never turns unknown trust into a zero-valued assertion");
    }
}
int main(void) {
    int cards = 0; if (cudaGetDeviceCount(&cards) != cudaSuccess || !cards) return 77;
    for (unsigned n : {1u, 64u}) { aotx_appraisal_model_authority(n); aotx_appraisal_model_unknown(n); }
    printf("appraisal model admission: %u checks, %u failures\n", aotx_checks, aotx_failures);
    return aotx_failures ? 1 : 0;
}
