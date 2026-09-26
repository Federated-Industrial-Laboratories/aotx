/* Purpose: Check versioned control selection and atomic refusal.
 * Owns: Distinct selection rows, exact evidence IDs and defect cases.
 * Launch shape: One thread for each row at N=1 and N=AOTX_SLOTS.
 * Lifetime: One synthetic qualification test. */
#ifndef AOTX_TEST_CONTROL_SELECTION_H
#define AOTX_TEST_CONTROL_SELECTION_H
#include "model/selection.cuh"
#ifdef AOTX_AFFECT
static const unsigned aotx_selection_case_count = 17;
#else
static const unsigned aotx_selection_case_count = 16;
#endif
__global__ void aotx_selection_fixture(unsigned defect) {
    for (unsigned v = 0; v < 2; ++v) aotx_conduct.vector[v].permit.digest[0] = (unsigned char)(11 + v);
    if (defect == 8 || defect == 9)
        for (unsigned v = 0; v < 2; ++v) aotx_conduct.vector[v].permit.status = defect == 8 ? 0 : 2;
    if (defect == 14) aotx_conduct.vector[1].permit.digest[0] = 11;
    if (defect == 15)
        for (unsigned v = 0; v < 2; ++v) aotx_conduct.vector[v].identity.model[23] ^= 1;
}
__global__ void aotx_selection_requests(unsigned count, unsigned defect) {
    unsigned i = threadIdx.x;
    if (i >= count) return;
    unsigned char p[48] = {};
    p[0] = 1; p[4] = 1; p[16] = (unsigned char)(11 + (defect == 14 ? 0 : i % 2));
    int dose = i % 2 ? 10000 : 5000;
    if (defect == 4) dose = 7500;
    if (defect == 5) dose = -5000;
    for (unsigned j = 0; j < 4; ++j) p[8 + j] = (unsigned char)((unsigned)dose >> (8 * j));
    if (defect == 1) p[0] = 2;
    if (defect == 2) p[4] = 2;
    if (defect == 3) p[16] = 0;
    if (defect == 6) p[16] = 93;
    if (defect == 12) p[12] = 1;
    if (defect == 13) for (unsigned j = 0; j < 48; ++j) p[j] = 0;
    aotx_model_how how = {}; how.seed = 501 + i; how.temperature = i * 0.01f;
    how.voice = AOTX_MODEL_CONDUCT_NONE; how.steer_from = 37 + i;
    for (unsigned j = 0; j < AOTX_MODEL_STEERS; ++j) how.steer[j] = AOTX_MODEL_CONDUCT_NONE;
    if (defect == 10) how.affect = 1;
    if (defect == 11) { how.voice = 0; how.voice_scale = 1; }
#ifdef AOTX_AFFECT
    if (defect == 16) how.steer[AOTX_MODEL_CONDUCT_AFFECT] = 0;
#endif
    aotx_model_how before = how;
    unsigned status = aotx_control_select(p, defect == 7 ? AOTX_MODEL_ROLES : AOTX_MODEL_LANGUAGE, &how);
    unsigned wanted = !defect || defect == 13 ? 200 :
        defect == 1 || defect == 2 || defect == 3 || defect == 12 ? 400 : 503;
    bool good = status == wanted;
    if (status != 200 || defect == 13) {
        const unsigned char *a = (const unsigned char *)&before, *b = (const unsigned char *)&how;
        for (unsigned j = 0; j < sizeof(how); ++j) good &= a[j] == b[j];
    } else good &= how.steer[0] == i % 2 && how.steer_strength[0] == dose / 10000.0f &&
        how.seed == before.seed && how.temperature == before.temperature && how.steer_from == before.steer_from;
    results[i] = good;
}
static void aotx_selection_cases(unsigned count, unsigned *cases, unsigned *failed) {
    unsigned out[AOTX_SLOTS];
    for (unsigned defect = 0; defect < aotx_selection_case_count; ++defect) {
        aotx_qualification_fixture<<<1, 1>>>(count, 0);
        aotx_selection_fixture<<<1, 1>>>(defect);
        aotx_selection_requests<<<1, AOTX_SLOTS>>>(count, defect);
        aotx_check_runtime(cudaMemcpyFromSymbol(out, results, count * sizeof(*out)), "cudaMemcpyFromSymbol");
        for (unsigned i = 0; i < count; ++i) {
            ++*cases;
            if (!out[i]) { ++*failed; printf("selection case %u row %u failed\n", defect, i); }
        }
    }
}
#endif
