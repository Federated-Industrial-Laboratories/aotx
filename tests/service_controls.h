/* Purpose: Check control availability through real service information requests.
 * Owns: Synthetic evidence rows and distinct principal request batches.
 * Launch shape: Service admission batches at N=1 and N=AOTX_SLOTS.
 * Lifetime: Fixtures release every mailbox and clear the control table. */
#ifndef AOTX_TEST_SERVICE_CONTROLS_H
#define AOTX_TEST_SERVICE_CONTROLS_H
#include "model/conduct.cuh"
#include "model/control.cuh"
__global__ void aotx_service_test_controls(unsigned count, unsigned defect) {
    aotx_model[AOTX_MODEL_LANGUAGE].layers = 1;
    aotx_conduct = {}; aotx_conduct.vectors = 1;
    aotx_steer_vector *v = aotx_conduct.vector;
    for (unsigned i = 0; i < 32; ++i) v->identity.model[i] = aotx_model_load.resident[AOTX_MODEL_LANGUAGE].body.digest[i];
    v->identity.wrap = aotx_model_wrap[AOTX_MODEL_LANGUAGE]; v->identity.wrap.usable = 0;
    v->name[0] = 't'; v->layers = 5; v->positions = 1;
    v->permit.status = defect == 1 ? 0 : 1; v->permit.count = 2;
    v->permit.dose[0] = 5000; v->permit.dose[1] = 10000; v->permit.digest[0] = 17;
    if (defect == 2) v->identity.model[21] ^= 1;
    if (defect == 3) for (unsigned i = 0; i <= count; ++i) aotx_service.grants[i].models = 1;
}
static void control_information(unsigned n) {
    for (unsigned defect = 0; defect < 4; ++defect) {
        fixture f(n);
        aotx_service_test_controls<<<1, 1>>>(n, defect); cu(cudaDeviceSynchronize());
        for (unsigned i = 0; i < n; ++i) f.send(i + 1, frame(AOTX_SERVICE_INFO, i + 1));
        f.tick();
        for (unsigned i = 0; i < n; ++i) {
            check(f.status(i + 1) == 200, "control information is available to the principal");
            const unsigned char *p = f.host[i + 1].bytes + AOTX_SERVICE_HEAD;
            check(aotx_service_get(p, 4) == 2 && aotx_service_get(p + 164, 4) == 160, "control schema and row size are explicit");
            check(aotx_service_get(p + 160, 4) == (defect == 3 ? 0 : 1), "foreign model controls are excluded");
            if (defect == 3) continue;
            const unsigned char *r = p + 192 + aotx_service_get(p + 44, 4) * 40;
            check(aotx_service_get(r + 8, 4) == (defect == 0), "control availability follows evidence and the live model");
            check(aotx_service_get(r + 20, 4) == (defect == 0 ? 2 : 0), "only accepted doses are reported");
            check(r[32] == 't' && r[33] == 0 && r[64] == 17, "control name and evidence identity remain exact");
            check(aotx_service_get(r + 96, 4) == (defect == 0 ? 5000 : 0), "the declared dose reaches the service");
        }
    }
    aotx_conduct_table none = {}; cu(cudaMemcpyToSymbol(aotx_conduct, &none, sizeof(none)));
}
#endif
