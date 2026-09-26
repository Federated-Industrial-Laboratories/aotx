/* Purpose: Check selected controls through the service mailbox and queue.
 * Owns: Distinct request content and explicit evidence changes.
 * Launch shape: Real service admission batches at N=1 and N=AOTX_SLOTS.
 * Lifetime: Each fixture releases its requests and control rows. */
#ifndef AOTX_TEST_SERVICE_SELECTION_H
#define AOTX_TEST_SERVICE_SELECTION_H
__global__ void aotx_service_test_control_change(void) { aotx_conduct.vector[0].permit.digest[0] ^= 1; }
static void control_selection(unsigned n) {
    for (unsigned defect = 0; defect < 5; ++defect) {
        fixture f(n);
        aotx_service_test_controls<<<1, 1>>>(n, defect == 3 ? 1 : 0); cu(cudaDeviceSynchronize());
        for (unsigned i = 0; i < n; ++i) {
            auto p = submit(i + 1, i + 201, "selection" + std::to_string(i));
            unsigned at = p.size(); p.resize(at + 48); put(p, 92, 48);
            put(p, at, defect == 2 ? 2 : 1); put(p, at + 4, 1);
            put(p, at + 8, defect == 1 ? 7500 : i % 2 ? 10000 : 5000); p[at + 16] = 17;
            put(p, 88, p.size() - AOTX_SERVICE_HEAD); f.send(i + 1, p);
        }
        f.tick(); auto jobs = f.jobs();
        for (unsigned i = 0; i < n; ++i) {
            check(f.status(i + 1) == (defect == 2 ? 400 : defect == 1 || defect == 3 ? 503 : 202),
                "selected control admission checks the exact evidence and dose");
            check(aotx_service_get(f.host[i + 1].bytes + 92, 4) == 0, "selection bytes cannot become a finish reason");
            if (defect && defect != 4) {
                check(jobs[i].phase == 0, "refused control does not create a request"); continue;
            }
            check(jobs[i].sample.steer[0] == 0 && jobs[i].sample.steer_strength[0] == (i % 2 ? 1.0f : 0.5f),
                "selected dose reaches the request consumer");
            check(jobs[i].control[16] == 17, "the queued request retains its qualification identity");
            check(aotx_service_get(f.host[i + 1].bytes + 92, 4) == 0, "admission does not report a false finish reason");
        }
        if (defect == 4) {
            aotx_service_test_control_change<<<1, 1>>>(); aotx_service_work<<<1, 1>>>(); aotx_service_work<<<1, 1>>>(); cu(cudaDeviceSynchronize());
            jobs = f.jobs();
            for (unsigned i = 0; i < n; ++i)
                check(jobs[i].phase == AOTX_SERVICE_FAILED && jobs[i].status == 503 && jobs[i].slot == AOTX_SLOTS,
                    "a changed qualification refuses queued execution before slot lease");
        }
    }
    aotx_conduct_table none = {}; cu(cudaMemcpyToSymbol(aotx_conduct, &none, sizeof(none)));
}
#endif
