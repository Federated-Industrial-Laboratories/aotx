/* Purpose: Check shared control selection, queued refusal and recorded recovery.
 * Owns: Distinct request text and immutable evidence references.
 * Launch shape: Real shared admission and work batches at N=1 and N=64.
 * Lifetime: A fixture retains its records across table reconstruction. */
#ifndef AOTX_TEST_SHARED_SELECTION_H
#define AOTX_TEST_SHARED_SELECTION_H
#include "model/control.cuh"
#include "model/conduct.cuh"
#include "shared/bridge.cuh"
__global__ void aotx_shared_selection_setup(unsigned available) {
    aotx_model[AOTX_MODEL_LANGUAGE].layers = 1;
    aotx_conduct = {}; aotx_conduct.vectors = 1;
    aotx_steer_vector &v = aotx_conduct.vector[0];
    for (unsigned i = 0; i < 32; ++i) v.identity.model[i] = aotx_model_load.resident[AOTX_MODEL_LANGUAGE].body.digest[i];
    v.identity.wrap = aotx_model_wrap[AOTX_MODEL_LANGUAGE]; v.identity.wrap.usable = 0;
    v.permit.status = available; v.permit.count = 2;
    v.permit.dose[0] = 5000; v.permit.dose[1] = 10000; v.permit.digest[0] = 17;
}
static void shared_selection(unsigned n) {
    fixture f(n); std::vector<std::vector<unsigned char>> p;
    aotx_shared_selection_setup<<<1,1>>>(1); cu(cudaDeviceSynchronize());
    for (unsigned i = 0; i < n; ++i) p.push_back(command(i+1, AOTX_SHARED_REGISTER, 1)); f.batch(p, 202);
    p.clear(); for (unsigned i = 0; i < n; ++i) p.push_back(command(i+1, AOTX_SHARED_SPACE, 2, i+100)); f.batch(p, 202);
    p.clear(); for (unsigned i = 0; i < n; ++i) p.push_back(command(i+1, AOTX_SHARED_CONVERSATION, 3, i+1000, i+100)); f.batch(p, 202);
    p.clear(); for (unsigned i = 0; i < n; ++i) {
        auto r = command(i+1, AOTX_SHARED_INPUT, 4, i+1000, 0, "selected "+std::to_string(i));
        unsigned h = AOTX_SERVICE_HEAD+144;
        put(r, h, 1); put(r, h+4, 1); put(r, h+8, i % 2 ? 10000 : 5000); r[h+16] = 17;
        p.push_back(r);
    }
    auto bad = p; for (auto &r : bad) put(r, AOTX_SERVICE_HEAD+152, 7500); f.batch(bad, 503);
    for (const auto &r : f.receipts()) check(r.operation != AOTX_SHARED_INPUT, "refused controls do not publish an input");
    bad = p; for (auto &r : bad) put(r, AOTX_SERVICE_HEAD+144, 2); f.batch(bad, 400);
    f.batch(p, 202); f.batch(p, 200);
    bad = p; for (auto &r : bad) r[AOTX_SERVICE_HEAD+160] ^= 1; f.batch(bad, 409);
    unsigned selected = 0;
    for (const auto &r : f.receipts()) if (r.operation == AOTX_SHARED_INPUT) {
        ++selected;
        unsigned i = (unsigned)aotx_service_get(r.actor, 4)-1;
        check(!memcmp(r.command, p[i].data()+AOTX_SERVICE_HEAD, p[i].size()-AOTX_SERVICE_HEAD),
            "admission retains every selected identity, dose and input byte");
    }
    check(selected == n, "all selected inputs have distinct receipts");
    aotx_shared_test_ack<<<1,1>>>(); aotx_shared_selection_setup<<<1,1>>>(0); cu(cudaDeviceSynchronize());
    for (unsigned i = 0; i < n; ++i) { aotx_shared_work<<<1,1>>>(); aotx_shared_emit<<<1,1>>>(); cu(cudaDeviceSynchronize()); }
    auto before = f.receipts();
    for (const auto &r : before) if (r.operation == AOTX_SHARED_INPUT)
        check(r.phase == AOTX_SHARED_FAILED && r.status == 503 && r.slot == AOTX_SLOTS && !r.sampled,
            "unavailable evidence refuses queued work before inference");
    cu(cudaMemcpyFromSymbol(&f.seam, aotx_seam, sizeof(f.seam)));
    unsigned count = (unsigned)f.seam.dev.tail;
    unsigned char *saved; cu(cudaMalloc(&saved, (size_t)count*AOTX_SLOT_BYTES));
    cu(cudaMemcpy(saved, f.seam.dev.base, (size_t)count*AOTX_SLOT_BYTES, cudaMemcpyDeviceToDevice));
    cu(cudaMemset(f.shared.participants, 0, f.shared.participant_capacity*sizeof(*f.shared.participants)));
    cu(cudaMemset(f.shared.spaces, 0, f.shared.space_capacity*sizeof(*f.shared.spaces)));
    cu(cudaMemset(f.shared.members, 0, f.shared.member_capacity*sizeof(*f.shared.members)));
    cu(cudaMemset(f.shared.conversations, 0, f.shared.conversation_capacity*sizeof(*f.shared.conversations)));
    cu(cudaMemset(f.shared.receipts, 0, f.shared.receipt_capacity*sizeof(*f.shared.receipts)));
    cu(cudaMemcpyToSymbol(aotx_shared, &f.shared, sizeof(f.shared)));
    aotx_shared_test_replay<<<1,1>>>(saved, count, f.result); cu(cudaDeviceSynchronize());
    check(f.value() == 1, "recorded selection recovers without current control availability");
    auto after = f.receipts(); selected = 0;
    for (unsigned i = 0; i < before.size(); ++i) if (before[i].operation == AOTX_SHARED_INPUT) {
        const auto &a = before[i], &b = after[i]; ++selected;
        check(a.phase == b.phase && a.status == b.status && b.saved_terminal &&
            !memcmp(a.command, b.command, a.length), "recovery preserves exact selection and terminal refusal");
    }
    check(selected == n, "recovery checks every selected request");
    cudaFree(saved); aotx_conduct_table none = {}; cu(cudaMemcpyToSymbol(aotx_conduct, &none, sizeof(none)));
}
#endif
