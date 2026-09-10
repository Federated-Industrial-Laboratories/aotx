/* Purpose: Verify bounded internal sequence ownership and checked cache release.
 * Owns: Independent tokenizer completion, timeout and full-queue controls.
 * Launch shape: N=1 and N=64 leases through real open, commit and cleanup kernels.
 * Lifetime: One internal pass without an ordinary conversation turn. */
#include "intake_fixture.h"

__global__ void aotx_intake_test_open(unsigned n, unsigned bad) {
    unsigned i = threadIdx.x; if (i >= n) return;
    aotx_say_count[i] = 2; aotx_say_gear.piece_count[i] = 1;
    aotx_say_gear.chunk[i * AOTX_SAY_PIECES] = 2 + (bad == 1 && i + 1 == n);
    aotx_say_id[i * AOTX_SAY_TOKENS] = 1; aotx_say_id[i * AOTX_SAY_TOKENS + 1] = 2;
    if (bad == 2 && i + 1 == n) aotx_live_bindings[i].pages = 0;
    aotx_kv.count[i] = aotx_kvl_pages(&aotx_model_space[AOTX_MODEL_LANGUAGE].shape, AOTX_SEQ_MAX_TOKENS);
    aotx_intake_open(i);
}
__global__ void aotx_intake_test_finish(unsigned n, unsigned timeout, unsigned full) {
    unsigned i = threadIdx.x;
    if (i < n) {
        auto *r = aotx_intake.rows + i;
        r->bytes = 2; r->reply[0] = '['; r->reply[1] = ']';
        if (timeout && r->state == 2) r->ticks = AOTX_INTAKE_TICKS;
        else if (r->state == 2) { aotx_seqs.slot[i].state = AOTX_SEQ_STATE_DONE; aotx_seqs.slot[i].last = aotx_seqs.slot[i].stop; }
        aotx_kv.count[i] = 2 + i % 5;
    }
    if (!i) { aotx_kv.served = aotx_kv.made; if (full) aotx_kv.made += AOTX_KV_QUEUE_MAX; }
}
__global__ void aotx_intake_test_drain(void) { if (!threadIdx.x) aotx_kv.served = aotx_kv.made; }
static void aotx_intake_lease(unsigned n, unsigned mode) {
    aotx_intake_device d(n); aotx_fixture empty;
    d.send(aotx_live_load_bytes(empty.wire(false, 0)), 1); d.send(aotx_intake_bind(n), 3);
    auto before = aotx_retain_store();
    d.process(aotx_live_parts(aotx_intake_query(n, 0, 1), 4, d.next_id++), false, false);
    aotx_check(d.state().phase == AOTX_INTAKE_RUN, "internal leases start after source recall");
    auto tail = d.seam().dev.tail;
    aotx_intake_test_open<<<1,64>>>(n, mode); AOTX_CUDA(cudaDeviceSynchronize());
    aotx_seq_table sequences; AOTX_CUDA(cudaMemcpyFromSymbol(&sequences, aotx_seqs, sizeof(sequences)));
    for (unsigned i = 0; i < n - (mode == 1 || mode == 2); ++i) {
        auto &seq = sequences.slot[i];
        aotx_check(seq.state == AOTX_SEQ_STATE_PREFILL && seq.prompt == 2 && seq.sample.temperature == 0 &&
            seq.sample.top_p == 1 && seq.sample.repeat_penalty == 1 && !seq.sample.affect,
            "internal sequences use fixed neutral greedy controls");
        for (unsigned j = 0; j < AOTX_MODEL_STEERS; ++j)
            aotx_check(seq.sample.steer[j] == AOTX_MODEL_CONDUCT_NONE, "internal sequence has no identity steering");
    }
    aotx_intake_test_finish<<<1,64>>>(n, mode == 3, mode == 4);
    aotx_intake_step<<<1,64>>>(); AOTX_CUDA(cudaDeviceSynchronize());
    if (mode == 3) { aotx_decode_commit<<<1,64>>>(0); aotx_intake_step<<<1,64>>>(); }
    if (mode == 4) {
        aotx_check(d.state().phase == AOTX_INTAKE_RUN && aotx_retain_store() == before, "full release queue blocks memory publication");
        aotx_kv_table held; AOTX_CUDA(cudaMemcpyFromSymbol(&held, aotx_kv, sizeof(held)));
        for (unsigned i = 0; i < n; ++i) aotx_check(held.count[i] == 2 + i % 5, "blocked release retains each slot's cache ownership");
        aotx_intake_test_drain<<<1,1>>>(); aotx_intake_step<<<1,64>>>();
    }
    AOTX_CUDA(cudaDeviceSynchronize());
    aotx_check(d.state().phase == AOTX_INTAKE_DONE, "complete cleanup ends every internal lease");
    aotx_kv_table kv; AOTX_CUDA(cudaMemcpyFromSymbol(&kv, aotx_kv, sizeof(kv)));
    AOTX_CUDA(cudaMemcpyFromSymbol(&sequences, aotx_seqs, sizeof(sequences)));
    for (unsigned i = 0; i < n; ++i) aotx_check(!kv.count[i] && sequences.slot[i].state == AOTX_SEQ_STATE_FREE,
        "terminal internal work retains no cache or language sequence");
    aotx_live_records emitted(d.seam().dev.tail - tail);
    AOTX_CUDA(cudaMemcpy(emitted.data(), d.out + tail * AOTX_SLOT_BYTES, emitted.size() * AOTX_SLOT_BYTES, cudaMemcpyDeviceToHost));
    for (const auto &record : emitted) aotx_check(((const aotx_record_header *)record.data())->cls != AOTX_CLASS_A,
        "internal open and cleanup produce no authoritative token record");
    d.process({});
    bool refused = mode >= 1 && mode <= 3;
    aotx_check(refused ? d.state().status && aotx_retain_store() == before : !d.state().status,
        "only complete internal output can admit the source batch");
    for (auto &b : d.bindings(n)) aotx_check(b.ordinal == !refused, "internal work cannot open an extra conversation turn");
}
int main(void) {
    int cards = 0; if (cudaGetDeviceCount(&cards) != cudaSuccess || !cards) return 77;
    for (unsigned n : {1u, 64u}) for (unsigned mode = 0; mode < 5; ++mode) aotx_intake_lease(n, mode);
    printf("interpretation leases: %u checks, %u failures\n", aotx_checks, aotx_failures);
    return aotx_failures ? 1 : 0;
}
