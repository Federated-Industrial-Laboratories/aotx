/* Purpose: Verify internal sequences wait for complete page leases under contention.
 * Owns: An independent finite page-pool model and exact batch publication checks.
 * Launch shape: N=1 and N=64 distinct internal sources with a constrained shared pool.
 * Lifetime: Tokenization completion through queued admission and final page release. */
#include "intake_fixture.h"
#include "media/prompt.cuh"

__global__ void aotx_intake_paging_prior(unsigned n) {
    unsigned i = threadIdx.x;
    if (i < n) aotx_seqs.slot[i].page_limit = 1;
}
__global__ void aotx_intake_paging_open(unsigned n) {
    unsigned i = threadIdx.x;
    if (!i) aotx_kvl_make(&aotx_model_space[AOTX_MODEL_LANGUAGE].shape, 28, 8, 128);
    __syncthreads();
    if (i >= n) return;
    aotx_say_count[i] = 2; aotx_say_gear.piece_count[i] = 1;
    aotx_say_gear.chunk[i * AOTX_SAY_PIECES] = 2;
    aotx_say_id[i * AOTX_SAY_TOKENS] = 1; aotx_say_id[i * AOTX_SAY_TOKENS + 1] = 2 + i;
    aotx_intake_open(i);
    aotx_say_count[i] = 0; aotx_say_gear.piece_count[i] = 0;
    aotx_say_id[i * AOTX_SAY_TOKENS + 1] = 99;
}
__global__ void aotx_intake_paging_finish(unsigned n, bool timeout) {
    unsigned i = threadIdx.x; if (i >= n) return;
    auto *r = aotx_intake.rows + i;
    if (timeout) r->ticks = AOTX_INTAKE_TICKS;
    else if (r->state == 2) {
        r->reply[0] = '['; r->reply[1] = ']'; r->bytes = 2;
        aotx_seqs.slot[i].state = AOTX_SEQ_STATE_DONE;
        aotx_seqs.slot[i].last = aotx_seqs.slot[i].stop;
    }
}
static void aotx_intake_paging_serve(std::vector<unsigned> &held, unsigned capacity) {
    aotx_kv_table kv; AOTX_CUDA(cudaMemcpyFromSymbol(&kv, aotx_kv, sizeof(kv)));
    unsigned used = 0; for (auto n : held) used += n;
    for (unsigned at = kv.served; at != kv.made; ++at) {
        auto entry = kv.queue[at & (AOTX_KV_QUEUE_MAX - 1)];
        if (!entry.pages) { used -= held[entry.agent]; held[entry.agent] = 0; }
        else {
            unsigned add = std::min(entry.pages, capacity - used);
            held[entry.agent] += add; used += add;
        }
    }
    for (unsigned i = 0; i < held.size(); ++i) kv.count[i] = held[i];
    kv.mapped_pages = AOTX_KV_PAGES - (capacity - used); kv.served = kv.made;
    AOTX_CUDA(cudaMemcpyToSymbol(aotx_kv, &kv, sizeof(kv)));
}
static void aotx_intake_paging(unsigned n, bool timeout) {
    aotx_intake_device d(n); aotx_fixture empty;
    d.send(aotx_live_load_bytes(empty.wire(false, 0)), 1);
    auto bind = aotx_intake_bind(n);
    for (unsigned i = 0; i < n; ++i) aotx_put(bind.data() + 120 + i * 64, 160, 4);
    d.send(bind, 3);
    auto before = aotx_retain_store();
    aotx_intake_paging_prior<<<1,64>>>(n);
    d.process(aotx_live_parts(aotx_intake_query(n, 0, 1), 4, d.next_id++), false, false);
    aotx_media_prepare<<<1,64>>>(); AOTX_CUDA(cudaDeviceSynchronize());
    aotx_intake_paging_open<<<1,64>>>(n); AOTX_CUDA(cudaDeviceSynchronize());
    aotx_kvl_shape shape; AOTX_CUDA(cudaMemcpyFromSymbol(&shape, aotx_model_space, sizeof(shape),
        AOTX_MODEL_LANGUAGE * sizeof(aotx_model_work) + offsetof(aotx_model_work, shape)));
    unsigned need = aotx_kvl_pages(&shape, AOTX_SEQ_MAX_TOKENS);
    aotx_check(need > 1 && need <= 160, "the fixture requires a nontrivial complete page lease");
    unsigned capacity = timeout ? need - 1 : n == 1 ? need : 2 * need + need / 2;
    std::vector<unsigned> held(n, 0); unsigned started = 0, waves = 0, peak = 0;
    aotx_seq_table seq; AOTX_CUDA(cudaMemcpyFromSymbol(&seq, aotx_seqs, sizeof(seq)));
    for (unsigned i = 0; i < n; ++i)
        aotx_check(seq.slot[i].state == AOTX_SEQ_STATE_FREE, "no sequence opens before its complete page lease arrives");
    for (unsigned tick = 0; tick < 4 * n + 4 && d.state().phase == AOTX_INTAKE_RUN; ++tick) {
        aotx_intake_paging_serve(held, capacity);
        if (timeout) aotx_intake_paging_finish<<<1,64>>>(n, true);
        aotx_intake_step<<<1,64>>>(); AOTX_CUDA(cudaDeviceSynchronize());
        AOTX_CUDA(cudaMemcpyFromSymbol(&seq, aotx_seqs, sizeof(seq)));
        unsigned active = 0;
        for (unsigned i = 0; i < n; ++i) if (seq.slot[i].state == AOTX_SEQ_STATE_PREFILL) {
            ++active;
            aotx_check(seq.slot[i].prompt == 2 && seq.slot[i].last == 2 + i,
                "queued leases preserve distinct tokens after tokenizer scratch is reused");
            aotx_check(held[i] >= need, "every running interpretation holds its whole page budget");
        }
        started += active; waves += active != 0; peak = std::max(peak, active);
        aotx_check(aotx_retain_store() == before, "earlier completed leases cannot publish part of the source batch");
        if (!timeout) aotx_intake_paging_finish<<<1,64>>>(n, false);
        aotx_intake_step<<<1,64>>>(); AOTX_CUDA(cudaDeviceSynchronize());
    }
    aotx_intake_paging_serve(held, capacity);
    aotx_check(d.state().phase == AOTX_INTAKE_DONE, "finite page contention completes or refuses the whole batch");
    aotx_check(timeout ? !started : started == n && (n == 1 || (waves > 1 && peak >= 2)),
        "available pages determine concurrent admission without a smaller source batch limit");
    for (auto pages : held) aotx_check(!pages, "completed and refused leases return every held page");
    d.process({});
    aotx_check(timeout ? d.state().status == AOTX_COG_CAPACITY && aotx_retain_store() == before : !d.state().status,
        "only complete successful batches enter memory after page cleanup");
    for (auto &b : d.bindings(n)) aotx_check(b.ordinal == !timeout, "page waits do not create conversation turns");
}
int main(void) {
    int cards = 0; if (cudaGetDeviceCount(&cards) != cudaSuccess || !cards) return 77;
    for (unsigned n : {1u, 64u}) for (bool timeout : {false, true}) aotx_intake_paging(n, timeout);
    printf("interpretation paging: %u checks, %u failures\n", aotx_checks, aotx_failures);
    return aotx_failures ? 1 : 0;
}
