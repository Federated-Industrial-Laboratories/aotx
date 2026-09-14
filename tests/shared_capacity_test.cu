/* Purpose: Verify shared page budgets, queued clocks, bounded groups and exact lease replay.
 * Owns: Distinct N=1 and N=64 requests and independent resource expectations.
 * Launch shape: Actual shared work and record kernels; no model weights are needed.
 * Lifetime: Saved queued input through resource release and resumed admission. */
#include "shared_capacity_fixture.h"

__global__ void aotx_capacity_bounds(unsigned count, unsigned *out)
{
    for (unsigned i = 0; i < count; ++i) {
        unsigned cap = i % 4 == 0 ? 1 : i % 4 == 1 ? 160 : i % 4 == 2 ? AOTX_KV_PAGES + 1 : ~0u;
        out[i] = aotx_shared_page_bound(AOTX_MODEL_LANGUAGE, cap);
    }
    out[count] = aotx_shared_page_available();
}
__global__ void aotx_capacity_owner(unsigned mode)
{
    if (mode == 1) {
        aotx_seqs.slot[0].state = AOTX_SEQ_STATE_PREFILL;
        aotx_seqs.slot[0].role = AOTX_MODEL_LANGUAGE;
        aotx_seqs.slot[0].prompt = 1000; aotx_seqs.slot[0].limit = 24;
        aotx_kv.count[0] = 10;
    } else if (mode == 2) {
        aotx_say.slot[0].wanted = 1; aotx_say.slot[0].page_limit = 160;
        aotx_prompt_roles[0] = AOTX_MODEL_LANGUAGE;
    } else if (mode == 3) {
        aotx_kvl_make(&aotx_model_space[AOTX_MODEL_LANGUAGE_Q4].shape, 48, 8, 128);
        aotx_model_load.resident[AOTX_MODEL_LANGUAGE_Q4].active = 1;
    } else if (mode == 4) {
        aotx_kvl_make(&aotx_model_space[AOTX_MODEL_EMBEDDING].shape, 28, 8, 128);
        aotx_model_load.resident[AOTX_MODEL_EMBEDDING].active = 1;
    } else if (mode == 5) {
        aotx_kvl_make(&aotx_model_space[AOTX_MODEL_LANGUAGE].shape, 36, 16, 512);
    }
}
static std::vector<unsigned> bounds(unsigned count)
{
    unsigned *d; cu(cudaMalloc(&d, (count + 1) * sizeof(*d)));
    aotx_capacity_bounds<<<1,1>>>(count, d); cu(cudaDeviceSynchronize());
    std::vector<unsigned> out(count + 1); cu(cudaMemcpy(out.data(), d, out.size() * sizeof(*d), cudaMemcpyDeviceToHost));
    cudaFree(d); return out;
}
static unsigned left(unsigned used) { return used < AOTX_KV_PAGES ? AOTX_KV_PAGES - used : 0; }
static unsigned whole(unsigned n) { return n < AOTX_KV_PAGES ? n : AOTX_KV_PAGES; }
static void page_cases(unsigned count)
{
    fixture f(count);
    for (unsigned mode = 0; mode < 6; ++mode) {
        aotx_capacity_seed<<<1,1>>>(count, 160, mode == 1 ? 10 : 0, false); aotx_capacity_owner<<<1,1>>>(mode);
        auto out = bounds(count);
        for (unsigned i = 0; i < count; ++i) {
            unsigned cap = i % 4 == 0 ? 1 : i % 4 == 1 ? 160 : i % 4 == 2 ? AOTX_KV_PAGES + 1 : ~0u;
            unsigned model = mode == 3 ? 199 : mode == 5 ? 1536 : 149;
            unsigned expected = cap < model ? cap : model;
            if (mode == 4 && expected < 116) expected = 116;
            check(out[i] == whole(expected), "page bounds cover capped language, routed intake and separate embedding");
        }
        unsigned wanted = mode == 1 ? left(75) : mode == 2 ? left(149) : AOTX_KV_PAGES;
        check(out.back() == wanted, "unmapped sequence and prompt claims reduce the available pool");
    }
    for (unsigned mapped : {0u, AOTX_KV_PAGES - 1, AOTX_KV_PAGES}) {
        aotx_capacity_seed<<<1,1>>>(count, 160, mapped, false);
        aotx_capacity_pool<<<1,1>>>(mapped, count, 2, ~0u - count / 2);
        check(bounds(count).back() == left(mapped + count * 2), "wrapped pending additions reserve their complete page claims");
        aotx_capacity_pool<<<1,1>>>(mapped, count, 0);
        check(bounds(count).back() == left(mapped), "queued releases do not free physical pages before service");
    }
    aotx_capacity_seed<<<1,1>>>(count, 160, 10, false); aotx_capacity_owner<<<1,1>>>(1);
    aotx_capacity_pool<<<1,1>>>(10, 1, 65);
    check(bounds(count).back() == left(75), "the same pending sequence claim is counted once");
    aotx_capacity_pool<<<1,1>>>(10, AOTX_KV_QUEUE_MAX + 1, 1);
    check(!bounds(count).back(), "an invalid pending queue cannot admit new page demand");
}

static void groups(unsigned count, bool varied, bool wide = false)
{
    fixture f(count); unsigned cap = wide ? AOTX_KV_PAGES + 1 : 160;
    aotx_capacity_seed<<<1,1>>>(count, cap, AOTX_KV_PAGES, varied);
    if (wide) aotx_capacity_owner<<<1,1>>>(5);
    aotx_shared_work<<<1,1>>>(); cu(cudaDeviceSynchronize());
    check(!capacity_state().kind, "a full pool starts no execution lease"); capacity_queued(f, count);
    aotx_capacity_pool<<<1,1>>>(0, 1, AOTX_KV_PAGES);
    aotx_shared_work<<<1,1>>>(); cu(cudaDeviceSynchronize());
    check(!capacity_state().kind, "a pending full-pool owner blocks new leases"); capacity_queued(f, count);
    aotx_capacity_pool<<<1,1>>>(0, 0, 0);
    unsigned completed = 0, rounds = 0; std::vector<bool> seen(count);
    std::vector<unsigned char> first_lease;
    while (completed < count && rounds++ <= count) {
        aotx_shared_work<<<1,1>>>(); cu(cudaDeviceSynchronize()); auto s = capacity_state();
        check(s.kind == AOTX_SHARED_LEASE_RECORD, "free capacity resumes a bounded lease group");
        if (s.kind != AOTX_SHARED_LEASE_RECORD) break;
        unsigned n = aotx_service_get(s.transfer, 4), used = 0;
        check(n && n < AOTX_SLOTS && s.total == 8 + n * 32, "the canonical lease contains the exact selected row count");
        std::vector<unsigned> ids;
        for (unsigned i = 0; i < n; ++i) {
            const unsigned char *r = s.transfer + 8 + i * 32;
            unsigned index = aotx_service_get(r, 4), slot = aotx_service_get(r + 4, 4);
            check(index < count && slot == i + 1 && !seen[index], "each lease uses a fresh slot and an unserved request");
            if (index >= count) continue;
            unsigned row_cap = varied ? 1 + index % cap : cap;
            unsigned model = wide ? 1536 : 149; used += whole(row_cap < model ? row_cap : model);
            check(aotx_service_get(r + 16, 4) == index + 1 && aotx_service_get(r + 8, 8) == index + 101,
                "lease bytes bind the admitted actor and sequence"); ids.push_back(index);
        }
        check(used <= AOTX_KV_PAGES, "the whole selected group fits its conservative page bound");
        capacity_queued(f, count, completed);
        if (first_lease.empty()) first_lease.assign(s.transfer, s.transfer + s.total);
        aotx_shared_emit<<<1,1>>>(); cu(cudaDeviceSynchronize()); auto rows = capacity_receipts(f, count);
        aotx_shared_execution execution[AOTX_SLOTS]; cu(cudaMemcpyFromSymbol(execution, aotx_shared_execution_slots, sizeof(execution)));
        for (unsigned i : ids) {
            check(rows[i].phase == AOTX_SHARED_RUNNING && rows[i].slot < AOTX_SLOTS,
                "only a complete lease changes the queued receipt to running");
            check(rows[i].slot < AOTX_SLOTS && execution[rows[i].slot].opened > 0, "execution time starts when the lease is applied");
        }
        check(!capacity_state().fatal, "bounded live lease publication succeeds");
        for (unsigned i : ids) {
            aotx_capacity_end<<<1,1>>>(i, 1000000000ull * (rounds + 1)); aotx_shared_emit<<<1,1>>>();
            cu(cudaDeviceSynchronize()); seen[i] = true; ++completed;
        }
        rows = capacity_receipts(f, count);
        for (unsigned i : ids) check(rows[i].phase == AOTX_SHARED_DONE && rows[i].status == 200 && rows[i].terminal_source,
            "completion records release the selected inputs for the next group");

    }
    check(completed == count, "every distinct queued input is served across resource groups");
    for (bool value : seen) check(value, "no request is skipped during capacity resumption");
    if (first_lease.empty()) return;
    aotx_capacity_seed<<<1,1>>>(count, cap, AOTX_KV_PAGES, varied);
    if (wide) aotx_capacity_owner<<<1,1>>>(5);
    unsigned char *bytes; cu(cudaMalloc(&bytes, first_lease.size()));
    cu(cudaMemcpy(bytes, first_lease.data(), first_lease.size(), cudaMemcpyHostToDevice));
    aotx_capacity_replay<<<1,1>>>(bytes, first_lease.size(), f.result); cu(cudaDeviceSynchronize());
    unsigned ok; cu(cudaMemcpy(&ok, f.result, sizeof(ok), cudaMemcpyDeviceToHost)); cudaFree(bytes);
    check(ok == 1 && !capacity_state().fatal, "recorded leases replay exactly even when live capacity is unavailable");
    auto rows = capacity_receipts(f, count);
    unsigned replayed = aotx_service_get(first_lease.data(), 4);
    for (unsigned i = 0; i < count; ++i) check(rows[i].phase ==
        (i < replayed ? AOTX_SHARED_RUNNING : AOTX_SHARED_QUEUED),
        "record replay retains the exact selected requests and leaves other inputs queued");

}
int main()
{
    for (unsigned n : {1u, 64u}) { page_cases(n); groups(n, false); groups(n, true); groups(n, false, true); }
    std::printf("shared-capacity checks=%u failures=%u\n", checks, failures);
    return failures ? 1 : 0;
}
