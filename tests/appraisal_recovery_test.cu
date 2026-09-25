/* Purpose: Verify both model-call interruptions and exact historical result recovery.
 * Owns: Distinct source outputs, saved byte comparisons and damaged record controls.
 * Launch shape: Real admission and replay at N=1 and N=64 without model weights.
 * Lifetime: Complete source batches and repeated recovery of their recorded results. */
#include "appraisal_work_fixture.h"

__global__ void aotx_recovery_refresh(void) { if (!threadIdx.x) aotx_appraisal_refresh(); }
__global__ void aotx_recovery_output(const unsigned char *first, const unsigned *lengths,
    unsigned n, unsigned mode, unsigned status) {
    unsigned i = threadIdx.x; if (i >= n) return;
    auto r = aotx_intake.rows + i;
    unsigned phase = mode == 4 ? i % 4 : mode;
    r->phase = phase < 2 ? phase : 2; r->second_call = phase == 3;
    r->first_bytes = phase >= 2 ? lengths[i] : 0;
    for (unsigned j = 0; j < lengths[i]; ++j) r->first_reply[j] = first[i * 4096 + j];
    r->bytes = phase == 1 ? lengths[i] - 1 : phase == 3 ? 1 : 0;
    for (unsigned j = 0; j < r->bytes; ++j) r->reply[j] = phase == 1 ? first[i * 4096 + j] : '{';
    for (unsigned j = 0; j < 32; ++j)
        r->model[j] = aotx_appraisal.rows[i].first_model[j] = phase ? 77 + j : 0;
    r->status = status;
    if (!i) { aotx_live.status = status; aotx_live.phase = AOTX_INTAKE_DONE; }
}
static void aotx_recovery_outputs(unsigned n, unsigned mode, unsigned status) {
    aotx_bytes first(n * 4096, 0); std::vector<unsigned> lengths(n);
    for (unsigned i = 0; i < n; ++i) {
        std::string text = "[\"retained source " + std::to_string(i) + " \"]";
        lengths[i] = text.size(); memcpy(first.data() + i * 4096, text.data(), text.size());
    }
    unsigned char *data; unsigned *sizes;
    AOTX_CUDA(cudaMalloc(&data, first.size())); AOTX_CUDA(cudaMalloc(&sizes, n * sizeof(unsigned)));
    AOTX_CUDA(cudaMemcpy(data, first.data(), first.size(), cudaMemcpyHostToDevice));
    AOTX_CUDA(cudaMemcpy(sizes, lengths.data(), n * sizeof(unsigned), cudaMemcpyHostToDevice));
    aotx_recovery_output<<<1,64>>>(data, sizes, n, mode, status); AOTX_CUDA(cudaDeviceSynchronize());
    cudaFree(data); cudaFree(sizes);
}
static void aotx_recovery_interruption(unsigned n, unsigned mode, unsigned status) {
    aotx_live_records start, request, decision; aotx_bytes expected, result;
    {
        aotx_appraisal_device d(n); start = aotx_appraisal_sources(d, n);
        request = d.process(aotx_live_parts(aotx_appraisal_work_request(n), 16, d.next_id++), false, false);
        auto before = aotx_retain_store();
        aotx_recovery_outputs(n, mode, status); decision = d.process({});
        expected = aotx_retain_store(); result = aotx_retain_result(decision, 17);
        aotx_check(!d.state().fatal && d.appraisal().last_status == status, "interruption publishes a complete bounded result");
        auto old = (const aotx_cognitive_store *)before.data(), now = (const aotx_cognitive_store *)expected.data();
        aotx_check(now->count == old->count + n && !memcmp(old->objects, now->objects, old->count * AOTX_COG_OBJECT) &&
            !memcmp(old->payload, now->payload, old->bytes), "interruption preserves prior memory and adds only queue versions");
        aotx_check(aotx_get(result.data() + 8, 4) == 2 && aotx_get(result.data() + 12, 4) == n, "all interrupted rows use the two-call result schema");
        for (unsigned i = 0; i < n; ++i) {
            const unsigned char *p = result.data() + 64 + i * AOTX_APPRAISAL_RESULT_ROW;
            unsigned phase = mode == 4 ? i % 4 : mode;
            std::string quote = "[\"retained source " + std::to_string(i) + " \"]";
            if (phase == 1) quote.pop_back();
            if (!phase) quote.clear();
            aotx_check(aotx_get(p + 4160, 4) == quote.size() && !memcmp(p + 4224, quote.data(), quote.size()),
                "each source retains its distinct complete or partial first output");
            aotx_check(aotx_get(p + 56, 4) == (phase == 3) && (phase != 3 || p[64] == '{') &&
                aotx_get(p + 4164, 4) == (phase == 3) && aotx_get(p + 4168, 4) == (phase < 2 ? phase : 2),
                "the recorded phase and second-call boundary match the actual partial output");
        }
    }
    for (unsigned pass = 0; pass < 2; ++pass) {
        aotx_appraisal_device d(n); d.process(start, true); d.idle(n); d.process(request, true);
        aotx_check(d.appraisal().active && d.state().phase == AOTX_LIVE_WAIT,
            "recovery reaches the recorded result boundary after foreground completion");
        d.process(decision, true);
        aotx_check(!d.state().fatal && !d.appraisal().calls && aotx_retain_store() == expected,
            "repeated recovery preserves both raw outputs and all memory bytes without model generation");
    }
    if (mode != 3 || status != AOTX_COG_DENIED) return;
    for (unsigned defect = 0; defect < 10; ++defect) {
        aotx_appraisal_device d(n); d.process(start, true); d.idle(n); d.process(request, true);
        aotx_check(d.appraisal().active && d.state().phase == AOTX_LIVE_WAIT,
            "damaged results are tested only after an admitted recorded request");
        auto before = aotx_retain_store(), changed = result;
        unsigned char *last = changed.data() + 64 + (n - 1) * AOTX_APPRAISAL_RESULT_ROW;
        unsigned offsets[] = {4192, 4160, 4164, 4168, 4172, 8319, 4226, 24, 60, 4159};
        last[offsets[defect]] ^= 1;
        d.process(aotx_live_parts(changed, 17, 5), true);
        aotx_check(d.state().fatal && aotx_retain_store() == before,
            "damaged call metadata, source evidence or model identities cannot publish during recovery");
    }
}
static aotx_bytes aotx_recovery_legacy_checkpoint(unsigned n) {
    aotx_appraisal_device d(n); aotx_appraisal_sources(d, n);
    auto saved = aotx_retain_store(); auto store = (const aotx_cognitive_store *)saved.data();
    const unsigned char legacy[32] = AOTX_APPRAISAL_LEGACY_PROCESSOR_BYTES;
    aotx_fixture f;
    for (unsigned i = 0; i < store->count; ++i) {
        aotx_row row; memcpy(row.data(), store->objects[i], AOTX_COG_OBJECT);
        auto begin = store->payload + aotx_get(row.data() + AOTX_CO_OFFSET);
        aotx_bytes p(begin, begin + aotx_get(row.data() + AOTX_CO_BYTES));
        if (!memcmp(p.data(), "AOTXAPC1", 8)) { memcpy(p.data() + 40, legacy, 32); aotx_put(p.data() + 12, 3, 4); }
        if (!memcmp(p.data(), "AOTXAPQ1", 8)) memcpy(p.data() + 64, legacy, 32);
        f.add(row, p);
    }
    return f.wire(false, store->sequence);
}
static void aotx_recovery_legacy(unsigned n) {
    auto checkpoint = aotx_recovery_legacy_checkpoint(n);
    aotx_live_records start, request, decision; aotx_bytes expected;
    {
        aotx_appraisal_device d(n); start = d.send(aotx_live_load_bytes(checkpoint), 1); d.idle(n);
        aotx_recovery_refresh<<<1,1>>>(); AOTX_CUDA(cudaDeviceSynchronize());
        aotx_check(!d.state().status && d.appraisal().write_flags == AOTX_APPRAISAL_RECALL && !d.appraisal().calls,
            "an imported historical configuration does not silently enable new generation");
        request = aotx_live_parts(aotx_appraisal_work_request(n), 16, 5);
        d.process(request, true); aotx_check(d.appraisal().active && d.appraisal().result_version == 1,
            "recorded historical requests retain their original single-call contract");
        aotx_appraisal_output(n); decision = d.process({}); expected = aotx_retain_store();
        auto result = aotx_retain_result(decision, 17);
        aotx_check(!d.state().fatal && !d.appraisal().last_status && aotx_get(result.data() + 8, 4) == 1 &&
            result.size() == 64 + n * AOTX_APPRAISAL_LEGACY_ROW + aotx_get(result.data() + 24),
            "historical response layout and typed processor identities remain unchanged");
    }
    for (unsigned pass = 0; pass < 2; ++pass) {
        aotx_appraisal_device d(n); d.process(start, true); d.process(request, true); d.process(decision, true);
        aotx_check(!d.state().fatal && !d.appraisal().calls && aotx_retain_store() == expected,
            "historical results recover exactly on the new runtime with no model call");
    }
    {
        aotx_appraisal_device d(n); d.send(aotx_live_load_bytes(checkpoint), 1); d.idle(n);
        d.send(aotx_appraisal_config_bytes(), 15);
        aotx_recovery_refresh<<<1,1>>>(); AOTX_CUDA(cudaDeviceSynchronize());
        aotx_check(!d.state().status && d.appraisal().write_flags == 1 && !d.appraisal().pending,
            "an explicit current configuration leaves historical queued sources ineligible for a new processor");
        auto input = aotx_appraisal_work_request(n); auto store = aotx_retain_store();
        auto view = (const aotx_cognitive_store *)store.data();
        aotx_put(input.data() + 16, view->sequence); aotx_put(input.data() + 40, 2);
        d.process(aotx_live_parts(input, 16, d.next_id++), false, false);
        aotx_check(!d.appraisal().active && !d.appraisal().calls && d.state().status,
            "an explicit old queue cannot receive the new processor by substitution");
    }
}
int main(int argc, char **argv) {
    bool cuda_check = argc == 2 && !strcmp(argv[1], "--cuda-check");
    if (argc != 1 && !cuda_check) {
        fprintf(stderr, "usage: aotx_appraisal_recovery_test [--cuda-check]\n"); return 2;
    }
    for (unsigned n : {1u, 64u}) {
        if (cuda_check) aotx_recovery_interruption(n, 4, AOTX_COG_DENIED);
        else {
            for (unsigned mode = 0; mode < 5; ++mode) aotx_recovery_interruption(n, mode, AOTX_COG_DENIED);
            aotx_recovery_interruption(n, 3, AOTX_COG_UNAVAILABLE);
        }
        aotx_recovery_legacy(n);
    }
    printf("appraisal recovery: %u checks, %u failures\n", aotx_checks, aotx_failures);
    return aotx_failures ? 1 : 0;
}
