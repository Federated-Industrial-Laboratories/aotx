/* Purpose: Verify complete appraisal at the configured object and payload limits.
 * Owns: Distinct retained rows, full-capacity filler and exact refusal expectations.
 * Launch shape: Real CUDA memory admission at N=1 and N=64.
 * Lifetime: One maintained test process with controlled decoder responses. */
#include "appraisal_work_fixture.h"

static void aotx_appraisal_capacity(unsigned n) {
    aotx_appraisal_device d(n); aotx_appraisal_sources(d, n);
    auto initial = aotx_retain_store();
    const auto *seed = (const aotx_cognitive_store *)initial.data();
    unsigned extra = AOTX_COG_OBJECTS - seed->count - 3 * n;
    uint64_t payload = AOTX_COG_PAYLOAD - seed->bytes - 480 * n;
    aotx_check(extra && payload >= extra, "the profile fits source batches and independently sized capacity data");
    if (!extra || payload < extra) return;
    aotx_fixture filler;
    for (unsigned i = 0; i < extra; ++i) {
        auto r = aotx_memory_row(i, AOTX_COG_COMPONENT, 6000000 + i, seed->sequence + i + 1, 2);
        filler.add(r, aotx_bytes(i ? 1 : payload - extra + 1, 1 + i % 251));
    }
    d.send(filler.wire(true, seed->sequence + 1, seed->tick + 1), AOTX_LIVE_UPDATE);
    aotx_check(!d.state().status, "the complete configured object and payload seed is admitted");
    if (d.state().status) return;
    auto before = aotx_retain_store(); auto prior = (const aotx_cognitive_store *)before.data();
    aotx_check(prior->count + 3 * n == AOTX_COG_OBJECTS && prior->bytes + 480 * n == AOTX_COG_PAYLOAD,
        "exactly one complete appraisal batch remains in both configured capacities");
    auto request = aotx_appraisal_work_request(n);
    d.process(aotx_live_parts(request, AOTX_APPRAISAL_REQUEST, d.next_id++), false, false);
    aotx_check(d.appraisal().active && d.state().phase == AOTX_INTAKE_RUN,
        "the last source batch reaches model work above all prior retained objects");
    if (!d.appraisal().active) return;
    aotx_appraisal_output(n);
    auto result = aotx_retain_result(d.process({}), AOTX_APPRAISAL_RESULT);
    auto saved = aotx_retain_store(); auto store = (const aotx_cognitive_store *)saved.data();
    aotx_check(!d.state().status && !d.state().fatal && d.appraisal().completed == n,
        "the final complete appraisal batch publishes successfully");
    aotx_check(store->count == AOTX_COG_OBJECTS && store->bytes == AOTX_COG_PAYLOAD,
        "appraisal uses the last object and payload byte of the configured allocation");
    aotx_check(result.size() >= 64 && aotx_get(result.data() + 12, 4) == n && !aotx_get(result.data() + 32, 4),
        "full-capacity publication records every source result");
    for (unsigned i = seed->count; i < prior->count; ++i) {
        auto old = prior->objects[i], now = store->objects[i];
        aotx_check(!memcmp(old, now, AOTX_COG_OBJECT), "appraisal preserves each distinct existing object record");
        uint64_t offset = aotx_get(old + AOTX_CO_OFFSET), bytes = aotx_get(old + AOTX_CO_BYTES);
        aotx_check(!memcmp(prior->payload + offset, store->payload + offset, bytes),
            "appraisal preserves all previously allocated payload bytes");
    }
    d.idle(n); auto bindings = d.bindings(n);
    d.process(aotx_live_parts(aotx_retain_query(n, store->sequence, 2, false, 0, 3, 80), 4, d.next_id++));
    auto after = d.bindings(n);
    aotx_check(d.state().status == AOTX_COG_CAPACITY && aotx_retain_store() == saved,
        "new source and queue admission at capacity publishes no partial objects");
    aotx_check(!memcmp(bindings.data(), after.data(), n * sizeof(bindings[0])),
        "capacity refusal preserves every conversation binding");
    printf("appraisal capacity N=%u objects=%u payload=%u\n", n, store->count, store->bytes);
}

int main(void) {
    int cards = 0; if (cudaGetDeviceCount(&cards) != cudaSuccess || !cards) return 77;
    for (unsigned n : {1u, 64u}) aotx_appraisal_capacity(n);
    printf("appraisal capacity: %u checks, %u failures\n", aotx_checks, aotx_failures);
    return aotx_failures || aotx_checks < AOTX_COG_OBJECTS ? 1 : 0;
}
