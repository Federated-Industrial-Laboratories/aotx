/* Purpose: Refuse malformed appraisal configuration and inaccessible source batches.
 * Owns: Independent format defects and exact whole-store refusal checks.
 * Launch shape: Real CUDA admission at one and 64 distinct source rows and three scopes.
 * Lifetime: One maintained test process; valid work stops before model execution. */
#include "appraisal_control_fixture.h"

static void aotx_appraisal_config_faults(unsigned n) {
    aotx_appraisal_device d(n); aotx_control_reset(); aotx_appraisal_sources(d, n);
    auto before = aotx_retain_store(); uint64_t accepted = d.state().accepted;
    for (unsigned fault = 0; fault < 11; ++fault) {
        auto p = aotx_appraisal_config_bytes();
        if (fault == 0) aotx_put(p.data() + 8, 2, 4);
        if (fault == 1) aotx_put(p.data() + 12, 8, 4);
        if (fault == 2) aotx_put(p.data() + 16, 0, 4);
        if (fault == 3) aotx_put(p.data() + 20, 0, 4);
        if (fault == 4) aotx_put(p.data() + 20, 4097, 4);
        if (fault == 5) aotx_put(p.data() + 24, 0, 4);
        if (fault == 6) aotx_put(p.data() + 28, 1000001, 4);
        if (fault == 7) aotx_put(p.data() + 32, 1000001, 4);
        if (fault == 8) aotx_put(p.data() + 36, AOTX_RECALL_BATCH + 1, 4);
        if (fault == 9) p[40] ^= 1;
        if (fault == 10) p[72] = 1;
        d.send(p, AOTX_APPRAISAL_CONTROL);
        aotx_check(d.state().status && !d.state().fatal && d.state().accepted == accepted && aotx_retain_store() == before,
            "unsupported schema, processor, reserved fields and invalid limits cannot change current configuration");
    }
}
static void aotx_appraisal_request_faults(unsigned n, unsigned scope, unsigned fault) {
    aotx_appraisal_device d(n); aotx_control_reset(); aotx_appraisal_sources(d, n, scope);
    aotx_control_metadata<<<1,AOTX_SLOTS>>>(n, 0);
    auto request = aotx_appraisal_work_request(n); unsigned last = 64 + (n - 1) * 32;
    if (fault == 1) aotx_put(request.data() + 16, aotx_get(request.data() + 16) + 1);
    if (fault == 2) request[24] ^= 1;
    if (fault == 3) aotx_put(request.data() + 40, 2);
    if (fault == 4) request[last] ^= 1;
    if (fault == 5) aotx_put(request.data() + last + 16, 2);
    if (fault == 6) aotx_put(request.data() + last + 24, AOTX_SLOTS, 4);
    if (fault == 7) request[last + 28] = 1;
    if (fault == 8 || fault == 9) {
        if (n == 1) {
            request.resize(128, 0); memcpy(request.data() + 96, request.data() + 64, 32);
            aotx_put(request.data() + 12, 2, 4); last = 96;
        }
        if (fault == 8) { memcpy(request.data() + last, request.data() + 64, 24);
            aotx_put(request.data() + last + 24, n == 1 ? 1 : n - 1, 4); }
        else aotx_put(request.data() + last + 24, 0, 4);
    }
    if (fault >= 10 && fault <= 13) aotx_control_mutate<<<1,1>>>(fault - 5, n);
    if (fault == 14) aotx_put(request.data() + 48, 2, 4);
    if (fault == 15) request[52] = 1;
    if (fault == 16) aotx_put(request.data() + 12, 0, 4);
    AOTX_CUDA(cudaDeviceSynchronize()); auto before = aotx_retain_store(); auto bindings = d.bindings(n);
    d.process(aotx_live_parts(request, AOTX_APPRAISAL_REQUEST, d.next_id++), false, false);
    auto state = d.appraisal(); auto observed = aotx_control_read();
    if (!fault) {
        aotx_check(state.active && state.count == n && !d.state().status && d.state().phase == AOTX_INTAKE_RUN,
            "valid exact source and scope references admit the whole batch");
        for (unsigned i = 0; i < n; ++i) aotx_check(observed.owned[i] == i + 1 && !observed.row_status[i],
            "every valid request leases its own source row");
    } else {
        aotx_check(!state.active && !state.calls && d.state().status && d.state().phase == AOTX_LIVE_IDLE,
            "wrong current references, duplicate rows and inaccessible evidence refuse the whole request");
        for (unsigned i = 0; i < n; ++i) aotx_check(!observed.owned[i] && !observed.wanted[i],
            "a refused source batch acquires no model or prompt lease");
    }
    auto after = d.bindings(n);
    aotx_check(aotx_retain_store() == before && !memcmp(after.data(), bindings.data(), n * sizeof(bindings[0])),
        "request admission and refusal leave source memory and user bindings unchanged");
}
static void aotx_appraisal_selection(unsigned n, unsigned unavailable) {
    aotx_appraisal_device d(n); aotx_control_reset(); aotx_appraisal_sources(d, n);
    unsigned limit = unavailable ? n : (n > 1 ? n / 2 : 1);
    aotx_control_line("appraisal limits 160 512 16384 " + std::to_string(limit)); d.process({});
    aotx_control_metadata<<<1,AOTX_SLOTS>>>(n, 0);
    if (unavailable) aotx_control_mutate<<<1,1>>>(5, n);
    aotx_control_line("appraisal run"); auto records = d.process({}, false, false);
    unsigned expected = unavailable ? n - 1 : limit;
    auto state = d.appraisal(); auto request = aotx_retain_result(records, AOTX_APPRAISAL_REQUEST);
    aotx_check(state.active == !!expected && state.calls == !!expected && !state.explicit_pending,
        "the explicit run selects only available queues within the current configured row limit");
    if (expected) {
        aotx_check(state.count == expected && request.size() == 64 + expected * 32,
            "the journal request contains the exact admitted row count");
        for (unsigned i = 0; i < expected; ++i)
            aotx_check(state.rows[i].queue == 1 + 3 * n + i && state.rows[i].source == 1 + 3 * i,
                "queue selection preserves distinct visible sources in order");
    } else aotx_check(request.empty() && !state.last_status, "no accessible pending source produces no request or model call");
}
static void aotx_appraisal_row_limit(unsigned n, unsigned excess) {
    unsigned sources = n == 1 ? 2 : n, limit = sources - 1, count = limit + excess;
    aotx_appraisal_device d(sources); aotx_control_reset(); aotx_appraisal_sources(d, sources);
    aotx_control_line("appraisal limits 160 512 16384 " + std::to_string(limit)); d.process({});
    aotx_control_metadata<<<1,AOTX_SLOTS>>>(sources, 0);
    auto request = aotx_appraisal_work_request(sources);
    request.resize(64 + count * 32); aotx_put(request.data() + 12, count, 4); aotx_put(request.data() + 40, 2);
    auto before = aotx_retain_store(); auto bindings = d.bindings(sources);
    d.process(aotx_live_parts(request, AOTX_APPRAISAL_REQUEST, d.next_id++), false, false);
    auto state = d.appraisal(); auto observed = aotx_control_read();
    aotx_check(state.active == !excess && state.calls == !excess && d.state().status == (excess ? AOTX_COG_CAPACITY : 0),
        "recorded request admission independently refuses a count above the current configured row limit");
    for (unsigned i = 0; i < sources; ++i)
        aotx_check(observed.owned[i] == (!excess && i < count ? i + 1 : 0) &&
            !!observed.wanted[i] == (!excess && i < count), "only an admitted batch can acquire internal prompt leases");
    auto after = d.bindings(sources);
    aotx_check(aotx_retain_store() == before && !memcmp(after.data(), bindings.data(), sources * sizeof(bindings[0])),
        "configured row-limit admission leaves source memory and user bindings unchanged");
}
int main(void) {
    int cards = 0; if (cudaGetDeviceCount(&cards) != cudaSuccess || !cards) return 77;
    for (unsigned n : {1u, 64u}) {
        aotx_appraisal_config_faults(n);
        for (unsigned scope = 0; scope < 3; ++scope)
            for (unsigned fault = 0; fault < 17; ++fault) aotx_appraisal_request_faults(n, scope, fault);
        aotx_appraisal_selection(n, 0); aotx_appraisal_selection(n, 1);
        aotx_appraisal_row_limit(n, 0); aotx_appraisal_row_limit(n, 1);
    }
    printf("appraisal schema: %u checks, %u failures\n", aotx_checks, aotx_failures);
    return aotx_failures || aotx_checks < 1000 ? 1 : 0;
}
