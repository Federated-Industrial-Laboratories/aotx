/* Purpose: Verify exact qualification, atomic refusal and replay without qualification.
 * Owns: Distinct synthetic sources and explicit test qualification rows.
 * Launch shape: N=1 and N=64 through the live memory kernels.
 * Lifetime: One admission or saved choice per case, without model weights. */
#include "intake_fixture.h"
#include "source_fixture.h"

static unsigned aotx_capability_withdraw_stage;
static bool aotx_capability_withdrawn;
static void aotx_capability_withdraw(bool replay) {
    unsigned phase = 0;
    AOTX_CUDA(cudaMemcpyFromSymbol(&phase, aotx_live, sizeof(phase), offsetof(aotx_live_state, phase)));
    bool withdraw = !replay && phase == AOTX_INTAKE_RUN;
    aotx_intake_capability empty[AOTX_INTAKE_CAPABILITIES] = {};
    if (withdraw && aotx_capability_withdraw_stage == 10)
        AOTX_CUDA(cudaMemcpyToSymbol(aotx_intake_capabilities, empty, sizeof(empty)));
    aotx_intake_fixture_service(replay);
    if (withdraw && aotx_capability_withdraw_stage == 11)
        AOTX_CUDA(cudaMemcpyToSymbol(aotx_intake_capabilities, empty, sizeof(empty)));
    aotx_capability_withdrawn |= withdraw;
}
static void aotx_capability_cases(unsigned n) {
    for (unsigned mode = 0; mode < 12; ++mode) {
        aotx_live_records start, choices; aotx_bytes expected;
        {
            aotx_intake_device d(n); aotx_fixture empty;
            start = d.send(aotx_live_load_bytes(empty.wire(false, 0)), 1);
            auto binding = d.send(aotx_intake_bind(n), 3);
            start.insert(start.end(), binding.begin(), binding.end());
            auto before = aotx_retain_store();
            aotx_intake_capability rows[AOTX_INTAKE_CAPABILITIES];
            AOTX_CUDA(cudaMemcpyFromSymbol(rows, aotx_intake_capabilities, sizeof(rows)));
            rows[0] = {};
            if (mode == 1) rows[1] = {};
            if (mode == 2) rows[1].model[31] ^= 1;
            if (mode == 3) rows[1].statement[31] ^= 1;
            if (mode == 4) rows[1].source[31] ^= 1;
            if (mode == 5) rows[1].profile[31] ^= 1;
            if (mode == 6) rows[1].wrapper.bytes[0] ^= 1;
            if (mode == 7) rows[1].wrapper.end_ids[0] ^= 1;
            if (mode == 8) rows[1].wrapper.think_close_id ^= 1;
            AOTX_CUDA(cudaMemcpyToSymbol(aotx_intake_capabilities, rows, sizeof(rows)));
            auto query = aotx_intake_query(n, 0, 1);
            if (mode != 9) for (unsigned i = 0; i < n; ++i)
                aotx_source_query(query.data() + 128 + i * AOTX_LIVE_QUERY_ROW, 8000 + i);
            std::vector<std::string> replies;
            for (unsigned i = 0; i < n; ++i)
                replies.push_back("[[3,\"Iris" + std::to_string(i) + " will cook.\",0]]");
            aotx_capability_withdraw_stage = mode; aotx_capability_withdrawn = false;
            choices = d.intake(query, replies, true,
                mode >= 10 ? aotx_capability_withdraw : aotx_intake_fixture_service);
            if (mode >= 10) aotx_check(aotx_capability_withdrawn,
                "qualification is removed only after first-call admission");
            expected = aotx_retain_store();
            aotx_check(mode ? d.state().status == AOTX_COG_UNAVAILABLE && expected == before :
                !d.state().status && expected != before,
                "only the exact qualified pair can publish new semantic memory");
            std::vector<aotx_intake_row> rows_after(n);
            AOTX_CUDA(cudaMemcpyFromSymbol(rows_after.data(), aotx_intake, n * sizeof(rows_after[0]),
                offsetof(aotx_intake_state, rows)));
            for (unsigned i = 0; i < n; ++i) {
                if (mode && mode < 10) aotx_check(!rows_after[i].first_bytes && !rows_after[i].second_call &&
                    (!i ? rows_after[i].status == AOTX_COG_UNAVAILABLE : !rows_after[i].status),
                    "an unavailable pair is refused before either internal call");
                if (mode >= 10) aotx_check(rows_after[i].first_bytes && rows_after[i].status ==
                    (mode == 10 ? AOTX_COG_UNAVAILABLE : AOTX_COG_OK),
                    "handoff refusal and publication refusal occur at their separate boundaries");
                aotx_live_binding b;
                AOTX_CUDA(cudaMemcpyFromSymbol(&b, aotx_live_bindings, sizeof(b), i * sizeof(b)));
                aotx_check(b.ordinal == (mode ? 0u : 1u), "refusal preserves every conversation input ordinal");
            }
        }
        if (!mode) {
            aotx_intake_device d(n); d.process(start, true);
            aotx_intake_capability empty[AOTX_INTAKE_CAPABILITIES] = {};
            AOTX_CUDA(cudaMemcpyToSymbol(aotx_intake_capabilities, empty, sizeof(empty)));
            d.process(choices, true);
            aotx_check(!d.state().fatal && aotx_retain_store() == expected,
                "saved choices replay exactly after current qualification is removed");
        }
    }
}
int main(void) {
    int cards = 0; if (cudaGetDeviceCount(&cards) != cudaSuccess || !cards) return 77;
    for (unsigned n : {1u, 64u}) aotx_capability_cases(n);
    printf("intake capability: %u checks, %u failures\n", aotx_checks, aotx_failures);
    return aotx_failures ? 1 : 0;
}
