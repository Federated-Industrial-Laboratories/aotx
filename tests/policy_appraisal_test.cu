/* Purpose: Verify admitted appraisal work, versioned replay and stale proposal refusal.
 * Owns: Exact policy journals, distinct work revisions and negative controls.
 * Launch shape: Finite data and native graphs at N=1 and N=64.
 * Lifetime: Test-owned memory and policy state; no model work is generated. */
#include "policy_appraisal_batch.h"
#include "appraisal/appraisal.cuh"
#include <stddef.h>

__global__ void aotx_policy_appraisal_set(unsigned pending, uint64_t revision, unsigned guard) {
    if (threadIdx.x) return;
    aotx_appraisal.observed = aotx_live_store.sequence; aotx_appraisal.observed_root = aotx_live_store.root_sequence;
    aotx_appraisal.observed_count = aotx_live_store.count; aotx_appraisal.observed_bytes = aotx_live_store.bytes;
    aotx_appraisal.observed_maintenance = aotx_maintenance.passes;
    aotx_appraisal.pending = pending; aotx_appraisal.revision = revision;
    aotx_appraisal.background = guard != 6;
    aotx_policy.paused = guard == 1; aotx_policy.stopped = guard == 5;
    aotx_sched.held = guard == 2; aotx_seam.replaying = guard == 4;
    aotx_agents.agent[0].state = guard == 3 ? AOTX_AGENT_STATE_RUN : AOTX_AGENT_STATE_IDLE;
}
__global__ void aotx_policy_appraisal_maintenance(unsigned enabled) {
    if (!threadIdx.x) aotx_live_store.maintenance = enabled;
}
__global__ void aotx_policy_appraisal_take(unsigned change, unsigned *out) {
    if (threadIdx.x) return;
    if (change == 1) ++aotx_appraisal.revision;
    if (change == 2) ++aotx_live_store.root_sequence;
    if (change == 3) { ++aotx_live_store.sequence; aotx_appraisal.observed = aotx_live_store.sequence; }
    if (change == 4) aotx_policy.paused = 1;
    if (change == 5) aotx_policy.stopped = 1;
    if (change == 6) aotx_appraisal.background = 0;
    if (change == 7) aotx_appraisal.pending = 0;
    if (change == 8) aotx_policy.fatal = 1;
    if (change == 9) --aotx_live_store.count;
    if (change == 10) --aotx_live_store.bytes;
    aotx_appraisal.observed_root = aotx_live_store.root_sequence;
    aotx_appraisal.observed_count = aotx_live_store.count; aotx_appraisal.observed_bytes = aotx_live_store.bytes;
    aotx_appraisal.observed_maintenance = aotx_maintenance.passes;
    out[0] = aotx_policy_appraisal(); out[1] = aotx_policy_appraisal();
}
__global__ void aotx_policy_appraisal_replaying(unsigned replay, unsigned *out) {
    if (threadIdx.x) return;
    aotx_seam.replaying = replay;
    if (out) *out = aotx_policy_restore_end();
}
static void aotx_policy_appraisal_seed(aotx_live_device &d, unsigned n, unsigned maintenance = 0, unsigned pressure = 95) {
    AOTX_LIVE_CLEAR(aotx_checkpoint); AOTX_LIVE_CLEAR(aotx_appraisal);
    aotx_checkpoint_test_clear<<<1,AOTX_SLOTS>>>();
    auto corpus = aotx_memory_corpus(n); auto image = corpus.wire(false, corpus.rows.size());
    if (pressure) {
        aotx_put(image.data() + 8, 2, 4); aotx_put(image.data() + 96, corpus.rows.size());
        aotx_put(image.data() + 112, n, 4); aotx_put(image.data() + 120, maintenance, 4);
        aotx_put(image.data() + 124, pressure, 4);
    }
    d.send(aotx_live_load_bytes(image), AOTX_LIVE_LOAD);
    aotx_check(!d.state().status && d.state().ready, "appraisal policy memory passes ordinary admission");
    if (d.state().status) exit(1);
    d.send(aotx_live_binding_bytes(n, corpus.rows.size()), AOTX_LIVE_BIND); d.idle(n);
}
static aotx_live_records aotx_policy_appraisal_collect(aotx_live_device &d) {
    uint64_t first = d.seam().dev.tail;
    aotx_policy_active_graph->tick();
    unsigned limit = AOTX_POLICY_EVENT_BYTES / ((AOTX_BODY_BYTES - AOTX_POLICY_PART) * AOTX_POLICY_EMIT) + 2;
    for (unsigned i = 0; i < limit && aotx_policy_read_state()->pending; ++i) aotx_policy_active_graph->tick();
    aotx_check(!aotx_policy_read_state()->pending, "the complete proposal fits its finite publication bound");
    auto seam = d.seam();
    aotx_check(seam.dev.tail < d.output_slots, "the policy journal fits the fixture ring");
    aotx_live_records out(seam.dev.tail - first);
    AOTX_CUDA(cudaMemcpy(out.data(), d.out + first * AOTX_SLOT_BYTES, out.size() * AOTX_SLOT_BYTES, cudaMemcpyDeviceToHost));
    return out;
}
static void aotx_policy_appraisal_consumed(unsigned change, bool expected) {
    unsigned *out, result[2]; AOTX_CUDA(cudaMalloc(&out, sizeof(result)));
    aotx_policy_appraisal_take<<<1,1>>>(change, out);
    AOTX_CUDA(cudaMemcpy(result, out, sizeof(result), cudaMemcpyDeviceToHost)); cudaFree(out);
    aotx_check(result[0] == expected && !result[1], "a current enabled proposal is consumed once and stale proposals are refused");
    aotx_check(!aotx_policy_read_state()->appraise, "consumption clears the appraisal proposal on every outcome");
}
static void aotx_policy_appraisal_live(unsigned n, unsigned mode) {
    for (unsigned change = 0; change <= 10; ++change) {
        aotx_live_device d(n); aotx_policy_appraisal_seed(d, n, 0, change & 1 ? 0 : 95);
        aotx_policy_asset asset(mode, 16, "aotx_policy_appraisal_native", AOTX_POLICY_TEST_CASES,
            1, 255, AOTX_ARCH, 2); asset.open();
        {
            aotx_policy_graph graph; aotx_policy_active_graph = &graph;
            uint64_t revision = 0x100000000ull + n * 73 + change;
            for (unsigned guard = 1; guard <= 6; ++guard) {
                aotx_policy_appraisal_set<<<1,1>>>(n + 1, revision, guard); graph.tick();
                auto s = aotx_policy_read_state();
                aotx_check(!s->calls && !s->decision && !s->candidate[0], "disabled, paused, held, foreground, replay and stopped ticks skip creator code");
            }
            aotx_policy_appraisal_set<<<1,1>>>(0, revision, 0); graph.tick();
            aotx_check(!aotx_policy_read_state()->calls, "an empty appraisal queue with maintenance off performs no work");
            aotx_policy_appraisal_set<<<1,1>>>(n + 1, revision, 0);
            aotx_policy_appraisal_collect(d);
            auto s = aotx_policy_read_state();
            aotx_check(s->decision == 1 && s->calls == 1 && !s->status && s->appraise && !s->maintain,
                "admitted appraisal runs without enabling maintenance");
            aotx_check(s->input.reserved0 == 2 && s->input.reserved1[0] == n + 1 &&
                !s->input.enabled && s->work_revision == revision && aotx_get(s->event + 20, 4) == 2,
                "the exact version and work revision enter the accepted event");
            for (unsigned i = 0; i < 12; ++i) graph.tick();
            aotx_check(aotx_policy_read_state()->calls == 1, "unchanged pending work does not repeat the policy evaluation");
            aotx_policy_appraisal_consumed(change, change == 0);
            if (!change) {
                aotx_policy_appraisal_set<<<1,1>>>(n + 1, revision + 1, 0);
                aotx_policy_appraisal_collect(d); auto next = aotx_policy_read_state();
                aotx_check(next->calls == 2 && next->decision == 2 && next->work_revision == revision + 1 &&
                    next->source == s->source && next->root == s->root,
                    "a new work revision is evaluated at the same memory source and root");
                aotx_policy_appraisal_consumed(0, true);
            }
        }
        aotx_policy_close();
    }
    printf("policy appraisal live N=%u mode=%u\n", n, mode);
}
static unsigned aotx_policy_appraisal_part(const aotx_live_record &record) {
    const auto *h = (const aotx_record_header *)record.data();
    unsigned char *part; unsigned *ok, result = 0;
    AOTX_CUDA(cudaMalloc(&part, AOTX_BODY_BYTES)); AOTX_CUDA(cudaMalloc(&ok, sizeof(*ok)));
    AOTX_CUDA(cudaMemcpy(part, record.data() + 64, h->body_len, cudaMemcpyHostToDevice));
    aotx_policy_test_part<<<1,1>>>(part, h->body_len, AOTX_FLAG_REPLAYED, ok);
    AOTX_CUDA(cudaMemcpy(&result, ok, sizeof(result), cudaMemcpyDeviceToHost)); cudaFree(part); cudaFree(ok);
    return result;
}
static void aotx_policy_appraisal_replay(unsigned n) {
    aotx_policy_asset asset(AOTX_POLICY_NATIVE, AOTX_POLICY_STATE_BYTES, "aotx_policy_appraisal_native",
        AOTX_POLICY_TEST_CASES, 1, 255, AOTX_ARCH, 2);
    aotx_live_records parts;
    std::unique_ptr<aotx_policy_state> expected;
    {
        aotx_live_device d(n); aotx_policy_appraisal_seed(d, n, 0, 0); asset.open();
        aotx_policy_graph graph; aotx_policy_active_graph = &graph;
        aotx_policy_appraisal_set<<<1,1>>>(n + 1, 0x200000000ull + 97 * n, 0);
        auto all = aotx_policy_appraisal_collect(d); expected = aotx_policy_read_state();
        for (const auto &r : all) if (((const aotx_record_header *)r.data())->type == AOTX_REC_POLICY) parts.push_back(r);
        aotx_check(expected->decision == 1 && expected->appraise && parts.size() > AOTX_POLICY_EMIT,
            "the complete native appraisal state crosses multiple journal publication ticks");
    }
    aotx_policy_close();
    if (parts.empty()) return;
    for (unsigned fault = 0; fault <= 8; ++fault) {
        asset.open(); aotx_policy_appraisal_replaying<<<1,1>>>(1, nullptr);
        auto changed = parts;
        if (fault == 1) aotx_policy_patch(changed, 20, 0, 4);
        if (fault == 2) aotx_policy_patch(changed, 20, 3, 4);
        if (fault == 3) aotx_policy_patch(changed, 64 + offsetof(aotx_policy_input, reserved0), 0, 4);
        if (fault == 4) aotx_policy_patch(changed, 64 + offsetof(aotx_policy_input, reserved1), 0, 4);
        if (fault == 5) aotx_policy_patch(changed, 192, AOTX_POLICY_MAINTAIN, 4);
        if (fault == 6) aotx_policy_patch(changed, 192, 3, 4);
        if (fault == 7) aotx_policy_patch(changed, 192 + offsetof(aotx_policy_output, reserved), 1, 8);
        if (fault == 8) aotx_policy_patch(changed, 64 + offsetof(aotx_policy_input, enabled), 2, 4);
        unsigned ok = 1;
        for (size_t i = 0; ok && i < changed.size(); ++i) ok = aotx_policy_appraisal_part(changed[i]);
        auto s = aotx_policy_read_state();
        if (!fault) {
            aotx_check(ok && !s->fatal && s->decision == expected->decision && s->work_revision == expected->work_revision &&
                s->state_hash == expected->state_hash && s->observed_objects == expected->observed_objects &&
                s->observed_bytes == expected->observed_bytes && !memcmp(s->current, expected->current, AOTX_POLICY_STATE_BYTES),
                "ABI 2 replay restores every native state byte and the exact accepted work revision");
            aotx_check(!s->calls && !s->appraise && !s->maintain && !s->candidate[0],
                "recorded recovery invokes no creator code and issues no fresh work proposal");
        } else aotx_check(!ok && s->fatal && !s->decision && !s->current[0],
            "wrong event versions, input markers, work admission and output fields refuse without state publication");
        aotx_policy_close();
    }
    asset.open(); aotx_policy_appraisal_replaying<<<1,1>>>(1, nullptr);
    aotx_check(aotx_policy_appraisal_part(parts.front()), "the first valid appraisal state fragment is admitted");
    auto partial = aotx_policy_read_state();
    aotx_check(partial->received && !partial->decision && !partial->current[0], "a partial appraisal decision has no accepted state");
    unsigned *ok, result; AOTX_CUDA(cudaMalloc(&ok, sizeof(*ok)));
    aotx_policy_appraisal_replaying<<<1,1>>>(1, ok);
    AOTX_CUDA(cudaMemcpy(&result, ok, sizeof(result), cudaMemcpyDeviceToHost)); cudaFree(ok);
    auto clean = aotx_policy_read_state();
    aotx_check(result && !clean->received && !clean->pending && !clean->appraise && !clean->decision,
        "restore end discards the incomplete appraisal proposal");
    aotx_policy_close();
    printf("policy appraisal replay N=%u\n", n);
}
static void aotx_policy_appraisal_version(unsigned n) {
    for (unsigned abi : {1u, 2u}) {
        aotx_live_device d(n); aotx_policy_appraisal_seed(d, n, 1);
        aotx_policy_asset asset(AOTX_POLICY_NATIVE, 16, "aotx_policy_appraisal_unchecked", AOTX_POLICY_TEST_CASES,
            1, 255, AOTX_ARCH, abi); asset.open();
        {
            aotx_policy_graph graph; aotx_policy_active_graph = &graph;
            if (abi == 1) {
                aotx_policy_appraisal_maintenance<<<1,1>>>(0);
                aotx_policy_appraisal_set<<<1,1>>>(n + 1, 19 + n, 0); graph.tick();
                aotx_check(!aotx_policy_read_state()->calls, "ABI 1 cannot start appraisal or silently select another policy");
                aotx_policy_appraisal_maintenance<<<1,1>>>(1);
            }
            aotx_policy_appraisal_set<<<1,1>>>(0, 19 + n, 0); aotx_policy_appraisal_collect(d);
            auto s = aotx_policy_read_state();
            aotx_check(s->decision == 1 && s->status == AOTX_COG_FORMAT && s->paused && !s->appraise && !s->current[0],
                "ABI 1 appraisal and ABI 2 appraisal without pending evidence preserve state and pause");
            if (abi == 1) {
                aotx_check(!aotx_get(s->event + 20, 4) && !s->input.reserved0 && !s->input.reserved1[0] &&
                    !s->input.reserved1[1] && !s->input.reserved1[2], "ABI 1 keeps its original event and input reserved bytes zero");
            }
            aotx_policy_appraisal_consumed(0, false);
        }
        aotx_policy_close();
    }
}
static void aotx_policy_appraisal_quiet(unsigned n) {
    aotx_live_device d(n); aotx_policy_appraisal_seed(d, n);
    aotx_policy_asset asset(AOTX_POLICY_NATIVE, 16, "aotx_policy_appraisal_native", AOTX_POLICY_TEST_CASES,
        1, 255, AOTX_ARCH, 2); asset.open();
    {
        aotx_policy_graph graph; aotx_policy_active_graph = &graph;
        aotx_policy_appraisal_set<<<1,1>>>(1, 91 + n, 0); aotx_policy_appraisal_collect(d);
        auto quiet = aotx_policy_read_state();
        aotx_check(quiet->decision == 1 && !quiet->status && !quiet->appraise && !quiet->output.action,
            "a native creator can choose quiet when admissible appraisal work exists");
        for (unsigned i = 0; i < 12; ++i) graph.tick();
        aotx_check(aotx_policy_read_state()->calls == 1, "an accepted quiet decision suppresses repeated unchanged work");
        aotx_policy_appraisal_set<<<1,1>>>(2, 92 + n, 0); aotx_policy_appraisal_collect(d);
        aotx_check(aotx_policy_read_state()->appraise && aotx_policy_read_state()->calls == 2,
            "new evidence revision permits a fresh native decision after quiet");
        aotx_policy_appraisal_consumed(0, true);
    }
    aotx_policy_close();
}
int main() {
    AOTX_CUDA(cudaFree(nullptr));
    for (unsigned n : {1u, 64u}) {
        for (unsigned abi : {1u, 2u}) for (unsigned mode : {AOTX_POLICY_SUPPLIED, AOTX_POLICY_RULES, AOTX_POLICY_NATIVE})
            aotx_policy_appraisal_batch(n, abi, mode);
        for (unsigned mode : {AOTX_POLICY_RULES, AOTX_POLICY_NATIVE}) aotx_policy_appraisal_live(n, mode);
        aotx_policy_appraisal_replay(n); aotx_policy_appraisal_version(n); aotx_policy_appraisal_quiet(n);
    }
    printf("policy appraisal: %u checks, %u failures\n", aotx_checks, aotx_failures);
    return aotx_failures ? 1 : 0;
}
