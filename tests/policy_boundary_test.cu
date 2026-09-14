/* Purpose: Verify incomplete, invalid and paused creator state transitions.
 * Owns: Exact recorded fragments and independent fault mutations.
 * Launch shape: Real native conditional nodes and ordered replay at N=1 and N=64.
 * Lifetime: One test process with isolated policy revisions. */
#include "policy_fixture.h"
#include "cli/cli.cuh"
#include "model/load.cuh"

__global__ void aotx_policy_test_replay(unsigned replay, unsigned *ok) {
    if (!threadIdx.x) {
        aotx_seam.replaying = replay;
        if (ok) *ok = aotx_policy_restore_end();
    }
}
__global__ void aotx_policy_test_dirty(void) {
    if (!threadIdx.x) { aotx_runtime_enabled = 1; ++aotx_runtime_dirty; }
}
static void aotx_policy_seed(aotx_live_device &d, unsigned n) {
    AOTX_LIVE_CLEAR(aotx_checkpoint);
    aotx_checkpoint_test_clear<<<1,AOTX_SLOTS>>>();
    auto corpus = aotx_memory_corpus(n); auto image = corpus.wire(false, corpus.rows.size());
    aotx_put(image.data() + 8, 2, 4); aotx_put(image.data() + 112, n, 4);
    aotx_put(image.data() + 96, corpus.rows.size());
    aotx_put(image.data() + 120, 1, 4); aotx_put(image.data() + 124, 95, 4);
    d.send(aotx_live_load_bytes(image), AOTX_LIVE_LOAD);
    aotx_check(!d.state().status && d.state().ready, "the boundary memory image passes complete admission");
    if (d.state().status) exit(1);
    d.send(aotx_live_binding_bytes(n, corpus.rows.size()), AOTX_LIVE_BIND);
    d.idle(n);
}
__global__ void aotx_policy_foreground_set(unsigned n, unsigned bound, unsigned runtime, unsigned kind) {
    if (threadIdx.x) return;
    unsigned slot = n - 1;
    aotx_runtime_enabled = runtime; aotx_live_bindings[slot].active = bound;
    aotx_agents.agent[slot].state = kind == 1 ? AOTX_AGENT_STATE_RUN : AOTX_AGENT_STATE_IDLE;
    aotx_agents.agent[slot].turn = 1000 + slot;
    aotx_agent_gear[slot].has_message = kind == 2; aotx_say.slot[slot].wanted = kind == 3;
    aotx_tool_embed.state[slot] = kind == 4 ? AOTX_TOOL_EMBED_WAIT : AOTX_TOOL_EMBED_NONE;
    aotx_seqs.slot[slot].state = kind == 5 ? AOTX_SEQ_STATE_DECODE : AOTX_SEQ_STATE_FREE;
    aotx_model_load.pending_count = kind == 6;
    aotx_task_used[AOTX_TASK_SLOTS - 1] = kind == 7;
    aotx_agents.task[AOTX_TASK_SLOTS - 1].state = AOTX_TASK_PENDING;
    aotx_catalog.arriving[AOTX_CATALOG_ARRIVING_MAX - 1].import = kind == 8;
#ifdef AOTX_AFFECT
    aotx_quality_state[slot].pending = kind == 9;
    aotx_quality_state[slot].ended = kind == 10;
#endif
}
static void aotx_policy_foreground(unsigned n) {
    for (unsigned bound : {0u, 1u}) for (unsigned runtime : {0u, 1u}) {
        aotx_live_device d(n); aotx_policy_seed(d, n);
        aotx_policy_asset asset(AOTX_POLICY_NATIVE, 16, "aotx_policy_malformed", AOTX_POLICY_TEST_CASES);
        asset.open();
        {
            aotx_policy_graph graph;
            unsigned kinds = 8;
#ifdef AOTX_AFFECT
            kinds = 10;
#endif
            for (unsigned kind = 1; kind <= kinds; ++kind) {
                aotx_policy_foreground_set<<<1,1>>>(n, bound, runtime, kind); graph.tick();
                auto state = aotx_policy_read_state();
                aotx_check(!state->calls && !state->decision && !state->input.valid && !state->candidate[0],
                    "foreground work skips native entry with and without memory bindings or complete runtime mode");
            }
            aotx_policy_foreground_set<<<1,1>>>(n, bound, runtime, 0); graph.tick();
            aotx_check(aotx_policy_read_state()->calls == 1, "the same native entry runs when all foreground work ends");
        }
        aotx_policy_close();
    }
    AOTX_LIVE_CLEAR(aotx_runtime_enabled);
}
static aotx_live_records aotx_policy_fragments(unsigned n, aotx_policy_asset &asset) {
    aotx_live_device d(n); aotx_policy_seed(d, n);
    asset.open(); aotx_policy_graph graph; aotx_policy_active_graph = &graph;
    aotx_live_records all = d.process({}, false, true, aotx_policy_test_hook);
    auto first = aotx_policy_read_state();
    aotx_check(first->pending && first->calls == 1 && !first->decision && !first->current[0],
        "a partial journal cannot publish native private state");
    unsigned each = (AOTX_BODY_BYTES - AOTX_POLICY_PART) * AOTX_POLICY_EMIT;
    unsigned remaining = (first->total - first->emitted + each - 1) / each;
    for (unsigned i = 0; i < remaining && aotx_policy_read_state()->pending; ++i)
        aotx_maint_append(all, d.process({}, false, true, aotx_policy_test_hook));
    auto last = aotx_policy_read_state();
    aotx_check(!last->pending && last->decision == 1 && last->current[0] == 1,
        "the complete final fragment publishes the candidate exactly once");
    if (last->pending) exit(1);
    aotx_live_records parts;
    for (const auto &r : all) if (((const aotx_record_header *)r.data())->type == AOTX_REC_POLICY) parts.push_back(r);
    aotx_check(parts.size() > AOTX_POLICY_EMIT, "declared native state requires multiple publication ticks");
    aotx_policy_close(); return parts;
}
static unsigned aotx_policy_apply_part(const aotx_live_record &r, unsigned flags = AOTX_FLAG_REPLAYED) {
    const auto *h = (const aotx_record_header *)r.data();
    unsigned char *part; unsigned *ok, actual = 0;
    AOTX_CUDA(cudaMalloc(&part, AOTX_BODY_BYTES)); AOTX_CUDA(cudaMalloc(&ok, sizeof(*ok)));
    AOTX_CUDA(cudaMemcpy(part, r.data() + 64, h->body_len, cudaMemcpyHostToDevice));
    aotx_policy_test_part<<<1,1>>>(part, h->body_len, flags, ok);
    AOTX_CUDA(cudaMemcpy(&actual, ok, sizeof(actual), cudaMemcpyDeviceToHost));
    cudaFree(ok); cudaFree(part); return actual;
}
static void aotx_policy_replay_faults(unsigned n, unsigned abi) {
    aotx_policy_asset asset(AOTX_POLICY_NATIVE, AOTX_POLICY_STATE_BYTES, "aotx_creator_maintenance",
        AOTX_POLICY_TEST_PTX, 1, 255, AOTX_ARCH, abi);
    auto parts = aotx_policy_fragments(n, asset);
    if (parts.empty()) return;
    for (unsigned fault = 0; fault < 9; ++fault) {
        asset.open(); aotx_policy_test_replay<<<1,1>>>(1, nullptr);
        auto changed = parts; unsigned char *p = changed.front().data() + 64;
        if (fault == 0) p[8] = 1;
        if (fault == 1) p[4] ^= 1;
        if (fault == 2) p[AOTX_POLICY_PART + 32] ^= 1;
        if (fault == 3) p[AOTX_POLICY_PART + 12] ^= 1;
        if (fault == 4) p[16] = 2;
        if (fault == 6) aotx_policy_patch(changed, 20, abi == 1 ? 2 : 0, 4);
        if (fault == 7) aotx_policy_patch(changed, 64 + offsetof(aotx_policy_input, reserved0), abi == 1 ? 2 : 0, 4);
        if (fault == 8) aotx_policy_patch(changed, 192, AOTX_POLICY_APPRAISE, 4);
        unsigned ok = aotx_policy_apply_part(changed.front(), fault == 5 ? 0 : AOTX_FLAG_REPLAYED);
        for (size_t i = 1; ok && i < changed.size(); ++i) ok = aotx_policy_apply_part(changed[i]);
        auto state = aotx_policy_read_state();
        aotx_check(!ok && state->fatal && !state->decision && !state->current[0],
            "framing, identity, replay and unsupported ABI faults preserve accepted state");
        aotx_policy_close();
    }
    asset.open(); aotx_policy_test_replay<<<1,1>>>(1, nullptr);
    aotx_check(aotx_policy_apply_part(parts.front()), "valid initial replay fragment is admitted");
    auto partial = aotx_policy_read_state();
    aotx_check(partial->received && !partial->decision && !partial->current[0],
        "an incomplete raw journal has no committed private state");
    unsigned *ok, result; AOTX_CUDA(cudaMalloc(&ok, sizeof(*ok)));
    aotx_policy_test_replay<<<1,1>>>(1, ok);
    AOTX_CUDA(cudaMemcpy(&result, ok, sizeof(result), cudaMemcpyDeviceToHost)); cudaFree(ok);
    auto stopped = aotx_policy_read_state();
    aotx_check(result && !stopped->received && !stopped->pending && !stopped->maintain && !stopped->decision,
        "restore end discards only the uncommitted candidate");
    aotx_policy_close();
}
static void aotx_policy_malformed(unsigned n) {
    aotx_live_device d(n); aotx_policy_seed(d, n);
    aotx_policy_asset asset(AOTX_POLICY_NATIVE, 16, "aotx_policy_malformed", AOTX_POLICY_TEST_CASES);
    asset.open(); aotx_policy_graph graph; aotx_policy_active_graph = &graph;
    for (unsigned guard : {1u, 2u, 3u}) {
        aotx_policy_test_control<<<1,1>>>(guard); graph.tick(); auto state = aotx_policy_read_state();
        aotx_check(!state->calls && !state->candidate[0] && !state->output.action,
            "conditional admission skips a native entry that ignores input validity");
    }
    aotx_policy_test_control<<<1,1>>>(0); aotx_policy_test_replay<<<1,1>>>(1, nullptr); graph.tick();
    aotx_check(!aotx_policy_read_state()->candidate[0], "replay skips the native entry itself");
    d.process({}, false, true, aotx_policy_test_hook);
    auto state = aotx_policy_read_state();
    aotx_check(state->decision == 1 && state->status == AOTX_COG_FORMAT && state->paused &&
        !state->current[0] && !state->maintain, "malformed native output records an error and preserves prior state");
    graph.tick(); aotx_check(aotx_policy_read_state()->calls == 1, "the recorded error pauses further native work");
    aotx_policy_close();
}
static void aotx_policy_checkpoint(unsigned n) {
    aotx_checkpoint_device d(n);
    aotx_checkpoint_ring *ring = d.ring();
    auto checkpoint = d.state();
    aotx_policy_seed(d.live, n);
    AOTX_CUDA(cudaMemcpyToSymbol(aotx_checkpoint, &checkpoint, sizeof(checkpoint)));
    d.publish(1); d.acknowledge(1); d.step();
    aotx_policy_asset asset(AOTX_POLICY_NATIVE, AOTX_POLICY_STATE_BYTES); asset.open();
    aotx_policy_graph graph; aotx_policy_active_graph = &graph;
    d.live.process({}, false, true, aotx_policy_test_hook);
    aotx_check(aotx_policy_read_state()->pending, "native state publication is still in progress");
    aotx_policy_test_dirty<<<1,1>>>(); d.step();
    aotx_check(!d.state().copying && ring->head == 1, "a dirty complete runtime waits for the final policy fragment");
    for (unsigned i = 0; i < AOTX_POLICY_EVENT_BYTES / 128 + 1 && aotx_policy_read_state()->pending; ++i)
        d.live.process({}, false, true, aotx_policy_test_hook);
    d.publish(2);
    aotx_check(ring->head == 2 && !aotx_policy_read_state()->pending,
        "complete runtime capture resumes after private-state publication");
    AOTX_LIVE_CLEAR(aotx_runtime_enabled); AOTX_LIVE_CLEAR(aotx_runtime_dirty); aotx_policy_close();
}
static void aotx_policy_console(unsigned n) {
    aotx_live_device d(n); aotx_policy_seed(d, n); AOTX_LIVE_CLEAR(aotx_cli);
    aotx_policy_asset asset(AOTX_POLICY_NATIVE); asset.open();
    const char *commands[] = {"status", "pause", "resume", "stop"};
    for (unsigned mode = 0; mode < 4; ++mode) {
        aotx_live_records records(n);
        for (unsigned i = 0; i < n; ++i) {
            auto *h = (aotx_record_header *)records[i].data();
            std::string text = "policy " + std::string(commands[mode]) + std::string(i, ' ');
            h->magic = AOTX_WIRE_MAGIC; h->layout = AOTX_WIRE_LAYOUT; h->header_bytes = 64;
            h->cls = AOTX_CLASS_A; h->type = AOTX_REC_INPUT_LINE; h->writer = AOTX_WRITER_CONSOLE;
            h->seq = i + 1; h->body_len = text.size();
            memcpy(records[i].data() + 64, text.data(), text.size());
        }
        uint64_t first = d.seam().dev.tail; d.process(records);
        aotx_live_records emitted(d.seam().dev.tail - first);
        AOTX_CUDA(cudaMemcpy(emitted.data(), d.out + first * AOTX_SLOT_BYTES,
            emitted.size() * AOTX_SLOT_BYTES, cudaMemcpyDeviceToHost));
        unsigned states = 0, counters = 0;
        std::string state = mode == 1 ? "paused" : mode == 3 ? "stopped" : "quiet";
        for (size_t j = 0; j < emitted.size(); ++j) {
            const auto &r = emitted[j];
            auto *h = (const aotx_record_header *)r.data();
            if (h->seq != first + j + 1 || h->cls != AOTX_CLASS_B ||
                h->type != AOTX_REC_CONSOLE || h->body_len > AOTX_BODY_BYTES) continue;
            std::string line((const char *)r.data() + 64, h->body_len);
            states += line.find("policy: " + state + " mode 3 decision 0") == 0;
            counters += line == "policy: calls 0 last ns 0 maximum ns 0 state hash 0 saved generation 0";
        }
        aotx_check(states == n && counters == n, "every control publishes both complete console status lines");
        auto current = aotx_policy_read_state();
        aotx_check(current->paused == (mode == 1 || mode == 3) && current->stopped == (mode == 3),
            "real console parsing changes the selected policy control state");
    }
    aotx_policy_close();
}
int main() {
    for (unsigned n : {1u, 64u}) {
        aotx_policy_foreground(n);
        for (unsigned abi : {1u, 2u}) aotx_policy_replay_faults(n, abi);
        aotx_policy_malformed(n); aotx_policy_checkpoint(n); aotx_policy_console(n);
    }
    printf("policy boundary: %u checks, %u failures\n", aotx_checks, aotx_failures);
    return aotx_failures ? 1 : 0;
}
