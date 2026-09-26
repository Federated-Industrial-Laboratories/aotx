/* Purpose: Check scoped affect isolation, cancellation, exact replay and slot reuse.
 * Owns: Distinct state rows and malformed completion controls without model weights.
 * Launch shape: N=1 and the profile slot limit through bounded shared lease groups.
 * Lifetime: One isolated set of recorded leases and completions. */
#include "shared_state_fixture.h"
#include "shared/affect.cuh"
#include "shared/bridge.cuh"
#include "agent/agent_state.cuh"
#include "cli/prompt.cuh"
#ifdef AOTX_AFFECT
struct aotx_affect_result {
    unsigned malformed, isolated, cleared, encoded, replayed, conflict;
    aotx_shared_affect_state state;
    unsigned char record[128], read[96];
};
__global__ void aotx_scope_seed(unsigned first, unsigned start, unsigned count, unsigned mode)
{
    unsigned slot = threadIdx.x;
    aotx_shared.slot[slot] = aotx_service.slot[slot] = 0;
    aotx_live_bindings[slot] = {}; aotx_agents.agent[slot] = {};
    aotx_say.slot[slot] = {}; aotx_seqs.slot[slot] = {};
    aotx_affect_state[slot].fast[0] = -12345;
    aotx_affect_acc[slot].events = AOTX_AFFECT_EVENT_MASK;
    if (!slot) {
        aotx_live.received = 0; aotx_live.phase = AOTX_LIVE_IDLE;
        aotx_setting_table.row[AOTX_SET_AFFECT_ON].value = mode == 2 ? 0 : 1;
    }
    if (slot < start || slot >= start + count) return;
    unsigned i = first + slot - start;
    auto &r = aotx_shared.receipts[i]; r = {};
    aotx_service_put(r.actor, i + 1, 4); r.sequence = i + 100; r.revision = 1;
    r.phase = AOTX_SHARED_QUEUED; r.operation = AOTX_SHARED_INPUT; r.saved_admission = 1;
    r.space = r.conversation = r.participant = i; r.slot = AOTX_SLOTS;
    r.role = AOTX_MODEL_LANGUAGE; r.pages = 4; r.model_digest[0] = 99;
    r.command[AOTX_SHARED_COMMAND_HEAD] = 'x'; aotx_service_put(r.command + 136, 1, 4);
    auto &person = aotx_shared.participants[i]; person = {}; person.active = 1; aotx_service_put(person.id, i + 1, 4);
    auto &space = aotx_shared.spaces[i]; space = {}; space.active = 1;
    aotx_service_put(space.id, i + 90, 4); aotx_service_put(space.owner, i + 1, 4);
    space.scope = mode == 5 ? 1 : mode == 6 ? 2 : 0;
    auto &c = aotx_shared.conversations[i]; c = {}; c.active = 1; c.space = i;
    aotx_service_put(c.id, i + 120, 4); c.request = i + 1;
    auto &owned = space.scope ? space.affect : c.affect;
    owned.value.scale = AOTX_AFFECT_SCALE_ONE; owned.value.axes = AOTX_AFFECT_DATA_AXES;
    if (mode != 2) {
        owned.enabled = 1; owned.revision = i + 7;
        owned.value.fast[0] = i + 100; owned.value.slow[1] = -int(i + 200);
    }
}
__global__ void aotx_scope_lease(unsigned first, unsigned start, unsigned count, unsigned *out)
{
    unsigned requests[AOTX_SLOTS], slots[AOTX_SLOTS];
    for (unsigned j = 0; j < count; ++j) { requests[j] = first + j; slots[j] = start + j; }
    *out = aotx_shared_lease(requests, slots, count);
}
__global__ void aotx_scope_complete(unsigned i, unsigned mode, aotx_affect_result *out)
{
    auto &r = aotx_shared.receipts[i]; unsigned slot = r.slot;
    auto &v = out[i]; v = {};
    if (slot >= AOTX_SLOTS) return;
    auto &state = *aotx_shared_affect_scope(&r);
    v.isolated = aotx_affect_state[slot].fast[0] == (mode == 2 ? 0 : int(i + 100)) &&
        !aotx_affect_acc[slot].events && !aotx_affect_laws[slot].on;
    aotx_seqs.slot[slot].role = r.role;
    aotx_shared_execution_slots[slot].model_opened = mode != 3;
    aotx_affect_acc[slot] = {}; aotx_affect_acc[slot].flag = mode == 0 || mode >= 4;
    aotx_affect_acc[slot].sampled = 4;
    aotx_affect_laws[slot] = {};
    aotx_affect_laws[slot].decay_fast = 0.9f; aotx_affect_laws[slot].decay_slow = 0.99f;
    aotx_affect_laws[slot].gain_fast = 0.1f; aotx_affect_laws[slot].gain_slow = 0.01f;
    aotx_affect_laws[slot].cap[0] = aotx_affect_laws[slot].cap[1] = 0.5f;
    r.cancel = mode == 3 || mode == 4;
    unsigned prompt = mode == 3 ? 0 : i + 10;
    if (!aotx_shared_complete(i, r.cancel ? 409 : 200, prompt, mode == 3 ? 0 : 4, r.cancel ? 0 : 1)) return;
    v.encoded = aotx_shared.total;
    aotx_service_bytes(v.record, aotx_shared.transfer, aotx_shared.total);
    if (aotx_shared.total == 128) {
        const unsigned offsets[] = {28, 56, 64, 72, 98, 100, 104, 108, 112, 116, 120};
        for (unsigned at : offsets) {
            unsigned char bad[128]; aotx_service_bytes(bad, v.record, 128);
            if (at == 104) aotx_service_put(bad + at, 0x7fc00000u, 4);
            else if (at == 108) aotx_service_put(bad + at, 0x80000000u, 4);
            else bad[at] ^= at == 112 ? 128 : 255;
            aotx_shared_affect_state before = state;
            bool accepted = aotx_shared_apply(AOTX_SHARED_COMPLETE_RECORD, bad, 128, 400 + i, true);
            if (!accepted && aotx_service_equal((unsigned char *)&before, (unsigned char *)&state, sizeof(state)) &&
                !r.terminal_source) ++v.malformed;
        }
    }
    auto other = r; other.conversation = aotx_shared.conversation_capacity - 1;
    auto &c = aotx_shared.conversations[other.conversation]; c = {}; c.active = 1; c.space = r.space;
    unsigned saved_scope = aotx_shared.spaces[r.space].scope;
    aotx_shared.spaces[r.space].scope = 0;
    bool private_clear = !aotx_shared_affect_conflict(&r, &other);
    aotx_shared.spaces[r.space].scope = 1;
    bool room = aotx_shared_affect_conflict(&r, &other);
    aotx_shared.spaces[r.space].scope = 2;
    bool instance = aotx_shared_affect_conflict(&r, &other);
    aotx_shared.spaces[r.space].scope = saved_scope;
    v.conflict = private_clear && (mode == 2 ? !room && !instance : room && instance);
}
__global__ void aotx_scope_observe(unsigned i, unsigned slot, aotx_affect_result *out)
{
    auto &v = out[i]; v.state = *aotx_shared_affect_scope(&aotx_shared.receipts[i]);
    v.cleared = !aotx_affect_state[slot].fast[0] && !aotx_affect_state[slot].slow[1] &&
        aotx_affect_state[slot].scale == AOTX_AFFECT_SCALE_ONE && !aotx_affect_acc[slot].flag &&
        !aotx_affect_acc[slot].sampled && !aotx_affect_acc[slot].events && !aotx_affect_laws[slot].gain_fast;
    aotx_shared_affect_read(i, v.read);
}
__global__ void aotx_scope_replay(unsigned i, unsigned slot, aotx_affect_result *out)
{
    auto &v = out[i]; auto &r = aotx_shared.receipts[i];
    auto &state = *aotx_shared_affect_scope(&r);
    aotx_shared_affect_state expected = state;
    state = {}; state.enabled = 1; state.revision = i + 7;
    state.value.fast[0] = i + 100; state.value.slow[1] = -int(i + 200);
    state.value.axes = AOTX_AFFECT_DATA_AXES; state.value.scale = AOTX_AFFECT_SCALE_ONE;
    r.phase = AOTX_SHARED_RUNNING; r.terminal_source = 0; r.slot = slot; r.sample.affect = 1;
    aotx_shared.slot[slot] = i + 1;
    aotx_affect_laws[slot].gain_fast = 1000; aotx_affect_state[slot].fast[0] = -17000;
    aotx_affect_acc[slot].events = AOTX_AFFECT_EVENT_MASK;
    bool applied = aotx_shared_apply(AOTX_SHARED_COMPLETE_RECORD, v.record, 128, 1000 + i, true);
    bool duplicate = aotx_shared_apply(AOTX_SHARED_COMPLETE_RECORD, v.record, 128, 1001 + i, true);
    v.replayed = applied && !duplicate && aotx_service_equal((unsigned char *)&state,
        (unsigned char *)&expected, sizeof(state)) && r.slot == AOTX_SLOTS && !aotx_affect_acc[slot].events;
}
static void run(unsigned n, unsigned mode)
{
    fixture f(n); aotx_affect_result *device;
    cu(cudaMalloc(&device, n * sizeof(*device))); cu(cudaMemset(device, 0, n * sizeof(*device)));
    for (unsigned first = 0; first < n;) {
        unsigned start = first % (AOTX_SLOTS - 1) + 1;
        unsigned count = std::min(std::min(n - first, AOTX_SLOTS - start), AOTX_RECALL_BATCH);
        aotx_scope_seed<<<1,AOTX_SLOTS>>>(first, start, count, mode);
        aotx_scope_lease<<<1,1>>>(first, start, count, f.result); cu(cudaDeviceSynchronize());
        check(f.value(), "complete affect lease batch admitted");
        aotx_shared_state pending; cu(cudaMemcpyFromSymbol(&pending, aotx_shared, sizeof(pending)));
        check(aotx_service_get(pending.transfer + 4, 4) == (mode == 2 ? 4u : 5u), "disabled lease retains old exact record version");
        for (unsigned j = 0; j < 1 + count * 40 / (AOTX_SHARED_RECORD_DATA * AOTX_SHARED_EMIT); ++j) aotx_shared_emit<<<1,1>>>();
        for (unsigned i = first; i < first + count; ++i) {
            aotx_scope_complete<<<1,1>>>(i, mode, device); aotx_shared_emit<<<1,1>>>();
            aotx_scope_observe<<<1,1>>>(i, i - first + start, device);
            if (mode == 0 || mode == 1 || mode >= 4) aotx_scope_replay<<<1,1>>>(i, i - first + start, device);
        }
        cu(cudaDeviceSynchronize());
        first += count;
    }
    std::vector<aotx_affect_result> rows(n);
    cu(cudaMemcpy(rows.data(), device, n * sizeof(rows[0]), cudaMemcpyDeviceToHost));
    for (unsigned i = 0; i < n; ++i) {
        auto &v = rows[i]; bool extension = mode == 0 || mode == 1 || mode >= 4;
        check(v.isolated, "lease replaces stale slot state and event flags");
        check(v.cleared, "completion clears state sums and law before slot reuse");
        check(v.conflict, "private state stays separate and shared state has one ordered lease");
        check(v.encoded == (extension ? 128u : 64u), "state record exists only for a managed started turn");
        check(v.malformed == (extension ? 11u : 0u), "malformed completions refuse without state or receipt mutation");
        check(v.state.revision == (mode == 2 ? 0u : i + 7u + extension), "scoped revision advances once per complete state record");
        check(v.state.available == 0, "absent probes stay unavailable");
        if (extension) check(v.replayed, "recorded state survives changed laws and rejects duplicate completion");
        if (mode == 0 || mode >= 4) {
            check(v.state.enabled && v.state.value.fast[0] != int(i + 100), "event law changes the owned state");
            check(!(v.state.reason & (1u << AOTX_AFFECT_EVENT_BUDGET)), "shared turn does not inherit an unused agent budget event");
            check(v.state.reason == 1u << (mode == 4 ? AOTX_AFFECT_EVENT_OPERATOR_STOP : AOTX_AFFECT_EVENT_STOP), "terminal event has its actual cause");
        }
        if (mode == 1 || mode == 2) check(!v.state.enabled && !v.state.value.fast[0], "off turn retains or restores neutral state");
        if (mode == 3) check(v.state.enabled && v.state.value.fast[0] == int(i + 100), "cancellation before inference leaves scope state unchanged");
    }
    aotx_shared_state final; cu(cudaMemcpyFromSymbol(&final, aotx_shared, sizeof(final)));
    check(!final.fatal && !final.kind, "all scoped transfers finish");
    cudaFree(device);
    std::printf("shared-affect N=%u mode=%u checks=%u failures=%u\n", n, mode, checks, failures);
}
#endif
int main(int argc, char **argv)
{
#ifdef AOTX_AFFECT
    bool memory = argc == 2 && !strcmp(argv[1], "--memory-check");
    if (argc != 1 && !memory) return 2;
    for (unsigned n : {1u, AOTX_SLOTS})
        for (unsigned mode = 0; mode < (memory ? 1u : 7u); ++mode) run(n, mode);
    std::printf("shared-affect total checks=%u failures=%u\n", checks, failures);
    return failures ? 1 : 0;
#else
    return 77;
#endif
}
