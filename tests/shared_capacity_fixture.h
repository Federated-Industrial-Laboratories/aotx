/* Purpose: Prepare distinct queued inputs and page owners for shared admission checks.
 * Owns: Test state only; the real bridge writes leases and completions.
 * Launch shape: One ordered setup batch before the production shared graph nodes.
 * Lifetime: Each test starts with separate receipts and an idle memory consumer. */
#ifndef AOTX_SHARED_CAPACITY_FIXTURE_H
#define AOTX_SHARED_CAPACITY_FIXTURE_H
#include "shared_state_fixture.h"
#include "shared/bridge.cuh"
#include "shared/capacity.cuh"
#include "cognitive/checkpoint.cuh"
#include "cognitive/codec.cuh"
#include "agent/agent_state.cuh"

__global__ void aotx_capacity_seed(unsigned count, unsigned cap, unsigned mapped, bool varied)
{
    aotx_sched.held = 0; aotx_sched.start_ns = 1000000000ull; aotx_time_tick = 1;
    aotx_live.ready = 1; aotx_live.fatal = 0; aotx_live.phase = AOTX_LIVE_IDLE; aotx_live.received = 0;
    aotx_live_store = {}; aotx_checkpoint = {}; aotx_runtime_enabled = 0;
    aotx_shared.kind = aotx_shared.total = aotx_shared.received = aotx_shared.fatal = 0;
    aotx_model_load.pending_count = 0;
    for (unsigned role = 0; role < AOTX_MODEL_ROLES; ++role) {
        aotx_model_load.resident[role] = {}; aotx_model_space[role].shape = {};
    }
    aotx_model_load.resident[AOTX_MODEL_LANGUAGE].active = 1;
    aotx_model_load.resident[AOTX_MODEL_LANGUAGE].body.digest[0] = 99;
    aotx_kvl_make(&aotx_model_space[AOTX_MODEL_LANGUAGE].shape, 36, 8, 128);
    aotx_kv.mapped_pages = mapped; aotx_kv.made = aotx_kv.served = aotx_kv.refused = 0;
    for (unsigned slot = 0; slot < AOTX_SLOTS; ++slot) {
        aotx_kv.count[slot] = 0; aotx_seqs.slot[slot] = {}; aotx_seq_asked[slot] = 0;
        aotx_say.slot[slot] = {}; aotx_shared.slot[slot] = aotx_service.slot[slot] = 0;
        aotx_shared_execution_slots[slot] = {}; aotx_live_bindings[slot] = {};
        aotx_agents.agent[slot] = {}; aotx_agents.agent[slot].state = AOTX_AGENT_STATE_FREE;
    }
    for (unsigned i = 0; i < count; ++i) {
        unsigned pages = varied ? 1 + i % cap : cap;
        aotx_shared_receipt &r = aotx_shared.receipts[i]; r = {};
        aotx_service_put(r.actor, i + 1, 4); r.sequence = i + 101; r.revision = 1;
        r.phase = AOTX_SHARED_QUEUED; r.saved_admission = 1; r.admission_source = i + 17;
        r.operation = AOTX_SHARED_INPUT; r.slot = AOTX_SLOTS; r.pages = pages;
        r.role = AOTX_MODEL_LANGUAGE; r.limit = 32; r.model_digest[0] = 99;
        r.participant = r.space = r.conversation = i;
        r.length = AOTX_SHARED_COMMAND_HEAD + 1; r.command[AOTX_SHARED_COMMAND_HEAD] = 'a' + i % 26;
        aotx_service_put(r.command + 136, 1, 4);
        auto &person = aotx_shared.participants[i]; person = {}; person.active = 1;
        aotx_service_put(person.id, i + 1, 4); person.next = r.sequence + 1;
        auto &space = aotx_shared.spaces[i]; space = {}; space.active = 1;
        aotx_service_put(space.owner, i + 1, 4); aotx_service_put(space.id, i + 201, 4);
        auto &c = aotx_shared.conversations[i]; c = {}; c.active = 1; c.space = i; c.request = i + 1;
        aotx_service_put(c.id, i + 301, 4);
        auto &g = aotx_service.grants[i]; g = {}; g.revision = 1; g.actions = 127;
        aotx_service_put(g.principal, i + 1, 4); g.pages = pages; g.tokens = 32; g.models = 1u << AOTX_MODEL_LANGUAGE;
    }
}

__global__ void aotx_capacity_end(unsigned request, unsigned long long now)
{
    aotx_live.phase = AOTX_LIVE_IDLE; aotx_live.received = 0; aotx_sched.start_ns = now;
    aotx_shared_complete(request, 200, 1, 1, 2);
}
__global__ void aotx_capacity_pool(unsigned mapped, unsigned pending, unsigned pages, unsigned first = 0)
{
    aotx_kv.mapped_pages = mapped; aotx_kv.made = first + pending; aotx_kv.served = first;
    for (unsigned i = 0; i < pending && i < AOTX_KV_QUEUE_MAX; ++i)
        aotx_kv.queue[(first + i) & (AOTX_KV_QUEUE_MAX - 1)] = {i % AOTX_SLOTS, pages};
}
__global__ void aotx_capacity_replay(const unsigned char *bytes, unsigned count, unsigned *result)
{
    aotx_seam.replaying = 1;
    *result = aotx_shared_apply(AOTX_SHARED_LEASE_RECORD, bytes, count, 5000, true);
    aotx_seam.replaying = 0;
}
static aotx_shared_state capacity_state()
{
    aotx_shared_state s; cu(cudaMemcpyFromSymbol(&s, aotx_shared, sizeof(s))); return s;
}
static std::vector<aotx_shared_receipt> capacity_receipts(fixture &f, unsigned count)
{
    std::vector<aotx_shared_receipt> rows(count);
    cu(cudaMemcpy(rows.data(), f.shared.receipts, count * sizeof(rows[0]), cudaMemcpyDeviceToHost)); return rows;
}
static void capacity_queued(fixture &f, unsigned count, unsigned completed = 0)
{
    auto rows = capacity_receipts(f, count);
    for (unsigned i = completed; i < count; ++i) {
        check(rows[i].phase == AOTX_SHARED_QUEUED && rows[i].slot == AOTX_SLOTS && !rows[i].terminal_source,
            "unleased input keeps its saved queued state");
        check(aotx_service_get(rows[i].actor, 4) == i + 1 && rows[i].sequence == i + 101 && rows[i].saved_admission,
            "queued input preserves its distinct actor and exact admission");
    }
    aotx_shared_execution execution[AOTX_SLOTS];
    cu(cudaMemcpyFromSymbol(execution, aotx_shared_execution_slots, sizeof(execution)));
    for (const auto &x : execution) check(!x.opened, "queued work has no execution clock");
}
#endif
