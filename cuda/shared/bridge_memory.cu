/* Purpose: Project saved conversation bindings into temporary memory query slots.
 * Owns: No conversation history; bindings return to their persistent table after execution.
 * Launch shape: One recorded lease batch followed by the existing parallel memory graph.
 * Lifetime: The exact lease and its recorded memory choice. */
#include "shared/bridge.cuh"
#include "shared/internal.cuh"
#include "agent/agent_state.cuh"
#include "cognitive/codec.cuh"
#include "cli/prompt.cuh"
__device__ aotx_shared_execution aotx_shared_execution_slots[AOTX_SLOTS];
__device__ bool aotx_shared_bridge_lease(const unsigned *requests, const unsigned *slots,
                                       unsigned count, bool replay)
{
    if (!count || count > AOTX_RECALL_BATCH || !aotx_live.ready || aotx_live.fatal ||
        aotx_live.phase != AOTX_LIVE_IDLE || aotx_live.received ||
        aotx_shared.transfer_serial > (~0ull - AOTX_SLOTS) / AOTX_SLOTS) return false;
    for (unsigned i = 0; i < count; ++i) {
        if (requests[i] >= aotx_shared.receipt_capacity || !slots[i] || slots[i] >= AOTX_SLOTS) return false;
        const aotx_shared_receipt &r = aotx_shared.receipts[requests[i]];
        if (r.conversation >= aotx_shared.conversation_capacity || r.space >= aotx_shared.space_capacity ||
            aotx_live_bound(slots[i]) || aotx_service_owns(slots[i]) ||
            aotx_agents.agent[slots[i]].state != AOTX_AGENT_STATE_FREE) return false;
        for (unsigned j = 0; j < i; ++j) if (slots[i] == slots[j] || requests[i] == requests[j]) return false;
    }
    unsigned total = 64 + count * AOTX_LIVE_QUERY_ROW;
    aotx_shared_zero(aotx_live.input, total);
    aotx_service_bytes(aotx_live.input, (const unsigned char *)"AOTXTXT1", 8);
    aotx_cog_put(aotx_live.input + 8, count, 4); aotx_cog_put(aotx_live.input + 12, 1, 4);
    aotx_service_bytes(aotx_live.input + 16, aotx_live_store.lineage, 16);
    aotx_cog_put(aotx_live.input + 32, aotx_live_store.sequence, 8);
    aotx_cog_put(aotx_live.input + 40, AOTX_LIVE_QUERY_ROW, 4);
    for (unsigned i = 0; i < count; ++i) {
        unsigned slot = slots[i];
        aotx_shared_receipt &r = aotx_shared.receipts[requests[i]];
        aotx_shared_conversation &c = aotx_shared.conversations[r.conversation];
        const aotx_shared_space &space = aotx_shared.spaces[r.space];
        aotx_live_binding &b = aotx_live_bindings[slot]; b = c.binding;
        b.active = 1; b.pages = r.pages; b.scope = space.scope; b.auto_retain = 2;
        aotx_service_bytes(b.principal, space.id, 16);
        aotx_shared_zero(b.room, 16);
        if (space.scope == 1) aotx_service_bytes(b.room, space.id, 16);
        aotx_service_bytes(b.conversation, c.id, 16);
        aotx_agents.agent[slot] = {}; aotx_agents.agent[slot].state = AOTX_AGENT_STATE_IDLE;
        aotx_agents.agent[slot].task = ~0u; aotx_agents.agent[slot].role = ~0u;
        aotx_agent_gear[slot] = {}; aotx_say.slot[slot] = {}; aotx_media_prompts[slot] = {};
        aotx_shared_execution_slots[slot] = {AOTX_SHARED_MEMORY, 0, 0, aotx_sched.start_ns};
        aotx_prompt_roles[slot] = r.role;
        unsigned char *row = aotx_live.input + 64 + i * AOTX_LIVE_QUERY_ROW, *q = row + 64;
        aotx_cog_put(row, slot, 4); aotx_cog_put(row + 4, 1, 4);
        aotx_service_bytes(row + 16, c.id, 16); aotx_cog_put(row + 32, b.ordinal + 1, 8);
        unsigned long long id = aotx_shared.transfer_serial * AOTX_SLOTS + i + 1;
        aotx_service_bytes(q, (const unsigned char *)"AOTXSHQ1", 8); aotx_cog_put(q + 8, id, 8);
        aotx_service_bytes(q + 16, b.principal, 16); aotx_service_bytes(q + 32, b.room, 16);
        aotx_service_bytes(q + 48, (const unsigned char *)"AOTXSHS1", 8); aotx_cog_put(q + 56, id, 8);
        aotx_cog_put(q + 132, AOTX_RECALL_LIMIT, 4); aotx_cog_put(q + 136, AOTX_RECALL_BUDGET, 4);
        unsigned length = aotx_shared_input_text(&r, q + 4640, AOTX_RECALL_TEXT);
        aotx_cog_put(q + 148, length, 4); aotx_cog_put(q + 152, b.scope, 4);
    }
    aotx_live.op = AOTX_LIVE_TEXT; aotx_live.total = total; aotx_live.received = total;
    aotx_live.source_seq = aotx_shared.source; aotx_live.admission = 0; aotx_live.pressure = 0;
    aotx_service_bytes(aotx_live.transfer_id, (const unsigned char *)"AOTXSHB1", 8);
    aotx_cog_put(aotx_live.transfer_id + 8, aotx_shared.transfer_serial, 8);
    aotx_live.phase = AOTX_LIVE_READY;
    (void)replay; return true;
}
__device__ void aotx_shared_memory_choice(unsigned slot, unsigned status)
{
    aotx_shared_receipt *r = aotx_shared_request(slot);
    if (!r) return;
    aotx_shared_execution &x = aotx_shared_execution_slots[slot];
    if (!x.status) x.status = status ? 503 : 0;
    x.stage = x.status ? AOTX_SHARED_END : AOTX_SHARED_PROMPT;
    if (!status) r->input_committed = 1;
}
__device__ void aotx_shared_bridge_release(unsigned request, bool replay)
{
    aotx_shared_receipt &r = aotx_shared.receipts[request];
    if (r.slot >= AOTX_SLOTS) return;
    unsigned slot = r.slot;
    if (r.conversation < aotx_shared.conversation_capacity && aotx_live_bound(slot))
        aotx_shared.conversations[r.conversation].binding = aotx_live_bindings[slot];
    if (!replay) aotx_seq_stop(slot);
    aotx_live_bindings[slot] = {}; aotx_shared_execution_slots[slot] = {};
    aotx_say.slot[slot] = {}; aotx_media_prompts[slot] = {}; aotx_agent_gear[slot] = {};
    aotx_agents.agent[slot] = {}; aotx_agents.agent[slot].state = AOTX_AGENT_STATE_FREE;
    aotx_agents.agent[slot].task = ~0u;
}
__device__ bool aotx_shared_memory_authorized(void)
{
    for (unsigned slot = 0; slot < AOTX_SLOTS; ++slot) {
        const aotx_shared_receipt *r = aotx_shared_request(slot);
        if (r && aotx_shared_execution_slots[slot].stage == AOTX_SHARED_MEMORY &&
            (!aotx_shared_authorized(r, 2) || r->cancel)) return false;
    }
    return true;
}
