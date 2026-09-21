/* Purpose: Render a shared model prompt from authored identity and selected memory.
 * Owns: Transient prompt bytes and frozen sampling options.
 * Launch shape: One calling thread for each leased slot.
 * Lifetime: One shared input, without an operator transcript or tool grant. */
#include "shared/bridge.cuh"
#include "shared/internal.cuh"
#include "catalog/catalog.cuh"
#include "agent/agent_state.cuh"
#include "cli/prompt.cuh"
#include "model/wrap.cuh"
#include "cognitive/recall_labels.cuh"
__device__ unsigned aotx_shared_model_prompt(unsigned slot)
{
    const aotx_shared_receipt *r = aotx_shared_request(slot);
    if (!r || aotx_live_prompt_check(slot)) return 503;
    const aotx_wrap *wrap = aotx_wrap_active(r->role);
    if (!wrap->usable) return 503;
    unsigned char *out = aotx_say.prompt[slot];
    unsigned at = aotx_wrap_put(out, 0, AOTX_SAY_BYTES, wrap, AOTX_WRAP_SYSTEM_HEAD);
    unsigned role = aotx_agents.agent[0].role;
    if (role < AOTX_MODULE_SLOTS) {
        const aotx_catalog_run overlay = aotx_catalog.entry[role].role.overlay;
        at = aotx_wrap_run(out, at, AOTX_SAY_BYTES, aotx_catalog_arena + overlay.at, overlay.length);
    }
    at = aotx_wrap_put(out, at, AOTX_SAY_BYTES, wrap, AOTX_WRAP_SYSTEM_TAIL);
    at = aotx_live_context(slot, out, at);
    unsigned turn = at;
    at = aotx_wrap_put(out, at, AOTX_SAY_BYTES, wrap, AOTX_WRAP_USER_HEAD);
    const unsigned char *q = aotx_live_bindings[slot].query;
    if (aotx_context_sources(q)) {
        if (!aotx_service_equal(q + AOTX_RECALL_ACTOR, r->actor, 16)) return 503;
        at = aotx_recall_word(out, at, AOTX_SAY_BYTES, "[source_actor=");
        at = aotx_recall_hex(out, at, AOTX_SAY_BYTES, r->actor);
        at = aotx_recall_word(out, at, AOTX_SAY_BYTES, "]\n");
    }
    if (at > AOTX_SAY_BYTES) return 413;
    unsigned n = aotx_shared_input_text(r, out + at, AOTX_SAY_BYTES - at);
    if (n > AOTX_SAY_BYTES - at) return 413;
    at += n;
    at = aotx_wrap_put(out, at, AOTX_SAY_BYTES, wrap, AOTX_WRAP_USER_TAIL);
    at = aotx_wrap_generation(out, at, AOTX_SAY_BYTES, wrap);
    if (at > AOTX_SAY_BYTES) return 413;
    aotx_say_slot &s = aotx_say.slot[slot]; s = {};
    s.length = at; s.turn_at = turn; s.page_limit = r->pages; s.wanted = 1;
    s.token_deadline = aotx_time_tick + AOTX_SAY_TOKEN_WAIT_TICKS;
    aotx_say_count[slot] = 0; aotx_media_prompts[slot] = {};
    return 0;
}
__device__ bool aotx_shared_sample(unsigned slot, aotx_model_how *sample)
{
    const aotx_shared_receipt *r = aotx_shared_request(slot);
    if (!r) return false;
    *sample = r->sample; return true;
}
__device__ unsigned aotx_shared_limit(unsigned slot, unsigned fallback)
{ const aotx_shared_receipt *r = aotx_shared_request(slot); return r ? r->limit : fallback; }
__device__ void aotx_shared_start_result(unsigned slot, unsigned status)
{
    if (!aotx_shared_owns(slot)) return;
    aotx_shared_execution_slots[slot].status = status;
    aotx_shared_execution_slots[slot].model_opened = !status;
    aotx_shared_execution_slots[slot].stage = status ? AOTX_SHARED_END : AOTX_SHARED_DECODE;
}
