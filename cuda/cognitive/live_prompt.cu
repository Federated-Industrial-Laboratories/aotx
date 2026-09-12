/* Purpose: Validate and render the current cognitive context for agent prompts.
 * Owns: No history; each binding holds its exact current request and selection.
 * Launch shape: One calling thread per conversation slot.
 * Lifetime: The bound request, including its tool continuations. */
#include "cognitive/live_validate.cuh"
#include "model/wrap.cuh"

__device__ uint32_t aotx_live_prompt_check(uint32_t slot) {
    if (!aotx_live_bound(slot)) return AOTX_COG_DENIED;
    aotx_live_binding *b = aotx_live_bindings + slot;
    if (!b->ordinal || !aotx_live.ready || aotx_live.fatal || b->choice.status) return AOTX_COG_MISSING;
    /* State updates require idle conversations. Any changed cut invalidates this context. */
    if (b->choice.cut != aotx_live_store.sequence) return AOTX_COG_STALE;
    return AOTX_COG_OK;
}
__device__ uint32_t aotx_live_context(uint32_t slot, unsigned char *out, uint32_t at) {
    const aotx_live_binding *b = aotx_live_bindings + slot;
    if (!b->context_bytes) return at;
    const aotx_wrap *wrap = aotx_wrap_active(aotx_prompt_role(slot));
    at = aotx_wrap_put(out, at, AOTX_SAY_BYTES, wrap, AOTX_WRAP_USER_HEAD);
    at = aotx_recall_run(out, at, AOTX_SAY_BYTES, b->choice.context, b->context_bytes);
    return aotx_wrap_put(out, at, AOTX_SAY_BYTES, wrap, AOTX_WRAP_USER_TAIL);
}
