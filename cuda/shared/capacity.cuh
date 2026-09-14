/* Purpose: Bound each live shared lease by the available page pool.
 * Owns: No allocation; the page queue and sequence owners keep their state.
 * Launch shape: One ordered admission batch over all slots and pending page requests.
 * Lifetime: One live lease decision; recorded leases retain their exact replay. */
#ifndef AOTX_SHARED_CAPACITY_CUH
#define AOTX_SHARED_CAPACITY_CUH
#include "model/decode_state.cuh"
#include "model/load.cuh"
#include "cli/prompt.cuh"
#include "cognitive/recall.h"

static __device__ unsigned aotx_shared_page_bound(unsigned role, unsigned cap)
{
    unsigned language = 0;
    for (unsigned i = 0; i < AOTX_MODEL_ROLES; ++i)
        if (i == role || (aotx_model_is_language(i) && aotx_model_load.resident[i].active))
            language = max(language, aotx_kvl_pages(&aotx_model_space[i].shape, AOTX_SEQ_MAX_TOKENS));
    unsigned embedding = aotx_model_load.resident[AOTX_MODEL_EMBEDDING].active ?
        aotx_kvl_pages(&aotx_model_space[AOTX_MODEL_EMBEDDING].shape, AOTX_RECALL_TEXT) : 0;
    /* Intake can select another resident language role. Embedding has its own text bound. */
    return min((unsigned)AOTX_KV_PAGES, max(embedding, min(cap, language)));
}

static __device__ unsigned long long aotx_shared_page_claims[AOTX_SLOTS];
static __device__ unsigned aotx_shared_page_available(void)
{
    unsigned pending = aotx_kv.made - aotx_kv.served;
    if (pending > AOTX_KV_QUEUE_MAX) return 0;
    unsigned long long *claims = aotx_shared_page_claims;
    for (unsigned i = 0; i < AOTX_SLOTS; ++i) claims[i] = 0;
    for (unsigned i = 0; i < pending; ++i) {
        const aotx_kv_entry &q = aotx_kv.queue[(aotx_kv.served + i) & (AOTX_KV_QUEUE_MAX - 1u)];
        if (q.agent >= AOTX_SLOTS) return 0;
        claims[q.agent] += q.pages;
    }
    /* A queued release does not free physical pages until the page owner serves it. */
    unsigned long long used = aotx_kv.mapped_pages;
    for (unsigned i = 0; i < AOTX_SLOTS; ++i) {
        const aotx_seq &s = aotx_seqs.slot[i];
        unsigned need = 0;
        if ((s.state == AOTX_SEQ_STATE_PREFILL || s.state == AOTX_SEQ_STATE_DECODE) && s.role < AOTX_MODEL_ROLES)
            need = aotx_kvl_pages(&aotx_model_space[s.role].shape, s.prompt + s.limit);
        else if (aotx_say.slot[i].wanted)
            need = aotx_shared_page_bound(aotx_prompt_role(i), aotx_say.slot[i].page_limit ?
                aotx_say.slot[i].page_limit : AOTX_KV_PAGES_EACH);
        unsigned extra = need > aotx_kv.count[i] ? need - aotx_kv.count[i] : 0;
        used += max(claims[i], (unsigned long long)extra);
    }
    return used < AOTX_KV_PAGES ? (unsigned)(AOTX_KV_PAGES - used) : 0;
}
#endif
