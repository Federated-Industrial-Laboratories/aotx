/* Purpose: Schedule saved shared inputs in finite execution groups.
 * Owns: Temporary slots and pressure handling; journal records own persistent results.
 * Launch shape: One ordered request batch before inference and one output batch after it.
 * Lifetime: A saved admission through a saved or interrupted terminal result. */
#include "shared/bridge.cuh"
#include "shared/internal.cuh"
#include "shared/affect.cuh"
#include "shared/capacity.cuh"
#include "cognitive/checkpoint.cuh"
#include "agent/agent_state.cuh"
#include "cli/prompt.cuh"
#include "model/load.cuh"
#include "model/selection.cuh"
static __device__ unsigned aotx_shared_result_cursor;
static __device__ unsigned aotx_shared_group_requests[AOTX_SLOTS], aotx_shared_group_slots[AOTX_SLOTS];
static __device__ bool aotx_shared_model_current(const aotx_shared_receipt &r)
{
    return r.role < AOTX_MODEL_ROLES && aotx_model_load.resident[r.role].active &&
        aotx_service_equal(r.model_digest, aotx_model_load.resident[r.role].body.digest, 32);
}
static __device__ unsigned aotx_shared_execution_status(aotx_shared_receipt &r)
{
    const aotx_service_grant *g = aotx_service_granted(r.actor);
    if (!g || g->revision != r.revision || !aotx_shared_authorized(&r, 2) ||
        g->tokens < r.limit || g->pages < r.pages) return 403;
    if (r.cancel) return 409;
    if (!aotx_shared_model_current(r)) return 503;
    aotx_model_how sample = r.sample;
#ifdef AOTX_AFFECT
    sample.affect = aotx_shared_affect_managed(&r);
#endif
    if (aotx_control_select(r.command + 144, r.role, &sample) != 200) return 503;
    for (unsigned i = 0; i < 2; ++i) {
        r.sample.steer[i] = sample.steer[i]; r.sample.steer_strength[i] = sample.steer_strength[i];
    }
    return 0;
}
__global__ void aotx_shared_work(void)
{
    if (threadIdx.x || blockIdx.x || !aotx_shared.enabled || aotx_seam.replaying) return;
    aotx_shared.pending_bytes = aotx_checkpoint_pending_bytes();
    aotx_shared.disk_error = aotx_checkpoint.error;
    aotx_shared.pressure = aotx_checkpoint_pressure();
    if (aotx_checkpoint.acknowledged)
        aotx_shared_ack(aotx_checkpoint.runtime_durable, aotx_checkpoint.generation,
            (const unsigned char *)aotx_checkpoint.incarnation, aotx_checkpoint.ack_boot,
            (const unsigned char *)aotx_checkpoint.commit_digest);
    if (aotx_sched.held || aotx_shared.fatal) return;
    bool busy = false;
    for (unsigned slot = 1; slot < AOTX_SLOTS; ++slot) {
        aotx_shared_receipt *r = aotx_shared_request(slot);
        if (!r) continue;
        busy = true;
        aotx_shared_execution &x = aotx_shared_execution_slots[slot];
        unsigned status = aotx_shared_execution_status(*r);
        if (!status && aotx_sched.start_ns - x.opened > AOTX_SERVICE_REQUEST_SECONDS * 1000000000ull) status = 504;
        if (status && !x.status) x.status = status;
        if (x.stage == AOTX_SHARED_MEMORY) continue;
        if (x.status) {
            aotx_seq_stop(slot); aotx_say.slot[slot].wanted = 0;
            x.stage = AOTX_SHARED_END;
        } else if (x.stage == AOTX_SHARED_PROMPT) {
            x.status = aotx_shared_model_prompt(slot);
            x.stage = x.status ? AOTX_SHARED_END : AOTX_SHARED_TOKENIZE;
        }
    }
    if (busy || aotx_shared.kind || aotx_shared.received || aotx_shared.pressure ||
        aotx_live.phase != AOTX_LIVE_IDLE || aotx_live.received || !aotx_checkpoint_quiet() ||
        aotx_model_load.pending_count) return;
    unsigned *requests = aotx_shared_group_requests, *slots = aotx_shared_group_slots, count = 0;
    unsigned pages = aotx_shared_page_available();
    for (unsigned i = 0; i < aotx_shared.receipt_capacity; ++i) {
        aotx_shared_receipt &r = aotx_shared.receipts[i];
        if (r.phase == AOTX_SHARED_INTERRUPTED && !r.terminal_source) {
            aotx_shared_complete(i, 598, r.prompt, r.sampled, 0); return;
        }
        if (r.phase != AOTX_SHARED_QUEUED || !r.saved_admission) continue;
        unsigned status = aotx_shared_execution_status(r);
        if (status) { aotx_shared_complete(i, status, 0, 0, 0); return; }
#ifdef AOTX_AFFECT
        bool conflict = false;
        for (unsigned j = 0; j < count; ++j)
            conflict |= aotx_shared_affect_conflict(&r, &aotx_shared.receipts[requests[j]]);
        if (conflict) continue;
#endif
        unsigned need = aotx_shared_page_bound(r.role, r.pages);
        if (need > pages) continue;
        unsigned slot = count ? slots[count - 1] + 1 : 1;
        for (; slot < AOTX_SLOTS; ++slot)
            if (aotx_agents.agent[slot].state == AOTX_AGENT_STATE_FREE && !aotx_service_owns(slot) &&
                !aotx_shared_owns(slot) && !aotx_live_bound(slot) && !aotx_say.slot[slot].wanted &&
                !aotx_say.slot[slot].live && (aotx_seqs.slot[slot].state == AOTX_SEQ_STATE_FREE ||
                aotx_seqs.slot[slot].state == AOTX_SEQ_STATE_DONE)) break;
        if (slot == AOTX_SLOTS) break;
        requests[count] = i; slots[count++] = slot; pages -= need;
    }
    if (count) aotx_shared_lease(requests, slots, count);
}
__global__ void aotx_shared_results(void)
{
    if (threadIdx.x || blockIdx.x || !aotx_shared.enabled || aotx_shared.fatal ||
        aotx_sched.held || aotx_seam.replaying || aotx_shared.kind || aotx_shared.received) return;
    for (unsigned pass = 0; pass < AOTX_SLOTS; ++pass) {
        unsigned slot = (aotx_shared_result_cursor + pass) % AOTX_SLOTS;
        aotx_shared_receipt *r = aotx_shared_request(slot);
        if (!r) continue;
        aotx_shared_execution &x = aotx_shared_execution_slots[slot];
        if (x.stage == AOTX_SHARED_MEMORY && aotx_live.phase == AOTX_LIVE_IDLE) {
            x.stage = AOTX_SHARED_END; x.status = 503;
        }
        unsigned index = aotx_shared.slot[slot] - 1;
        aotx_seq &seq = aotx_seqs.slot[slot];
        if (x.stage == AOTX_SHARED_DECODE) {
            unsigned error = aotx_shared_execution_status(*r);
            if (error) { x.status = error; aotx_seq_stop(slot); x.stage = AOTX_SHARED_END; }
            else {
                aotx_say_slot &s = aotx_say.slot[slot];
                unsigned got = aotx_seq_take_text(slot, s.text, AOTX_SAY_TAKE);
                if (got) {
                    if (got > AOTX_SHARED_RESULT_BYTES - r->output) {
                        x.status = 413; aotx_seq_stop(slot); x.stage = AOTX_SHARED_END;
                    } else if (aotx_shared_output(index, s.text, got)) {
                        aotx_shared_result_cursor = (slot + 1) % AOTX_SLOTS; return;
                    } else { x.status = 503; aotx_seq_stop(slot); x.stage = AOTX_SHARED_END; }
                } else if (seq.state == AOTX_SEQ_STATE_DONE || seq.state == AOTX_SEQ_STATE_FREE) {
                    x.stage = AOTX_SHARED_END;
                    if (seq.state == AOTX_SEQ_STATE_FREE) x.status = 503;
                }
            }
        }
        if (x.stage == AOTX_SHARED_END) {
            if (seq.state == AOTX_SEQ_STATE_PREFILL || seq.state == AOTX_SEQ_STATE_DECODE) {
                aotx_seq_stop(slot); continue;
            }
            unsigned finish = x.status ? 0 : (seq.last == seq.stop || aotx_wrap_end(seq.role, seq.last)) ? 1 : 2;
            unsigned prompt = x.model_opened ? seq.prompt : 0;
            unsigned sampled = x.model_opened ? seq.sampled : 0;
            if (aotx_shared_complete(index, x.status ? x.status : 200, prompt, sampled, finish)) {
                aotx_shared_result_cursor = (slot + 1) % AOTX_SLOTS; return;
            }
        }
    }
}
