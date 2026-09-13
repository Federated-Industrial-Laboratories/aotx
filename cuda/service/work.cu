/* Purpose: Lease free execution slots for bounded groups of service requests.
 * Owns: Request-to-slot bindings and frozen sequence parameters.
 * Launch shape: One ordered request batch; the normal model graph batches the work.
 * Lifetime: Admission through terminal sequence release. */
#include "service/internal.cuh"
#include "agent/agent_state.cuh"
#include "cli/prompt.cuh"
#include "cognitive/live.cuh"
#include "model/load.cuh"
#include "media/prompt.cuh"

__device__ bool aotx_service_owns(unsigned slot)
{ return aotx_service.enabled && slot < AOTX_SLOTS && aotx_service.slot[slot] != 0; }
__device__ const unsigned char *aotx_service_principal(unsigned slot)
{
    return aotx_service_owns(slot) ? aotx_service.jobs[aotx_service.slot[slot] - 1].principal : 0;
}
__device__ unsigned aotx_service_limit(unsigned slot, unsigned fallback)
{ return aotx_service_owns(slot) ? aotx_service.jobs[aotx_service.slot[slot] - 1].limit : fallback; }
__device__ bool aotx_service_sample(unsigned slot, aotx_model_how *sample)
{
    if (!aotx_service_owns(slot)) return false;
    *sample = aotx_service.jobs[aotx_service.slot[slot] - 1].sample; return true;
}
__device__ void aotx_service_start_result(unsigned slot, unsigned status)
{
    if (!aotx_service_owns(slot)) return;
    aotx_service_job &j = aotx_service.jobs[aotx_service.slot[slot] - 1];
    j.status = status;
    j.phase = status ? AOTX_SERVICE_FAILED : AOTX_SERVICE_RUNNING;
}
__global__ void aotx_service_work(void)
{
    if (!aotx_service.enabled || aotx_sched.held || aotx_seam.replaying) return;
    bool busy = false;
    for (unsigned i = 0; i < AOTX_SERVICE_REQUESTS; ++i) {
        aotx_service_job &j = aotx_service.jobs[i];
        if (!j.phase || j.phase >= AOTX_SERVICE_DONE) continue;
        const aotx_service_grant *g = aotx_service_granted(j.principal);
        bool denied = !g || g->revision != j.revision;
        bool expired = aotx_service.clock - j.opened > AOTX_SERVICE_REQUEST_SECONDS * 1000000000ull;
        if (denied || expired || j.cancel) {
            j.cancel = 1;
            if (!j.status) j.status = denied ? 403 : expired ? 504 : 409;
            if (j.slot == AOTX_SLOTS) j.phase = AOTX_SERVICE_CANCELLED;
            else {
                aotx_seq_stop(j.slot);
                if (j.phase == AOTX_SERVICE_PREPARE) {
                    aotx_say.slot[j.slot].wanted = 0; j.phase = AOTX_SERVICE_CANCELLED;
                }
            }
            j.changed = aotx_service.clock;
        }
    }
    for (unsigned slot = 0; slot < AOTX_SLOTS; ++slot) busy |= aotx_service_owns(slot);
    /* A complete group ends before the next starts, so checkpoints can observe quiet slots. */
    if (busy || aotx_model_load.pending_count) return;
    for (unsigned slot = 1; slot < AOTX_SLOTS; ++slot) {
        if (aotx_agents.agent[slot].state != AOTX_AGENT_STATE_FREE || aotx_live_bound(slot) ||
            aotx_say.slot[slot].wanted || aotx_say.slot[slot].live ||
            (aotx_seqs.slot[slot].state != AOTX_SEQ_STATE_FREE && aotx_seqs.slot[slot].state != AOTX_SEQ_STATE_DONE)) continue;
        unsigned index = AOTX_SERVICE_REQUESTS;
        unsigned long long first = ~0ull;
        for (unsigned i = 0; i < AOTX_SERVICE_REQUESTS; ++i) {
            const aotx_service_job &j = aotx_service.jobs[i];
            if (j.phase == AOTX_SERVICE_QUEUED && j.opened < first) { index = i; first = j.opened; }
        }
        if (index == AOTX_SERVICE_REQUESTS) break;
        aotx_service_job &j = aotx_service.jobs[index];
        if (!aotx_model_load.resident[j.role].active ||
            !aotx_service_equal(j.model_digest, aotx_model_load.resident[j.role].body.digest, 32)) {
            j.status = 503; j.phase = AOTX_SERVICE_FAILED; j.changed = aotx_service.clock; continue;
        }
        j.slot = slot; j.phase = AOTX_SERVICE_PREPARE; aotx_service.slot[slot] = index + 1;
        aotx_prompt_roles[slot] = j.role;
        aotx_say_slot &s = aotx_say.slot[slot]; s = {};
        s.length = j.length; s.page_limit = j.pages; s.wanted = 1;
        aotx_service_bytes(aotx_say.prompt[slot], j.text, j.length);
        aotx_media_prompts[slot] = {};
    }
}
__global__ void aotx_service_reply(void)
{
    unsigned slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (!aotx_service_owns(slot)) return;
    aotx_service_job &j = aotx_service.jobs[aotx_service.slot[slot] - 1];
    aotx_seq &seq = aotx_seqs.slot[slot];
    if (j.phase == AOTX_SERVICE_RUNNING) {
        aotx_say_slot &s = aotx_say.slot[slot];
        unsigned got = aotx_seq_take_text(slot, s.text, AOTX_SAY_TAKE);
        if (got > AOTX_SERVICE_OUTPUT_BYTES - j.output) {
            j.cancel = 1; j.status = 413; aotx_seq_stop(slot);
        } else {
            aotx_service_bytes(j.result + j.output, s.text, got); j.output += got;
        }
        j.prompt = seq.prompt; j.sampled = seq.sampled; j.changed = aotx_service.clock;
        if (seq.state == AOTX_SEQ_STATE_DONE || seq.state == AOTX_SEQ_STATE_FREE) {
            if (seq.state == AOTX_SEQ_STATE_FREE && !j.cancel) { j.cancel = 1; j.status = 503; }
            j.phase = j.cancel ? AOTX_SERVICE_CANCELLED : AOTX_SERVICE_DONE;
            j.finish = j.cancel ? 0 : (seq.last == seq.stop || aotx_wrap_end(seq.role, seq.last)) ? 1 : 2;
        }
    }
    if (j.phase >= AOTX_SERVICE_DONE) {
        aotx_say.slot[slot].wanted = aotx_say.slot[slot].live = 0;
        aotx_media_prompts[slot] = {};
        j.slot = AOTX_SLOTS; j.changed = aotx_service.clock; aotx_service.slot[slot] = 0;
    }
}
