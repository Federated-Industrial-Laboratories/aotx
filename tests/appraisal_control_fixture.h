/* Purpose: Exercise staged appraisal controls and model lease boundaries.
 * Owns: Test command bytes, control guards and observable lease summaries.
 * Launch shape: Real admission kernels with distinct source batches at one and 64 rows.
 * Lifetime: One case; no model weights or forward kernels are used. */
#ifndef AOTX_APPRAISAL_CONTROL_FIXTURE_H
#define AOTX_APPRAISAL_CONTROL_FIXTURE_H
#include "appraisal_work_fixture.h"
#include "appraisal/schema.cuh"
#include "cognitive/checkpoint.cuh"
#include "cli/cli.cuh"
#include "shared/state.cuh"
#include "media/runtime.cuh"

struct aotx_control_view {
    uint64_t revision, sequence, root, passes, calls;
    unsigned pending, enabled, flags, current, taken[2], status, phase;
    unsigned owned[AOTX_SLOTS], row_state[AOTX_SLOTS], row_status[AOTX_SLOTS], ticks[AOTX_SLOTS];
    unsigned sequence_state[AOTX_SLOTS], sequence_flags[AOTX_SLOTS], pages[AOTX_SLOTS], wanted[AOTX_SLOTS];
    unsigned turn[AOTX_SLOTS];
    unsigned live, made, served, released[AOTX_SLOTS];
};
__global__ void aotx_control_observe(aotx_control_view *out, unsigned take) {
    unsigned i = threadIdx.x;
    if (!i) {
        *out = {}; out->pending = aotx_appraisal_pending(); out->enabled = aotx_appraisal_enabled();
        out->revision = aotx_appraisal_revision(); out->sequence = aotx_live_store.sequence;
        out->root = aotx_live_store.root_sequence; out->passes = aotx_maintenance.passes;
        out->flags = aotx_appraisal.write_flags; out->current = aotx_appraisal.config; out->calls = aotx_appraisal.calls;
        out->status = aotx_live.status; out->phase = aotx_live.phase;
        if (take) { out->taken[0] = aotx_policy_appraisal(); out->taken[1] = aotx_policy_appraisal(); }
        out->live = aotx_seqs.live; out->made = aotx_kv.made; out->served = aotx_kv.served;
        for (unsigned j = aotx_kv.served; j < aotx_kv.made && j - aotx_kv.served < AOTX_KV_QUEUE_MAX; ++j) {
            const aotx_kv_entry *q = aotx_kv.queue + (j & (AOTX_KV_QUEUE_MAX - 1));
            if (q->agent < AOTX_SLOTS && !q->pages) ++out->released[q->agent];
        }
    }
    __syncthreads();
    if (i < AOTX_SLOTS) {
        out->owned[i] = aotx_intake.row[i]; out->row_state[i] = aotx_intake.rows[i].state;
        out->row_status[i] = aotx_intake.rows[i].status; out->ticks[i] = aotx_intake.rows[i].ticks;
        out->sequence_state[i] = aotx_seqs.slot[i].state; out->sequence_flags[i] = aotx_seqs.slot[i].flags;
        out->pages[i] = aotx_kv.count[i]; out->wanted[i] = aotx_say.slot[i].wanted;
        out->turn[i] = aotx_agents.agent[i].turn;
    }
}
static aotx_control_view aotx_control_read(unsigned take = 0) {
    aotx_control_view *p, out; AOTX_CUDA(cudaMalloc(&p, sizeof(out)));
    aotx_control_observe<<<1,AOTX_SLOTS>>>(p, take);
    AOTX_CUDA(cudaMemcpy(&out, p, sizeof(out), cudaMemcpyDeviceToHost)); cudaFree(p); return out;
}
__global__ void aotx_control_command(const unsigned char *text, unsigned bytes) {
    if (!threadIdx.x) aotx_cli_line(text, bytes, aotx_time_tick);
}
static void aotx_control_line(const std::string &line) {
    unsigned char *p; AOTX_CUDA(cudaMalloc(&p, line.size()));
    AOTX_CUDA(cudaMemcpy(p, line.data(), line.size(), cudaMemcpyHostToDevice));
    aotx_control_command<<<1,1>>>(p, (unsigned)line.size()); AOTX_CUDA(cudaDeviceSynchronize()); cudaFree(p);
}
__global__ void aotx_control_metadata(unsigned n, unsigned leased) {
    unsigned i = threadIdx.x;
    if (!i) {
        aotx_decode.ready = 1; aotx_decode.role = AOTX_MODEL_LANGUAGE;
        aotx_model_load.resident[AOTX_MODEL_LANGUAGE].active = 1;
        for (unsigned j = 0; j < 32; ++j) aotx_model_load.resident[AOTX_MODEL_LANGUAGE].body.digest[j] = 77 + j;
        if (leased) aotx_seqs.live = n;
    }
    if (leased && i < n) {
        aotx_intake.rows[i].state = 2; aotx_intake.rows[i].status = 0;
        aotx_say.slot[i].wanted = 0; aotx_decode.rows[i] = aotx_decode.first[i] = 0;
        aotx_seqs.slot[i].state = AOTX_SEQ_STATE_DECODE; aotx_seqs.slot[i].role = AOTX_MODEL_LANGUAGE;
        aotx_seqs.slot[i].prompt = 1; aotx_seqs.slot[i].sampled = 0; aotx_seqs.slot[i].flags = 0;
        aotx_kv.count[i] = 1 + i;
    }
}
__global__ void aotx_control_guard(unsigned guard, unsigned n) {
    if (threadIdx.x) return;
    aotx_sched.held = guard == 1; aotx_policy.paused = guard == 4; aotx_policy.stopped = guard == 5;
    aotx_agents.agent[n - 1].state = guard == 2 ? AOTX_AGENT_STATE_RUN : AOTX_AGENT_STATE_IDLE;
    aotx_seam.apply.available = aotx_seam.apply.this_tick + (guard == 3);
    aotx_policy.pending = guard == 6;
}
__global__ void aotx_control_policy(unsigned abi, unsigned change) {
    if (threadIdx.x) return;
    aotx_policy.enabled = 1; aotx_policy.config.mode = AOTX_POLICY_SUPPLIED; aotx_policy.config.abi = abi;
    aotx_policy.appraise = 1; aotx_policy.source = aotx_live_store.sequence; aotx_policy.root = aotx_live_store.root_sequence;
    aotx_policy.observed_objects = aotx_live_store.count; aotx_policy.observed_bytes = aotx_live_store.bytes;
    aotx_policy.work_revision = aotx_appraisal_revision();
    if (change == 1) ++aotx_policy.work_revision;
    if (change == 2) ++aotx_policy.source;
    if (change == 3) ++aotx_policy.root;
    if (change == 4) ++aotx_policy.observed_objects;
    if (change == 5) ++aotx_policy.observed_bytes;
}
__global__ void aotx_control_mutate(unsigned mode, unsigned n, uint64_t value = 0) {
    if (threadIdx.x) return;
    if (mode == 1) ++aotx_maintenance.passes;
    if (mode == 2) aotx_live_store.bytes = (uint32_t)value;
    if (mode == 3) aotx_kv.made = AOTX_KV_QUEUE_MAX;
    if (mode == 4) aotx_kv.served = aotx_kv.made;
    if (mode >= 5 && mode <= 8) {
        unsigned row = mode == 5 ? 1 + 3 * n + n - 1 : 1 + 3 * (n - 1);
        unsigned char *r = aotx_live_store.objects[row];
        if (mode == 5 || mode == 6) aotx_cog_put(r + AOTX_CO_EVIDENCE, 3, 4);
        if (mode == 7) r[AOTX_CO_SUBJECT] ^= 1;
        if (mode == 8) aotx_cog_put(r + AOTX_CO_EXPIRY, aotx_live_store.sequence, 8);
        aotx_appraisal.observed = UINT64_MAX;
    }
    if (mode == 9) {
        aotx_appraisal_refresh();
        aotx_cog_put(aotx_live_store.objects[aotx_appraisal.config] + AOTX_CO_FLAGS, AOTX_COG_TOMBSTONE, 4);
        aotx_appraisal.observed = UINT64_MAX;
    }
}
static void aotx_control_reset(void) {
    AOTX_LIVE_CLEAR(aotx_checkpoint); AOTX_LIVE_CLEAR(aotx_shared); AOTX_LIVE_CLEAR(aotx_service);
    AOTX_LIVE_CLEAR(aotx_media); AOTX_LIVE_CLEAR(aotx_task_used);
#ifdef AOTX_AFFECT
    AOTX_LIVE_CLEAR(aotx_quality_state);
#endif
}
static void aotx_control_finish(aotx_appraisal_device &d, const aotx_bytes &before) {
    uint64_t journal = d.seam().dev.tail;
    aotx_live_decide<<<1,64>>>(); AOTX_CUDA(cudaDeviceSynchronize());
    aotx_check(d.state().phase == AOTX_LIVE_WRITE && !d.state().written && d.state().choice_bytes &&
        aotx_retain_store() == before && d.seam().dev.tail == journal,
        "staged results leave every memory and journal byte unchanged before commit");
    d.process({});
    aotx_check(!d.state().fatal && d.state().phase == AOTX_LIVE_IDLE && !d.appraisal().active,
        "the complete recorded result releases the work batch");
}
#endif
