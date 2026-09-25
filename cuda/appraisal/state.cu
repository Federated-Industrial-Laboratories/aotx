/* Purpose: Observe current appraisal configuration and retained pending work.
 * Owns: Disposable lookup caches; authoritative state stays in memory objects.
 * Launch shape: Serial metadata calls over the configured store at operation boundaries.
 * Lifetime: One store cut; compaction and new writes invalidate cached indices. */
#include "appraisal/appraisal.cuh"
#include "appraisal/schema.cuh"
#include "cognitive/lookup.cuh"
#include "cognitive/checkpoint.cuh"
#include "cognitive/maintenance.cuh"
#include "cli/prompt.cuh"
#include "shared/state.cuh"
#include "service/service.cuh"
#include "policy/state.cuh"
#include "sched/sched.cuh"
#include "media/runtime.cuh"
#include "model/load.cuh"

__device__ aotx_appraisal_state aotx_appraisal;
/* Tick metadata calls are serial. Dependency marks use device storage sized with the store. */
static __device__ uint32_t aotx_appraisal_need[AOTX_COG_WORDS], aotx_appraisal_done[AOTX_COG_WORDS];
__device__ uint32_t aotx_appraisal_resolve(const aotx_cognitive_query *query) {
    return aotx_cog_resolve_scratch(&aotx_live_store, query, true, aotx_live_store.sequence,
        aotx_appraisal_need, aotx_appraisal_done).status;
}
__device__ void aotx_appraisal_refresh(void) {
    if (aotx_appraisal.active || (aotx_appraisal.observed == aotx_live_store.sequence &&
        aotx_appraisal.observed_root == aotx_live_store.root_sequence &&
        aotx_appraisal.observed_count == aotx_live_store.count &&
        aotx_appraisal.observed_bytes == aotx_live_store.bytes &&
        aotx_appraisal.observed_maintenance == aotx_maintenance.passes &&
        aotx_appraisal.observed_replay == (uint32_t)aotx_seam.replaying)) return;
    aotx_appraisal.observed = aotx_live_store.sequence;
    aotx_appraisal.observed_root = aotx_live_store.root_sequence;
    aotx_appraisal.observed_count = aotx_live_store.count; aotx_appraisal.observed_bytes = aotx_live_store.bytes;
    aotx_appraisal.observed_maintenance = aotx_maintenance.passes;
    aotx_appraisal.observed_replay = (uint32_t)aotx_seam.replaying;
    aotx_appraisal.config = UINT32_MAX; aotx_appraisal.pending = aotx_appraisal.background = aotx_appraisal.write_flags = 0;
    aotx_appraisal.revision = 0;
    for (uint32_t i = 0; i < aotx_live_store.count; ++i) {
        const unsigned char *r = aotx_live_store.objects[i];
        if (aotx_cog_cold(r)) continue;
        if (aotx_cog_u16(r + AOTX_CO_KIND) != AOTX_COG_POLICY) continue;
        const unsigned char *p = aotx_live_store.payload + aotx_cog_u64(r + AOTX_CO_OFFSET);
        if (aotx_appraisal_magic(p, aotx_cog_u64(r + AOTX_CO_BYTES), "AOTXAPC1") &&
            (aotx_appraisal.config == UINT32_MAX || aotx_cog_u64(r + AOTX_CO_UPDATED) > aotx_appraisal.revision)) {
            aotx_appraisal.config = i; aotx_appraisal.revision = aotx_cog_u64(r + AOTX_CO_UPDATED);
        }
    }
    if (aotx_appraisal.config == UINT32_MAX) return;
    const unsigned char *c = aotx_live_store.objects[aotx_appraisal.config];
    aotx_cognitive_query access = {};
    for (uint32_t j = 0; j < 16; ++j) {
        access.id[j] = c[AOTX_CO_ID + j]; access.principal[j] = c[AOTX_CO_OWNER + j];
    }
    access.version = aotx_cog_u64(c + AOTX_CO_VERSION);
    if (aotx_cog_latest(&aotx_live_store, access.id) != (int)aotx_appraisal.config ||
        aotx_appraisal_resolve(&access)) {
        aotx_appraisal.config = UINT32_MAX; return;
    }
    const unsigned char *config = aotx_live_store.payload + aotx_cog_u64(c + AOTX_CO_OFFSET);
    uint32_t contract = aotx_appraisal_contract(config + 40);
    if (!contract) {
        aotx_appraisal.last_status = AOTX_COG_LAYOUT; return;
    }
    aotx_appraisal.write_flags = aotx_cog_u32(config + 12);
    if (contract == 1 && !aotx_seam.replaying) aotx_appraisal.write_flags &= AOTX_APPRAISAL_RECALL;
    aotx_appraisal.background = !!(aotx_appraisal.write_flags & AOTX_APPRAISAL_BACKGROUND);
    aotx_appraisal.pages = aotx_cog_u32(config + 16); aotx_appraisal.tokens = aotx_cog_u32(config + 20);
    aotx_appraisal.ticks = aotx_cog_u32(config + 24);
    if (!(aotx_appraisal.write_flags & AOTX_APPRAISAL_WRITE)) { aotx_appraisal.background = 0; return; }
    for (uint32_t i = 0; i < aotx_live_store.count; ++i) {
        const unsigned char *r = aotx_live_store.objects[i];
        if (aotx_cog_cold(r)) continue;
        const unsigned char *p = aotx_live_store.payload + aotx_cog_u64(r + AOTX_CO_OFFSET);
        if (!aotx_appraisal_magic(p, aotx_cog_u64(r + AOTX_CO_BYTES), "AOTXAPQ1") ||
            !aotx_cog_equal(p + 64, config + 40, 32) ||
            aotx_cog_u32(p + 12) != AOTX_APPRAISAL_PENDING ||
            aotx_cog_latest(&aotx_live_store, r + AOTX_CO_ID) != (int)i) continue;
        aotx_cognitive_query q = {};
        for (uint32_t j = 0; j < 16; ++j) {
            q.id[j] = r[AOTX_CO_ID + j]; q.principal[j] = r[AOTX_CO_OWNER + j]; q.room[j] = r[AOTX_CO_ROOM + j];
        }
        q.version = aotx_cog_u64(r + AOTX_CO_VERSION);
        if (aotx_appraisal_resolve(&q)) continue;
        ++aotx_appraisal.pending;
        uint64_t revision = aotx_cog_u64(r + AOTX_CO_UPDATED);
        if (revision > aotx_appraisal.revision) aotx_appraisal.revision = revision;
    }
    if (aotx_maintenance.passes > UINT64_MAX - aotx_appraisal.revision) {
        aotx_appraisal.pending = aotx_appraisal.background = 0; aotx_appraisal.last_status = AOTX_COG_CAPACITY;
    } else aotx_appraisal.revision += aotx_maintenance.passes;
}
__device__ uint32_t aotx_appraisal_pending(void) {
    aotx_appraisal_refresh();
    bool blocked = aotx_appraisal.blocked_sequence == aotx_live_store.sequence &&
        aotx_appraisal.blocked_root == aotx_live_store.root_sequence &&
        aotx_appraisal.blocked_count == aotx_live_store.count && aotx_appraisal.blocked_bytes == aotx_live_store.bytes;
    return aotx_appraisal.background && !blocked ? aotx_appraisal.pending : 0;
}
__device__ uint64_t aotx_appraisal_revision(void) { aotx_appraisal_refresh(); return aotx_appraisal.revision; }
__device__ bool aotx_appraisal_enabled(void) { aotx_appraisal_refresh(); return aotx_appraisal.background != 0; }
__device__ bool aotx_appraisal_interrupted(void) {
    if (!aotx_appraisal.active) return false;
    if (aotx_appraisal.control_pending || aotx_policy.paused || aotx_policy.stopped || aotx_sched.held) return true;
    return aotx_appraisal_foreground();
}
__device__ bool aotx_appraisal_foreground(void) {
    if (aotx_seam.apply.available > aotx_seam.apply.this_tick || !aotx_shared_quiet()) return true;
    if (!aotx_media_quiet() || aotx_model_load.pending_count) return true;
    if (aotx_service.enabled && aotx_service.jobs)
        for (uint32_t j = 0; j < AOTX_SERVICE_REQUESTS; ++j)
            if (aotx_service.jobs[j].phase && aotx_service.jobs[j].phase < AOTX_SERVICE_DONE) return true;
    for (uint32_t slot = 0; slot < AOTX_SLOTS; ++slot) {
        if (aotx_agents.agent[slot].state != AOTX_AGENT_STATE_FREE &&
            (aotx_agents.agent[slot].state != AOTX_AGENT_STATE_IDLE ||
            aotx_agents.agent[slot].task != ~0u || aotx_agent_gear[slot].has_message)) return true;
        if (aotx_intake_owns(slot)) continue;
        if (aotx_say.slot[slot].wanted || aotx_say.slot[slot].live || aotx_service_owns(slot)) return true;
    }
    for (uint32_t i = 0; i < AOTX_TASK_SLOTS; ++i)
        if (aotx_task_used[i] && aotx_agents.task[i].state == AOTX_TASK_PENDING) return true;
    return false;
}
