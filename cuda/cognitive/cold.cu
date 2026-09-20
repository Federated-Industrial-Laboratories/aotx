/* Purpose: Admit scoped residency requests and publish complete store changes.
 * Owns: Resident capacity, active references and atomic device installation.
 * Launch shape: One 64-thread block operates on the complete selected batch.
 * Lifetime: One recorded residency operation. */
#include "cognitive/cold.cuh"
#include "cognitive/cold_plan.cuh"
#include "shared/state.cuh"
#include "cli/cli.cuh"

__device__ aotx_cold_state aotx_cold;

static __device__ void binding_roots(const aotx_live_binding *b) {
    if (!b->active) return;
    for (uint32_t i = 0; i < b->focus_count; ++i)
        aotx_cold_mark_ref(&aotx_live_store, aotx_cold.needed, b->focus[i], aotx_cog_u64(b->focus[i] + 16));
    for (uint32_t group = 0; group < 2; ++group)
        for (uint32_t i = 0; i < aotx_cog_u32(b->query + 140 + group * 4); ++i) {
            const unsigned char *p = b->query + 4256 + group * 192 + i * 24;
            aotx_cold_mark_ref(&aotx_live_store, aotx_cold.needed, p, aotx_cog_u64(p + 16));
        }
    for (uint32_t i = 0; i < b->choice.count; ++i) {
        const unsigned char *p = b->choice.selection + 16 + i * 32;
        aotx_cold_mark_ref(&aotx_live_store, aotx_cold.needed, p, aotx_cog_u64(p + 16));
    }
}
__device__ void aotx_cold_begin(void) {
    if (!threadIdx.x) {
        aotx_cold.active = 1; aotx_cold.recovery = 0; aotx_cold.ticks = aotx_cold.copied = 0;
        aotx_cold.mode = aotx_live.total >= AOTX_COLD_HEADER ? aotx_cog_u32(aotx_live.input + 12) : 0;
        aotx_cold.status = aotx_cold_select(&aotx_live_store, aotx_live.input, aotx_live.total,
            aotx_cold.selected, aotx_cold.needed, aotx_cold.done);
        if (!aotx_cold.status && (!aotx_live.ready || !aotx_checkpoint_quiet())) aotx_cold.status = AOTX_COG_DENIED;
        if (!aotx_cold.status && aotx_live_admission_pressure()) aotx_cold.status = AOTX_COG_CAPACITY;
        if (!aotx_cold.status && !aotx_seam.replaying && !aotx_checkpoint.ring) aotx_cold.status = AOTX_COG_UNAVAILABLE;
        if (!aotx_cold.status && aotx_cold.mode == AOTX_COLD_OFFLOAD) {
            aotx_cold.status = aotx_cold_closure(&aotx_live_store, aotx_cold.selected, aotx_cold.needed);
            for (uint32_t i = 0; i < AOTX_SLOTS; ++i) binding_roots(aotx_live_bindings + i);
            if (aotx_shared.enabled) for (uint32_t i = 0; i < aotx_shared.conversation_capacity; ++i)
                if (aotx_shared.conversations[i].active) binding_roots(&aotx_shared.conversations[i].binding);
            for (uint32_t i = 0; i < aotx_live_store.count; ++i)
                if (aotx_cold.selected[i] && aotx_cold.needed[i]) aotx_cold.status = AOTX_COG_REFERENCE;
        }
        aotx_cold.count = aotx_cold.bytes = 0;
        if (!aotx_cold.status && (aotx_cold.mode == AOTX_COLD_FETCH || aotx_cold.mode == AOTX_COLD_GPU))
            for (uint32_t i = 0; i < aotx_live_store.count; ++i) if (aotx_cold.selected[i]) {
                ++aotx_cold.count; aotx_cold.bytes += (uint32_t)aotx_cog_u64(aotx_live_store.objects[i] + AOTX_CO_BYTES);
            }
        for (uint32_t i = 0; i < 16; ++i) aotx_live.query_id[i] = aotx_live.transfer_id[i];
        aotx_live.auto_mode = aotx_live.text_mode = aotx_live.intake_mode = 0;
        aotx_live.count = 0; aotx_live.received = 0;
        aotx_live.phase = aotx_seam.replaying ? AOTX_LIVE_WAIT : AOTX_COLD_BUILD;
        if (!aotx_seam.replaying && !aotx_cold.status && aotx_cold.count) {
            aotx_cold_transport *r = (aotx_cold_transport *)((unsigned char *)aotx_checkpoint.ring + AOTX_CP_COLD_OFFSET);
            uint64_t prior = aotx_seam_acquire_sys(&r->request);
            if (prior == UINT64_MAX || prior != aotx_seam_acquire_sys(&r->response)) aotx_cold.status = AOTX_COG_UNAVAILABLE;
            else {
                r->boot = aotx_seam.boot_id; r->generation = aotx_checkpoint.generation;
                for (uint32_t i = 0; i < 2; ++i) r->incarnation[i] = aotx_checkpoint.incarnation[i];
                r->count = aotx_cold.count; r->reserved = 0;
                uint32_t at = 0;
                for (uint32_t i = 0; i < aotx_live_store.count; ++i) if (aotx_cold.selected[i]) {
                    for (uint32_t j = 0; j < AOTX_COG_OBJECT; ++j) r->rows[at][j] = aotx_live_store.objects[i][j];
                    ++at;
                }
                aotx_cold.serial = prior + 1;
                aotx_seam_release_sys(&r->request, aotx_cold.serial);
                aotx_live.phase = AOTX_COLD_WAIT;
            }
        }
    }
}
__device__ void aotx_cold_candidate(const unsigned char *payload) {
    if (aotx_cold.status) return;
    for (uint32_t i = threadIdx.x; i < sizeof(aotx_live_store); i += blockDim.x)
        ((unsigned char *)&aotx_live_candidate)[i] = ((const unsigned char *)&aotx_live_store)[i];
    __syncthreads();
    uint32_t at = 0, read = 0;
    for (uint32_t i = 0; i < aotx_live_store.count; ++i) {
        const unsigned char *r = aotx_live_store.objects[i];
        unsigned char *out = aotx_live_candidate.objects[i];
        bool cold = aotx_cog_cold(r), selected = aotx_cold.selected[i] != 0;
        bool offload = selected && aotx_cold.mode == AOTX_COLD_OFFLOAD;
        bool fetched = selected && cold && !offload;
        uint32_t bytes = (uint32_t)aotx_cog_u64(r + AOTX_CO_BYTES);
        if ((!cold && !offload) || fetched) {
            const unsigned char *source = fetched ? payload + read : aotx_live_store.payload + aotx_cog_u64(r + AOTX_CO_OFFSET);
            for (uint32_t j = threadIdx.x; j < bytes; j += blockDim.x) aotx_live_candidate.payload[at + j] = source[j];
            if (!threadIdx.x) {
                aotx_cog_put(out + AOTX_CO_OFFSET, bytes ? at : 0, 8);
                aotx_cog_put(out + AOTX_CO_FLAGS, aotx_cog_u32(r + AOTX_CO_FLAGS) & ~AOTX_COG_COLD, 4);
            }
            at += bytes; if (fetched) read += bytes;
        } else if (!threadIdx.x) {
            aotx_cog_put(out + AOTX_CO_OFFSET, 0, 8);
            aotx_cog_put(out + AOTX_CO_FLAGS, aotx_cog_u32(r + AOTX_CO_FLAGS) | AOTX_COG_COLD, 4);
        }
    }
    for (uint32_t i = at + threadIdx.x; i < AOTX_COG_PAYLOAD; i += blockDim.x) aotx_live_candidate.payload[i] = 0;
    if (!threadIdx.x) { aotx_live_candidate.bytes = at; aotx_live_candidate.tiered = aotx_cold.mode != AOTX_COLD_GPU; }
    __syncthreads();
    for (uint32_t i = threadIdx.x; i < aotx_live_candidate.count; i += blockDim.x) {
        uint32_t status = aotx_cog_validate(&aotx_live_candidate, i);
        if (status) atomicCAS(&aotx_cold.status, 0u, status);
    }
    __syncthreads();
}
__device__ void aotx_cold_publish(void) {
    if (!aotx_live.status) for (uint32_t i = threadIdx.x; i < sizeof(aotx_live_store); i += blockDim.x)
        ((unsigned char *)&aotx_live_store)[i] = ((const unsigned char *)&aotx_live_candidate)[i];
    __syncthreads();
    if (!threadIdx.x) {
        if (aotx_live.status) ++aotx_live.refused; else ++aotx_live.accepted;
        uint32_t changed = 0;
        if (!aotx_live.status) for (uint32_t i = 0; i < aotx_live_store.count; ++i) changed += !!aotx_cold.selected[i];
        aotx_live_note(AOTX_COLD_CONTROL, aotx_live.status, changed);
        aotx_live.phase = AOTX_LIVE_IDLE; aotx_cold.active = 0;
    }
}
__device__ void aotx_cold_cancel(void) {
    if (aotx_cold.active && aotx_live.phase == AOTX_COLD_WAIT && !aotx_seam.replaying) {
        aotx_cold.status = AOTX_COG_UNAVAILABLE; aotx_live.phase = AOTX_COLD_BUILD;
    }
}
__device__ void aotx_cold_status(aotx_cli_out *out) {
    uint64_t bytes = 0; uint32_t count = 0;
    for (uint32_t i = 0; i < aotx_live_store.count; ++i) if (aotx_cog_cold(aotx_live_store.objects[i])) {
        ++count; bytes += aotx_cog_u64(aotx_live_store.objects[i] + AOTX_CO_BYTES);
    }
    aotx_cli_say(out, "memory tier: "); aotx_cli_num(out, aotx_live_store.tiered);
    aotx_cli_say(out, " cold objects "); aotx_cli_num(out, count);
    aotx_cli_say(out, " cold bytes "); aotx_cli_num(out, bytes);
    aotx_cli_say(out, " read pending "); aotx_cli_num(out, aotx_cold.active);
}
