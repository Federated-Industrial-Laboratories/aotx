/* Purpose: Repack retained objects and payloads without a CPU state loop.
 * Owns: Maintenance plans and atomic publication into the live store.
 * Launch shape: Parallel root and offset scans, one closure block, parallel copies and publication.
 * Lifetime: One recorded maintenance operation at a quiescent graph boundary. */
#include "cognitive/maintenance_roots.cuh"
#include "sched/sched.cuh"

__device__ aotx_memory_maintenance aotx_maintenance;

__global__ void aotx_memory_seed(void) {
    if (aotx_sched.held || aotx_live.phase != AOTX_LIVE_MAINTENANCE) return;
    const unsigned char *p = aotx_live.input;
    uint64_t floor = aotx_live_store.sequence > aotx_cog_u32(p + 16) ?
        aotx_live_store.sequence - aotx_cog_u32(p + 16) : 0;
    if (floor < aotx_live_store.retry_floor) floor = aotx_live_store.retry_floor;
    if (!blockIdx.x && !threadIdx.x) { aotx_maintenance.status = 0; aotx_maintenance.running = 0; }
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < aotx_live_store.count; i += blockDim.x * gridDim.x) {
        aotx_maintenance.latest[i] = (uint32_t)aotx_cog_latest(&aotx_live_store,
            aotx_live_store.objects[i] + AOTX_CO_ID);
        aotx_maintenance.marks[i] = aotx_memory_root(i, floor, aotx_cog_u32(p + 20)) ? 1u : 0u;
    }
}
__global__ void aotx_memory_plan(void) {
    if (aotx_sched.held || aotx_live.phase != AOTX_LIVE_MAINTENANCE) return;
    __shared__ uint32_t progress;
    const unsigned char *p = aotx_live.input;
    uint64_t floor = aotx_live_store.sequence > aotx_cog_u32(p + 16) ?
        aotx_live_store.sequence - aotx_cog_u32(p + 16) : 0;
    if (floor < aotx_live_store.retry_floor) floor = aotx_live_store.retry_floor;
    for (uint32_t i = threadIdx.x; i < AOTX_SLOTS; i += blockDim.x) aotx_memory_binding_roots(i);
    __syncthreads();
    for (uint32_t pass = 0; pass < aotx_live_store.count; ++pass) {
        if (!threadIdx.x) progress = 0;
        __syncthreads();
        for (uint32_t i = threadIdx.x; i < aotx_live_store.count; i += blockDim.x) {
            if (atomicCAS(aotx_maintenance.marks + i, 1u, 3u) != 1u) continue;
            atomicExch(&progress, 1u); aotx_memory_dependencies(i);
        }
        __syncthreads();
        if (!progress) break;
    }
    if (!threadIdx.x) {
        uint32_t count = 0, bytes = 0, covered = 0;
        for (uint32_t i = 0; i < aotx_live_store.count; ++i) if (aotx_maintenance.marks[i]) {
            ++count; bytes += (uint32_t)aotx_cog_u64(aotx_live_store.objects[i] + AOTX_CO_BYTES);
            if (aotx_cog_u64(aotx_live_store.objects[i] + AOTX_CO_UPDATED) > floor) ++covered;
        }
        if (covered != aotx_live_store.sequence - floor) aotx_maintenance.status = AOTX_COG_SEQUENCE;
        aotx_live_candidate.count = count; aotx_live_candidate.bytes = bytes;
        aotx_live_candidate.sequence = aotx_live_store.sequence; aotx_live_candidate.tick = aotx_live_store.tick;
        for (uint32_t j = 0; j < 16; ++j) aotx_live_candidate.lineage[j] = aotx_live_store.lineage[j];
        aotx_live_candidate.root_sequence = aotx_live_store.sequence; aotx_live_candidate.retry_floor = floor;
        aotx_live_candidate.keep_recent = aotx_cog_u32(p + 16); aotx_live_candidate.max_age = aotx_cog_u32(p + 20);
        aotx_live_candidate.maintenance = aotx_cog_u32(p + 24); aotx_live_candidate.pressure_percent = aotx_cog_u32(p + 28);
        aotx_maintenance.removed = aotx_live_store.count - count;
        aotx_maintenance.released = aotx_live_store.bytes - bytes;
        aotx_maintenance.running = !aotx_maintenance.status;
    }
}
__global__ void aotx_memory_offsets(void) {
    if (aotx_sched.held || aotx_live.phase != AOTX_LIVE_MAINTENANCE || !aotx_maintenance.running) return;
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < aotx_live_store.count; i += blockDim.x * gridDim.x) {
        uint32_t index = 0, offset = 0;
        for (uint32_t j = 0; j < i; ++j) if (aotx_maintenance.marks[j]) {
            ++index; offset += (uint32_t)aotx_cog_u64(aotx_live_store.objects[j] + AOTX_CO_BYTES);
        }
        aotx_maintenance.indices[i] = aotx_maintenance.marks[i] ? index : UINT32_MAX;
        aotx_maintenance.offsets[i] = offset;
    }
}
__global__ void aotx_memory_copy(void) {
    if (aotx_sched.held || aotx_live.phase != AOTX_LIVE_MAINTENANCE || !aotx_maintenance.running) return;
    for (uint32_t i = blockIdx.x; i < aotx_live_store.count; i += gridDim.x) {
        if (!aotx_maintenance.marks[i]) continue;
        const unsigned char *r = aotx_live_store.objects[i];
        unsigned char *out = aotx_live_candidate.objects[aotx_maintenance.indices[i]];
        for (uint32_t j = threadIdx.x; j < AOTX_COG_OBJECT; j += blockDim.x) out[j] = r[j];
        uint32_t bytes = (uint32_t)aotx_cog_u64(r + AOTX_CO_BYTES);
        uint32_t source = (uint32_t)aotx_cog_u64(r + AOTX_CO_OFFSET), offset = aotx_maintenance.offsets[i];
        for (uint32_t j = threadIdx.x; j < bytes; j += blockDim.x)
            aotx_live_candidate.payload[offset + j] = aotx_live_store.payload[source + j];
        __syncthreads();
        if (!threadIdx.x) aotx_cog_put(out + AOTX_CO_OFFSET, bytes ? offset : 0, 8);
        __syncthreads();
    }
    uint32_t first = blockIdx.x * blockDim.x + threadIdx.x, stride = blockDim.x * gridDim.x;
    for (uint32_t j = first + aotx_live_candidate.count * AOTX_COG_OBJECT;
         j < AOTX_COG_OBJECTS * AOTX_COG_OBJECT; j += stride) (&aotx_live_candidate.objects[0][0])[j] = 0;
    for (uint32_t j = first + aotx_live_candidate.bytes; j < AOTX_COG_PAYLOAD; j += stride)
        aotx_live_candidate.payload[j] = 0;
}
__global__ void aotx_memory_install(void) {
    if (aotx_sched.held || aotx_live.phase != AOTX_LIVE_MAINTENANCE || !aotx_maintenance.running) return;
    for (uint32_t j = blockIdx.x * blockDim.x + threadIdx.x; j < sizeof(aotx_live_store); j += blockDim.x * gridDim.x)
        ((unsigned char *)&aotx_live_store)[j] = ((const unsigned char *)&aotx_live_candidate)[j];
}
__global__ void aotx_memory_publish(void) {
    if (aotx_sched.held || aotx_live.phase != AOTX_LIVE_MAINTENANCE) return;
    if (aotx_maintenance.running) for (uint32_t slot = threadIdx.x; slot < AOTX_SLOTS; slot += blockDim.x) {
        if (!aotx_live_bound(slot)) continue;
        aotx_recall_result *r = &aotx_live_bindings[slot].choice;
        for (uint32_t i = 0; i < r->count; ++i) r->index[i] = aotx_maintenance.indices[r->index[i]];
    }
    __syncthreads();
    if (!threadIdx.x) {
        aotx_live.status = aotx_maintenance.status;
        aotx_maintenance.last_attempt = aotx_live_store.sequence;
        if (aotx_live.status) ++aotx_live.refused;
        else { ++aotx_live.accepted; ++aotx_maintenance.passes; }
        aotx_live_note(AOTX_LIVE_MAINTAIN, aotx_live.status, aotx_maintenance.removed);
        aotx_live.received = 0; aotx_live.phase = AOTX_LIVE_IDLE; aotx_maintenance.running = 0;
    }
}
