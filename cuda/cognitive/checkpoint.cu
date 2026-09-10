/* Purpose: Advance the bounded checkpoint ring and enforce persistence pressure.
 * Owns: Snapshot publication and device copies of durable progress.
 * Launch shape: One metadata block, parallel device and transport copies, then publication.
 * Lifetime: The optional memory mirror; base mode leaves the ring unbound. */
#include "cognitive/checkpoint.cuh"
#include "cognitive/codec.cuh"
#include "sched/sched.cuh"
#include "cli/cli.cuh"

__device__ aotx_checkpoint_state aotx_checkpoint;

static __device__ uint64_t aotx_cp_ack(void) {
    aotx_checkpoint_ring *r = aotx_checkpoint.ring;
    uint64_t consumed = aotx_seam_acquire_sys(&r->consumed);
    uint64_t boot = aotx_seam_acquire_sys(&r->ack_boot);
    uint64_t error = aotx_seam_acquire_sys(&r->error);
    if (consumed > aotx_checkpoint.head || (consumed && boot != aotx_seam.boot_id)) {
        aotx_checkpoint.error = AOTX_COG_SEQUENCE;
        return 0;
    }
    aotx_checkpoint.error = error;
    if (consumed && consumed == aotx_seam_acquire_sys(&r->consumed)) {
        aotx_checkpoint.durable = aotx_seam_acquire_sys(&r->durable_revision);
        aotx_checkpoint.generation = aotx_seam_acquire_sys(&r->generation);
    }
    return consumed;
}
__device__ bool aotx_checkpoint_pressure(void) {
    if (!aotx_checkpoint.ring || aotx_seam.replaying) return false;
    uint64_t consumed = aotx_cp_ack();
    return aotx_checkpoint.error || aotx_checkpoint.head - consumed + aotx_checkpoint.copying >= AOTX_MEMORY_SNAPSHOTS;
}
__device__ bool aotx_checkpoint_maintenance_pressure(void) {
    bool pressure = aotx_checkpoint_pressure();
    return pressure || (aotx_checkpoint.ring && !aotx_seam.replaying &&
        (aotx_checkpoint.copying || aotx_checkpoint.durable != aotx_live.accepted));
}
__device__ void aotx_checkpoint_status(aotx_cli_out *out) {
    if (!aotx_checkpoint.ring) { aotx_cli_say(out, "memory mirror: off"); return; }
    uint64_t consumed = aotx_cp_ack();
    aotx_cli_say(out, "memory mirror: committed "); aotx_cli_num(out, aotx_live.accepted);
    aotx_cli_say(out, " durable "); aotx_cli_num(out, aotx_checkpoint.durable);
    aotx_cli_say(out, " generation "); aotx_cli_num(out, aotx_checkpoint.generation);
    aotx_cli_say(out, " pending "); aotx_cli_num(out, aotx_checkpoint.head - consumed + aotx_checkpoint.copying);
    aotx_cli_say(out, " error "); aotx_cli_num(out, aotx_checkpoint.error);
}
__global__ void aotx_checkpoint_step(void) {
    if (aotx_sched.held || aotx_seam.replaying || !aotx_checkpoint.ring) return;
    if (!threadIdx.x) aotx_checkpoint.capturing = !aotx_checkpoint.copying && !aotx_checkpoint_pressure() &&
        aotx_live.accepted != aotx_checkpoint.captured && aotx_checkpoint_idle();
    __syncthreads();
    if (aotx_checkpoint.capturing) {
        aotx_checkpoint_encode();
        __syncthreads();
        if (!threadIdx.x) { aotx_checkpoint.copying = 1; aotx_checkpoint.copied = 0; }
    }
}
/* These nodes are adjacent in one stream. No live mutation can occur between them. */
__global__ void aotx_checkpoint_fill(void) {
    if (aotx_sched.held || aotx_seam.replaying || !aotx_checkpoint.ring || !aotx_checkpoint.capturing) return;
    uint32_t base = AOTX_CP_HEADER + aotx_checkpoint.bindings * AOTX_CP_ROW + AOTX_COG_HEADER;
    uint32_t objects = aotx_live_store.count * AOTX_COG_OBJECT;
    uint32_t first = blockIdx.x * blockDim.x + threadIdx.x, stride = blockDim.x * gridDim.x;
    const unsigned char *rows = &aotx_live_store.objects[0][0];
    for (uint32_t j = first; j < objects; j += stride) aotx_checkpoint_image[base + j] = rows[j];
    for (uint32_t j = first; j < aotx_live_store.bytes; j += stride)
        aotx_checkpoint_image[base + objects + j] = aotx_live_store.payload[j];
}
__global__ void aotx_checkpoint_copy(void) {
    if (aotx_sched.held || aotx_seam.replaying || !aotx_checkpoint.ring || !aotx_checkpoint.copying) return;
    unsigned char *slot = (unsigned char *)(aotx_checkpoint.ring + 1) +
        (aotx_checkpoint.head % AOTX_MEMORY_SNAPSHOTS) * AOTX_CP_SLOT_BYTES;
    uint32_t begin = aotx_checkpoint.copied, count = aotx_checkpoint.bytes - begin;
    if (count > AOTX_CP_COPY) count = AOTX_CP_COPY;
    for (uint32_t j = blockIdx.x * blockDim.x + threadIdx.x; j < count; j += blockDim.x * gridDim.x)
        slot[AOTX_CP_SLOT_HEADER + begin + j] = aotx_checkpoint_image[begin + j];
    __threadfence_system();
}
__global__ void aotx_checkpoint_publish(void) {
    if (aotx_sched.held || aotx_seam.replaying || !aotx_checkpoint.ring || !aotx_checkpoint.copying) return;
    uint32_t count = aotx_checkpoint.bytes - aotx_checkpoint.copied;
    if (count > AOTX_CP_COPY) count = AOTX_CP_COPY;
    aotx_checkpoint.copied += count;
    if (aotx_checkpoint.copied != aotx_checkpoint.bytes) return;
    aotx_checkpoint_ring *ring = aotx_checkpoint.ring;
    unsigned char *slot = (unsigned char *)(ring + 1) +
        (aotx_checkpoint.head % AOTX_MEMORY_SNAPSHOTS) * AOTX_CP_SLOT_BYTES;
    for (uint32_t j = 0; j < AOTX_CP_SLOT_HEADER; ++j) slot[j] = 0;
    aotx_cog_put(slot, aotx_seam.boot_id, 8);
    aotx_cog_put(slot + 8, ++aotx_checkpoint.head, 8);
    aotx_cog_put(slot + 16, aotx_checkpoint.bytes, 8);
    __threadfence_system();
    aotx_seam_release_sys(&ring->head, aotx_checkpoint.head);
    aotx_checkpoint.copying = 0;
}
