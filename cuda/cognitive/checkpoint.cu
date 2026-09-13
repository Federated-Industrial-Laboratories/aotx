/* Purpose: Advance the bounded checkpoint ring and enforce persistence pressure.
 * Owns: Snapshot publication and device copies of durable progress.
 * Launch shape: One metadata block, parallel device and transport copies, then publication.
 * Lifetime: The optional memory mirror; base mode leaves the ring unbound. */
#include "cognitive/checkpoint.cuh"
#include "cognitive/codec.cuh"
#include "sched/sched.cuh"
#include "cli/cli.cuh"
#include "policy/state.cuh"

__device__ aotx_checkpoint_state aotx_checkpoint;

static __device__ uint64_t aotx_cp_ack(void) {
    aotx_checkpoint_state *s = &aotx_checkpoint;
    aotx_checkpoint_ring *r = s->ring;
    uint64_t serial = aotx_seam_acquire_sys(&r->ack_serial);
    if (serial & 1) return s->acknowledged;
    uint64_t consumed = aotx_seam_acquire_sys(&r->consumed);
    uint64_t boot = aotx_seam_acquire_sys(&r->ack_boot);
    uint64_t error = aotx_seam_acquire_sys(&r->error);
    uint64_t revision = aotx_seam_acquire_sys(&r->durable_revision);
    uint64_t sequence = aotx_seam_acquire_sys(&r->durable_sequence);
    uint64_t generation = aotx_seam_acquire_sys(&r->generation);
    uint64_t runtime = aotx_seam_acquire_sys(&r->reserved[1]);
    uint64_t incarnation[2], digest[4];
    for (unsigned i = 0; i < 2; ++i) incarnation[i] = aotx_seam_acquire_sys(r->incarnation + i);
    for (unsigned i = 0; i < 4; ++i) digest[i] = aotx_seam_acquire_sys(r->commit_digest + i);
    __threadfence_system();
    if (serial != aotx_seam_acquire_sys(&r->ack_serial)) return s->acknowledged;
    if (r->magic != AOTX_CP_MAGIC || r->layout != AOTX_CP_LAYOUT || serial / 2 != consumed ||
        consumed < s->acknowledged || consumed > s->head ||
        (consumed && (boot != aotx_seam.boot_id || !generation ||
         !(incarnation[0] | incarnation[1]) || !(digest[0] | digest[1] | digest[2] | digest[3])))) {
        s->error = AOTX_COG_SEQUENCE;
        return s->acknowledged;
    }
    if (consumed > s->acknowledged) {
        const aotx_checkpoint_cut *cut = s->cuts + (consumed - 1) % AOTX_MEMORY_SNAPSHOTS;
        bool same = incarnation[0] == s->incarnation[0] && incarnation[1] == s->incarnation[1];
        bool changed = false;
        for (unsigned i = 0; i < 4; ++i) changed |= digest[i] != s->commit_digest[i];
        if (revision != cut->revision || sequence != cut->sequence || runtime != cut->runtime ||
            (same && (generation < s->generation || (generation == s->generation && changed)))) {
            s->error = AOTX_COG_SEQUENCE;
            return s->acknowledged;
        }
        for (uint64_t at = s->acknowledged; at < consumed; ++at)
            s->pending_bytes -= s->cuts[at % AOTX_MEMORY_SNAPSHOTS].bytes;
        s->acknowledged = consumed; s->ack_boot = boot; s->durable_sequence = sequence;
        s->durable = revision; s->generation = generation; s->runtime_durable = runtime;
        for (unsigned i = 0; i < 2; ++i) s->incarnation[i] = incarnation[i];
        for (unsigned i = 0; i < 4; ++i) s->commit_digest[i] = digest[i];
    } else if (consumed) {
        bool changed = revision != s->durable || sequence != s->durable_sequence ||
            generation != s->generation || runtime != s->runtime_durable || boot != s->ack_boot;
        for (unsigned i = 0; i < 2; ++i) changed |= incarnation[i] != s->incarnation[i];
        for (unsigned i = 0; i < 4; ++i) changed |= digest[i] != s->commit_digest[i];
        if (changed) { s->error = AOTX_COG_SEQUENCE; return s->acknowledged; }
    }
    s->error = error;
    return s->acknowledged;
}
__device__ uint64_t aotx_checkpoint_pending_bytes(void) {
    if (!aotx_checkpoint.ring) return 0;
    aotx_cp_ack();
    return aotx_checkpoint.pending_bytes + (aotx_checkpoint.copying ? aotx_checkpoint.bytes : 0);
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
    if (aotx_runtime_enabled) {
        aotx_cli_say(out, " runtime source "); aotx_cli_num(out, aotx_runtime_dirty);
        aotx_cli_say(out, " durable "); aotx_cli_num(out, aotx_checkpoint.runtime_durable);
    }
    aotx_cli_say(out, " pending bytes "); aotx_cli_num(out, aotx_checkpoint_pending_bytes());
}
__global__ void aotx_checkpoint_step(void) {
    if (aotx_sched.held || aotx_seam.replaying || !aotx_checkpoint.ring) return;
    if (!threadIdx.x) aotx_checkpoint.capturing = !aotx_checkpoint.copying && !aotx_checkpoint_pressure() &&
        (aotx_live.accepted != aotx_checkpoint.captured ||
         (aotx_runtime_enabled && aotx_runtime_dirty != aotx_checkpoint.runtime_captured)) &&
        aotx_checkpoint_idle() && aotx_policy_quiet();
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
    aotx_checkpoint_cut *cut = aotx_checkpoint.cuts + aotx_checkpoint.head % AOTX_MEMORY_SNAPSHOTS;
    cut->sequence = aotx_cog_u64(aotx_checkpoint_image + 48);
    cut->revision = aotx_checkpoint.captured; cut->runtime = aotx_runtime_enabled ? aotx_checkpoint.runtime_captured : 0;
    cut->bytes = aotx_checkpoint.bytes; aotx_checkpoint.pending_bytes += cut->bytes;
    aotx_cog_put(slot, aotx_seam.boot_id, 8);
    aotx_cog_put(slot + 8, ++aotx_checkpoint.head, 8);
    aotx_cog_put(slot + 16, aotx_checkpoint.bytes, 8);
    if (aotx_runtime_enabled) aotx_cog_put(slot + 24, aotx_checkpoint.runtime_captured, 8);
    __threadfence_system();
    aotx_seam_release_sys(&ring->head, aotx_checkpoint.head);
    aotx_checkpoint.copying = 0;
}
