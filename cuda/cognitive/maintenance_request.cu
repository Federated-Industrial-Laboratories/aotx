/* Purpose: Record explicit and automatic maintenance through one admission path.
 * Owns: Complete policy requests and visible maintenance status.
 * Launch shape: Serial control within the live-stage block; no disk polling loop.
 * Lifetime: An opted-in store and its exact journal replay. */
#include "cognitive/maintenance.cuh"
#include "cognitive/codec.cuh"
#include "seam/seam.cuh"
#include "cli/cli.cuh"
#include "policy/state.cuh"

static __device__ unsigned char aotx_memory_request[96];

__device__ void aotx_memory_auto_request(void) {
    if (aotx_seam.replaying || !aotx_live_store.maintenance || !aotx_checkpoint_idle() ||
        aotx_maintenance.last_attempt == aotx_live_store.sequence) return;
    uint32_t percent = aotx_live_store.pressure_percent;
    if (!aotx_policy.enabled && (uint64_t)aotx_live_store.count * 100 < (uint64_t)AOTX_COG_OBJECTS * percent &&
        (uint64_t)aotx_live_store.bytes * 100 < (uint64_t)AOTX_COG_PAYLOAD * percent) return;
    if (aotx_checkpoint_maintenance_pressure()) return;
    if (!aotx_policy_maintenance()) return;
    unsigned char *body = aotx_memory_request, *p = body + AOTX_LIVE_PART;
    for (uint32_t j = 0; j < 96; ++j) body[j] = 0;
    aotx_cog_put(body, 1, 4); aotx_cog_put(body + 4, AOTX_LIVE_MAINTAIN, 4);
    for (uint32_t j = 0; j < 8; ++j) body[8 + j] = "AOTXMNT1"[j];
    aotx_cog_put(body + 16, aotx_live.accepted, 8); aotx_cog_put(body + 24, 64, 4);
    for (uint32_t j = 0; j < 8; ++j) p[j] = "AOTXMNT1"[j];
    aotx_cog_put(p + 8, 1, 4); aotx_cog_put(p + 12, 1, 4);
    aotx_cog_put(p + 16, aotx_live_store.keep_recent, 4); aotx_cog_put(p + 20, aotx_live_store.max_age, 4);
    aotx_cog_put(p + 24, aotx_live_store.maintenance, 4); aotx_cog_put(p + 28, percent, 4);
    aotx_cog_put(p + 32, aotx_live_store.sequence, 8); aotx_cog_put(p + 40, aotx_live_store.root_sequence, 8);
    for (uint32_t j = 0; j < 16; ++j) p[48 + j] = aotx_live_store.lineage[j];
    uint64_t seq = aotx_seam_write(AOTX_WRITER_SYSTEM, AOTX_CLASS_A, AOTX_LIVE_RECORD,
        AOTX_FLAG_ADMISSION, body, 96);
    aotx_seam.apply.state_hash = aotx_seam_fnv1a(aotx_seam.apply.state_hash, body, 96);
    ++aotx_seam.apply.applied_count;
    aotx_live_part(body, 96, seq, AOTX_FLAG_ADMISSION);
}
__device__ void aotx_memory_maintain_begin(void) {
    if (threadIdx.x) return;
    const unsigned char *p = aotx_live.input;
    uint32_t status = 0;
    if (aotx_live.total != 64 || !aotx_cog_equal(p, (const unsigned char *)"AOTXMNT1", 8) ||
        aotx_cog_u32(p + 8) != 1 || aotx_cog_u32(p + 12) > 1 ||
        aotx_cog_u32(p + 24) > 1 || !aotx_cog_u32(p + 28) || aotx_cog_u32(p + 28) > 100)
        status = AOTX_COG_FORMAT;
    else if (!aotx_live.ready || !aotx_checkpoint_quiet()) status = AOTX_COG_DENIED;
    else if (!aotx_cog_equal(p + 48, aotx_live_store.lineage) ||
        aotx_cog_u64(p + 32) != aotx_live_store.sequence ||
        aotx_cog_u64(p + 40) != aotx_live_store.root_sequence) status = AOTX_COG_STALE;
    else if (aotx_live_admission_pressure()) status = AOTX_COG_CAPACITY;
    if (status) {
        aotx_live.status = status; ++aotx_live.refused; aotx_live.received = 0;
        aotx_live_note(AOTX_LIVE_MAINTAIN, status, 0); aotx_live.phase = AOTX_LIVE_IDLE;
    } else aotx_live.phase = AOTX_LIVE_MAINTENANCE;
}
__device__ void aotx_memory_status(aotx_cli_out *out) {
    aotx_cli_say(out, "memory store: objects "); aotx_cli_num(out, aotx_live_store.count);
    aotx_cli_say(out, " of "); aotx_cli_num(out, AOTX_COG_OBJECTS);
    aotx_cli_say(out, " payload "); aotx_cli_num(out, aotx_live_store.bytes);
    aotx_cli_say(out, " of "); aotx_cli_num(out, AOTX_COG_PAYLOAD);
    aotx_console_write(out->text, out->at); aotx_cli_clear(out);
    aotx_cli_say(out, "memory lifecycle: root "); aotx_cli_num(out, aotx_live_store.root_sequence);
    aotx_cli_say(out, " retry floor "); aotx_cli_num(out, aotx_live_store.retry_floor);
    aotx_cli_say(out, " automatic "); aotx_cli_num(out, aotx_live_store.maintenance);
    aotx_cli_say(out, " removed "); aotx_cli_num(out, aotx_maintenance.removed);
    aotx_cli_say(out, " released bytes "); aotx_cli_num(out, aotx_maintenance.released);
}
