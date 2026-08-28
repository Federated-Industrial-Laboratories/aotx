/* Purpose: Start a tick, make the tick load, and close the tick with a commit record.
 * Owns: The tick state.
 * Launch shape: One thread starts and commits; a grid makes the load.
 * Lifetime: One node of every tick. */
#include "rng/rng.cuh"
#include "sched/sched.cuh"
#include "settings/settings.cuh"

__device__ aotx_sched_state aotx_sched =
    { 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull };

/* A hold writes one stall record when it starts, and one more when it ends. The second
 * record carries the count of ticks that were held. A held tick writes no other record and
 * no commit record, because a held tick is not a complete tick. */
static __device__ __forceinline__ void aotx_sched_stall(unsigned long long room,
                                                        unsigned long long overrun)
{
    aotx_stall_body body;
    body.host_ring_free = room;
    body.held_count = aotx_sched.held_count;
    if (overrun != 0ull) {
        body.held_count |= AOTX_STALL_OVERRUN;
    }
    aotx_seam_write(AOTX_WRITER_SYSTEM, AOTX_CLASS_B, AOTX_REC_STALL, 0u,
                    &body, (unsigned int)sizeof body);
}

/* The tick starts with one read of the drain cursor and one read of the inbound head. The
 * tick is held when the host ring or the device ring lacks room for the worst case of this
 * tick. */
__global__ void aotx_sched_tick_start(unsigned long long workload)
{
    const aotx_host_ring_preamble *host =
        (const aotx_host_ring_preamble *)aotx_seam.host.preamble;
    const aotx_inbound_preamble *inbound =
        (const aotx_inbound_preamble *)aotx_seam.in.preamble;

    aotx_time_tick += 1ull;
    aotx_sched.start_ns = aotx_time_globaltimer();

    unsigned long long room = aotx_seam_host_free(aotx_seam_acquire_sys(&host->cursor));
    unsigned long long ready = aotx_seam_acquire_sys(&inbound->head) - aotx_seam.in.consumed;
    if (ready > AOTX_INBOUND_MAX_TICK) {
        ready = AOTX_INBOUND_MAX_TICK;
    }
    if (workload > AOTX_TICK_RECORDS_MAX) {
        workload = AOTX_TICK_RECORDS_MAX;
    }

    /* The worst case of a tick has six parts. The first part is the records of the tick.
     * The second and the third are the journal record and the echo of every input. The
     * others are the answer, the tick load, the records of the decode, and the records of
     * the agents and the tools. */
    unsigned long long backlog = aotx_seam.dev.tail - aotx_seam.dev.flushed;
    unsigned long long worst = AOTX_TICK_RECORDS_OWN
                             + ready * (AOTX_APPLY_RECORDS_EACH + AOTX_CLI_RECORDS_EACH)
                             + workload + AOTX_DECODE_RECORDS_MAX
                             + AOTX_AGENT_RECORDS_MAX;
    unsigned long long need = 2ull * aotx_seam_block_bytes(backlog + worst);
    unsigned long long held = 0ull;
    if (need > room || backlog + worst > aotx_seam.dev.slot_count) {
        held = 1ull;
    }

    aotx_sched.free_bytes = room;
    aotx_sched.held = held;
    /* A replay takes the records of one tick of the journal in one tick of this run. The
     * apply then takes fewer records this tick and the rest wait in the ring. */
    unsigned long long takes = held ? 0ull
                             : (unsigned long long)aotx_seam_replay_take(
                                   aotx_seam.in.consumed, (unsigned int)ready);
    aotx_seam.apply.this_tick = takes;

    /* The apply owns two sequences for each input it takes: the journal record and the
     * echo. The reservation stands before the apply runs. A record that the command layer
     * writes then takes a sequence after the run, and never one inside it. */
    aotx_seam.apply.first_seq = aotx_seam.dev.tail + 1ull;
    aotx_seam.dev.tail += AOTX_APPLY_RECORDS_EACH * takes;
    aotx_bulk_tick_start();
    if (held != 0ull) {
        aotx_sched.held_count += 1ull;
    }

    /* One record for the start of a hold, one for its end, and one for a drop that the
     * flush found. Nothing is written while the hold lasts. */
    unsigned long long overrun = aotx_seam.dev.overrun;
    int changed = (overrun != aotx_sched.overrun_seen) ? 1 : 0;
    if (held != aotx_sched.holding || changed) {
        aotx_sched_stall(room, overrun);
        aotx_sched.overrun_seen = overrun;
    }
    aotx_sched.holding = held;
}

/* The tick load. Each record carries the tick, the sequence, the writer position and two
 * random words, so no two records hold the same body. */
__global__ void aotx_sched_workload(unsigned long long workload)
{
    if (aotx_sched.held != 0ull) {
        return;
    }
    unsigned long long lane = (unsigned long long)(blockIdx.x * blockDim.x + threadIdx.x);
    unsigned long long stride = (unsigned long long)(gridDim.x * blockDim.x);
    for (unsigned long long i = lane; i < workload; i += stride) {
        unsigned long long seq = aotx_seam_claim(1u);
        aotx_record_header *header = aotx_seam_slot(seq);
        unsigned long long *body = (unsigned long long *)aotx_seam_body(header);
        uint4 word = aotx_rng_lane(aotx_seam.boot_id, (unsigned int)aotx_time_tick,
                                   (unsigned int)lane, i);
        body[0] = aotx_time_tick;
        body[1] = seq;
        body[2] = (unsigned long long)blockIdx.x;
        body[3] = (unsigned long long)threadIdx.x;
        body[4] = i;
        body[5] = ((unsigned long long)word.y << 32) | (unsigned long long)word.x;
        body[6] = ((unsigned long long)word.w << 32) | (unsigned long long)word.z;
        body[7] = workload;
        aotx_seam_publish(header, seq, AOTX_WRITER_SYSTEM, AOTX_CLASS_B,
                          AOTX_REC_NOTE, 0u, 64u);
    }
}

/* The commit record is the last record of a complete tick and of the block that the flush
 * writes. Its record count is the count of that block, so it covers every record that no
 * block holds yet. A held tick writes no commit record. */
__global__ void aotx_sched_commit(void)
{
    if (aotx_sched.held != 0ull) {
        return;
    }

    /* The statistics record states what the tick took and what it carried. The commit
     * record follows it, so the block of the tick ends with the commit. The record cannot
     * wait for the flush that carries it. The time of the tick counts the kernels up to the
     * commit, and the two flush nodes of the tick before. */
    unsigned long long stats_seq = aotx_seam_claim(1u);
    aotx_record_header *stats_header = aotx_seam_slot(stats_seq);
    aotx_stats_body *stats = (aotx_stats_body *)aotx_seam_body(stats_header);
    stats->tick_ns = (aotx_time_globaltimer() - aotx_sched.start_ns) + aotx_sched.flush_ns;
    stats->records = (stats_seq + 1ull) - aotx_seam.dev.flushed;
    stats->inbound = aotx_seam.apply.this_tick;
    aotx_seam_publish(stats_header, stats_seq, AOTX_WRITER_SYSTEM, AOTX_CLASS_B,
                      AOTX_REC_STATS, 0u, (unsigned int)sizeof(aotx_stats_body));

    unsigned long long seq = aotx_seam_claim(1u);
    aotx_record_header *header = aotx_seam_slot(seq);
    aotx_commit_body *body = (aotx_commit_body *)aotx_seam_body(header);
    body->state_hash = aotx_seam.apply.state_hash;
    body->applied_count = aotx_seam.apply.applied_count;
    body->inbound_consumed = aotx_seam.in.consumed;
    body->records_this_tick = seq - aotx_seam.dev.flushed;
    aotx_seam_publish(header, seq, AOTX_WRITER_SYSTEM, AOTX_CLASS_A,
                      AOTX_REC_TICK_COMMIT, 0u, (unsigned int)sizeof(aotx_commit_body));
    aotx_sched.records = seq;

    /* The two values the pump reads go in the control page with a release store. The
     * pace of the next tick therefore takes the settings of this one. */
    aotx_settings_publish();

    /* The flush of this tick starts here, and the last flush node measures it. */
    aotx_sched.commit_ns = aotx_time_globaltimer();
}
