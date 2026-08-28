/* Purpose: Fill the arena, tick and seam panels from the state the device holds.
 * Owns: Nothing; the kernels write the cells of their own panel.
 * Launch shape: One block for each panel; one thread for each cell and for each row.
 * Lifetime: One node of every frame. */
#include "kvcache/kvcache.cuh"
#include "mem/mem.cuh"
#include "sched/sched.cuh"
#include "ui/ui.cuh"

/* Give the name of a memory region kind. */
static __device__ __forceinline__ const char *aotx_ui_kind_of(unsigned int kind)
{
    switch (kind) {
    case AOTX_MEM_KIND_RING:    return "ring";
    case AOTX_MEM_KIND_SCRATCH: return "scratch";
    case AOTX_MEM_KIND_WEIGHTS: return "weights";
    default:                    return "none";
    }
}

/* Put the title on the first row of a panel. */
static __device__ __forceinline__ void aotx_ui_head(const aotx_ui_panel *panel,
                                                    const char *name)
{
    aotx_ui_say(panel, 0u, 1u, name, AOTX_UI_HIGH);
}

/* The arena panel shows the region table and the budget the regions take. */
__global__ void aotx_ui_arena(void)
{
    const aotx_ui_panel *panel = &aotx_ui_panel_table[AOTX_UI_ARENA];
    aotx_ui_blank(panel);
    __syncthreads();

    unsigned int count = aotx_mem_region_table.count;
    for (unsigned int i = threadIdx.x; i < count; i += blockDim.x) {
        const aotx_mem_region *region = &aotx_mem_region_table.region[i];
        unsigned int row = i + 2u;
        unsigned int col = aotx_ui_say(panel, row, 1u, aotx_ui_kind_of(region->kind),
                                       AOTX_UI_NORMAL);
        col = aotx_ui_number(panel, row, col + 1u, region->bytes >> 20, AOTX_UI_NORMAL);
        aotx_ui_say(panel, row, col + 1u, "MB", AOTX_UI_DIM);
    }

    if (threadIdx.x == 0u) {
        unsigned long long mapped = 0ull;
        for (unsigned int i = 0u; i < count; ++i) {
            mapped += aotx_mem_region_table.region[i].bytes;
        }
        unsigned long long used = aotx_seam.dev.tail - aotx_seam.dev.flushed;
        aotx_ui_head(panel, "arena");
        aotx_ui_say(panel, 1u, 1u, "region size", AOTX_UI_DIM);
        unsigned int row = count + 3u;
        aotx_ui_field(panel, row, "mapped MB", mapped >> 20);
        aotx_ui_field(panel, row + 1u, "held MB", aotx_mem_budget_table.reserved >> 20);
        aotx_ui_field(panel, row + 2u, "free MB", aotx_mem_budget_table.free_now >> 20);
        aotx_ui_field(panel, row + 3u, "total MB", aotx_mem_budget_table.total >> 20);
        aotx_ui_field(panel, row + 4u, "ring used", used);
        /* The table holds the pages of each agent slot, so the sum is the count mapped. */
        unsigned long long pages = 0ull;
        for (unsigned int slot = 0u; slot < AOTX_KV_AGENTS; ++slot) {
            pages += (unsigned long long)aotx_kv.count[slot];
        }
        aotx_ui_field(panel, row + 5u, "kv pages mapped", pages);
    }
}

/* The tick panel shows what the tick did and what the last statistics record holds. */
__global__ void aotx_ui_tick(void)
{
    __shared__ unsigned long long seqs[1];
    const aotx_ui_panel *panel = &aotx_ui_panel_table[AOTX_UI_TICK];
    aotx_ui_blank(panel);
    /* The block walks the ring together. A walk by one thread over a full ring costs tens
     * of milliseconds and holds the frame. */
    unsigned int count = aotx_ui_recent(AOTX_REC_STATS, 1u, seqs);
    __syncthreads();
    if (threadIdx.x != 0u) {
        return;
    }

    aotx_ui_head(panel, "tick");
    aotx_ui_field(panel, 1u, "tick", aotx_time_tick);
    aotx_ui_field(panel, 2u, "records", aotx_sched.records);
    aotx_ui_field(panel, 3u, "blocks", aotx_sched.blocks);
    aotx_ui_field(panel, 4u, "held", aotx_sched.held_count);
    aotx_ui_field(panel, 5u, "applied", aotx_seam.apply.applied_count);
    aotx_ui_field(panel, 6u, "start ns", aotx_sched.start_ns);
    /* The decode: the sequences that are live, the calls it refused, and the pages the
     * slots hold. The last row of the panel holds the three of them. */
    unsigned int col = aotx_ui_say(panel, 10u, 1u, "decode", AOTX_UI_DIM);
    col = aotx_ui_number(panel, 10u, col + 1u, aotx_seqs.live, AOTX_UI_NORMAL);
    col = aotx_ui_say(panel, 10u, col + 1u, "refused", AOTX_UI_DIM);
    col = aotx_ui_number(panel, 10u, col + 1u, aotx_seqs.refused, AOTX_UI_NORMAL);
    col = aotx_ui_say(panel, 10u, col + 1u, "pages", AOTX_UI_DIM);
    aotx_ui_number(panel, 10u, col + 1u, aotx_kv.mapped_pages, AOTX_UI_NORMAL);
    /* The agents: the agents that are not free, the tasks the table holds, and the tool
     * requests that wait for the operator. */
    col = aotx_ui_say(panel, 11u, 1u, "agents", AOTX_UI_DIM);
    col = aotx_ui_number(panel, 11u, col + 1u, aotx_agents.live, AOTX_UI_NORMAL);
    col = aotx_ui_say(panel, 11u, col + 1u, "tasks", AOTX_UI_DIM);
    col = aotx_ui_number(panel, 11u, col + 1u, aotx_agents.tasks, AOTX_UI_NORMAL);
    col = aotx_ui_say(panel, 11u, col + 1u, "pending", AOTX_UI_DIM);
    aotx_ui_number(panel, 11u, col + 1u, aotx_cli_pending_count(), AOTX_UI_NORMAL);

    unsigned long long seq = (count != 0u) ? seqs[0] : 0ull;
    if (seq == 0ull) {
        aotx_ui_say(panel, 7u, 1u, "no statistics record", AOTX_UI_DIM);
        return;
    }
    const volatile aotx_record_header *header = aotx_cli_slot(seq);
    const volatile aotx_stats_body *body =
        (const volatile aotx_stats_body *)((const volatile unsigned char *)header
                                           + AOTX_HEADER_BYTES);
    unsigned long long tick_ns = body->tick_ns;
    unsigned long long records = body->records;
    unsigned long long inbound = body->inbound;
    if (!aotx_cli_holds(header, seq, AOTX_REC_STATS)) {
        aotx_ui_say(panel, 7u, 1u, "no statistics record", AOTX_UI_DIM);
        return;
    }
    aotx_ui_field(panel, 7u, "last ns", tick_ns);
    aotx_ui_field(panel, 8u, "last records", records);
    aotx_ui_field(panel, 9u, "last inbound", inbound);
}

/* The seam panel shows how far the disk side is behind. The lag in bytes comes from the
 * cursor of the drain, and the lag in ticks comes from the block the cursor points at. */
__global__ void aotx_ui_seam(void)
{
    __shared__ unsigned long long seqs[1];
    const aotx_ui_panel *panel = &aotx_ui_panel_table[AOTX_UI_SEAM];
    aotx_ui_blank(panel);
    unsigned int count = aotx_ui_recent(AOTX_REC_STALL, 1u, seqs);
    __syncthreads();
    if (threadIdx.x != 0u) {
        return;
    }

    const aotx_host_ring_preamble *host =
        (const aotx_host_ring_preamble *)aotx_seam.host.preamble;
    if (host == 0) {
        aotx_ui_head(panel, "seam");
        aotx_ui_say(panel, 1u, 1u, "the rings are not open", AOTX_UI_DIM);
        return;
    }
    unsigned long long cursor = aotx_seam_acquire_sys(&host->cursor);
    unsigned long long head = aotx_seam.host.head;
    unsigned long long behind = (head > cursor) ? (head - cursor) : 0ull;

    aotx_ui_head(panel, "seam");
    aotx_ui_field(panel, 1u, "head bytes", head);
    aotx_ui_field(panel, 2u, "drain bytes", cursor);
    aotx_ui_field(panel, 3u, "lag bytes", behind);
    aotx_ui_field(panel, 4u, "block seq", aotx_seam.host.block_seq);
    aotx_ui_field(panel, 5u, "free bytes", aotx_sched.free_bytes);

    /* The block at the cursor is the next one the drain takes, and its tick states how far
     * behind the disk is. A cursor at the head means the disk holds every block. */
    unsigned long long lag = 0ull;
    if (behind != 0ull) {
        const volatile aotx_block_header *block =
            (const volatile aotx_block_header *)(aotx_seam.host.data
                                                 + (cursor & aotx_seam.host.mask));
        if (block->magic == AOTX_BLOCK_MAGIC && block->block_seq != 0ull
            && aotx_time_tick > block->tick) {
            lag = aotx_time_tick - block->tick;
        }
    }
    aotx_ui_field(panel, 6u, "lag ticks", lag);

    unsigned long long seq = (count != 0u) ? seqs[0] : 0ull;
    if (seq == 0ull) {
        aotx_ui_say(panel, 7u, 1u, "no stall record", AOTX_UI_DIM);
        return;
    }
    const volatile aotx_record_header *header = aotx_cli_slot(seq);
    const volatile aotx_stall_body *body =
        (const volatile aotx_stall_body *)((const volatile unsigned char *)header
                                           + AOTX_HEADER_BYTES);
    unsigned long long room = body->host_ring_free;
    unsigned long long held = body->held_count;
    if (!aotx_cli_holds(header, seq, AOTX_REC_STALL)) {
        aotx_ui_say(panel, 7u, 1u, "no stall record", AOTX_UI_DIM);
        return;
    }
    aotx_ui_field(panel, 7u, "stall free", room);
    aotx_ui_field(panel, 8u, "stall held", held & ~AOTX_STALL_OVERRUN);
    aotx_ui_field(panel, 9u, "dropped runs", aotx_seam.dev.overrun);
}
