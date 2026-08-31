/* Purpose: Write one snapshot of the grid, the status figures and the live tables.
 * Owns: The mirror state; the bytes of the snapshot belong to the mapped file.
 * Launch shape: One block of AOTX_MIRROR_THREADS; the last node of the raster graph.
 * Lifetime: The whole run; the node writes one slot for each frame. */
#include "agent/agent.cuh"
#include "agent/transcript.cuh"
#include "catalog/catalog.cuh"
#include "cli/agents.cuh"
#include "model/model.cuh"
#include "model/load.cuh"
#include "sched/sched.cuh"
#include "settings/settings.cuh"
#include "tool/tool.cuh"
#include "ui/mirror.cuh"

__device__ aotx_mirror_state aotx_mirror;

/* The architecture of the build, as the version line writes it. */
#define AOTX_MIRROR_QUOTE(x)  #x
#define AOTX_MIRROR_TEXT_OF(x) AOTX_MIRROR_QUOTE(x)
#define AOTX_MIRROR_ARCH      "sm_" AOTX_MIRROR_TEXT_OF(AOTX_ARCH)

/* The tables hold one row for each row of the device tables of this profile. */
typedef char aotx_mirror_check_rows[(AOTX_SLOTS <= AOTX_MIRROR_AGENT_ROWS
                                     && AOTX_MODULE_SLOTS <= AOTX_MIRROR_MODULE_ROWS
                                     && AOTX_MODEL_ROLES <= AOTX_MIRROR_MODEL_ROWS
                                     && (unsigned int)AOTX_SETTING_NUMBER_COUNT
                                        <= AOTX_MIRROR_SETTING_ROWS) ? 1 : -1];

/* Copy a text that ends with a zero byte into a fixed field and clear the bytes after it.
 * A field of the mirror holds the same bytes for the same state. A reader that compares two
 * snapshots therefore sees no difference that the state does not have. */
static __device__ __forceinline__ void aotx_mirror_put(char *to, unsigned int bytes,
                                                       const char *from)
{
    unsigned int at = 0u;
    while (at + 1u < bytes && from[at] != '\0') {
        to[at] = from[at];
        at += 1u;
    }
    while (at < bytes) {
        to[at] = '\0';
        at += 1u;
    }
}

/* The ticks the disk side is behind. The block at the cursor of the drain is the next one
 * the drain takes, and its tick states the lag. The seam panel reads the same two fields. */
static __device__ __forceinline__ unsigned long long aotx_mirror_lag_ticks(void)
{
    const aotx_host_ring_preamble *host =
        (const aotx_host_ring_preamble *)aotx_seam.host.preamble;
    if (host == 0) {
        return 0ull;
    }
    unsigned long long cursor = aotx_seam_acquire_sys(&host->cursor);
    unsigned long long head = aotx_seam.host.head;
    if (head <= cursor) {
        return 0ull;
    }
    const volatile aotx_block_header *block =
        (const volatile aotx_block_header *)(aotx_seam.host.data
                                             + (cursor & aotx_seam.host.mask));
    if (block->magic != AOTX_BLOCK_MAGIC || block->block_seq == 0ull
        || aotx_time_tick <= block->tick) {
        return 0ull;
    }
    return aotx_time_tick - block->tick;
}

/* The language model that is resident, or the value that stands for none. */
static __device__ __forceinline__ unsigned int aotx_mirror_language(void)
{
    if (aotx_model[AOTX_MODEL_LANGUAGE].layers != 0u) {
        return AOTX_MODEL_LANGUAGE;
    }
    if (aotx_model[AOTX_MODEL_LANGUAGE_Q4].layers != 0u) {
        return AOTX_MODEL_LANGUAGE_Q4;
    }
    return 0xffffffffu;
}

/* The status figures and the panel table. One thread writes the head, while the other
 * threads of the block write the cells. */
static __device__ void aotx_mirror_write_head(aotx_mirror_head *head,
                                              unsigned long long tables_at)
{
    head->tick = aotx_time_tick;
    head->boot_id = aotx_seam.boot_id;
    head->drain_lag_ms = aotx_mirror_lag_ticks()
                       * (unsigned long long)aotx_setting_count(AOTX_SET_TICK_PERIOD_MS);
    head->held = AOTX_STALL_HELD(aotx_sched.held_count);
    head->model_mb = aotx_model_load.placed_bytes >> 20;
    head->reserved_model = 0ull;
    head->agents_live = aotx_agents.live;
    head->slots = (unsigned int)AOTX_SLOTS;
    head->requests_waiting = aotx_cli_pending_count();
    head->language = aotx_mirror_language();

    /* The cursor cell is the cell of the editor in the grid. A terminal puts its own
     * cursor where the window shows the bright cell. The prompt takes two columns after
     * column one of the console panel. */
    const aotx_ui_panel *panel = &aotx_ui_panel_table[AOTX_UI_CONSOLE];
    unsigned int first = 0u;
    unsigned int row = 0u;
    unsigned int col = 0u;
    unsigned int shown = 0u;
    unsigned int top = 0u;
    aotx_ui_editor_place(panel, &first, &row, &col, &shown, &top);
    head->cursor_row = (unsigned int)panel->row + row;
    head->cursor_col = (unsigned int)panel->col + col;
    head->focus = aotx_cli_focus;
    head->tables_sequence = (unsigned int)tables_at;
    aotx_mirror_put(head->profile, (unsigned int)sizeof head->profile, AOTX_PROFILE_NAME);
    aotx_mirror_put(head->arch, (unsigned int)sizeof head->arch, AOTX_MIRROR_ARCH);
    for (unsigned int i = 0u; i < AOTX_MIRROR_PANELS; ++i) {
        const aotx_ui_panel *at = &aotx_ui_panel_table[i];
        aotx_mirror_put(head->panel[i].name, AOTX_MIRROR_NAME_BYTES, aotx_ui_panel_name(i));
        head->panel[i].row = at->row;
        head->panel[i].col = at->col;
        head->panel[i].rows = at->rows;
        head->panel[i].cols = at->cols;
    }
}

/* The cells of the frame. Every thread copies 16 bytes at a time from the grid the panel
 * kernels wrote and the raster kernel read. */
static __device__ __forceinline__ void aotx_mirror_write_cells(aotx_mirror_cell *cell)
{
    const uint4 *from = (const uint4 *)aotx_ui_grid;
    uint4 *to = (uint4 *)cell;
    const unsigned int units = (AOTX_MIRROR_CELLS * 2u) / 16u;
    for (unsigned int at = threadIdx.x; at < units; at += blockDim.x) {
        to[at] = from[at];
    }
}

/* One row for each agent slot. A row of a free slot holds an empty name. */
static __device__ void aotx_mirror_write_agents(aotx_mirror_tables *tables)
{
    for (unsigned int row = threadIdx.x; row < AOTX_MIRROR_AGENT_ROWS; row += blockDim.x) {
        aotx_mirror_agent_row *to = &tables->agent[row];
        const aotx_agent *agent = (row < AOTX_SLOTS) ? &aotx_agents.agent[row] : 0;
        to->id = row;
        if (agent == 0 || agent->state == AOTX_AGENT_STATE_FREE) {
            to->role = AOTX_ROLE_NONE;
            to->state = AOTX_AGENT_STATE_FREE;
            to->task = 0xffffffffu;
            to->request = 0u;
            to->turn = 0u;
            to->pages = 0u;
            aotx_mirror_put(to->role_name, AOTX_MIRROR_NAME_BYTES, "");
            continue;
        }
        to->role = agent->role;
        to->state = agent->state;
        to->task = agent->task;
        to->request = agent->request;
        to->turn = agent->turn;
        to->pages = aotx_transcript_page_limit(row);
        aotx_mirror_put(to->role_name, AOTX_MIRROR_NAME_BYTES,
                        aotx_cli_role_name(agent->role));
    }
}

/* The requests that wait for the operator, the lowest number first. The order is the order
 * of the agents panel, so the two surfaces name the same first request. */
static __device__ void aotx_mirror_write_requests(aotx_mirror_tables *tables)
{
    for (unsigned int rank = threadIdx.x; rank < AOTX_MIRROR_REQUEST_ROWS;
         rank += blockDim.x) {
        aotx_mirror_request_row *to = &tables->request[rank];
        unsigned int at = aotx_cli_pending_at(rank);
        if (at >= AOTX_SLOTS) {
            to->request = 0u;
            to->agent = 0u;
            to->tool = AOTX_CATALOG_NO_ENTRY;
            aotx_mirror_put(to->tool_name, AOTX_MIRROR_NAME_BYTES, "");
            aotx_mirror_put(to->argument, AOTX_MIRROR_TEXT_BYTES, "");
            continue;
        }
        const aotx_request *slot = &aotx_requests.slot[at];
        to->request = slot->request;
        to->agent = slot->agent;
        to->tool = slot->entry;
        aotx_mirror_put(to->tool_name, AOTX_MIRROR_NAME_BYTES,
                        aotx_cli_tool_name(slot->entry));
        unsigned int length = slot->arg_len;
        if (length > AOTX_TOOL_ARG_BYTES) {
            length = AOTX_TOOL_ARG_BYTES;
        }
        if (length > AOTX_MIRROR_TEXT_BYTES - 1u) {
            length = AOTX_MIRROR_TEXT_BYTES - 1u;
        }
        for (unsigned int i = 0u; i < length; ++i) {
            to->argument[i] = slot->arg[i];
        }
        for (unsigned int i = length; i < AOTX_MIRROR_TEXT_BYTES; ++i) {
            to->argument[i] = '\0';
        }
    }
}

/* One row for each model role. The device holds the role and the block type of the
 * weights, and the store on disk holds the file names. */
static __device__ void aotx_mirror_write_models(aotx_mirror_tables *tables)
{
    for (unsigned int role = threadIdx.x; role < AOTX_MIRROR_MODEL_ROWS;
         role += blockDim.x) {
        aotx_mirror_model_row *to = &tables->model[role];
        int in = (role < AOTX_MODEL_ROLES) && (aotx_model[role].layers != 0u);
        to->role = role;
        to->quant = (role < AOTX_MODEL_ROLES) ? aotx_model[role].weight_type : 0u;
        to->resident = (in != 0) ? 1u : 0u;
        aotx_mirror_put(to->name, AOTX_MIRROR_TEXT_BYTES,
                        (in != 0) ? aotx_say_role_name(role) : "");
    }
}

/* One row for each catalog entry, with the reason of a refusal. */
static __device__ void aotx_mirror_write_modules(aotx_mirror_tables *tables)
{
    for (unsigned int row = threadIdx.x; row < AOTX_MIRROR_MODULE_ROWS;
         row += blockDim.x) {
        aotx_mirror_module_row *to = &tables->module[row];
        const aotx_catalog_entry *entry = (row < AOTX_MODULE_SLOTS)
                                        ? &aotx_catalog.entry[row] : 0;
        if (entry == 0 || entry->state == AOTX_CATALOG_FREE) {
            to->kind = 0u;
            to->state = AOTX_CATALOG_FREE;
            aotx_mirror_put(to->name, AOTX_MIRROR_TEXT_BYTES, "");
            aotx_mirror_put(to->reason, AOTX_MIRROR_TEXT_BYTES, "");
            continue;
        }
        to->kind = entry->kind;
        to->state = entry->state;
        aotx_mirror_put(to->name, AOTX_MIRROR_TEXT_BYTES, entry->name);
        aotx_mirror_put(to->reason, AOTX_MIRROR_TEXT_BYTES,
                        (entry->state == AOTX_CATALOG_REFUSED)
                        ? aotx_catalog_why_name(entry->why) : "");
    }
}

/* One row for each number setting, in the order of the key list. */
static __device__ void aotx_mirror_write_settings(aotx_mirror_tables *tables)
{
    for (unsigned int row = threadIdx.x; row < AOTX_MIRROR_SETTING_ROWS;
         row += blockDim.x) {
        aotx_mirror_setting_row *to = &tables->setting[row];
        if (row >= (unsigned int)AOTX_SETTING_NUMBER_COUNT) {
            to->value = 0;
            to->scale = 1u;
            to->effect = 0u;
            aotx_mirror_put(to->key, AOTX_MIRROR_TEXT_BYTES, "");
            continue;
        }
        to->value = aotx_setting_value(row);
        to->scale = (unsigned int)aotx_setting_scale(row);
        to->effect = aotx_setting_effect(row);
        aotx_mirror_put(to->key, AOTX_MIRROR_TEXT_BYTES, aotx_setting_name(row));
    }
}

__global__ void aotx_ui_mirror(void)
{
    __shared__ unsigned long long start_ns;
    if (aotx_mirror.slot == 0) {
        return;
    }
    const unsigned long long frame = aotx_mirror.frame + 1ull;
    const unsigned int which =
        (unsigned int)((frame - 1ull) % (unsigned long long)AOTX_MIRROR_SLOTS);
    aotx_mirror_snapshot *at =
        (aotx_mirror_snapshot *)(aotx_mirror.slot + which * aotx_mirror.slot_bytes);
    /* The tables go into every slot at the start of each run of frames. A reader that
     * takes either slot therefore finds tables of the same age. */
    const int tables = (((frame - 1ull) % (unsigned long long)AOTX_MIRROR_TABLES_EVERY)
                        < (unsigned long long)AOTX_MIRROR_SLOTS) ? 1 : 0;

    /* The zero says the slot is being written. The fence puts it in front of the bytes. */
    if (threadIdx.x == 0u) {
        start_ns = aotx_time_globaltimer();
        aotx_seam_release_sys(&at->head.sequence, 0ull);
        __threadfence_system();
    }
    __syncthreads();

    if (threadIdx.x == 0u) {
        aotx_mirror_write_head(&at->head,
                               (tables != 0) ? frame : aotx_mirror.tables[which]);
    }
    aotx_mirror_write_cells(at->cell);
    if (tables != 0) {
        aotx_mirror_write_agents(&at->tables);
        aotx_mirror_write_requests(&at->tables);
        aotx_mirror_write_models(&at->tables);
        aotx_mirror_write_modules(&at->tables);
        aotx_mirror_write_settings(&at->tables);
    }

    /* The barrier orders every write of the block before the release store of the frame. */
    __syncthreads();
    if (threadIdx.x == 0u) {
        __threadfence_system();
        aotx_seam_release_sys(&at->head.sequence, frame);
        aotx_mirror.frame = frame;
        if (tables != 0) {
            aotx_mirror.tables[which] = frame;
        }
        aotx_mirror.node_ns += aotx_time_globaltimer() - start_ns;
    }
}
