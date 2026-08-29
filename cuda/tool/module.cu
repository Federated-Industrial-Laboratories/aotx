/* Purpose: Fill the batch of every device tool module and take the answers back.
 * Owns: The module tables of the tick and the plan the host glue reads.
 * Launch shape: One thread for each request slot; the plan kernels take one thread.
 * Lifetime: The whole run.
 *
 * The fill gives a module the rows of the requests that name it. A module reads the rows
 * whose take is 1 and writes the output rows of those and no other. The tool step reads
 * the done word of each row it gave and hands the text to the agent. */
#include "bus/bus.cuh"
#include "seam/seam.cuh"
#include "catalog/console.cuh"
#include "tool/module.cuh"
#include "tool/tool_state.cuh"

__device__ aotx_tool_module_state aotx_tool_modules;
__device__ aotx_tool_module_plan aotx_tool_module_list;

/* The line a refusal builds. One thread makes a refusal, so one line is enough. */
static __device__ aotx_cli_out aotx_tool_module_line;

/* Copy one run of the arena into a field that ends with a zero byte. */
__device__ __forceinline__ static void aotx_tool_module_copy(char *out, unsigned int max,
                                                             aotx_catalog_run run)
{
    unsigned int length = (run.length < max) ? run.length : (max - 1u);
    for (unsigned int i = 0u; i < max; ++i) {
        out[i] = (i < length) ? (char)aotx_catalog_arena[run.at + i] : '\0';
    }
}

/* Write the kernel name of a module: the name the manifest gives, or aotx_tool_ and the
 * name of the module. */
__device__ __forceinline__ static void aotx_tool_module_kernel(char *out, unsigned int max,
                                                               const aotx_catalog_entry *row)
{
    if (row->tool.entry.length != 0u) {
        aotx_tool_module_copy(out, max, row->tool.entry);
        return;
    }
    const char *head = "aotx_tool_";
    unsigned int at = 0u;
    while (head[at] != '\0' && at + 1u < max) {
        out[at] = head[at];
        at += 1u;
    }
    for (unsigned int i = 0u; i < row->name_len && at + 1u < max; ++i) {
        out[at] = row->name[i];
        at += 1u;
    }
    while (at < max) {
        out[at] = '\0';
        at += 1u;
    }
}

__global__ void aotx_tool_module_scan(void)
{
    if (blockIdx.x != 0u || threadIdx.x != 0u) {
        return;
    }
    aotx_tool_module_plan *plan = &aotx_tool_module_list;
    plan->rows = 0u;
    plan->gen = aotx_catalog.count.device_gen;
    for (unsigned int i = 0u; i < AOTX_MODULE_SLOTS; ++i) {
        if (aotx_catalog_is_module(i) == 0) {
            continue;
        }
        if (plan->rows >= (unsigned int)AOTX_TOOL_MODULES) {
            return;
        }
        const aotx_catalog_entry *row = &aotx_catalog.entry[i];
        aotx_tool_module_row *out = &plan->row[plan->rows];
        out->entry = i;
        out->import = row->import;
        for (unsigned int b = 0u; b < AOTX_CATALOG_NAME_BYTES; ++b) {
            out->name[b] = (b < row->name_len) ? row->name[b] : '\0';
        }
        for (unsigned int b = 0u; b < (unsigned int)AOTX_IMPORT_PATH_BYTES; ++b) {
            out->path[b] = row->path[b];
        }
        out->path[AOTX_IMPORT_PATH_BYTES - 1u] = '\0';
        aotx_tool_module_copy(out->file, AOTX_TOOL_MODULE_FILE, row->tool.module);
        aotx_tool_module_kernel(out->kernel, AOTX_CATALOG_NAME_BYTES, row);
        for (unsigned int b = 0u; b < 32u; ++b) {
            out->digest[b] = row->digest[b];
        }
        plan->rows += 1u;
    }
}

__global__ void aotx_tool_module_bind(unsigned int nodes)
{
    if (blockIdx.x != 0u || threadIdx.x != 0u) {
        return;
    }
    aotx_tool_module_state *state = &aotx_tool_modules;
    unsigned int held = (nodes < (unsigned int)AOTX_TOOL_MODULES)
                      ? nodes : (unsigned int)AOTX_TOOL_MODULES;
    state->out.abi = AOTX_TOOL_ABI;
    state->out.rows = AOTX_SLOTS;
    state->out.head = state->head;
    state->out.text = state->text;
    for (unsigned int m = 0u; m < (unsigned int)AOTX_TOOL_MODULES; ++m) {
        aotx_tool_batch *batch = &state->batch[m];
        batch->abi = AOTX_TOOL_ABI;
        batch->rows = AOTX_SLOTS;
        batch->tick = 0ull;
        batch->boot_id = 0ull;
        batch->scratch_bytes = (unsigned long long)AOTX_TOOL_SCRATCH_BYTES;
        batch->out_bytes = (unsigned long long)AOTX_TOOL_RESULT_BYTES;
        batch->row = state->row + (unsigned long long)m * AOTX_SLOTS;
        batch->scratch = state->scratch;
        state->entry[m] = (m < held) ? aotx_tool_module_list.row[m].entry
                                     : AOTX_CATALOG_NO_ENTRY;
    }
    state->nodes = held;
    state->gen = aotx_tool_module_list.gen;
}

__global__ void aotx_tool_module_note(unsigned int before, unsigned int after,
                                      unsigned int took_us)
{
    if (blockIdx.x != 0u || threadIdx.x != 0u) {
        return;
    }
    aotx_tool_module_state *state = &aotx_tool_modules;
    aotx_cli_out *out = &aotx_tool_module_line;
    state->captures += 1u;
    state->before = before;
    state->after = after;
    state->took_us = took_us;
    aotx_cli_clear(out);
    aotx_cli_say(out, "tools: the tick graph was captured again at tick ");
    aotx_cli_num(out, aotx_time_tick);
    aotx_cli_say(out, ": ");
    aotx_cli_num(out, (unsigned long long)before);
    aotx_cli_say(out, " nodes before, ");
    aotx_cli_num(out, (unsigned long long)after);
    aotx_cli_say(out, " after, ");
    aotx_cli_num(out, (unsigned long long)took_us);
    aotx_cli_say(out, " microseconds");
    aotx_console_write(out->text, out->at);
    aotx_bus_append(AOTX_WRITER_SYSTEM, AOTX_BUS_NOTE, 0u, out->text, out->at, 0ull, 0ull,
                    0.0f, aotx_time_tick);
    aotx_cli_clear(out);
}

__global__ void aotx_tool_module_refuse(unsigned int entry, unsigned int why)
{
    if (blockIdx.x != 0u || threadIdx.x != 0u || entry >= AOTX_MODULE_SLOTS) {
        return;
    }
    aotx_catalog_entry *row = &aotx_catalog.entry[entry];
    aotx_cli_out *out = &aotx_tool_module_line;
    row->state = AOTX_CATALOG_REFUSED;
    row->why = why;
    row->figure = 0u;
    aotx_catalog.count.refused += 1u;
    aotx_catalog_anchor();
    aotx_cli_clear(out);
    aotx_cli_say(out, "import: ");
    aotx_cli_add(out, row->name, row->name_len);
    aotx_cli_say(out, " refused: ");
    aotx_cli_say(out, aotx_catalog_why_name(why));
    aotx_console_write(out->text, out->at);
    aotx_bus_append(AOTX_WRITER_SYSTEM, AOTX_BUS_NOTE, 0u, out->text, out->at, 0ull, 0ull,
                    0.0f, aotx_time_tick);
    aotx_cli_clear(out);
}

/* Put the arguments of a request in the row of a module. The request holds them as one
 * line of key=value pairs, and the row holds one key and one value for each. */
__device__ __noinline__ static void aotx_tool_module_arguments(aotx_tool_row *row,
                                                               const aotx_request *hold)
{
    const aotx_catalog_tool *tool = &aotx_catalog.entry[hold->entry].tool;
    unsigned int keys = (tool->arguments < AOTX_TOOL_ARGS_MAX) ? tool->arguments
                                                               : AOTX_TOOL_ARGS_MAX;
    row->arguments = keys;
    for (unsigned int k = 0u; k < AOTX_TOOL_ARGS_MAX; ++k) {
        aotx_tool_argument *arg = &row->argument[k];
        aotx_catalog_run key;
        key.at = 0u;
        key.length = 0u;
        if (k < keys) {
            key = tool->key[k];
        }
        unsigned int span = (key.length < AOTX_TOOL_KEY_BYTES) ? key.length
                                                               : (AOTX_TOOL_KEY_BYTES - 1u);
        for (unsigned int i = 0u; i < AOTX_TOOL_KEY_BYTES; ++i) {
            arg->key[i] = (i < span) ? (char)aotx_catalog_arena[key.at + i] : '\0';
        }
        unsigned int at = 0u;
        unsigned int length = 0u;
        if (k < keys && aotx_tool_argument_of(hold->arg, hold->arg_len,
                                              (const char *)aotx_catalog_arena + key.at,
                                              key.length, &at, &length) != 0) {
            if (length > AOTX_TOOL_VALUE_BYTES) {
                length = AOTX_TOOL_VALUE_BYTES;
            }
            for (unsigned int i = 0u; i < length; ++i) {
                arg->value[i] = hold->arg[at + i];
            }
        }
        arg->length = length;
    }
}

__device__ void aotx_tool_module_fill(unsigned int slot)
{
    if (slot >= AOTX_SLOTS) {
        return;
    }
    aotx_tool_module_state *state = &aotx_tool_modules;
    /* The tick and the boot id of the batch belong to this tick. One thread writes them
     * for every node, because the nodes of one tick read one tick. */
    if (slot == 0u) {
        for (unsigned int m = 0u; m < (unsigned int)AOTX_TOOL_MODULES; ++m) {
            state->batch[m].tick = aotx_time_tick;
            state->batch[m].boot_id = aotx_seam.boot_id;
        }
    }
    const aotx_request *hold = &aotx_requests.slot[slot];
    /* A request runs on a module when the tool of its entry is a device module. The reply
     * must not be in hand, and the operator must not have stopped it. */
    unsigned int live = (hold->request != 0u && aotx_tool_done[slot] == 0u
                         && hold->auth != AOTX_AUTH_PENDING
                         && hold->auth != AOTX_AUTH_REFUSED
                         && aotx_catalog_is_module(hold->entry) != 0) ? 1u : 0u;
    unsigned int mine = live ? aotx_tool_module_node(hold->entry)
                             : (unsigned int)AOTX_TOOL_MODULE_NONE;
    if (mine < state->nodes) {
        /* The output row of a request that runs belongs to that request alone, because one
         * request names one tool. The fill clears the done word before the node runs. */
        state->head[slot].status = AOTX_TOOL_STATUS_OK;
        state->head[slot].length = 0u;
        state->head[slot].done = 0u;
        state->head[slot].reserved = 0u;
        /* The scratch of a row that runs holds no byte of the tick before it. The clear
         * costs the row alone, so a tick with no module call pays nothing. */
        unsigned long long *words = (unsigned long long *)(state->scratch
                                    + (unsigned long long)slot * AOTX_TOOL_SCRATCH_BYTES);
        for (unsigned int i = 0u; i < AOTX_TOOL_SCRATCH_BYTES / 8u; ++i) {
            words[i] = 0ull;
        }
        atomicAdd(&state->took, 1u);
    }
    for (unsigned int m = 0u; m < state->nodes; ++m) {
        aotx_tool_row *row = &state->row[(unsigned long long)m * AOTX_SLOTS + slot];
        if (m != mine) {
            row->take = 0u;
            continue;
        }
        row->take = 1u;
        row->request = hold->request;
        row->agent = hold->agent;
        row->seed = ((unsigned long long)aotx_time_tick << 16) ^ (unsigned long long)slot;
        aotx_tool_module_arguments(row, hold);
    }
}

__device__ int aotx_tool_module_reap(unsigned int slot, aotx_request *hold)
{
    aotx_tool_module_state *state = &aotx_tool_modules;
    if (slot >= AOTX_SLOTS || hold == 0) {
        return 0;
    }
    unsigned int mine = aotx_tool_module_node(hold->entry);
    if (mine >= state->nodes) {
        /* The nodes stand for the catalog the pump captured. A call to a device tool that
         * holds no node of that capture ends with the reason. */
        if (state->gen != aotx_catalog.count.device_gen) {
            return 0;
        }
        hold->status = AOTX_TOOL_ERROR;
        hold->result_len = aotx_tool_put(hold->result, 0u,
                                         "this tool holds no node of the tick graph");
        atomicAdd(&state->bare, 1u);
        return 1;
    }
    if (state->head[slot].done == 0u) {
        return 0;
    }
    unsigned int length = state->head[slot].length;
    if (length > (unsigned int)AOTX_TOOL_RESULT_BYTES) {
        length = (unsigned int)AOTX_TOOL_RESULT_BYTES;
        atomicAdd(&state->over, 1u);
    }
    const char *text = state->text + (unsigned long long)slot * AOTX_TOOL_RESULT_BYTES;
    for (unsigned int i = 0u; i < length; ++i) {
        hold->result[i] = text[i];
    }
    hold->result_len = length;
    hold->status = (state->head[slot].status == AOTX_TOOL_STATUS_OK) ? AOTX_TOOL_OK
                                                                     : AOTX_TOOL_ERROR;
    state->head[slot].done = 0u;
    atomicAdd(&state->gave, 1u);
    return 1;
}
