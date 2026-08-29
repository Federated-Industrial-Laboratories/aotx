/* Purpose: Import one module, fill a synthetic batch for it and judge what it wrote.
 * Owns: The verdict of the check and the canary of the untaken rows.
 * Launch shape: One thread for each request row; the import takes one thread.
 * Lifetime: One run of the check program.
 *
 * The check imports the manifest through the reader the apply uses, so a manifest that
 * loads here loads in a run. The batch it fills is the batch of the tick, so a module that
 * passes the check reads the same bytes in the tick graph. */
#include "catalog/check.cuh"
#include "tool/tool_state.cuh"

__device__ aotx_check_verdict aotx_check_out;

/* The pattern the check writes in an untaken output row. A module that writes such a row
 * changes the pattern, and the judge counts it. */
#define AOTX_CHECK_CANARY 0xA5A5A5A5u
#define AOTX_CHECK_BYTE   ((char)0x5a)

__global__ void aotx_check_import(const unsigned char *bytes, unsigned int length,
                                  unsigned int kind, const char *name,
                                  const char *path)
{
    if (blockIdx.x != 0u || threadIdx.x != 0u) {
        return;
    }
    aotx_import_head head;
    head.import = 1u;
    head.part = 0u;
    head.kind = kind;
    head.files = 1u;
    head.file_bytes[0] = length;
    head.file_bytes[1] = 0u;
    for (unsigned int i = 0u; i < 32u; ++i) {
        head.digest[i] = 0u;
    }
    for (unsigned int i = 0u; i < (unsigned int)AOTX_IMPORT_NAME_BYTES; ++i) {
        head.name[i] = name[i];
    }
    for (unsigned int i = 0u; i < (unsigned int)AOTX_IMPORT_PATH_BYTES; ++i) {
        head.path[i] = path[i];
    }
    if (aotx_catalog_apply(AOTX_REC_IMPORT, &head, (unsigned int)sizeof head, 1ull) != 0) {
        return;
    }
    aotx_import_part part;
    unsigned int number = 1u;
    for (unsigned int at = 0u; at < length; at += (unsigned int)AOTX_IMPORT_TEXT_BYTES) {
        unsigned int span = length - at;
        if (span > (unsigned int)AOTX_IMPORT_TEXT_BYTES) {
            span = (unsigned int)AOTX_IMPORT_TEXT_BYTES;
        }
        part.import = 1u;
        part.part = number;
        part.file = 0u;
        part.offset = at;
        part.length = span;
        for (unsigned int i = 0u; i < (unsigned int)AOTX_IMPORT_TEXT_BYTES; ++i) {
            part.text[i] = (i < span) ? (char)bytes[at + i] : '\0';
        }
        number += 1u;
        aotx_catalog_apply(AOTX_REC_IMPORT, &part, (unsigned int)sizeof part, 1ull);
    }
}

__global__ void aotx_check_digest(unsigned int entry, const unsigned char *digest)
{
    if (blockIdx.x != 0u || threadIdx.x != 0u || entry >= AOTX_MODULE_SLOTS) {
        return;
    }
    for (unsigned int i = 0u; i < 32u; ++i) {
        aotx_catalog.entry[entry].digest[i] = digest[i];
    }
    aotx_catalog_anchor();
}

__global__ void aotx_check_report(unsigned int entry, aotx_check_entry *out)
{
    if (blockIdx.x != 0u || threadIdx.x != 0u || out == 0) {
        return;
    }
    out->state = AOTX_CATALOG_FREE;
    out->why = AOTX_CATALOG_WHY_NONE;
    out->figure = 0u;
    out->side = AOTX_CATALOG_SIDE_DEVICE;
    out->arguments = 0u;
    out->timeout = 0u;
    out->deadline = 0u;
    out->authorize = 0u;
    out->example_len = 0u;
    if (entry >= AOTX_MODULE_SLOTS) {
        return;
    }
    const aotx_catalog_entry *row = &aotx_catalog.entry[entry];
    out->state = row->state;
    out->why = row->why;
    out->figure = row->figure;
    out->side = row->tool.side;
    out->arguments = row->tool.arguments;
    out->timeout = row->tool.timeout;
    out->deadline = row->tool.deadline;
    out->authorize = row->tool.authorize;
    for (unsigned int i = 0u; i < AOTX_CHECK_TEXT_BYTES; ++i) {
        out->reason[i] = '\0';
        out->example[i] = '\0';
    }
    unsigned int at = 0u;
    const char *why = aotx_catalog_why_name(row->why);
    while (why[at] != '\0' && at + 1u < AOTX_CHECK_TEXT_BYTES) {
        out->reason[at] = why[at];
        at += 1u;
    }
    unsigned int span = row->tool.example.length;
    if (span >= AOTX_CHECK_TEXT_BYTES) {
        span = AOTX_CHECK_TEXT_BYTES - 1u;
    }
    for (unsigned int i = 0u; i < span; ++i) {
        out->example[i] = (char)aotx_catalog_arena[row->tool.example.at + i];
    }
    out->example_len = span;
    /* The program of a host tool, so the check runs the file the manifest names. */
    span = row->tool.program.length;
    if (span >= AOTX_CHECK_TEXT_BYTES) {
        span = AOTX_CHECK_TEXT_BYTES - 1u;
    }
    for (unsigned int i = 0u; i < AOTX_CHECK_TEXT_BYTES; ++i) {
        out->program[i] = (i < span)
                        ? (char)aotx_catalog_arena[row->tool.program.at + i] : '\0';
    }
}

/* Write the value of one key of a row from the example line. The content of each row is
 * distinct, and the row number makes it so. That number stands in the value of the second
 * key, when the tool has a second key. It stands at the end of the value of the first key,
 * when the tool has one key alone. */
__device__ __forceinline__ static void aotx_check_value(aotx_tool_argument *arg,
                                                        const char *example,
                                                        unsigned int length,
                                                        aotx_catalog_run key,
                                                        unsigned int row, int number)
{
    unsigned int span = (key.length < AOTX_TOOL_KEY_BYTES) ? key.length
                                                           : (AOTX_TOOL_KEY_BYTES - 1u);
    for (unsigned int i = 0u; i < AOTX_TOOL_KEY_BYTES; ++i) {
        arg->key[i] = (i < span) ? (char)aotx_catalog_arena[key.at + i] : '\0';
    }
    unsigned int at = 0u;
    unsigned int made = 0u;
    unsigned int held = 0u;
    if (aotx_tool_argument_of(example, length, (const char *)aotx_catalog_arena + key.at,
                              key.length, &at, &made) != 0) {
        held = (made > AOTX_TOOL_VALUE_BYTES) ? AOTX_TOOL_VALUE_BYTES : made;
        for (unsigned int i = 0u; i < held; ++i) {
            arg->value[i] = example[at + i];
        }
    }
    if (number != 0) {
        if (held == 0u || held + 8u < AOTX_TOOL_VALUE_BYTES) {
            if (held != 0u) {
                arg->value[held] = ' ';
                held += 1u;
            }
            held += aotx_text_utoa((unsigned long long)row, arg->value + held,
                                   AOTX_TOOL_VALUE_BYTES - held);
        }
    }
    arg->length = held;
}

__global__ void aotx_check_fill(unsigned int node, unsigned int entry, unsigned int rows)
{
    unsigned int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= AOTX_SLOTS || node >= (unsigned int)AOTX_TOOL_MODULES
        || entry >= AOTX_MODULE_SLOTS) {
        return;
    }
    aotx_tool_module_state *state = &aotx_tool_modules;
    aotx_tool_row *row = &state->row[(unsigned long long)node * AOTX_SLOTS + slot];
    char *text = state->text + (unsigned long long)slot * AOTX_TOOL_RESULT_BYTES;
    if (slot == 0u) {
        aotx_check_out.rows = AOTX_SLOTS;
        aotx_check_out.taken = rows;
        aotx_check_out.done = 0u;
        aotx_check_out.status_bad = 0u;
        aotx_check_out.over = 0u;
        aotx_check_out.untaken = 0u;
        aotx_check_out.empty = 0u;
        aotx_check_out.longest = 0u;
    }
    if (slot >= rows) {
        /* An untaken row keeps the canary. A module that writes it changes the pattern. */
        row->take = 0u;
        state->head[slot].status = AOTX_CHECK_CANARY;
        state->head[slot].length = AOTX_CHECK_CANARY;
        state->head[slot].done = 0u;
        state->head[slot].reserved = AOTX_CHECK_CANARY;
        text[0] = AOTX_CHECK_BYTE;
        return;
    }
    const aotx_catalog_tool *tool = &aotx_catalog.entry[entry].tool;
    const char *example = (const char *)aotx_catalog_arena
                        + aotx_catalog.entry[entry].tool.example.at;
    unsigned int length = aotx_catalog.entry[entry].tool.example.length;
    unsigned int keys = (tool->arguments < AOTX_TOOL_ARGS_MAX) ? tool->arguments
                                                               : AOTX_TOOL_ARGS_MAX;
    row->take = 1u;
    row->request = slot + 1u;
    row->agent = slot;
    row->arguments = keys;
    row->seed = 0x9e3779b97f4a7c15ull ^ (unsigned long long)slot;
    for (unsigned int k = 0u; k < AOTX_TOOL_ARGS_MAX; ++k) {
        aotx_catalog_run key;
        key.at = 0u;
        key.length = 0u;
        if (k < keys) {
            key = tool->key[k];
        }
        int number = (keys > 1u) ? ((k == 1u) ? 1 : 0) : ((k == 0u) ? 1 : 0);
        aotx_check_value(&row->argument[k], example, length, key, slot, number);
    }
    state->head[slot].status = AOTX_TOOL_STATUS_OK;
    state->head[slot].length = 0u;
    state->head[slot].done = 0u;
    state->head[slot].reserved = 0u;
    text[0] = AOTX_CHECK_BYTE;
}

__global__ void aotx_check_judge(unsigned int rows)
{
    unsigned int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= AOTX_SLOTS) {
        return;
    }
    aotx_tool_module_state *state = &aotx_tool_modules;
    const aotx_tool_output_row *head = &state->head[slot];
    const char *text = state->text + (unsigned long long)slot * AOTX_TOOL_RESULT_BYTES;
    if (slot >= rows) {
        if (head->done != 0u || head->status != AOTX_CHECK_CANARY
            || head->length != AOTX_CHECK_CANARY || text[0] != AOTX_CHECK_BYTE) {
            atomicAdd(&aotx_check_out.untaken, 1u);
        }
        return;
    }
    if (head->done == 0u) {
        return;
    }
    atomicAdd(&aotx_check_out.done, 1u);
    if (head->status != AOTX_TOOL_STATUS_OK && head->status != AOTX_TOOL_STATUS_ERROR) {
        atomicAdd(&aotx_check_out.status_bad, 1u);
    }
    if (head->length > (unsigned int)AOTX_TOOL_RESULT_BYTES) {
        atomicAdd(&aotx_check_out.over, 1u);
    } else if (head->length == 0u) {
        atomicAdd(&aotx_check_out.empty, 1u);
    }
    atomicMax(&aotx_check_out.longest, head->length);
}
