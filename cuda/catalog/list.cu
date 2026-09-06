/* Purpose: Build the tool list and the skill list of a role from the catalog.
 * Owns: Nothing; the caller holds the prompt table the lists go in.
 * Launch shape: Device functions; the agent step calls them once for each prompt.
 * Lifetime: Each call.
 *
 * The list of one prompt is built from the entries the mask of the role allows, in the
 * order of the catalog. A tool installed in one tick is therefore in the prompt of the
 * tick that follows, with no host in the path. The block takes AOTX_CATALOG_LIST_BYTES at
 * the most: a role that allows more gets the first that fit, and the cut is counted. */
#include "agent/overlays.cuh"
#include "catalog/catalog.cuh"
#include "tool/tool.cuh"
#include "model/call_format.cuh"

/* Add a text that ends with a zero byte to a prompt. */
__device__ __forceinline__ static unsigned int aotx_catalog_put(unsigned char *out,
                                                                unsigned int at,
                                                                const char *text)
{
    unsigned int length = 0u;
    while (text[length] != '\0') {
        length += 1u;
    }
    if (at > AOTX_SAY_BYTES || length > AOTX_SAY_BYTES - at) {
        return AOTX_SAY_BYTES + 1u;
    }
    for (unsigned int i = 0u; i < length; ++i) {
        out[at] = (unsigned char)text[i];
        at += 1u;
    }
    return at;
}

/* Add a run of the arena to a prompt. */
__device__ __forceinline__ static unsigned int aotx_catalog_put_run(unsigned char *out,
                                                                    unsigned int at,
                                                                    aotx_catalog_run run)
{
    if (at > AOTX_SAY_BYTES || run.length > AOTX_SAY_BYTES - at) {
        return AOTX_SAY_BYTES + 1u;
    }
    for (unsigned int i = 0u; i < run.length; ++i) {
        out[at] = aotx_catalog_arena[run.at + i];
        at += 1u;
    }
    return at;
}

/* Add a run of the arena as the content of a JSON string. The three bytes the schema does
 * not take in a string get their escape. */
__device__ __forceinline__ static unsigned int aotx_catalog_put_json(unsigned char *out,
                                                                     unsigned int at,
                                                                     aotx_catalog_run run)
{
    unsigned int need = 0u;
    for (unsigned int i = 0u; i < run.length; ++i) {
        unsigned char byte = aotx_catalog_arena[run.at + i];
        if (byte == (unsigned char)'"' || byte == (unsigned char)'\\'
            || byte == (unsigned char)'\n') {
            need += 2u;
        } else if (byte >= 0x20u) {
            need += 1u;
        }
    }
    if (at > AOTX_SAY_BYTES || need > AOTX_SAY_BYTES - at) {
        return AOTX_SAY_BYTES + 1u;
    }
    for (unsigned int i = 0u; i < run.length; ++i) {
        unsigned char byte = aotx_catalog_arena[run.at + i];
        if (byte == (unsigned char)'"' || byte == (unsigned char)'\\') {
            out[at++] = (unsigned char)'\\';
            out[at++] = byte;
        } else if (byte == (unsigned char)'\n') {
            out[at++] = (unsigned char)'\\';
            out[at++] = (unsigned char)'n';
        } else if (byte >= 0x20u) {
            out[at++] = byte;
        }
    }
    return at;
}

/* Add the name of an entry to a prompt. */
__device__ __forceinline__ static unsigned int aotx_catalog_put_name(unsigned char *out,
                                                                     unsigned int at,
                                                                     const aotx_catalog_entry *row)
{
    if (at > AOTX_SAY_BYTES || row->name_len > AOTX_SAY_BYTES - at) {
        return AOTX_SAY_BYTES + 1u;
    }
    for (unsigned int i = 0u; i < row->name_len; ++i) {
        out[at] = (unsigned char)row->name[i];
        at += 1u;
    }
    return at;
}

/* Write one tool as the JSON object the chat template of the model file gives it. The
 * parameters object comes from the argument keys, which are string values. */
__device__ __forceinline__ static unsigned int aotx_catalog_one_tool(unsigned char *out,
                                                                     unsigned int at,
                                                                     const aotx_catalog_entry *row)
{
    at = aotx_catalog_put(out, at, "{\"name\": \"");
    at = aotx_catalog_put_name(out, at, row);
    at = aotx_catalog_put(out, at, "\", \"description\": \"");
    at = aotx_catalog_put_json(out, at, row->description);
    at = aotx_catalog_put(out, at, "\", \"parameters\": {\"type\": \"object\", "
                                   "\"properties\": {");
    for (unsigned int k = 0u; k < row->tool.arguments; ++k) {
        if (k != 0u) {
            at = aotx_catalog_put(out, at, ", ");
        }
        at = aotx_catalog_put(out, at, "\"");
        at = aotx_catalog_put_run(out, at, row->tool.key[k]);
        at = aotx_catalog_put(out, at, "\": {\"type\": \"string\"}");
    }
    at = aotx_catalog_put(out, at, "}, \"required\": [");
    for (unsigned int k = 0u; k < row->tool.arguments; ++k) {
        if (k != 0u) {
            at = aotx_catalog_put(out, at, ", ");
        }
        at = aotx_catalog_put(out, at, "\"");
        at = aotx_catalog_put_run(out, at, row->tool.key[k]);
        at = aotx_catalog_put(out, at, "\"");
    }
    return aotx_catalog_put(out, at, "]}}\n");
}

__device__ unsigned int aotx_catalog_tool_list(unsigned char *out, unsigned int at,
                                               unsigned int role)
{
    const aotx_call_format *format = aotx_call_format_active();
    if (format->kind == AOTX_CALL_NONE || format->kind >= AOTX_CALL_FORMAT_KINDS) return at;
    unsigned int fixed = format->length[AOTX_CALL_TOOLS_HEAD]
                       + format->length[AOTX_CALL_TOOLS_TAIL]
                       + format->length[AOTX_CALL_INSTRUCTION];
    if (fixed > AOTX_CATALOG_LIST_BYTES) return AOTX_SAY_BYTES + 1u;
    unsigned int reserve = format->length[AOTX_CALL_TOOLS_TAIL]
                         + format->length[AOTX_CALL_INSTRUCTION];
    unsigned int start = at;
    unsigned int cut = 0u;
    at = aotx_call_format_put(out, at, AOTX_SAY_BYTES, format, AOTX_CALL_TOOLS_HEAD);
    if (format->kind == AOTX_CALL_LLAMA_JSON) {
        at = aotx_call_format_put(out, at, AOTX_SAY_BYTES, format, AOTX_CALL_INSTRUCTION);
        reserve -= format->length[AOTX_CALL_INSTRUCTION];
    }
    const unsigned int *mask = (role < AOTX_MODULE_SLOTS)
                             ? aotx_catalog.entry[role].role.tools : 0;
    if (mask != 0) {
        for (unsigned int i = 0u; i < AOTX_MODULE_SLOTS; ++i) {
            if (aotx_tool_available(i) == 0
                || aotx_catalog_mask_has(mask, i) == 0) {
                continue;
            }
            /* The block takes the bound at the most. A tool that would cross it is not
             * written and the cut is counted, so the prompt keeps its shape. */
            unsigned int again = aotx_catalog_one_tool(out, at, &aotx_catalog.entry[i]);
            if (again > AOTX_SAY_BYTES || again - start > AOTX_CATALOG_LIST_BYTES - reserve) {
                cut += 1u;
                continue;
            }
            at = again;
        }
    }
    at = aotx_call_format_put(out, at, AOTX_SAY_BYTES, format, AOTX_CALL_TOOLS_TAIL);
    reserve -= format->length[AOTX_CALL_TOOLS_TAIL];

    /* The skills the catalog holds, with the name and the description of each one. The
     * model reads the list and asks for a body with skill_use. */
    unsigned int said = 0u;
    for (unsigned int i = 0u; i < AOTX_MODULE_SLOTS; ++i) {
        if (aotx_catalog_is(i, AOTX_MODULE_SKILL) == 0) {
            continue;
        }
        const aotx_catalog_entry *row = &aotx_catalog.entry[i];
        unsigned int again = (said == 0u)
                           ? aotx_catalog_put(out, at, AOTX_OVERLAY_SKILLS_HEAD) : at;
        again = aotx_catalog_put(out, again, "- ");
        again = aotx_catalog_put_name(out, again, row);
        again = aotx_catalog_put(out, again, ": ");
        again = aotx_catalog_put_run(out, again, row->description);
        again = aotx_catalog_put(out, again, "\n");
        if (again > AOTX_SAY_BYTES || again - start > AOTX_CATALOG_LIST_BYTES - reserve) {
            cut += 1u;
            continue;
        }
        at = again;
        said = 1u;
    }
    if (cut != 0u) {
        atomicAdd(&aotx_catalog.count.list_cut, cut);
    }
    if (format->kind != AOTX_CALL_LLAMA_JSON)
        at = aotx_call_format_put(out, at, AOTX_SAY_BYTES, format, AOTX_CALL_INSTRUCTION);
    return at;
}

__device__ unsigned int aotx_catalog_skill_bodies(unsigned char *out, unsigned int at,
                                                  unsigned int role)
{
    if (role >= AOTX_MODULE_SLOTS) {
        return at;
    }
    /* The turn keeps room for its own text: the task or the message, the result of a tool
     * and the wrap. A body that does not fit whole in the room that is left is not
     * written. The cut is counted, so no prompt holds half a skill. */
    unsigned int keep = 2u * (unsigned int)AOTX_TASK_TEXT_BYTES + 256u;
    unsigned int room = (AOTX_SAY_BYTES > keep) ? (AOTX_SAY_BYTES - keep) : 0u;
    unsigned int cut = 0u;
    const aotx_catalog_role *row = &aotx_catalog.entry[role].role;
    for (unsigned int s = 0u; s < row->skills && s < AOTX_CATALOG_ROLE_SKILLS; ++s) {
        unsigned int which = row->skill[s];
        if (aotx_catalog_is(which, AOTX_MODULE_SKILL) == 0) {
            continue;
        }
        if (at + 2u + aotx_catalog.entry[which].body.length > room) {
            cut += 1u;
            continue;
        }
        at = aotx_catalog_put(out, at, "\n\n");
        at = aotx_catalog_put_run(out, at, aotx_catalog.entry[which].body);
    }
    if (cut != 0u) {
        atomicAdd(&aotx_catalog.count.list_cut, cut);
    }
    return at;
}
