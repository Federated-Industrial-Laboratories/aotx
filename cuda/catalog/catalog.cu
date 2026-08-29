/* Purpose: Hold the catalog table and the arena, and take the records that change them.
 * Owns: The catalog table, the arena of texts and the free list of arena runs.
 * Launch shape: Device functions; the apply step and the commit node call them.
 * Lifetime: The whole run. */
#include "agent/agent.cuh"
#include "bus/bus.cuh"
#include "catalog/catalog.cuh"
#include "cli/cli.cuh"
#include "tool/tool.cuh"

__device__ aotx_catalog_state aotx_catalog;
__device__ unsigned char aotx_catalog_arena[AOTX_CATALOGUE_BYTES];

/* The line a refusal builds. One thread changes the catalog, so one line is enough and no
 * frame of the kernel holds it. */
static __device__ aotx_cli_out aotx_catalog_out;

__device__ const char *aotx_catalog_why_name(unsigned int why)
{
    switch (why) {
    case AOTX_CATALOG_WHY_NAME:    return "the name takes 1 to 63 bytes of a to z, 0 to 9 "
                                          "and the low line";
    case AOTX_CATALOG_WHY_HEAD:    return "the name of the manifest is not the name of the "
                                          "directory";
    case AOTX_CATALOG_WHY_KIND:    return "the kind of the manifest is not the kind of the "
                                          "import";
    case AOTX_CATALOG_WHY_KEY:     return "the manifest holds a key this kind does not take";
    case AOTX_CATALOG_WHY_SHAPE:   return "a line of the manifest is not a key and a value";
    case AOTX_CATALOG_WHY_BODY:    return "the body is longer than the bound";
    case AOTX_CATALOG_WHY_ARGS:    return "the tool names more argument keys than the bound";
    case AOTX_CATALOG_WHY_ARENA:   return "the arena holds no run for the text";
    case AOTX_CATALOG_WHY_TABLE:   return "the catalog holds no free entry";
    case AOTX_CATALOG_WHY_TWICE:   return "an import of that number arrives already";
    case AOTX_CATALOG_WHY_PART:    return "a part names a file or a run the head does not "
                                          "hold";
    case AOTX_CATALOG_WHY_MISSING: return "the manifest gives no kind or no name";
    case AOTX_CATALOG_WHY_VALUE:   return "a value is not one the key takes";
    case AOTX_CATALOG_WHY_SKILLS:  return "the role names more skills than the bound";
    case AOTX_CATALOG_WHY_EMPTY:   return "the kind of this module needs a body and the "
                                          "import carries none";
    default:                       return "no reason";
    }
}

__device__ const char *aotx_catalog_gone_name(unsigned int gone)
{
    switch (gone) {
    case AOTX_CATALOG_GONE_UNKNOWN: return "no module holds that name";
    case AOTX_CATALOG_GONE_ROLE:    return "an agent runs on that role";
    case AOTX_CATALOG_GONE_TOOL:    return "a request of that tool is in flight";
    case AOTX_CATALOG_GONE_BUILT:   return "a built-in tool does not go";
    default:                        return "no reason";
    }
}

/* Write one console line and put the same text on the bus as a note. Every refusal of the
 * catalog takes this path, so the operator and the bus hold the same reason. */
__device__ static void aotx_catalog_report(aotx_cli_out *out, unsigned long long tick)
{
    aotx_console_write(out->text, out->at);
    aotx_bus_append(AOTX_WRITER_SYSTEM, AOTX_BUS_NOTE, 0u, out->text, out->at,
                    0ull, 0ull, 0.0f, tick);
    aotx_cli_clear(out);
}

/* Take a run of the arena by first fit. The return is 0 when the run is in hand. A length
 * of zero gives a run of no length and takes no bytes. */
__device__ static int aotx_catalog_take(unsigned int length, aotx_catalog_run *run)
{
    run->at = 0u;
    run->length = 0u;
    if (length == 0u) {
        return 0;
    }
    for (unsigned int i = 0u; i < aotx_catalog.frees; ++i) {
        aotx_catalog_run *free_run = &aotx_catalog.free_run[i];
        if (free_run->length < length) {
            continue;
        }
        run->at = free_run->at;
        run->length = length;
        free_run->at += length;
        free_run->length -= length;
        if (free_run->length == 0u) {
            for (unsigned int k = i + 1u; k < aotx_catalog.frees; ++k) {
                aotx_catalog.free_run[k - 1u] = aotx_catalog.free_run[k];
            }
            aotx_catalog.frees -= 1u;
        }
        aotx_catalog.used += length;
        return 0;
    }
    return 1;
}

/* Give a run of the arena back. The list stands in the order of the offsets. A run beside
 * another one joins it, so the count of the list does not grow without a bound. */
__device__ static void aotx_catalog_give(aotx_catalog_run run)
{
    if (run.length == 0u) {
        return;
    }
    aotx_catalog.used -= (run.length <= aotx_catalog.used) ? run.length : aotx_catalog.used;
    unsigned int at = 0u;
    while (at < aotx_catalog.frees && aotx_catalog.free_run[at].at < run.at) {
        at += 1u;
    }
    /* The run before this one takes it when the two touch. */
    if (at > 0u) {
        aotx_catalog_run *before = &aotx_catalog.free_run[at - 1u];
        if (before->at + before->length == run.at) {
            before->length += run.length;
            /* The run after may now touch the one that grew. */
            if (at < aotx_catalog.frees
                && before->at + before->length == aotx_catalog.free_run[at].at) {
                before->length += aotx_catalog.free_run[at].length;
                for (unsigned int k = at + 1u; k < aotx_catalog.frees; ++k) {
                    aotx_catalog.free_run[k - 1u] = aotx_catalog.free_run[k];
                }
                aotx_catalog.frees -= 1u;
            }
            return;
        }
    }
    /* The run after this one takes it when the two touch. */
    if (at < aotx_catalog.frees && run.at + run.length == aotx_catalog.free_run[at].at) {
        aotx_catalog.free_run[at].at = run.at;
        aotx_catalog.free_run[at].length += run.length;
        return;
    }
    /* A list that is full loses the bytes of this run. The arena then holds less than the
     * profile names, and the next import that does not fit says so with the figure. */
    if (aotx_catalog.frees >= AOTX_CATALOG_FREE_RUNS) {
        return;
    }
    for (unsigned int k = aotx_catalog.frees; k > at; --k) {
        aotx_catalog.free_run[k] = aotx_catalog.free_run[k - 1u];
    }
    aotx_catalog.free_run[at] = run;
    aotx_catalog.frees += 1u;
}

/* The arena runs of one entry go back to the free list, and the entry keeps its name. */
__device__ void aotx_catalog_release(aotx_catalog_entry *row)
{
    aotx_catalog_give(row->manifest);
    aotx_catalog_give(row->body);
    aotx_catalog_give(row->role.overlay);
    row->manifest.at = 0u;
    row->manifest.length = 0u;
    row->body.at = 0u;
    row->body.length = 0u;
    row->role.overlay.at = 0u;
    row->role.overlay.length = 0u;
    row->description.at = 0u;
    row->description.length = 0u;
    row->version.at = 0u;
    row->version.length = 0u;
}

/* Take an arena run and give it back at once. The caller of an import that the reader
 * refused frees the runs that arrived, because a refused entry keeps its name alone. */
__device__ void aotx_catalog_free_run(aotx_catalog_run run)
{
    aotx_catalog_give(run);
}

__device__ int aotx_catalog_take_run(unsigned int length, aotx_catalog_run *run)
{
    return aotx_catalog_take(length, run);
}

/* Find the entry of the role of a name, whatever the state of the entry. The engine keeps
 * the entry of the conductor and of the verifier. A path of the engine names each one. The
 * console speaks to the conductor. A verifier judges a result. */
__device__ void aotx_catalog_anchor(void)
{
    aotx_catalog.conductor = aotx_catalog_find("conductor", 9u, AOTX_MODULE_ROLE);
    aotx_catalog.verifier = aotx_catalog_find("verifier", 8u, AOTX_MODULE_ROLE);
}

/* The manifest of one built-in tool. A built-in tool goes in the catalog as an entry of
 * the shape an imported tool takes. One path therefore serves the list, the mask and the
 * parser. The text stands in the arena and the reader of the import reads it. */
__device__ __forceinline__ static const char *aotx_catalog_built_text(unsigned int which)
{
    switch (which) {
    case 0u: return "kind: tool\n"
                    "name: memory_recall\n"
                    "version: built in\n"
                    "side: device\n"
                    "arguments: text\n"
                    "authorise: never\n"
                    "description: Find the notes in memory that are nearest to a text.\n";
    case 1u: return "kind: tool\n"
                    "name: memory_write\n"
                    "version: built in\n"
                    "side: device\n"
                    "arguments: provenance,text\n"
                    "authorise: never\n"
                    "description: Put one note in memory with its source. The source is "
                    "one of computed, fetched, recalled or testimony.\n";
    case 2u: return "kind: tool\n"
                    "name: fs_read\n"
                    "version: built in\n"
                    "side: host\n"
                    "arguments: path\n"
                    "authorise: never\n"
                    "description: Read a file below the allowed root. The operator must "
                    "permit this tool.\n";
    default: return "kind: tool\n"
                    "name: skill_use\n"
                    "version: built in\n"
                    "side: device\n"
                    "arguments: name\n"
                    "authorise: never\n"
                    "description: Ask for the text of one skill by its name in the skill "
                    "list.\n";
    }
}

/* The tool number that the tool module knows a built-in tool by. */
__device__ __forceinline__ static unsigned int aotx_catalog_built_tool(unsigned int which)
{
    switch (which) {
    case 0u: return AOTX_TOOL_MEMORY_RECALL;
    case 1u: return AOTX_TOOL_MEMORY_WRITE;
    case 2u: return AOTX_TOOL_FS_READ;
    default: return AOTX_TOOL_SKILL_USE;
    }
}

__device__ void aotx_catalog_built_in(void)
{
    /* A second call changes nothing, because the first entry stands after the first. */
    if (aotx_catalog.entry[0].state != AOTX_CATALOG_FREE) {
        return;
    }
    if (aotx_catalog.frees == 0u && aotx_catalog.used == 0u) {
        aotx_catalog.free_run[0].at = 0u;
        aotx_catalog.free_run[0].length = (unsigned int)AOTX_CATALOGUE_BYTES;
        aotx_catalog.frees = 1u;
    }
    for (unsigned int which = 0u; which < AOTX_CATALOG_BUILT_IN; ++which) {
        const char *text = aotx_catalog_built_text(which);
        unsigned int length = 0u;
        while (text[length] != '\0') {
            length += 1u;
        }
        aotx_catalog_entry *row = &aotx_catalog.entry[which];
        aotx_catalog_run run;
        if (aotx_catalog_take(length, &run) != 0) {
            return;
        }
        for (unsigned int i = 0u; i < length; ++i) {
            aotx_catalog_arena[run.at + i] = (unsigned char)text[i];
        }
        row->manifest = run;
        row->kind = AOTX_MODULE_TOOL;
        unsigned int figure = 0u;
        unsigned int why = aotx_catalog_manifest_read(row, run.at, length,
                                                      AOTX_MODULE_TOOL, &figure);
        if (why != AOTX_CATALOG_WHY_NONE) {
            return;
        }
        row->tool.side = AOTX_CATALOG_SIDE_BUILT;
        row->tool.built_in = aotx_catalog_built_tool(which);
        row->state = AOTX_CATALOG_INSTALLED;
        row->tick = 0ull;
        row->seq = 0ull;
        row->why = AOTX_CATALOG_WHY_NONE;
        row->figure = 0u;
    }
}

__global__ void aotx_catalog_boot(void)
{
    if (blockIdx.x != 0u || threadIdx.x != 0u) {
        return;
    }
    aotx_catalog_built_in();
    aotx_catalog_anchor();
}

__device__ unsigned int aotx_catalog_remove_judge(const char *name, unsigned int length,
                                                  unsigned int *entry)
{
    unsigned int at = aotx_catalog_find_any(name, length);
    *entry = at;
    if (at >= AOTX_MODULE_SLOTS) {
        return AOTX_CATALOG_GONE_UNKNOWN;
    }
    const aotx_catalog_entry *row = &aotx_catalog.entry[at];
    if (row->kind == AOTX_MODULE_TOOL && row->tool.side == AOTX_CATALOG_SIDE_BUILT) {
        return AOTX_CATALOG_GONE_BUILT;
    }
    if (row->kind == AOTX_MODULE_ROLE) {
        for (unsigned int a = 0u; a < AOTX_SLOTS; ++a) {
            if (aotx_agents.agent[a].state != AOTX_AGENT_STATE_FREE
                && aotx_agents.agent[a].role == at) {
                return AOTX_CATALOG_GONE_ROLE;
            }
        }
    }
    if (row->kind == AOTX_MODULE_TOOL) {
        for (unsigned int r = 0u; r < AOTX_SLOTS; ++r) {
            if (aotx_requests.slot[r].request != 0u
                && aotx_requests.slot[r].entry == at) {
                return AOTX_CATALOG_GONE_TOOL;
            }
        }
    }
    return AOTX_CATALOG_GONE_NONE;
}

__device__ int aotx_catalog_remove(const aotx_remove_body *body)
{
    aotx_cli_out *out = &aotx_catalog_out;
    unsigned long long tick = aotx_time_tick;
    if (body == 0) {
        return 1;
    }
    unsigned int length = 0u;
    while (length < AOTX_CATALOG_NAME_BYTES && body->name[length] != '\0') {
        length += 1u;
    }
    unsigned int entry = AOTX_MODULE_SLOTS;
    unsigned int gone = aotx_catalog_remove_judge(body->name, length, &entry);
    aotx_cli_clear(out);
    if (gone != AOTX_CATALOG_GONE_NONE) {
        aotx_cli_say(out, "remove: ");
        aotx_cli_add(out, body->name, length);
        aotx_cli_say(out, ": ");
        aotx_cli_say(out, aotx_catalog_gone_name(gone));
        aotx_catalog_report(out, tick);
        aotx_catalog.count.gone += 1u;
        return 1;
    }
    aotx_catalog_entry *row = &aotx_catalog.entry[entry];
    /* A role that goes takes its name out of every role list that named it. A mask bit of
     * a free entry would let a later module of another name take the place of this one. */
    for (unsigned int i = 0u; i < AOTX_MODULE_SLOTS; ++i) {
        aotx_catalog_role *other = &aotx_catalog.entry[i].role;
        other->tools[entry >> 5] &= ~(1u << (entry & 31u));
        other->needs_auth[entry >> 5] &= ~(1u << (entry & 31u));
        other->skill_mask[entry >> 5] &= ~(1u << (entry & 31u));
        unsigned int kept = 0u;
        for (unsigned int s = 0u; s < other->skills; ++s) {
            if (other->skill[s] != entry) {
                other->skill[kept++] = other->skill[s];
            }
        }
        other->skills = kept;
    }
    aotx_catalog_release(row);
    row->state = AOTX_CATALOG_FREE;
    row->name_len = 0u;
    row->kind = 0u;
    row->why = AOTX_CATALOG_WHY_NONE;
    row->figure = 0u;
    row->unknown = 0u;
    aotx_catalog.count.removed += 1u;
    aotx_catalog_anchor();
    aotx_cli_say(out, "remove: ");
    aotx_cli_add(out, body->name, length);
    aotx_cli_say(out, " is out of the catalog");
    aotx_catalog_report(out, tick);
    return 0;
}

__device__ void aotx_catalog_commit(unsigned long long tick)
{
    unsigned int count = aotx_catalog.pending_count;
    for (unsigned int i = 0u; i < count; ++i) {
        /* The body is built in the slot of the ring and not in a frame of this kernel. The
         * record is class A, so the state hash folds it and a restore applies it again. */
        unsigned long long seq = aotx_seam_claim(1u);
        aotx_record_header *header = aotx_seam_slot(seq);
        aotx_remove_body *body = (aotx_remove_body *)aotx_seam_body(header);
        for (unsigned int b = 0u; b < AOTX_CATALOG_NAME_BYTES; ++b) {
            body->name[b] = aotx_catalog.pending[i][b];
        }
        aotx_seam_publish(header, seq, AOTX_WRITER_CONSOLE, AOTX_CLASS_A, AOTX_REC_REMOVE,
                          0u, (unsigned int)sizeof *body);
        aotx_seam.apply.state_hash = aotx_seam_fnv1a(aotx_seam.apply.state_hash,
                                                     aotx_seam_body_of(seq),
                                                     (unsigned int)sizeof *body);
        aotx_seam.apply.applied_count += 1ull;
        /* The apply of the record takes the entry out. The live run and the replay
         * therefore take one path. */
        aotx_catalog_remove(body);
    }
    aotx_catalog.pending_count = 0u;
}
