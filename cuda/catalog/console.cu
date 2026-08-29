/* Purpose: Act on the console commands that read the catalog and take a module out.
 * Owns: Nothing; the catalog holds the entries and the console buffer holds the lines.
 * Launch shape: One thread; the apply step calls these in slot order.
 * Lifetime: The whole run. */
#include "catalog/console.cuh"

/* Lines of the body of a skill that the module command shows. */
#define AOTX_CATALOG_BODY_LINES  6u

/* Add a run of the arena to a line. */
__device__ __forceinline__ static void aotx_catalog_add(aotx_cli_out *out,
                                                        aotx_catalog_run run)
{
    aotx_cli_add(out, (const char *)aotx_catalog_arena + run.at, run.length);
}

/* Write one line of the modules command. */
__device__ __forceinline__ static void aotx_catalog_row_line(aotx_cli_out *out,
                                                             const aotx_catalog_entry *row)
{
    aotx_cli_say(out, "  ");
    aotx_cli_add(out, row->name, row->name_len);
    aotx_cli_say(out, " ");
    aotx_cli_say(out, aotx_catalog_kind_name(row->kind));
    aotx_cli_say(out, " ");
    aotx_cli_say(out, aotx_catalog_state_name(row->state));
    aotx_cli_say(out, " ");
    if (row->version.length != 0u) {
        aotx_catalog_add(out, row->version);
    } else {
        aotx_cli_say(out, "-");
    }
    if (row->state == AOTX_CATALOG_REFUSED) {
        aotx_cli_say(out, ": ");
        aotx_cli_say(out, aotx_catalog_why_name(row->why));
        if (row->figure != 0u) {
            aotx_cli_say(out, " of ");
            aotx_cli_num(out, (unsigned long long)row->figure);
        }
    }
    aotx_cli_console(out);
}

__device__ void aotx_catalog_modules_command(aotx_cli_out *out, unsigned int kind)
{
    aotx_cli_say(out, "modules: name kind state version");
    aotx_cli_console(out);
    unsigned int shown = 0u;
    for (unsigned int i = 0u; i < AOTX_MODULE_SLOTS; ++i) {
        const aotx_catalog_entry *row = &aotx_catalog.entry[i];
        if (row->state == AOTX_CATALOG_FREE) {
            continue;
        }
        if (kind != 0u && row->kind != kind) {
            continue;
        }
        aotx_catalog_row_line(out, row);
        shown += 1u;
    }
    if (shown == 0u) {
        aotx_cli_say(out, "  no modules");
        aotx_cli_console(out);
        return;
    }
    aotx_cli_say(out, "arena: bytes ");
    aotx_cli_num(out, (unsigned long long)aotx_catalog.used);
    aotx_cli_say(out, " of ");
    aotx_cli_num(out, (unsigned long long)AOTX_CATALOGUE_BYTES);
    aotx_cli_say(out, " free runs ");
    aotx_cli_num(out, (unsigned long long)aotx_catalog.frees);
    aotx_cli_say(out, " list cut ");
    aotx_cli_num(out, (unsigned long long)aotx_catalog.count.list_cut);
    aotx_cli_console(out);
}

__device__ void aotx_catalog_module_command(aotx_cli_out *out, const char *name,
                                            unsigned int length)
{
    unsigned int at = aotx_catalog_find_any(name, length);
    if (at >= AOTX_MODULE_SLOTS) {
        aotx_cli_say(out, "module: no module holds that name");
        aotx_cli_console(out);
        aotx_cli_count.refused += 1u;
        return;
    }
    const aotx_catalog_entry *row = &aotx_catalog.entry[at];
    aotx_cli_say(out, "module: ");
    aotx_catalog_row_line(out, row);

    /* The manifest of the module, one line at a time. The text stands in the arena, so
     * the console shows the bytes the import carried. */
    unsigned int walk = row->manifest.at;
    unsigned int stop = walk + row->manifest.length;
    while (walk < stop) {
        unsigned int line = walk;
        while (walk < stop && aotx_catalog_arena[walk] != (unsigned char)'\n') {
            walk += 1u;
        }
        if (walk > line) {
            aotx_cli_say(out, "  ");
            aotx_cli_add(out, (const char *)aotx_catalog_arena + line, walk - line);
            aotx_cli_console(out);
        }
        walk += 1u;
    }
    if (row->kind != AOTX_MODULE_SKILL || row->body.length == 0u) {
        return;
    }
    aotx_cli_say(out, "  body: bytes ");
    aotx_cli_num(out, (unsigned long long)row->body.length);
    aotx_cli_console(out);
    walk = row->body.at;
    stop = walk + row->body.length;
    for (unsigned int shown = 0u; shown < AOTX_CATALOG_BODY_LINES && walk < stop; ++shown) {
        unsigned int line = walk;
        while (walk < stop && aotx_catalog_arena[walk] != (unsigned char)'\n') {
            walk += 1u;
        }
        aotx_cli_say(out, "  | ");
        aotx_cli_add(out, (const char *)aotx_catalog_arena + line, walk - line);
        aotx_cli_console(out);
        walk += 1u;
    }
}

__device__ void aotx_catalog_remove_command(aotx_cli_out *out, const char *name,
                                            unsigned int length, unsigned long long tick)
{
    (void)tick;
    /* A replay of the journal sends every line again. The remove record of a remove line
     * stands in the journal beside that line, and the apply of the record makes the
     * change. This command therefore writes no record while a replay runs. */
    if (aotx_seam.replaying != 0ull) {
        return;
    }
    if (length == 0u || length >= AOTX_CATALOG_NAME_BYTES) {
        aotx_cli_say(out, "remove: give the name of one module");
        aotx_cli_console(out);
        aotx_cli_count.refused += 1u;
        return;
    }
    /* The judgement comes before the record, so a line the catalog refuses writes no
     * record and the journal holds the changes that landed. */
    unsigned int entry = AOTX_MODULE_SLOTS;
    unsigned int gone = aotx_catalog_remove_judge(name, length, &entry);
    if (gone != AOTX_CATALOG_GONE_NONE) {
        aotx_cli_say(out, "remove: ");
        aotx_cli_add(out, name, length);
        aotx_cli_say(out, ": ");
        aotx_cli_say(out, aotx_catalog_gone_name(gone));
        aotx_cli_console(out);
        aotx_cli_count.refused += 1u;
        aotx_catalog.count.gone += 1u;
        return;
    }
    unsigned int wait = aotx_catalog.pending_count;
    if (wait >= AOTX_CATALOG_PENDING_MAX) {
        aotx_cli_say(out, "remove: the tick holds too many remove lines; give it again");
        aotx_cli_console(out);
        aotx_cli_count.refused += 1u;
        return;
    }
    for (unsigned int i = 0u; i < AOTX_CATALOG_NAME_BYTES; ++i) {
        aotx_catalog.pending[wait][i] = (i < length) ? name[i] : '\0';
    }
    aotx_catalog.pending_count = wait + 1u;
    if (!aotx_cli_allow()) {
        return;
    }
    aotx_cli_say(out, "remove: ");
    aotx_cli_add(out, name, length);
    aotx_cli_say(out, " goes out at the commit of this tick");
    aotx_console_write(out->text, out->at);
    aotx_cli_clear(out);
}
