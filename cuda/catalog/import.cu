/* Purpose: Take the head and the parts of one import and commit the module they carry.
 * Owns: Nothing; the catalog holds the entries, the arena and the imports that arrive.
 * Launch shape: A device function; the apply step calls it from its serial thread.
 * Lifetime: Each call.
 *
 * A head claims a free entry, or the entry of the same name. The parts then fill the runs
 * the head took. The commit reads the manifest, checks the entry and gives it the state
 * INSTALLED or REFUSED. An import of a name that stands replaces that entry whole at the
 * commit. The parts fill runs of their own, so the entry never changes in pieces. */
#include "agent/agent_state.cuh"
#include "bus/bus.cuh"
#include "catalog/console.cuh"

/* The line the import builds. One thread applies an import, so one line is enough and no
 * frame of the kernel holds it. */
static __device__ aotx_cli_out aotx_catalog_line;

/* Write one console line and put the same text on the bus as a note. */
__device__ static void aotx_catalog_tell(aotx_cli_out *out, unsigned long long tick)
{
    aotx_console_write(out->text, out->at);
    aotx_bus_append(AOTX_WRITER_SYSTEM, AOTX_BUS_NOTE, 0u, out->text, out->at,
                    0ull, 0ull, 0.0f, tick);
    aotx_cli_clear(out);
}

/* Say that an import did not go in, and why. */
__device__ static int aotx_catalog_no(const char *name, unsigned int length,
                                      unsigned int why, unsigned int figure,
                                      unsigned long long tick)
{
    aotx_cli_out *out = &aotx_catalog_line;
    aotx_cli_clear(out);
    aotx_cli_say(out, "import: ");
    aotx_cli_add(out, name, length);
    aotx_cli_say(out, ": ");
    aotx_cli_say(out, aotx_catalog_why_name(why));
    if (figure != 0u) {
        aotx_cli_say(out, " of ");
        aotx_cli_num(out, (unsigned long long)figure);
    }
    aotx_catalog_tell(out, tick);
    aotx_catalog.count.refused += 1u;
    aotx_catalog.count.last_why = why;
    return 1;
}

/* Give the row of an import that arrives, or the row count. */
__device__ __forceinline__ static unsigned int aotx_catalog_row_of(unsigned int import)
{
    for (unsigned int i = 0u; i < AOTX_CATALOG_ARRIVING_MAX; ++i) {
        if (aotx_catalog.arriving[i].import == import) {
            return i;
        }
    }
    return AOTX_CATALOG_ARRIVING_MAX;
}

/* The bytes of a name that ends with a zero byte, up to the bound of the field. */
__device__ __forceinline__ static unsigned int aotx_catalog_name_len(const char *name)
{
    unsigned int at = 0u;
    while (at < AOTX_CATALOG_NAME_BYTES && name[at] != '\0') {
        at += 1u;
    }
    return at;
}

/* Mark the entry of an import the head refused. A refused entry keeps its name and its
 * reason until the next import of that name, so the modules command shows what went
 * wrong. Every run of the module that stood goes back to the free list with it. */
__device__ static int aotx_catalog_head_no(unsigned int at, const char *name,
                                           unsigned int length, unsigned int why,
                                           unsigned int figure, unsigned long long tick)
{
    aotx_catalog_entry *row = &aotx_catalog.entry[at];
    aotx_catalog_release(row);
    for (unsigned int i = 0u; i < AOTX_CATALOG_NAME_BYTES; ++i) {
        row->name[i] = (i < length) ? name[i] : '\0';
    }
    row->name_len = length;
    row->state = AOTX_CATALOG_REFUSED;
    row->why = why;
    row->figure = figure;
    row->unknown = 0u;
    aotx_catalog_anchor();
    return aotx_catalog_no(name, length, why, figure, tick);
}

/* Take the head of an import. */
__device__ static int aotx_catalog_head(const aotx_import_head *head,
                                        unsigned long long tick)
{
    unsigned int length = aotx_catalog_name_len(head->name);
    if (length == 0u || length >= AOTX_CATALOG_NAME_BYTES) {
        return aotx_catalog_no(head->name, length, AOTX_CATALOG_WHY_NAME,
                               AOTX_CATALOG_NAME_BYTES - 1u, tick);
    }
    /* A second head of an import number that arrives takes no entry, because the entry of
     * that number belongs to the import which arrives. */
    if (head->import == 0u
        || aotx_catalog_row_of(head->import) < AOTX_CATALOG_ARRIVING_MAX) {
        return aotx_catalog_no(head->name, length, AOTX_CATALOG_WHY_TWICE, 0u, tick);
    }
    unsigned int at = aotx_catalog_find_any(head->name, length);
    if (at < AOTX_MODULE_SLOTS
        && aotx_catalog.entry[at].state == AOTX_CATALOG_ARRIVING) {
        /* An import of that name arrives already. This head cancels that arrival whole:
         * the runs it took go back and its row goes free. No run of the arena therefore
         * goes back twice. A restore that replays half an import leaves such an entry,
         * and the next import of that name clears it. */
        for (unsigned int i = 0u; i < AOTX_CATALOG_ARRIVING_MAX; ++i) {
            aotx_catalog_arriving *old = &aotx_catalog.arriving[i];
            if (old->import == 0u || old->entry != at) {
                continue;
            }
            aotx_catalog_free_run(old->run[0]);
            aotx_catalog_free_run(old->run[1]);
            old->import = 0u;
            aotx_catalog.count.cancelled += 1u;
        }
        /* The runs of the module that stood belong to the entry, and the entry keeps
         * them for the head that follows. */
        aotx_catalog.entry[at].state = AOTX_CATALOG_REFUSED;
        aotx_catalog.entry[at].why = AOTX_CATALOG_WHY_TWICE;
    }
    if (at >= AOTX_MODULE_SLOTS) {
        for (unsigned int i = 0u; i < AOTX_MODULE_SLOTS; ++i) {
            if (aotx_catalog.entry[i].state == AOTX_CATALOG_FREE) {
                at = i;
                break;
            }
        }
    }
    if (at >= AOTX_MODULE_SLOTS) {
        return aotx_catalog_no(head->name, length, AOTX_CATALOG_WHY_TABLE,
                               (unsigned int)AOTX_MODULE_SLOTS, tick);
    }
    unsigned int row_at = aotx_catalog_row_of(0u);
    if (row_at >= AOTX_CATALOG_ARRIVING_MAX) {
        return aotx_catalog_no(head->name, length, AOTX_CATALOG_WHY_BUSY,
                               AOTX_CATALOG_ARRIVING_MAX, tick);
    }
    if (head->kind != AOTX_MODULE_SKILL && head->kind != AOTX_MODULE_ROLE
        && head->kind != AOTX_MODULE_TOOL) {
        return aotx_catalog_head_no(at, head->name, length, AOTX_CATALOG_WHY_KIND, 0u,
                                    tick);
    }
    /* The head counts the files that carry bytes, and a file of no bytes carries none.
     * A skill directory that holds the skill file alone therefore gives one file, whose
     * byte count stands in the second place. The two counts must agree. */
    unsigned int carried = 0u;
    for (unsigned int f = 0u; f < (unsigned int)AOTX_IMPORT_FILES; ++f) {
        carried += (head->file_bytes[f] != 0u) ? 1u : 0u;
    }
    if (head->files == 0u || head->files > (unsigned int)AOTX_IMPORT_FILES
        || carried != head->files) {
        return aotx_catalog_head_no(at, head->name, length, AOTX_CATALOG_WHY_FILES,
                                    (unsigned int)AOTX_IMPORT_FILES, tick);
    }
    /* A skill body must fit the bound of a body. The prompt of a turn then holds it
     * beside the overlay and the result of a tool. A skill file that carries its head as
     * well takes the bound of a head beside it. The commit then checks the body it splits
     * out against the bound of a body. A role overlay has its own bound. */
    if (head->kind == AOTX_MODULE_SKILL
        && head->file_bytes[1] > (unsigned int)AOTX_SKILL_BYTES
                                 + AOTX_CATALOG_HEAD_BYTES) {
        return aotx_catalog_head_no(at, head->name, length, AOTX_CATALOG_WHY_BODY,
                                    (unsigned int)AOTX_SKILL_BYTES, tick);
    }
    if (head->kind == AOTX_MODULE_ROLE
        && head->file_bytes[1] > AOTX_CATALOG_OVERLAY_BYTES) {
        return aotx_catalog_head_no(at, head->name, length, AOTX_CATALOG_WHY_BODY,
                                    AOTX_CATALOG_OVERLAY_BYTES, tick);
    }
    aotx_catalog_arriving *hold = &aotx_catalog.arriving[row_at];
    aotx_catalog_entry *row = &aotx_catalog.entry[at];
    hold->import = head->import;
    hold->entry = at;
    hold->kind = head->kind;
    hold->files = head->files;
    hold->tick = tick;
    for (unsigned int f = 0u; f < (unsigned int)AOTX_IMPORT_FILES; ++f) {
        hold->file_bytes[f] = head->file_bytes[f];
        hold->got[f] = 0u;
        hold->run[f].at = 0u;
        hold->run[f].length = 0u;
    }
    for (unsigned int f = 0u; f < (unsigned int)AOTX_IMPORT_FILES; ++f) {
        if (aotx_catalog_take_run(hold->file_bytes[f], &hold->run[f]) != 0) {
            for (unsigned int k = 0u; k < f; ++k) {
                aotx_catalog_free_run(hold->run[k]);
            }
            hold->import = 0u;
            return aotx_catalog_head_no(at, head->name, length, AOTX_CATALOG_WHY_ARENA,
                                        hold->file_bytes[f], tick);
        }
    }
    /* The runs of the entry that stands are kept until the commit. An import the checks
     * refuse therefore leaves no run of the module that went. */
    hold->held[0] = row->manifest;
    hold->held[1] = (row->kind == AOTX_MODULE_ROLE) ? row->role.overlay : row->body;
    for (unsigned int b = 0u; b < 32u; ++b) {
        hold->digest[b] = head->digest[b];
    }
    if (row->state == AOTX_CATALOG_INSTALLED) {
        aotx_catalog.count.replaced += 1u;
    }
    for (unsigned int i = 0u; i < AOTX_CATALOG_NAME_BYTES; ++i) {
        row->name[i] = (i < length) ? head->name[i] : '\0';
    }
    row->name_len = length;
    row->kind = head->kind;
    row->state = AOTX_CATALOG_ARRIVING;
    row->why = AOTX_CATALOG_WHY_NONE;
    row->figure = 0u;
    row->unknown = 0u;
    aotx_catalog.count.heads += 1u;
    /* An entry that arrives is not installed, so no role, no prompt and no parser sees
     * it. The anchors of the engine therefore drop a role that is under replacement. */
    aotx_catalog_anchor();
    return 0;
}

/* Commit one import: read the manifest, check the entry, and give it its state. */
__device__ static int aotx_catalog_land(aotx_catalog_arriving *hold,
                                        unsigned long long seq, unsigned long long tick)
{
    aotx_cli_out *out = &aotx_catalog_line;
    aotx_catalog_entry *row = &aotx_catalog.entry[hold->entry];
    unsigned int figure = 0u;
    aotx_catalog_run manifest = hold->run[0];
    aotx_catalog_run body = hold->run[1];
    /* A skill directory may hold the skill file alone. The feeder then sends no manifest
     * and the whole file in the second place. The head of that file is the manifest and
     * the text after the head is the body. The device therefore splits the run it took.
     * The two parts touch, so each one goes back to the free list and they join again. */
    if (manifest.length == 0u && body.length != 0u && hold->kind == AOTX_MODULE_SKILL) {
        unsigned int end = aotx_catalog_head_end(body.at, body.length);
        manifest.at = body.at;
        manifest.length = end - body.at;
        body.length = (body.at + body.length) - end;
        body.at = end;
    }
    row->manifest = manifest;
    if (hold->kind == AOTX_MODULE_ROLE) {
        row->role.overlay = body;
        row->body.at = 0u;
        row->body.length = 0u;
    } else {
        row->body = body;
        row->role.overlay.at = 0u;
        row->role.overlay.length = 0u;
    }
    for (unsigned int b = 0u; b < 32u; ++b) {
        row->digest[b] = hold->digest[b];
    }
    unsigned int why = aotx_catalog_manifest_read(row, manifest.at, manifest.length,
                                                  hold->kind, &figure);
    /* A skill needs a body and a role needs an overlay. A tool carries no text of its
     * own, so an import of one file is a tool. */
    if (why == AOTX_CATALOG_WHY_NONE && hold->kind != AOTX_MODULE_TOOL
        && body.length == 0u) {
        why = AOTX_CATALOG_WHY_EMPTY;
    }
    /* The body of a skill file stands after its head. Its bound is checked here and not
     * at the head, where the head bytes stand in the same count. */
    if (why == AOTX_CATALOG_WHY_NONE && hold->kind == AOTX_MODULE_SKILL
        && body.length > (unsigned int)AOTX_SKILL_BYTES) {
        figure = (unsigned int)AOTX_SKILL_BYTES;
        why = AOTX_CATALOG_WHY_BODY;
    }
    row->tick = tick;
    row->seq = seq;
    aotx_cli_clear(out);
    aotx_cli_say(out, "import: ");
    aotx_cli_add(out, row->name, row->name_len);
    aotx_cli_say(out, " ");
    aotx_cli_say(out, aotx_catalog_kind_name(hold->kind));
    if (why != AOTX_CATALOG_WHY_NONE) {
        /* A refused entry keeps its name and its reason until the next import of that
         * name, so the modules command shows what went wrong. Every run of the module
         * goes back, the run of the module that stood before it as well. */
        aotx_catalog_free_run(hold->run[0]);
        aotx_catalog_free_run(hold->run[1]);
        aotx_catalog_free_run(hold->held[0]);
        aotx_catalog_free_run(hold->held[1]);
        row->manifest.at = 0u;
        row->manifest.length = 0u;
        row->body.at = 0u;
        row->body.length = 0u;
        row->role.overlay.at = 0u;
        row->role.overlay.length = 0u;
        /* The runs of the description and of the version point into the bytes that went
         * back, so a refused entry keeps neither. */
        row->description.at = 0u;
        row->description.length = 0u;
        row->version.at = 0u;
        row->version.length = 0u;
        row->state = AOTX_CATALOG_REFUSED;
        row->why = why;
        row->figure = figure;
        aotx_catalog.count.refused += 1u;
        aotx_catalog.count.last_why = why;
        aotx_cli_say(out, " refused: ");
        aotx_cli_say(out, aotx_catalog_why_name(why));
        if (figure != 0u) {
            aotx_cli_say(out, " of ");
            aotx_cli_num(out, (unsigned long long)figure);
        }
        aotx_catalog_tell(out, tick);
        aotx_catalog_anchor();
        return 1;
    }
    aotx_catalog_free_run(hold->held[0]);
    aotx_catalog_free_run(hold->held[1]);
    row->state = AOTX_CATALOG_INSTALLED;
    row->why = AOTX_CATALOG_WHY_NONE;
    row->figure = 0u;
    aotx_catalog.count.installed += 1u;
    aotx_cli_say(out, " installed");
    if (row->unknown != 0u) {
        aotx_catalog_unknown_names(row, out);
    }
    aotx_catalog_tell(out, tick);
    aotx_catalog_anchor();
    /* The role of the console may have landed with this record. Its agent takes slot 0
     * here, on the serial thread of the apply. A line that stands after this import in
     * the same batch of inputs therefore finds that agent. The spawn writes derived
     * records alone, so the state hash keeps the order the journal holds. */
    aotx_agent_boot_spawn();
    return 0;
}

/* Take one part of an import. */
__device__ static int aotx_catalog_part(const aotx_import_part *part,
                                        unsigned long long seq, unsigned long long tick)
{
    unsigned int row_at = aotx_catalog_row_of(part->import);
    if (part->import == 0u || row_at >= AOTX_CATALOG_ARRIVING_MAX) {
        aotx_cli_out *out = &aotx_catalog_line;
        aotx_cli_clear(out);
        aotx_cli_say(out, "import: part of import ");
        aotx_cli_num(out, (unsigned long long)part->import);
        aotx_cli_say(out, ": no import of that number arrives");
        aotx_catalog_tell(out, tick);
        return 1;
    }
    aotx_catalog_arriving *hold = &aotx_catalog.arriving[row_at];
    unsigned int file = part->file;
    /* The run of a file holds the bytes the head named for it. A part therefore writes
     * inside a run the head took, and never at the start of the arena. */
    if (file >= (unsigned int)AOTX_IMPORT_FILES || hold->file_bytes[file] == 0u
        || hold->run[file].length != hold->file_bytes[file]
        || part->length > (unsigned int)AOTX_IMPORT_TEXT_BYTES
        || part->offset > hold->file_bytes[file]
        || part->offset + part->length > hold->file_bytes[file]) {
        unsigned int entry = hold->entry;
        aotx_catalog_free_run(hold->run[0]);
        aotx_catalog_free_run(hold->run[1]);
        aotx_catalog_free_run(hold->held[0]);
        aotx_catalog_free_run(hold->held[1]);
        aotx_catalog.entry[entry].state = AOTX_CATALOG_REFUSED;
        aotx_catalog.entry[entry].why = AOTX_CATALOG_WHY_PART;
        aotx_catalog.entry[entry].figure = 0u;
        aotx_catalog.entry[entry].manifest.length = 0u;
        aotx_catalog.entry[entry].body.length = 0u;
        aotx_catalog.entry[entry].role.overlay.length = 0u;
        /* The runs of the description and of the version point into the bytes that went
         * back, so a refused entry keeps neither. */
        aotx_catalog.entry[entry].description.length = 0u;
        aotx_catalog.entry[entry].version.length = 0u;
        hold->import = 0u;
        aotx_catalog_anchor();
        return aotx_catalog_no(aotx_catalog.entry[entry].name,
                               aotx_catalog.entry[entry].name_len,
                               AOTX_CATALOG_WHY_PART, 0u, tick);
    }
    for (unsigned int i = 0u; i < part->length; ++i) {
        aotx_catalog_arena[hold->run[file].at + part->offset + i] =
            (unsigned char)part->text[i];
    }
    unsigned int end = part->offset + part->length;
    if (end > hold->got[file]) {
        hold->got[file] = end;
    }
    aotx_catalog.count.parts += 1u;
    for (unsigned int f = 0u; f < (unsigned int)AOTX_IMPORT_FILES; ++f) {
        if (hold->got[f] < hold->file_bytes[f]) {
            return 0;
        }
    }
    int state = aotx_catalog_land(hold, seq, tick);
    hold->import = 0u;
    return state;
}

/* The apply of the tick calls this. The mark keeps the body of the call out of the frame
 * of that kernel, which the spill gate holds to a figure. */
__device__ __noinline__ int aotx_catalog_apply(unsigned int type, const void *body,
                                               unsigned int body_len,
                                               unsigned long long seq)
{
    unsigned long long tick = aotx_time_tick;
    if (body == 0) {
        return 1;
    }
    if (type == (unsigned int)AOTX_REC_REMOVE) {
        return (body_len >= (unsigned int)sizeof(aotx_remove_body))
             ? aotx_catalog_remove((const aotx_remove_body *)body) : 1;
    }
    /* The head and the part share the first two fields, so the part field says which body
     * the record carries. A head is part zero. */
    const unsigned int *fields = (const unsigned int *)body;
    if (fields[1] == 0u) {
        if (body_len < (unsigned int)sizeof(aotx_import_head)) {
            return 1;
        }
        return aotx_catalog_head((const aotx_import_head *)body, tick);
    }
    if (body_len < (unsigned int)sizeof(aotx_import_part)) {
        return 1;
    }
    return aotx_catalog_part((const aotx_import_part *)body, seq, tick);
}
