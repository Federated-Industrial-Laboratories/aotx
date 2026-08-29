/* Purpose: Read one module manifest on the device and fill the entry it describes.
 * Owns: Nothing; the caller holds the entry and the arena holds the text.
 * Launch shape: A device function; the apply step calls it once for each commit.
 * Lifetime: Each call.
 *
 * The shape is one key and one value a line. A value runs to the end of its line. A
 * number sign at the start of a line makes a comment. The reader takes that shape and the
 * two-key head of a skill file, and no other. It keeps no array of its own: a key is
 * compared where it stands, and every text it keeps is a run of the arena. */
#include "catalog/console.cuh"
#include "model/model.cuh"

/* Report whether the bytes from start to end are the word. */
__device__ __forceinline__ static int aotx_catalog_word_is(unsigned int start,
                                                           unsigned int end,
                                                           const char *word)
{
    unsigned int at = start;
    unsigned int i = 0u;
    while (at < end && word[i] != '\0'
           && aotx_catalog_arena[at] == (unsigned char)word[i]) {
        at += 1u;
        i += 1u;
    }
    return (at == end && word[i] == '\0') ? 1 : 0;
}

/* Move past the spaces and the tabs from a position. */
__device__ __forceinline__ static unsigned int aotx_catalog_blank(unsigned int at,
                                                                  unsigned int end)
{
    while (at < end && (aotx_catalog_arena[at] == (unsigned char)' '
                        || aotx_catalog_arena[at] == (unsigned char)'\t')) {
        at += 1u;
    }
    return at;
}

/* Move back over the spaces, the tabs and the carriage return at the end of a run. */
__device__ __forceinline__ static unsigned int aotx_catalog_trim(unsigned int start,
                                                                 unsigned int end)
{
    while (end > start && (aotx_catalog_arena[end - 1u] == (unsigned char)' '
                           || aotx_catalog_arena[end - 1u] == (unsigned char)'\t'
                           || aotx_catalog_arena[end - 1u] == (unsigned char)'\r')) {
        end -= 1u;
    }
    return end;
}

/* Read a run as a decimal count. The return is 1 when every byte is a figure and the
 * value stays under a million. */
__device__ __forceinline__ static int aotx_catalog_count_of(unsigned int start,
                                                            unsigned int end,
                                                            unsigned int *value)
{
    unsigned int got = 0u;
    if (start >= end || end - start > 7u) {
        return 0;
    }
    for (unsigned int at = start; at < end; ++at) {
        unsigned char byte = aotx_catalog_arena[at];
        if (byte < (unsigned char)'0' || byte > (unsigned char)'9') {
            return 0;
        }
        got = got * 10u + (unsigned int)(byte - (unsigned char)'0');
    }
    *value = got;
    return 1;
}

/* Report whether a run is a name: 1 to 63 bytes of a to z, 0 to 9 and the low line. */
__device__ __forceinline__ static int aotx_catalog_name_ok(unsigned int start,
                                                           unsigned int end)
{
    if (start >= end || end - start >= AOTX_CATALOG_NAME_BYTES) {
        return 0;
    }
    for (unsigned int at = start; at < end; ++at) {
        unsigned char byte = aotx_catalog_arena[at];
        int letter = (byte >= (unsigned char)'a' && byte <= (unsigned char)'z');
        int figure = (byte >= (unsigned char)'0' && byte <= (unsigned char)'9');
        if (!letter && !figure && byte != (unsigned char)'_') {
            return 0;
        }
    }
    return 1;
}

/* Give the end of the piece that starts at a position, up to the next comma. */
__device__ __forceinline__ static unsigned int aotx_catalog_piece(unsigned int at,
                                                                  unsigned int end)
{
    while (at < end && aotx_catalog_arena[at] != (unsigned char)',') {
        at += 1u;
    }
    return at;
}

/* Add the entries a comma list names to a mask, and give the count of the names the
 * catalog does not hold. An unknown name fails safe to no entry. */
__device__ __forceinline__ static unsigned int aotx_catalog_mask_of(unsigned int start,
                                                                    unsigned int end,
                                                                    unsigned int *mask,
                                                                    unsigned int kind)
{
    unsigned int unknown = 0u;
    unsigned int at = start;
    while (at < end) {
        unsigned int piece = aotx_catalog_piece(at, end);
        unsigned int first = aotx_catalog_blank(at, piece);
        unsigned int last = aotx_catalog_trim(first, piece);
        if (last > first) {
            unsigned int found = aotx_catalog_find((const char *)aotx_catalog_arena + first,
                                                   last - first, kind);
            if (found < AOTX_MODULE_SLOTS) {
                aotx_catalog_mask_set(mask, found);
            } else {
                unknown += 1u;
            }
        }
        at = piece + 1u;
    }
    return unknown;
}

/* Report whether a key stands in the list of keys of a kind. */
__device__ __forceinline__ static int aotx_catalog_key_of_kind(unsigned int start,
                                                               unsigned int end,
                                                               unsigned int kind)
{
    if (aotx_catalog_word_is(start, end, "kind")
        || aotx_catalog_word_is(start, end, "name")
        || aotx_catalog_word_is(start, end, "description")
        || aotx_catalog_word_is(start, end, "version")) {
        return 1;
    }
    if (kind == AOTX_MODULE_TOOL) {
        return aotx_catalog_word_is(start, end, "side")
            || aotx_catalog_word_is(start, end, "arguments")
            || aotx_catalog_word_is(start, end, "authorise")
            || aotx_catalog_word_is(start, end, "deadline")
            || aotx_catalog_word_is(start, end, "timeout")
            || aotx_catalog_word_is(start, end, "module")
            || aotx_catalog_word_is(start, end, "entry")
            || aotx_catalog_word_is(start, end, "program")
            || aotx_catalog_word_is(start, end, "example")
            || aotx_catalog_word_is(start, end, "sha256");
    }
    if (kind == AOTX_MODULE_ROLE) {
        return aotx_catalog_word_is(start, end, "model")
            || aotx_catalog_word_is(start, end, "tools")
            || aotx_catalog_word_is(start, end, "authorise")
            || aotx_catalog_word_is(start, end, "budget")
            || aotx_catalog_word_is(start, end, "pages")
            || aotx_catalog_word_is(start, end, "skills")
            || aotx_catalog_word_is(start, end, "body");
    }
    return aotx_catalog_word_is(start, end, "body");
}

/* Take one key and one value into the entry. The return is a reason, or none. */
__device__ static unsigned int aotx_catalog_pair(aotx_catalog_entry *row,
                                                 unsigned int key_at, unsigned int key_end,
                                                 unsigned int at, unsigned int end,
                                                 unsigned int kind, unsigned int head,
                                                 unsigned int *figure)
{
    aotx_catalog_run run;
    run.at = at;
    run.length = end - at;
    if (aotx_catalog_word_is(key_at, key_end, "kind")) {
        unsigned int said = aotx_catalog_word_is(at, end, "skill") ? AOTX_MODULE_SKILL
                          : (aotx_catalog_word_is(at, end, "role") ? AOTX_MODULE_ROLE
                             : (aotx_catalog_word_is(at, end, "tool") ? AOTX_MODULE_TOOL
                                : 0u));
        if (said == 0u) {
            return AOTX_CATALOG_WHY_VALUE;
        }
        return (said == kind) ? AOTX_CATALOG_WHY_NONE : AOTX_CATALOG_WHY_KIND;
    }
    if (aotx_catalog_word_is(key_at, key_end, "name")) {
        if (aotx_catalog_name_ok(at, end) == 0) {
            *figure = AOTX_CATALOG_NAME_BYTES - 1u;
            return AOTX_CATALOG_WHY_NAME;
        }
        unsigned int length = end - at;
        if (head != 0u) {
            /* The head of the import already named the module. A manifest that names
             * another module is not the manifest of this directory. */
            if (row->name_len != length) {
                return AOTX_CATALOG_WHY_HEAD;
            }
            for (unsigned int i = 0u; i < length; ++i) {
                if ((unsigned char)row->name[i] != aotx_catalog_arena[at + i]) {
                    return AOTX_CATALOG_WHY_HEAD;
                }
            }
            return AOTX_CATALOG_WHY_NONE;
        }
        for (unsigned int i = 0u; i < AOTX_CATALOG_NAME_BYTES; ++i) {
            row->name[i] = (i < length) ? (char)aotx_catalog_arena[at + i] : '\0';
        }
        row->name_len = length;
        return AOTX_CATALOG_WHY_NONE;
    }
    if (aotx_catalog_word_is(key_at, key_end, "description")) {
        row->description = run;
        return AOTX_CATALOG_WHY_NONE;
    }
    if (aotx_catalog_word_is(key_at, key_end, "version")) {
        row->version = run;
        return AOTX_CATALOG_WHY_NONE;
    }
    if (kind == AOTX_MODULE_TOOL) {
        if (aotx_catalog_word_is(key_at, key_end, "side")) {
            if (aotx_catalog_word_is(at, end, "device")) {
                row->tool.side = AOTX_CATALOG_SIDE_DEVICE;
            } else if (aotx_catalog_word_is(at, end, "host")) {
                row->tool.side = AOTX_CATALOG_SIDE_HOST;
            } else {
                return AOTX_CATALOG_WHY_VALUE;
            }
            return AOTX_CATALOG_WHY_NONE;
        }
        if (aotx_catalog_word_is(key_at, key_end, "arguments")) {
            unsigned int walk = at;
            row->tool.arguments = 0u;
            while (walk < end) {
                unsigned int piece = aotx_catalog_piece(walk, end);
                unsigned int first = aotx_catalog_blank(walk, piece);
                unsigned int last = aotx_catalog_trim(first, piece);
                if (last > first) {
                    if (row->tool.arguments >= AOTX_CATALOG_ARGS) {
                        *figure = AOTX_CATALOG_ARGS;
                        return AOTX_CATALOG_WHY_ARGS;
                    }
                    row->tool.key[row->tool.arguments].at = first;
                    row->tool.key[row->tool.arguments].length = last - first;
                    row->tool.arguments += 1u;
                }
                walk = piece + 1u;
            }
            return AOTX_CATALOG_WHY_NONE;
        }
        if (aotx_catalog_word_is(key_at, key_end, "authorise")) {
            if (aotx_catalog_word_is(at, end, "always")) {
                row->tool.authorize = AOTX_CATALOG_AUTH_ALWAYS;
            } else if (aotx_catalog_word_is(at, end, "never")) {
                row->tool.authorize = AOTX_CATALOG_AUTH_NEVER;
            } else {
                return AOTX_CATALOG_WHY_VALUE;
            }
            return AOTX_CATALOG_WHY_NONE;
        }
        if (aotx_catalog_word_is(key_at, key_end, "deadline")) {
            return aotx_catalog_count_of(at, end, &row->tool.deadline)
                 ? AOTX_CATALOG_WHY_NONE : AOTX_CATALOG_WHY_VALUE;
        }
        if (aotx_catalog_word_is(key_at, key_end, "timeout")) {
            return aotx_catalog_count_of(at, end, &row->tool.timeout)
                 ? AOTX_CATALOG_WHY_NONE : AOTX_CATALOG_WHY_VALUE;
        }
        if (aotx_catalog_word_is(key_at, key_end, "entry")) {
            row->tool.entry = run;
            return AOTX_CATALOG_WHY_NONE;
        }
        /* The module file, the program and the example line are runs of the manifest text
         * in the arena. The host glue reads the module file name from the entry, and the
         * check program reads the program and the example line the same way. */
        if (aotx_catalog_word_is(key_at, key_end, "module")) {
            row->tool.module = run;
            return AOTX_CATALOG_WHY_NONE;
        }
        if (aotx_catalog_word_is(key_at, key_end, "program")) {
            row->tool.program = run;
            return AOTX_CATALOG_WHY_NONE;
        }
        if (aotx_catalog_word_is(key_at, key_end, "example")) {
            row->tool.example = run;
            return AOTX_CATALOG_WHY_NONE;
        }
        return AOTX_CATALOG_WHY_NONE;
    }
    if (kind == AOTX_MODULE_ROLE) {
        if (aotx_catalog_word_is(key_at, key_end, "model")) {
            /* The value is a role name of the model file list. The four names of that
             * list are the four the device knows. A role of the two small models opens
             * no reply, so a role of a run names a language file. */
            if (aotx_catalog_word_is(at, end, "language")) {
                row->role.model = AOTX_MODEL_LANGUAGE;
            } else if (aotx_catalog_word_is(at, end, "language-q4")) {
                row->role.model = AOTX_MODEL_LANGUAGE_Q4;
            } else if (aotx_catalog_word_is(at, end, "embedding")) {
                row->role.model = AOTX_MODEL_EMBEDDING;
            } else if (aotx_catalog_word_is(at, end, "reranker")) {
                row->role.model = AOTX_MODEL_RERANKER;
            } else {
                return AOTX_CATALOG_WHY_VALUE;
            }
            return AOTX_CATALOG_WHY_NONE;
        }
        if (aotx_catalog_word_is(key_at, key_end, "tools")) {
            row->unknown += aotx_catalog_mask_of(at, end, row->role.tools,
                                                 AOTX_MODULE_TOOL);
            return AOTX_CATALOG_WHY_NONE;
        }
        if (aotx_catalog_word_is(key_at, key_end, "authorise")) {
            row->unknown += aotx_catalog_mask_of(at, end, row->role.needs_auth,
                                                 AOTX_MODULE_TOOL);
            return AOTX_CATALOG_WHY_NONE;
        }
        if (aotx_catalog_word_is(key_at, key_end, "budget")) {
            return aotx_catalog_count_of(at, end, &row->role.budget)
                 ? AOTX_CATALOG_WHY_NONE : AOTX_CATALOG_WHY_VALUE;
        }
        if (aotx_catalog_word_is(key_at, key_end, "pages")) {
            return aotx_catalog_count_of(at, end, &row->role.pages)
                 ? AOTX_CATALOG_WHY_NONE : AOTX_CATALOG_WHY_VALUE;
        }
        if (aotx_catalog_word_is(key_at, key_end, "skills")) {
            unsigned int walk = at;
            while (walk < end) {
                unsigned int piece = aotx_catalog_piece(walk, end);
                unsigned int first = aotx_catalog_blank(walk, piece);
                unsigned int last = aotx_catalog_trim(first, piece);
                if (last > first) {
                    unsigned int found =
                        aotx_catalog_find((const char *)aotx_catalog_arena + first,
                                          last - first, AOTX_MODULE_SKILL);
                    if (found >= AOTX_MODULE_SLOTS) {
                        row->unknown += 1u;
                    } else if (row->role.skills >= AOTX_CATALOG_ROLE_SKILLS) {
                        *figure = AOTX_CATALOG_ROLE_SKILLS;
                        return AOTX_CATALOG_WHY_SKILLS;
                    } else {
                        row->role.skill[row->role.skills] = found;
                        row->role.skills += 1u;
                        aotx_catalog_mask_set(row->role.skill_mask, found);
                    }
                }
                walk = piece + 1u;
            }
            return AOTX_CATALOG_WHY_NONE;
        }
    }
    /* The body key names the file the feeder read. The device opens no file, so the value
     * stands in the manifest and the run of the body comes from the import. */
    return AOTX_CATALOG_WHY_NONE;
}

__device__ unsigned int aotx_catalog_head_end(unsigned int at, unsigned int length)
{
    unsigned int end = at + length;
    unsigned int fences = 0u;
    unsigned int walk = at;
    while (walk < end) {
        unsigned int line = walk;
        while (walk < end && aotx_catalog_arena[walk] != (unsigned char)'\n') {
            walk += 1u;
        }
        unsigned int stop = aotx_catalog_trim(line, walk);
        walk += 1u;
        unsigned int first = aotx_catalog_blank(line, stop);
        if (first >= stop) {
            continue;
        }
        if (aotx_catalog_word_is(first, stop, "---") == 0) {
            /* A file whose first line of data is not a fence carries no head. The
             * caller then reads a manifest of no bytes and the checks refuse it. */
            if (fences == 0u) {
                return at;
            }
            continue;
        }
        fences += 1u;
        if (fences == 2u) {
            return (walk < end) ? walk : end;
        }
    }
    return at;
}

__device__ unsigned int aotx_catalog_manifest_read(aotx_catalog_entry *row,
                                                   unsigned int at, unsigned int length,
                                                   unsigned int kind, unsigned int *figure)
{
    unsigned int end = at + length;
    unsigned int head = row->name_len;
    unsigned int said_kind = 0u;
    unsigned int said_name = 0u;
    unsigned int said_other = 0u;
    unsigned int fence = 0u;
    *figure = 0u;

    row->unknown = 0u;
    row->description.at = 0u;
    row->description.length = 0u;
    row->version.at = 0u;
    row->version.length = 0u;
    row->tool.side = AOTX_CATALOG_SIDE_DEVICE;
    row->tool.authorize = AOTX_CATALOG_AUTH_NEVER;
    row->tool.deadline = 0u;
    row->tool.timeout = 0u;
    row->tool.arguments = 0u;
    row->tool.built_in = 0u;
    row->tool.entry.at = 0u;
    row->tool.entry.length = 0u;
    row->tool.module.at = 0u;
    row->tool.module.length = 0u;
    row->tool.program.at = 0u;
    row->tool.program.length = 0u;
    row->tool.example.at = 0u;
    row->tool.example.length = 0u;
    row->role.model = AOTX_MODEL_LANGUAGE;
    row->role.budget = 0u;
    row->role.pages = 0u;
    row->role.skills = 0u;
    aotx_catalog_mask_clear(row->role.tools);
    aotx_catalog_mask_clear(row->role.needs_auth);
    aotx_catalog_mask_clear(row->role.skill_mask);

    unsigned int walk = at;
    while (walk < end) {
        unsigned int line = walk;
        while (walk < end && aotx_catalog_arena[walk] != (unsigned char)'\n') {
            walk += 1u;
        }
        unsigned int stop = aotx_catalog_trim(line, walk);
        walk += 1u;
        unsigned int first = aotx_catalog_blank(line, stop);
        if (first >= stop || aotx_catalog_arena[first] == (unsigned char)'#') {
            continue;
        }
        /* A file of a skill starts with a fence of three dashes and holds two keys. The
         * text after the second fence is the body, which the import carries as a file of
         * its own. A fence therefore ends the head. */
        if (aotx_catalog_word_is(first, stop, "---")) {
            if (fence == 0u && said_kind == 0u && said_name == 0u) {
                if (kind != AOTX_MODULE_SKILL) {
                    return AOTX_CATALOG_WHY_KIND;
                }
                fence = 1u;
                said_kind = 1u;
                continue;
            }
            break;
        }
        unsigned int key_end = first;
        while (key_end < stop && aotx_catalog_arena[key_end] != (unsigned char)':') {
            key_end += 1u;
        }
        if (key_end >= stop) {
            return AOTX_CATALOG_WHY_SHAPE;
        }
        unsigned int key_last = aotx_catalog_trim(first, key_end);
        unsigned int value = aotx_catalog_blank(key_end + 1u, stop);
        if (key_last == first) {
            return AOTX_CATALOG_WHY_SHAPE;
        }
        /* The head of a skill file takes two keys and no other. */
        if (fence != 0u) {
            if (aotx_catalog_word_is(first, key_last, "name") == 0
                && aotx_catalog_word_is(first, key_last, "description") == 0) {
                return AOTX_CATALOG_WHY_KEY;
            }
        } else if (aotx_catalog_key_of_kind(first, key_last, kind) == 0) {
            return AOTX_CATALOG_WHY_KEY;
        }
        unsigned int why = aotx_catalog_pair(row, first, key_last, value, stop, kind, head,
                                             figure);
        if (why != AOTX_CATALOG_WHY_NONE) {
            return why;
        }
        if (aotx_catalog_word_is(first, key_last, "kind")) {
            said_kind = 1u;
        } else if (aotx_catalog_word_is(first, key_last, "name")) {
            said_name = 1u;
        } else if (aotx_catalog_word_is(first, key_last, "description") == 0) {
            said_other = 1u;
        }
    }
    /* The head of a skill file gives a name and a description and no kind. The head
     * record of the import gives the kind. A text of those two keys alone is therefore
     * the head of a skill, whether the fence lines stand around it or not. */
    if (said_kind == 0u && kind == AOTX_MODULE_SKILL && said_name != 0u
        && said_other == 0u) {
        said_kind = 1u;
    }
    if (said_kind == 0u || said_name == 0u) {
        return AOTX_CATALOG_WHY_MISSING;
    }
    row->kind = kind;
    return AOTX_CATALOG_WHY_NONE;
}

__device__ void aotx_catalog_unknown_names(const aotx_catalog_entry *row,
                                           aotx_cli_out *out)
{
    unsigned int at = row->manifest.at;
    unsigned int end = at + row->manifest.length;
    unsigned int said = 0u;
    while (at < end) {
        unsigned int line = at;
        while (at < end && aotx_catalog_arena[at] != (unsigned char)'\n') {
            at += 1u;
        }
        unsigned int stop = aotx_catalog_trim(line, at);
        at += 1u;
        unsigned int first = aotx_catalog_blank(line, stop);
        unsigned int key_end = first;
        while (key_end < stop && aotx_catalog_arena[key_end] != (unsigned char)':') {
            key_end += 1u;
        }
        if (key_end >= stop) {
            continue;
        }
        unsigned int key_last = aotx_catalog_trim(first, key_end);
        unsigned int kind = 0u;
        if (aotx_catalog_word_is(first, key_last, "tools")
            || aotx_catalog_word_is(first, key_last, "authorise")) {
            kind = AOTX_MODULE_TOOL;
        } else if (aotx_catalog_word_is(first, key_last, "skills")) {
            kind = AOTX_MODULE_SKILL;
        } else {
            continue;
        }
        unsigned int walk = aotx_catalog_blank(key_end + 1u, stop);
        while (walk < stop) {
            unsigned int piece = aotx_catalog_piece(walk, stop);
            unsigned int name = aotx_catalog_blank(walk, piece);
            unsigned int last = aotx_catalog_trim(name, piece);
            if (last > name
                && aotx_catalog_find((const char *)aotx_catalog_arena + name, last - name,
                                     kind) >= AOTX_MODULE_SLOTS) {
                aotx_cli_say(out, (said == 0u) ? "; not known: " : ", ");
                aotx_cli_add(out, (const char *)aotx_catalog_arena + name, last - name);
                said = 1u;
            }
            walk = piece + 1u;
        }
    }
}
