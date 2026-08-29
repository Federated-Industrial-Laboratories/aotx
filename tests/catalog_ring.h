/* Purpose: Give the catalog check the cases that go over the inbound ring and the apply.
 * Owns: The module texts of each case.
 * Threading: One host thread writes the ring while the pump makes ticks.
 * Lifetime: One run of the test program.
 *
 * The file is a part of the catalog check. It reads the kernels of that check, so it
 * comes after them in the same translation unit. */
#ifndef AOTX_TESTS_CATALOG_RING_H
#define AOTX_TESTS_CATALOG_RING_H

#include "catalog_kernels.h"

/* Read the counts of the catalog. */
static aotx_catalog_counts aotx_catalog_test_counts(void)
{
    aotx_catalog_state *state = aotx_test_catalog_read();
    aotx_catalog_counts counts = state->count;
    free(state);
    return counts;
}

/* Ticks a case waits for the ring to empty. */
#define AOTX_CATALOG_TEST_WAIT  600u

/* Report whether the free list of the arena is sound, and name the case when it is not.
 * A run that went back twice, or a run that overlaps the run beside it, fails here. */
static void aotx_catalog_test_sound(const char *what, unsigned int *applied,
                                    unsigned int *failed)
{
    unsigned int *out = (unsigned int *)aotx_catalog_test_take(sizeof(unsigned int));
    unsigned int sound = 0u;
    aotx_catalog_test_arena<<<1, 1>>>(out);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_check_runtime(cudaMemcpy(&sound, out, sizeof sound, cudaMemcpyDeviceToHost),
                       "cudaMemcpy");
    cudaFree(out);
    aotx_catalog_test_check(sound == 1u, what, applied, failed);
}

/* Run ticks until the apply took every record of the ring. */
static void aotx_catalog_test_settle(aotx_pump *pump, aotx_seam_rings *rings)
{
    aotx_inbound_preamble *preamble = (aotx_inbound_preamble *)rings->inbound_map;
    for (unsigned int i = 0u; i < AOTX_CATALOG_TEST_WAIT; ++i) {
        aotx_pump_tick(pump);
        unsigned long long head = __atomic_load_n(&preamble->head, __ATOMIC_ACQUIRE);
        unsigned long long took = __atomic_load_n(&preamble->consumed, __ATOMIC_ACQUIRE);
        if (took >= head) {
            aotx_pump_tick(pump);
            return;
        }
    }
}

/* Give the catalog back to the built-in tools alone. The free list must hold one run
 * after that, which proves a run that goes back joins the run beside it. */
static unsigned int aotx_catalog_test_reset(void)
{
    unsigned int *out = (unsigned int *)aotx_catalog_test_take(sizeof(unsigned int));
    unsigned int frees = 0u;
    aotx_catalog_test_clear<<<1, 1>>>(out);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_check_runtime(cudaMemcpy(&frees, out, sizeof frees, cudaMemcpyDeviceToHost),
                       "cudaMemcpy");
    cudaFree(out);
    return frees;
}

/* Build one skill module of a number, with its own name, description and body. */
static void aotx_catalog_test_skill(aotx_test_module *module, unsigned int which,
                                    unsigned int body_bytes)
{
    char name[64];
    char manifest[512];
    char *body = (char *)calloc(body_bytes + 1u, 1u);
    snprintf(name, sizeof name, "skill_%u", which);
    snprintf(manifest, sizeof manifest,
             "kind: skill\nname: %s\nversion: %u\ndescription: The skill of number %u.\n"
             "body: SKILL.md\n", name, which + 1u, which);
    for (unsigned int i = 0u; i < body_bytes; ++i) {
        body[i] = (char)('a' + (char)((which + i) % 26u));
    }
    aotx_test_module_text(module, AOTX_MODULE_SKILL, name, manifest, body);
    free(body);
}

/* The import case: a run of skills, each with its own name and its own body, over the
 * inbound ring and the apply node. */
static void aotx_catalog_test_import(aotx_pump *pump, aotx_seam_rings *rings,
                                     unsigned long long boot_id, unsigned int count,
                                     unsigned int *applied, unsigned int *failed)
{
    aotx_pump_report before;
    aotx_pump_report after;
    unsigned int frees = aotx_catalog_test_reset();
    aotx_catalog_test_check(frees == 1u,
                            "the arena free list holds one run after every module goes",
                            applied, failed);
    aotx_pump_read(&before);
    unsigned int records = 0u;
    for (unsigned int i = 0u; i < count; ++i) {
        aotx_test_module module;
        aotx_catalog_test_skill(&module, i, 64u + i);
        unsigned char *bodies = (unsigned char *)calloc(AOTX_TEST_IMPORT_MAX,
                                                        AOTX_BODY_BYTES);
        unsigned int *sizes = (unsigned int *)calloc(AOTX_TEST_IMPORT_MAX,
                                                     sizeof(unsigned int));
        records += aotx_test_import_build(&module, i + 1u, bodies, sizes);
        free(bodies);
        free(sizes);
        aotx_test_import_feed(rings, &module, i + 1u, boot_id);
        aotx_test_module_free(&module);
        /* The ring holds AOTX_INBOUND_SLOTS records. A run of imports longer than that
         * needs the apply to take some of them first. */
        if ((i % 64u) == 63u) {
            aotx_catalog_test_settle(pump, rings);
        }
    }
    aotx_catalog_test_settle(pump, rings);
    aotx_pump_read(&after);

    aotx_catalog_state *state = aotx_test_catalog_read();
    unsigned int good = 0u;
    unsigned int bytes = 0u;
    for (unsigned int i = 0u; i < count; ++i) {
        char name[64];
        snprintf(name, sizeof name, "skill_%u", i);
        unsigned int at = AOTX_MODULE_SLOTS;
        unsigned int length = (unsigned int)strlen(name);
        for (unsigned int k = 0u; k < AOTX_MODULE_SLOTS; ++k) {
            if (state->entry[k].state == AOTX_CATALOG_INSTALLED
                && state->entry[k].name_len == length
                && strncmp(state->entry[k].name, name, length) == 0) {
                at = k;
                break;
            }
        }
        if (at < AOTX_MODULE_SLOTS && state->entry[at].kind == AOTX_MODULE_SKILL
            && state->entry[at].body.length == 64u + i
            && state->entry[at].description.length != 0u
            && state->entry[at].version.length != 0u) {
            good += 1u;
            bytes += state->entry[at].body.length;
        }
    }
    aotx_catalog_test_check(good == count,
                            "every skill of the run went in with its own body", applied,
                            failed);
    /* Every record of an import is class A, so the apply folded each one into the state
     * hash and counted it. A record the fold missed changes the count. */
    aotx_catalog_test_check(after.applied - before.applied
                            == (unsigned long long)records,
                            "the apply folded every import record into the state hash",
                            applied, failed);
    aotx_catalog_test_check(after.state_hash != before.state_hash,
                            "the state hash moved with the imports", applied, failed);
    printf("catalog: %u of %u skills went in over %u records, %u body bytes, hash %llx to "
           "%llx\n", good, count, records, bytes, before.state_hash, after.state_hash);
    /* A table that holds every entry refuses one more import with the count of the
     * table. The bound of the profile is the bound the operator reads. */
    unsigned int taken = 0u;
    for (unsigned int k = 0u; k < AOTX_MODULE_SLOTS; ++k) {
        taken += (state->entry[k].state != AOTX_CATALOG_FREE) ? 1u : 0u;
    }
    free(state);
    if (taken != (unsigned int)AOTX_MODULE_SLOTS) {
        return;
    }
    aotx_test_module one;
    aotx_catalog_test_skill(&one, count + 1u, 32u);
    unsigned int gone = aotx_catalog_test_counts().refused;
    aotx_test_import_feed(rings, &one, count + 2u, boot_id);
    aotx_test_module_free(&one);
    aotx_catalog_test_settle(pump, rings);
    aotx_catalog_counts full = aotx_catalog_test_counts();
    aotx_catalog_test_check(full.refused == gone + 1u
                            && full.last_why == AOTX_CATALOG_WHY_TABLE,
                            "a table that holds every entry refuses one more import with "
                            "the reason of the table", applied, failed);
    printf("catalog: the table holds %u entries and refuses the next import\n", taken);
}

/* The three kinds over the ring: a skill, a role and a tool manifest. */
static void aotx_catalog_test_kinds(aotx_pump *pump, aotx_seam_rings *rings,
                                    unsigned long long boot_id, unsigned int *applied,
                                    unsigned int *failed)
{
    aotx_test_module module;
    aotx_catalog_test_reset();
    aotx_test_module_text(&module, AOTX_MODULE_SKILL, "how_to_count",
                          AOTX_CATALOG_TEST_SKILL, "Count the rows, then the columns.");
    aotx_test_import_feed(rings, &module, 1u, boot_id);
    aotx_test_module_free(&module);
    aotx_test_module_text(&module, AOTX_MODULE_TOOL, "word_count",
                          AOTX_CATALOG_TEST_TOOL_HOST, NULL);
    aotx_test_import_feed(rings, &module, 2u, boot_id);
    aotx_test_module_free(&module);
    aotx_test_module_text(&module, AOTX_MODULE_ROLE, "scribe", AOTX_CATALOG_TEST_ROLE,
                          "You are a scribe. You write things down.");
    aotx_test_import_feed(rings, &module, 3u, boot_id);
    aotx_test_module_free(&module);
    aotx_catalog_test_settle(pump, rings);

    unsigned int skill = aotx_test_catalog_entry("how_to_count", AOTX_MODULE_SKILL);
    unsigned int tool = aotx_test_catalog_entry("word_count", AOTX_MODULE_TOOL);
    unsigned int role = aotx_test_catalog_entry("scribe", AOTX_MODULE_ROLE);
    aotx_catalog_test_check(skill < AOTX_MODULE_SLOTS && tool < AOTX_MODULE_SLOTS
                            && role < AOTX_MODULE_SLOTS,
                            "the three kinds of module go in over the ring", applied,
                            failed);
    if (skill >= AOTX_MODULE_SLOTS || tool >= AOTX_MODULE_SLOTS
        || role >= AOTX_MODULE_SLOTS) {
        return;
    }
    aotx_catalog_state *state = aotx_test_catalog_read();
    unsigned int recall = aotx_test_catalog_entry("memory_recall", AOTX_MODULE_TOOL);
    unsigned int write = aotx_test_catalog_entry("memory_write", AOTX_MODULE_TOOL);
    unsigned int read = aotx_test_catalog_entry("fs_read", AOTX_MODULE_TOOL);
    const aotx_catalog_role *row = &state->entry[role].role;
    int holds = ((row->tools[recall >> 5] & (1u << (recall & 31u))) != 0u)
             && ((row->tools[write >> 5] & (1u << (write & 31u))) != 0u);
    int leaves = ((row->tools[read >> 5] & (1u << (read & 31u))) == 0u);
    int needs = ((row->needs_auth[write >> 5] & (1u << (write & 31u))) != 0u);
    aotx_catalog_test_check(holds && leaves && needs,
                            "the tool mask of a role names the tools of its manifest",
                            applied, failed);
    aotx_catalog_test_check(state->entry[tool].tool.arguments == 1u
                            && state->entry[tool].tool.authorize
                               == AOTX_CATALOG_AUTH_ALWAYS
                            && state->entry[tool].tool.deadline == 400u,
                            "the tool row holds the values of its manifest", applied,
                            failed);
    aotx_catalog_test_check(state->entry[role].role.overlay.length != 0u
                            && state->entry[skill].body.length != 0u,
                            "a role keeps an overlay run and a skill keeps a body run",
                            applied, failed);
    printf("catalog: the skill, the role and the tool are entries %u %u %u\n", skill, role,
           tool);
    free(state);
}

/* Every refusal the head and the parts can give, each with its reason. */
static void aotx_catalog_test_refusals(aotx_pump *pump, aotx_seam_rings *rings,
                                       unsigned long long boot_id, unsigned int *applied,
                                       unsigned int *failed)
{
    aotx_test_module module;
    aotx_catalog_test_reset();

    /* A body longer than the bound of a skill body. */
    unsigned int over = (unsigned int)AOTX_SKILL_BYTES + 1u;
    char *body = (char *)calloc(over + 1u, 1u);
    memset(body, 'x', over);
    aotx_test_module_text(&module, AOTX_MODULE_SKILL, "too_long",
                          "kind: skill\nname: too_long\n", body);
    aotx_test_import_feed(rings, &module, 11u, boot_id);
    aotx_test_module_free(&module);
    free(body);

    /* A manifest with a key the kind does not take. */
    aotx_test_module_text(&module, AOTX_MODULE_SKILL, "odd_key",
                          "kind: skill\nname: odd_key\ncolor: red\n", "a body");
    aotx_test_import_feed(rings, &module, 12u, boot_id);
    aotx_test_module_free(&module);

    /* A manifest whose name is not the name of the directory. */
    aotx_test_module_text(&module, AOTX_MODULE_SKILL, "outer",
                          "kind: skill\nname: inner\n", "a body");
    aotx_test_import_feed(rings, &module, 13u, boot_id);
    aotx_test_module_free(&module);

    /* A skill of one file, which carries no body. */
    aotx_test_module_text(&module, AOTX_MODULE_SKILL, "no_body",
                          "kind: skill\nname: no_body\n", NULL);
    aotx_test_import_feed(rings, &module, 14u, boot_id);
    aotx_test_module_free(&module);
    aotx_catalog_test_settle(pump, rings);

    aotx_catalog_state *state = aotx_test_catalog_read();
    unsigned int refused[4] = { AOTX_CATALOG_WHY_NONE, AOTX_CATALOG_WHY_NONE,
                                AOTX_CATALOG_WHY_NONE, AOTX_CATALOG_WHY_NONE };
    const char *names[4] = { "too_long", "odd_key", "outer", "no_body" };
    for (unsigned int i = 0u; i < 4u; ++i) {
        for (unsigned int k = 0u; k < AOTX_MODULE_SLOTS; ++k) {
            if (state->entry[k].state != AOTX_CATALOG_FREE
                && state->entry[k].name_len == (unsigned int)strlen(names[i])
                && strncmp(state->entry[k].name, names[i],
                           strlen(names[i])) == 0) {
                refused[i] = (state->entry[k].state == AOTX_CATALOG_REFUSED)
                           ? state->entry[k].why : AOTX_CATALOG_WHY_NONE;
                break;
            }
        }
    }
    free(state);
    aotx_catalog_test_check(refused[0] == AOTX_CATALOG_WHY_BODY,
                            "a body over the bound is refused with the bound", applied,
                            failed);
    aotx_catalog_test_check(refused[1] == AOTX_CATALOG_WHY_KEY,
                            "a manifest with an unknown key is refused", applied, failed);
    aotx_catalog_test_check(refused[2] == AOTX_CATALOG_WHY_HEAD,
                            "a manifest of another name is refused", applied, failed);
    aotx_catalog_test_check(refused[3] == AOTX_CATALOG_WHY_EMPTY,
                            "a skill that carries no body is refused", applied, failed);

    /* A second head of an import that arrives, and a part outside its file. The two
     * records are built by hand, because no directory gives them. */
    unsigned char *bodies = (unsigned char *)calloc(AOTX_TEST_IMPORT_MAX, AOTX_BODY_BYTES);
    unsigned int *sizes = (unsigned int *)calloc(AOTX_TEST_IMPORT_MAX,
                                                 sizeof(unsigned int));
    aotx_test_module_text(&module, AOTX_MODULE_SKILL, "twice",
                          "kind: skill\nname: twice\n", "a body of some length");
    unsigned int made = aotx_test_import_build(&module, 21u, bodies, sizes);
    aotx_test_feed_records(rings, AOTX_REC_IMPORT, AOTX_CLASS_A, AOTX_WRITER_FEEDER, 0u,
                           bodies, sizes[0], 1u, boot_id);
    aotx_test_feed_records(rings, AOTX_REC_IMPORT, AOTX_CLASS_A, AOTX_WRITER_FEEDER, 0u,
                           bodies, sizes[0], 1u, boot_id);
    aotx_catalog_test_settle(pump, rings);
    unsigned int arriving = aotx_test_catalog_entry("twice", AOTX_MODULE_SKILL);
    aotx_catalog_test_check(arriving >= AOTX_MODULE_SLOTS,
                            "a second head of one import number takes no entry", applied,
                            failed);
    /* A part that names bytes outside the file of the head. */
    aotx_import_part bad;
    memcpy(&bad, bodies + AOTX_BODY_BYTES, sizeof bad);
    bad.offset = 4096u;
    memcpy(bodies + AOTX_BODY_BYTES, &bad, sizeof bad);
    aotx_test_feed_records(rings, AOTX_REC_IMPORT, AOTX_CLASS_A, AOTX_WRITER_FEEDER, 0u,
                           bodies + AOTX_BODY_BYTES, sizes[1], 1u, boot_id);
    aotx_catalog_test_settle(pump, rings);
    state = aotx_test_catalog_read();
    unsigned int why = AOTX_CATALOG_WHY_NONE;
    for (unsigned int k = 0u; k < AOTX_MODULE_SLOTS; ++k) {
        if (state->entry[k].name_len == 5u
            && strncmp(state->entry[k].name, "twice", 5u) == 0) {
            why = state->entry[k].why;
        }
    }
    free(state);
    aotx_catalog_test_check(why == AOTX_CATALOG_WHY_PART,
                            "a part outside its file refuses the import", applied, failed);
    aotx_test_module_free(&module);
    (void)made;

    /* An arena that cannot hold the manifest. The head names the bytes and no part goes. */
    aotx_import_head huge;
    memset(&huge, 0, sizeof huge);
    huge.import = 31u;
    huge.part = 0u;
    huge.kind = AOTX_MODULE_TOOL;
    huge.files = 1u;
    /* The head asks for one byte more than the arena holds. The figure of the refusal
     * then tells the bytes that were asked for from the bytes the arena has. */
    huge.file_bytes[0] = (unsigned int)AOTX_CATALOGUE_BYTES + 1u;
    snprintf(huge.name, sizeof huge.name, "enormous");
    aotx_test_feed_records(rings, AOTX_REC_IMPORT, AOTX_CLASS_A, AOTX_WRITER_FEEDER, 0u,
                           &huge, (unsigned int)sizeof huge, 1u, boot_id);
    aotx_catalog_test_settle(pump, rings);
    state = aotx_test_catalog_read();
    unsigned int arena_why = AOTX_CATALOG_WHY_NONE;
    unsigned int arena_figure = 0u;
    for (unsigned int k = 0u; k < AOTX_MODULE_SLOTS; ++k) {
        if (state->entry[k].name_len == 8u
            && strncmp(state->entry[k].name, "enormous", 8u) == 0) {
            arena_why = state->entry[k].why;
            arena_figure = state->entry[k].figure;
        }
    }
    free(state);
    aotx_catalog_test_check(arena_why == AOTX_CATALOG_WHY_ARENA
                            && arena_figure == (unsigned int)AOTX_CATALOGUE_BYTES + 1u,
                            "an arena that holds no run refuses the import with the "
                            "figure", applied, failed);
    printf("catalog: the commit refused a long body, an unknown key, a name that is not "
           "the directory, a body that is missing, a second head, a part outside its file "
           "and an arena that is full\n");
    free(bodies);
    free(sizes);
}

/* An import of a name that stands replaces that module whole. */
static void aotx_catalog_test_replace(aotx_pump *pump, aotx_seam_rings *rings,
                                      unsigned long long boot_id, unsigned int *applied,
                                      unsigned int *failed)
{
    aotx_test_module module;
    aotx_catalog_test_reset();
    aotx_catalog_state *empty = aotx_test_catalog_read();
    unsigned int base = empty->used;
    free(empty);

    aotx_test_module_text(&module, AOTX_MODULE_SKILL, "shared",
                          "kind: skill\nname: shared\nversion: 1\n"
                          "description: The first text.\n", "the first body");
    aotx_test_import_feed(rings, &module, 41u, boot_id);
    aotx_test_module_free(&module);
    aotx_catalog_test_settle(pump, rings);
    unsigned int first = aotx_test_catalog_entry("shared", AOTX_MODULE_SKILL);

    aotx_test_module_text(&module, AOTX_MODULE_SKILL, "shared",
                          "kind: skill\nname: shared\nversion: 2\n"
                          "description: The second text, which is longer.\n",
                          "the second body, which is longer than the first");
    aotx_test_import_feed(rings, &module, 42u, boot_id);
    aotx_test_module_free(&module);
    aotx_catalog_test_settle(pump, rings);
    unsigned int again = aotx_test_catalog_entry("shared", AOTX_MODULE_SKILL);

    aotx_catalog_state *state = aotx_test_catalog_read();
    int same = (first == again) && (first < AOTX_MODULE_SLOTS);
    int grew = same && state->entry[first].body.length
               == (unsigned int)strlen("the second body, which is longer than the first");
    unsigned int held = state->count.replaced;
    unsigned int used = state->used;
    free(state);
    aotx_catalog_test_check(same, "a second import of a name takes the entry of the first",
                            applied, failed);
    aotx_catalog_test_check(grew, "the entry holds the body of the second import alone",
                            applied, failed);
    aotx_catalog_test_check(held != 0u, "the catalog counts the module it replaced",
                            applied, failed);

    /* The runs of the module that went are back in the free list. A remove of the entry
     * then gives the arena the bytes it had before the first import. */
    aotx_test_remove_feed(rings, "shared", boot_id);
    aotx_catalog_test_settle(pump, rings);
    state = aotx_test_catalog_read();
    aotx_catalog_test_check(state->used == base && state->frees == 1u,
                            "the arena holds the bytes it held before the replace",
                            applied, failed);
    printf("catalog: replace took entry %u twice, the arena went from %u to %u bytes and "
           "back to %u\n", first, base, used, state->used);
    free(state);
}

#endif
