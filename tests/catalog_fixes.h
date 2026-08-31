/* Purpose: Give the catalog check the skill file, the arrival and the import line.
 * Owns: The module texts and the lines of each case.
 * Threading: One host thread writes the ring while the pump makes ticks.
 * Lifetime: One run of the test program.
 *
 * The file is a part of the catalog check. It reads the ring cases of that check, so it
 * comes after them in the same translation unit. */
#ifndef AOTX_TESTS_CATALOG_FIXES_H
#define AOTX_TESTS_CATALOG_FIXES_H

#include "catalog_ring.h"

/* The head and the body of a skill file, as an operator writes them. */
#define AOTX_CATALOG_TEST_FILE \
    "---\n" \
    "name: open_skill\n" \
    "description: The shape a skill file of another agent system holds.\n" \
    "---\n" \
    "Read the head of the table first.\nThen read the rows.\n"

#define AOTX_CATALOG_TEST_FILE_BODY "Read the head of the table first.\nThen read the rows.\n"

/* Put one input line in the inbound ring, as the feeder does. */
static void aotx_test_line_feed(aotx_seam_rings *rings, const char *line,
                                unsigned long long boot_id)
{
    char body[AOTX_BODY_BYTES];
    unsigned int length = (unsigned int)strlen(line);
    memset(body, 0, sizeof body);
    if (length > (unsigned int)AOTX_BODY_BYTES) {
        length = (unsigned int)AOTX_BODY_BYTES;
    }
    memcpy(body, line, length);
    aotx_test_feed_records(rings, AOTX_REC_INPUT_LINE, AOTX_CLASS_A, AOTX_WRITER_FEEDER,
                           0u, body, length, 1u, boot_id);
}

/* The skill file case: a directory that holds the skill file alone. The feeder sends no
 * manifest and the whole file in the second place, and the device splits the head from
 * the body. */
static void aotx_catalog_test_skill_file(aotx_pump *pump, aotx_seam_rings *rings,
                                         unsigned long long boot_id, unsigned int *applied,
                                         unsigned int *failed)
{
    aotx_test_module module;
    aotx_catalog_test_reset();

    aotx_test_module_text(&module, AOTX_MODULE_SKILL, "open_skill", "",
                          AOTX_CATALOG_TEST_FILE);
    aotx_test_import_feed(rings, &module, 71u, boot_id);
    aotx_test_module_free(&module);
    aotx_catalog_test_settle(pump, rings);

    unsigned int at = aotx_test_catalog_entry("open_skill", AOTX_MODULE_SKILL);
    aotx_catalog_test_check(at < AOTX_MODULE_SLOTS,
                            "a directory of the skill file alone installs", applied,
                            failed);
    if (at < AOTX_MODULE_SLOTS) {
        aotx_catalog_state *state = aotx_test_catalog_read();
        unsigned int body = state->entry[at].body.length;
        unsigned int head = state->entry[at].manifest.length;
        aotx_catalog_test_check(body == (unsigned int)strlen(AOTX_CATALOG_TEST_FILE_BODY)
                                && head == (unsigned int)strlen(AOTX_CATALOG_TEST_FILE)
                                           - body
                                && state->entry[at].description.length != 0u,
                                "the device splits the head of the file from its body",
                                applied, failed);
        printf("catalog: the skill file gave a head of %u bytes and a body of %u\n", head,
               body);
        free(state);
    }

    /* The guard bites: a head of three keys, and a file with no head at all. */
    aotx_test_module_text(&module, AOTX_MODULE_SKILL, "third_key", "",
                          "---\nname: third_key\ndescription: A skill.\nlicense: none\n"
                          "---\nA body.\n");
    aotx_test_import_feed(rings, &module, 72u, boot_id);
    aotx_test_module_free(&module);
    aotx_test_module_text(&module, AOTX_MODULE_SKILL, "no_head", "",
                          "There is no head on this file.\n");
    aotx_test_import_feed(rings, &module, 73u, boot_id);
    aotx_test_module_free(&module);
    aotx_catalog_test_settle(pump, rings);

    aotx_catalog_state *state = aotx_test_catalog_read();
    unsigned int third = AOTX_CATALOG_WHY_NONE;
    unsigned int bare = AOTX_CATALOG_WHY_NONE;
    for (unsigned int k = 0u; k < AOTX_MODULE_SLOTS; ++k) {
        if (state->entry[k].name_len == 9u
            && strncmp(state->entry[k].name, "third_key", 9u) == 0) {
            third = state->entry[k].why;
        }
        if (state->entry[k].name_len == 7u
            && strncmp(state->entry[k].name, "no_head", 7u) == 0) {
            bare = state->entry[k].why;
        }
    }
    free(state);
    aotx_catalog_test_check(third == AOTX_CATALOG_WHY_KEY,
                            "a head of a skill file with a third key is refused", applied,
                            failed);
    aotx_catalog_test_check(bare == AOTX_CATALOG_WHY_MISSING,
                            "a skill file with no head is refused", applied, failed);
    aotx_catalog_test_sound("the arena is sound after the skill file case", applied,
                            failed);
}

/* The arrival case: a remove and a second head that reach an import between its head and
 * its last part. Neither may give a run of the arena back twice. */
static void aotx_catalog_test_arriving(aotx_pump *pump, aotx_seam_rings *rings,
                                       unsigned long long boot_id, unsigned int *applied,
                                       unsigned int *failed)
{
    unsigned char *bodies = (unsigned char *)calloc(AOTX_TEST_IMPORT_MAX, AOTX_BODY_BYTES);
    unsigned int *sizes = (unsigned int *)calloc(AOTX_TEST_IMPORT_MAX,
                                                 sizeof(unsigned int));
    aotx_test_module module;
    aotx_catalog_test_reset();
    aotx_catalog_test_free_requests<<<1, AOTX_SLOTS>>>();
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    /* One module of that name stands. */
    aotx_test_module_text(&module, AOTX_MODULE_SKILL, "twice_x",
                          "kind: skill\nname: twice_x\nversion: 1\n"
                          "description: The first text.\n", "the first body of twice_x");
    aotx_test_import_feed(rings, &module, 81u, boot_id);
    aotx_test_module_free(&module);
    aotx_catalog_test_settle(pump, rings);
    unsigned int first = aotx_test_catalog_entry("twice_x", AOTX_MODULE_SKILL);

    /* A second import of that name arrives, all but its last part. */
    aotx_test_module_text(&module, AOTX_MODULE_SKILL, "twice_x",
                          "kind: skill\nname: twice_x\nversion: 2\n"
                          "description: The second text.\n",
                          "the second body of twice_x, which is longer than the first");
    unsigned int made = aotx_test_import_build(&module, 82u, bodies, sizes);
    for (unsigned int i = 0u; i + 1u < made; ++i) {
        aotx_test_feed_records(rings, AOTX_REC_IMPORT, AOTX_CLASS_A, AOTX_WRITER_FEEDER,
                               0u, bodies + (size_t)i * AOTX_BODY_BYTES, sizes[i], 1u,
                               boot_id);
    }
    aotx_catalog_test_settle(pump, rings);

    /* A remove of that name reaches the entry while it arrives. */
    aotx_catalog_counts before = aotx_catalog_test_counts();
    aotx_test_remove_feed(rings, "twice_x", boot_id);
    aotx_catalog_test_settle(pump, rings);
    aotx_catalog_counts after = aotx_catalog_test_counts();
    aotx_catalog_state *state = aotx_test_catalog_read();
    int arriving = (first < AOTX_MODULE_SLOTS)
                && (state->entry[first].state == AOTX_CATALOG_ARRIVING);
    free(state);
    aotx_catalog_test_check(after.gone == before.gone + 1u
                            && after.removed == before.removed && arriving != 0,
                            "a remove of a name whose import arrives is held back",
                            applied, failed);
    aotx_catalog_test_sound("the arena is sound after the remove of an arrival", applied,
                            failed);

    /* The last part lands and the module of the second import stands alone. */
    aotx_test_feed_records(rings, AOTX_REC_IMPORT, AOTX_CLASS_A, AOTX_WRITER_FEEDER, 0u,
                           bodies + (size_t)(made - 1u) * AOTX_BODY_BYTES,
                           sizes[made - 1u], 1u, boot_id);
    aotx_catalog_test_settle(pump, rings);
    unsigned int again = aotx_test_catalog_entry("twice_x", AOTX_MODULE_SKILL);
    state = aotx_test_catalog_read();
    int right = (again == first) && (again < AOTX_MODULE_SLOTS)
             && (state->entry[again].body.length
                 == (unsigned int)strlen("the second body of twice_x, which is longer "
                                         "than the first"));
    free(state);
    aotx_catalog_test_check(right, "the import that arrived commits its own body", applied,
                            failed);
    aotx_catalog_test_sound("the arena is sound after the commit of that import", applied,
                            failed);
    aotx_test_module_free(&module);

    /* A second head of the same name cancels an arrival whole. */
    aotx_test_module_text(&module, AOTX_MODULE_SKILL, "twice_x",
                          "kind: skill\nname: twice_x\nversion: 3\n"
                          "description: The third text.\n", "the third body of twice_x");
    /* The head of import 83 claims the entry and its parts stay away. The head of import
     * 84, of the same name, then cancels that arrival and takes the entry itself. */
    made = aotx_test_import_build(&module, 83u, bodies, sizes);
    aotx_test_feed_records(rings, AOTX_REC_IMPORT, AOTX_CLASS_A, AOTX_WRITER_FEEDER, 0u,
                           bodies, sizes[0], 1u, boot_id);
    aotx_catalog_test_settle(pump, rings);
    before = aotx_catalog_test_counts();
    made = aotx_test_import_build(&module, 84u, bodies, sizes);
    for (unsigned int i = 0u; i < made; ++i) {
        aotx_test_feed_records(rings, AOTX_REC_IMPORT, AOTX_CLASS_A, AOTX_WRITER_FEEDER,
                               0u, bodies + (size_t)i * AOTX_BODY_BYTES, sizes[i], 1u,
                               boot_id);
    }
    aotx_catalog_test_settle(pump, rings);
    after = aotx_catalog_test_counts();
    aotx_catalog_test_check(after.cancelled == before.cancelled + 1u
                            && after.installed == before.installed + 1u,
                            "a head of a name that arrives cancels that arrival and takes "
                            "the entry", applied, failed);
    aotx_catalog_test_sound("the arena is sound after the cancelled arrival", applied,
                            failed);
    aotx_test_module_free(&module);
    printf("catalog: the arrival held back one remove and one head cancelled it, and the "
           "arena stayed sound\n");
    free(bodies);
    free(sizes);
}


/* The batch case. A line that stands after the import of the role of the console, in one
 * batch of inputs, finds the agent of that role. */
static void aotx_catalog_test_same_batch(aotx_pump *pump, aotx_seam_rings *rings,
                                         unsigned long long boot_id, const char *modules,
                                         unsigned int *applied, unsigned int *failed)
{
    static const char *const names[3] = { "conductor", "verifier", "worker" };
    char dir[1024];
    aotx_test_module module;
    aotx_catalog_test_reset();
    aotx_catalog_test_free_requests<<<1, AOTX_SLOTS>>>();
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    /* Every record of the three imports and the line after them go in the ring first.
     * The apply of the next tick then reads them as one batch. */
    unsigned long long mark = aotx_catalog_test_mark();
    for (unsigned int i = 0u; i < 3u; ++i) {
        snprintf(dir, sizeof dir, "%s/roles/%s", modules, names[i]);
        if (aotx_test_module_dir(&module, dir) != 0) {
            continue;
        }
        aotx_test_import_feed(rings, &module, 91u + i, boot_id);
        aotx_test_module_free(&module);
    }
    aotx_test_line_feed(rings, "agents", boot_id);
    aotx_catalog_test_settle(pump, rings);

    aotx_agent_table *table = (aotx_agent_table *)calloc(1, sizeof *table);
    aotx_check_runtime(cudaMemcpyFromSymbol(table, aotx_agents, sizeof *table),
                       "cudaMemcpyFromSymbol");
    unsigned int conductor = aotx_test_catalog_entry("conductor", AOTX_MODULE_ROLE);
    aotx_catalog_test_check(conductor < AOTX_MODULE_SLOTS
                            && table->agent[0].state != AOTX_AGENT_STATE_FREE
                            && table->agent[0].role == conductor,
                            "the agent of the console stands after the import of its role",
                            applied, failed);
    free(table);
    aotx_catalog_test_check(aotx_catalog_test_said(mark, "0 conductor idle", 1)
                            && aotx_catalog_test_said(mark, "no agents", 0),
                            "a line of the same batch as that import sees the agent",
                            applied, failed);
    printf("catalog: the three roles and one line went in one batch, and the line saw the "
           "agent of the console\n");
}

/* The import line case. A line of a surface the feeder does not read gives one request
 * record. It states no result before the feeder answers. */
static void aotx_catalog_test_import_line(aotx_pump *pump, aotx_seam_rings *rings,
                                          unsigned long long boot_id,
                                          unsigned int *applied, unsigned int *failed)
{
    static const char path[] = "/modules/one";
    unsigned char *bytes = (unsigned char *)aotx_catalog_test_take(AOTX_BODY_BYTES);
    unsigned int *mark = (unsigned int *)aotx_catalog_test_take(sizeof(unsigned int));
    aotx_tool_request_body body;
    unsigned int held = 0u;
    unsigned long long line = 0ull;

    line = aotx_catalog_test_mark();
    aotx_catalog_counts before = aotx_catalog_test_counts();
    aotx_test_line_feed(rings, "import /modules/one", boot_id);
    aotx_catalog_test_settle(pump, rings);
    aotx_catalog_counts after = aotx_catalog_test_counts();
    aotx_catalog_test_last<<<1, 1>>>(AOTX_REC_TOOL_REQUEST, bytes, mark);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_check_runtime(cudaMemcpy(&held, mark, sizeof held, cudaMemcpyDeviceToHost),
                       "cudaMemcpy");
    aotx_check_runtime(cudaMemcpy(&body, bytes, sizeof body, cudaMemcpyDeviceToHost),
                       "cudaMemcpy");
    aotx_catalog_test_check(held == 1u && body.tool == AOTX_TOOL_IMPORT
                            && body.agent == AOTX_REQUEST_NO_AGENT
                            && body.arg_len == (unsigned int)strlen(path)
                            && memcmp(body.arg, path, strlen(path)) == 0
                            && after.asked == before.asked + 1u,
                            "an import line writes one request record for the feeder",
                            applied, failed);
    aotx_catalog_test_check(aotx_catalog_test_said(line, "import:", 0),
                            "the console states no result before the feeder answers",
                            applied, failed);

    /* A replay of the journal writes no request. The import records of the run stand in
     * the journal, and the feeder reads no directory a second time. */
    before = aotx_catalog_test_counts();
    aotx_seam_set_replaying(1);
    aotx_test_line_feed(rings, "import /modules/two", boot_id);
    aotx_catalog_test_settle(pump, rings);
    aotx_seam_set_replaying(0);
    after = aotx_catalog_test_counts();
    aotx_catalog_test_check(after.asked == before.asked,
                            "a replayed import line writes no request", applied, failed);

    /* The report of an import the feeder refused reaches the console and the bus. */
    line = aotx_catalog_test_mark();
    before = aotx_catalog_test_counts();
    aotx_test_line_feed(rings,
                        "import /modules/three refused: the file is not there", boot_id);
    aotx_catalog_test_settle(pump, rings);
    after = aotx_catalog_test_counts();
    aotx_catalog_test_last<<<1, 1>>>(AOTX_REC_NOTE, bytes, mark);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_catalog_test_check(after.asked == before.asked,
                            "the report of a refused import writes no request", applied,
                            failed);
    aotx_catalog_test_check(aotx_catalog_test_said(
                                line,
                                "import: the directory is not readable: the file is not there",
                                1),
                            "the report of a refused import reaches the console", applied,
                            failed);
    printf("catalog: the import line wrote one request for the feeder and the report of a "
           "refusal reached the console\n");
    cudaFree(bytes);
    cudaFree(mark);
}

/* The reasons case: the row table of the imports that arrive, and the runs of a refused
 * entry. */
static void aotx_catalog_test_reasons(aotx_pump *pump, aotx_seam_rings *rings,
                                      unsigned long long boot_id, unsigned int *applied,
                                      unsigned int *failed)
{
    unsigned char *bodies = (unsigned char *)calloc(AOTX_TEST_IMPORT_MAX, AOTX_BODY_BYTES);
    unsigned int *sizes = (unsigned int *)calloc(AOTX_TEST_IMPORT_MAX,
                                                 sizeof(unsigned int));
    char name[64];
    char manifest[256];
    aotx_test_module module;
    aotx_catalog_test_reset();

    /* Every row of the imports that arrive is taken, and one more head names that. */
    for (unsigned int i = 0u; i <= AOTX_CATALOG_ARRIVING_MAX; ++i) {
        snprintf(name, sizeof name, "waits_%u", i);
        snprintf(manifest, sizeof manifest, "kind: skill\nname: %s\n", name);
        aotx_test_module_text(&module, AOTX_MODULE_SKILL, name, manifest, "a body");
        aotx_test_import_build(&module, 201u + i, bodies, sizes);
        aotx_test_feed_records(rings, AOTX_REC_IMPORT, AOTX_CLASS_A, AOTX_WRITER_FEEDER,
                               0u, bodies, sizes[0], 1u, boot_id);
        aotx_test_module_free(&module);
    }
    aotx_catalog_test_settle(pump, rings);
    aotx_catalog_counts counts = aotx_catalog_test_counts();
    aotx_catalog_test_check(counts.last_why == AOTX_CATALOG_WHY_BUSY,
                            "a head that finds no free row names the rows and not the "
                            "entries", applied, failed);

    /* A part outside its file leaves an entry with no run of the arena in any field. The
     * entry held a module before that import, so its description and its version pointed
     * into runs that went back with it. */
    aotx_catalog_test_reset();
    aotx_test_module_text(&module, AOTX_MODULE_SKILL, "stale_v",
                          "kind: skill\nname: stale_v\nversion: 9\n"
                          "description: A text that goes back.\n", "a body of some length");
    aotx_test_import_feed(rings, &module, 210u, boot_id);
    aotx_catalog_test_settle(pump, rings);
    aotx_catalog_state *stood = aotx_test_catalog_read();
    unsigned int before_runs = 0u;
    for (unsigned int k = 0u; k < AOTX_MODULE_SLOTS; ++k) {
        if (stood->entry[k].name_len == 7u
            && strncmp(stood->entry[k].name, "stale_v", 7u) == 0) {
            before_runs = stood->entry[k].description.length + stood->entry[k].version.length;
        }
    }
    free(stood);
    aotx_catalog_test_check(before_runs != 0u,
                            "the entry held a description and a version before the import "
                            "that failed", applied, failed);
    unsigned int made = aotx_test_import_build(&module, 211u, bodies, sizes);
    aotx_test_feed_records(rings, AOTX_REC_IMPORT, AOTX_CLASS_A, AOTX_WRITER_FEEDER, 0u,
                           bodies, sizes[0], 1u, boot_id);
    aotx_import_part bad;
    memcpy(&bad, bodies + AOTX_BODY_BYTES, sizeof bad);
    bad.offset = 4096u;
    memcpy(bodies + AOTX_BODY_BYTES, &bad, sizeof bad);
    aotx_test_feed_records(rings, AOTX_REC_IMPORT, AOTX_CLASS_A, AOTX_WRITER_FEEDER, 0u,
                           bodies + AOTX_BODY_BYTES, sizes[1], 1u, boot_id);
    aotx_catalog_test_settle(pump, rings);
    (void)made;
    aotx_catalog_state *state = aotx_test_catalog_read();
    unsigned int runs = 1u;
    for (unsigned int k = 0u; k < AOTX_MODULE_SLOTS; ++k) {
        if (state->entry[k].name_len == 7u
            && strncmp(state->entry[k].name, "stale_v", 7u) == 0) {
            runs = state->entry[k].description.length + state->entry[k].version.length
                 + state->entry[k].manifest.length + state->entry[k].body.length;
        }
    }
    free(state);
    aotx_catalog_test_check(runs == 0u,
                            "a refused entry keeps no run of the arena in any field",
                            applied, failed);
    aotx_catalog_test_sound("the arena is sound after the reasons case", applied, failed);
    printf("catalog: the row table and the refused entry each name what they hold\n");
    free(bodies);
    free(sizes);
}


/* The restore case. An import whose last part is not in the journal never lands, and the
 * number of an import is unique while that import arrives. Every such import goes out of
 * the catalog when the replay ends. */
static void aotx_catalog_test_restore(aotx_pump *pump, aotx_seam_rings *rings,
                                      unsigned long long boot_id, unsigned int *applied,
                                      unsigned int *failed)
{
    unsigned char *bodies = (unsigned char *)calloc(AOTX_TEST_IMPORT_MAX, AOTX_BODY_BYTES);
    unsigned int *sizes = (unsigned int *)calloc(AOTX_TEST_IMPORT_MAX,
                                                 sizeof(unsigned int));
    aotx_test_module module;
    aotx_catalog_test_reset();

    /* The head of an import goes in and its parts do not. A run that was killed in the
     * middle of an import leaves the journal in that state. */
    aotx_test_module_text(&module, AOTX_MODULE_SKILL, "half_way",
                          "kind: skill\nname: half_way\n", "a body that never arrives");
    aotx_test_import_build(&module, 221u, bodies, sizes);
    aotx_test_feed_records(rings, AOTX_REC_IMPORT, AOTX_CLASS_A, AOTX_WRITER_FEEDER, 0u,
                           bodies, sizes[0], 1u, boot_id);
    aotx_test_module_free(&module);
    aotx_catalog_test_settle(pump, rings);
    aotx_catalog_state *state = aotx_test_catalog_read();
    unsigned int held = 0u;
    for (unsigned int k = 0u; k < AOTX_MODULE_SLOTS; ++k) {
        held += (state->entry[k].state == AOTX_CATALOG_ARRIVING) ? 1u : 0u;
    }
    free(state);
    aotx_catalog_test_check(held == 1u, "the head of an import that does not land arrives",
                            applied, failed);

    /* The record of the restore ends the replay. */
    unsigned long long mark = aotx_catalog_test_mark();
    aotx_catalog_counts before = aotx_catalog_test_counts();
    aotx_restore_body report;
    memset(&report, 0, sizeof report);
    report.restored_boot_id = boot_id;
    report.last_tick = 1ull;
    report.replayed_count = 1ull;
    aotx_test_feed_records(rings, AOTX_REC_RESTORE, AOTX_CLASS_B, AOTX_WRITER_RESTORE, 0u,
                           &report, (unsigned int)sizeof report, 1u, boot_id);
    aotx_catalog_test_settle(pump, rings);
    aotx_catalog_counts after = aotx_catalog_test_counts();
    state = aotx_test_catalog_read();
    unsigned int left = 0u;
    for (unsigned int k = 0u; k < AOTX_MODULE_SLOTS; ++k) {
        left += (state->entry[k].state == AOTX_CATALOG_ARRIVING) ? 1u : 0u;
    }
    unsigned int rows = 0u;
    for (unsigned int k = 0u; k < AOTX_CATALOG_ARRIVING_MAX; ++k) {
        rows += (state->arriving[k].import != 0u) ? 1u : 0u;
    }
    free(state);
    aotx_catalog_test_check(left == 0u && rows == 0u
                            && after.dropped == before.dropped + 1u,
                            "the end of a replay drops the import that did not land",
                            applied, failed);
    aotx_catalog_test_check(aotx_catalog_test_said(mark, "incomplete imports were removed", 1),
                            "the console names the imports that went out", applied,
                            failed);
    aotx_catalog_test_sound("the arena is sound after the end of the replay", applied,
                            failed);

    /* The number of that import is free again, so an import of the same number lands. */
    aotx_test_module_text(&module, AOTX_MODULE_SKILL, "half_way",
                          "kind: skill\nname: half_way\n", "a body that arrives whole");
    aotx_test_import_feed(rings, &module, 221u, boot_id);
    aotx_test_module_free(&module);
    aotx_catalog_test_settle(pump, rings);
    aotx_catalog_test_check(aotx_test_catalog_entry("half_way", AOTX_MODULE_SKILL)
                            < AOTX_MODULE_SLOTS,
                            "an import of that number lands after the replay ended",
                            applied, failed);
    printf("catalog: the end of the replay dropped one import and its number came free\n");
    free(bodies);
    free(sizes);
}

/* The room case. A body that the room of a prompt cannot hold is cut, and the bytes that
 * go in say so. The role of the case allows every tool of the catalog, so its system block
 * is long and the room its turn keeps is short. */
static void aotx_catalog_test_room(aotx_pump *pump, aotx_seam_rings *rings,
                                   unsigned long long boot_id, unsigned int *applied,
                                   unsigned int *failed)
{
    static const char words[] = "the result is cut to the room of this prompt";
    static const char turn[] = "count the rows";
    char manifest[1024];
    char name[64];
    aotx_test_module module;
    aotx_catalog_test_reset();
    aotx_catalog_test_free_requests<<<1, AOTX_SLOTS>>>();
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    /* Twenty host tools with a long description, and a role that allows every one of them. A
     * device tool takes a node of the tick graph, and the graph holds a bounded count of
     * them. The tools of this case therefore run on the disk side, which has no such bound. */
    for (unsigned int i = 0u; i < 20u; ++i) {
        snprintf(name, sizeof name, "wide_%u", i);
        snprintf(manifest, sizeof manifest,
                 "kind: tool\nname: %s\nside: host\nprogram: run\narguments: text\n"
                 "description: The tool of number %u, which has a description that is "
                 "long enough to fill the block of a prompt after a few of them.\n",
                 name, i);
        aotx_test_module_text(&module, AOTX_MODULE_TOOL, name, manifest, NULL);
        aotx_test_import_feed(rings, &module, 241u + i, boot_id);
        aotx_test_module_free(&module);
    }
    aotx_catalog_test_settle(pump, rings);
    snprintf(manifest, sizeof manifest, "kind: role\nname: broad\nmodel: language\n"
             "tools: wide_0,wide_1,wide_2,wide_3,wide_4,wide_5,wide_6,wide_7,wide_8,"
             "wide_9,wide_10,wide_11,wide_12,wide_13,wide_14,wide_15,wide_16,wide_17,"
             "wide_18,wide_19\nbody: overlay.txt\n");
    aotx_test_module_text(&module, AOTX_MODULE_ROLE, "broad", manifest,
                          "You are broad. You call every tool.");
    aotx_test_import_feed(rings, &module, 271u, boot_id);
    aotx_test_module_free(&module);

    /* A body of the whole bound of a skill, and a body of a few bytes. */
    char *big = (char *)calloc((size_t)AOTX_SKILL_BYTES + 1u, 1u);
    for (unsigned int i = 0u; i < (unsigned int)AOTX_SKILL_BYTES; ++i) {
        big[i] = (char)('a' + (char)(i % 26u));
    }
    aotx_test_module_text(&module, AOTX_MODULE_SKILL, "long_one",
                          "kind: skill\nname: long_one\ndescription: A long skill.\n", big);
    aotx_test_import_feed(rings, &module, 272u, boot_id);
    aotx_test_module_free(&module);
    aotx_test_module_text(&module, AOTX_MODULE_SKILL, "short_one",
                          "kind: skill\nname: short_one\ndescription: A short skill.\n",
                          "Read the head of the table.");
    aotx_test_import_feed(rings, &module, 273u, boot_id);
    aotx_test_module_free(&module);
    aotx_catalog_test_settle(pump, rings);
    free(big);

    unsigned int broad = aotx_test_catalog_entry("broad", AOTX_MODULE_ROLE);
    aotx_catalog_test_check(broad < AOTX_MODULE_SLOTS
                            && aotx_test_catalog_entry("long_one", AOTX_MODULE_SKILL)
                               < AOTX_MODULE_SLOTS,
                            "the role and the long skill of the room case went in",
                            applied, failed);
    if (broad >= AOTX_MODULE_SLOTS) {
        return;
    }

    unsigned int *out = (unsigned int *)aotx_catalog_test_take(sizeof(unsigned int));
    unsigned int *room = (unsigned int *)aotx_catalog_test_take(sizeof(unsigned int));
    unsigned int *length = (unsigned int *)aotx_catalog_test_take(sizeof(unsigned int));
    unsigned long long *tick = (unsigned long long *)aotx_catalog_test_take(
        sizeof(unsigned long long));
    char *name_on = (char *)aotx_catalog_test_take(64u);
    char *text = (char *)aotx_catalog_test_take(64u);
    char *bytes = (char *)aotx_catalog_test_take(AOTX_TOOL_RESULT_BYTES);
    char *held = (char *)calloc((size_t)AOTX_TOOL_RESULT_BYTES + 1u, 1u);
    unsigned int agent = AOTX_SLOTS;
    unsigned int had = 0u;
    unsigned int made = 0u;

    aotx_catalog_test_spawn<<<1, 1>>>(broad, out, 1ull);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_check_runtime(cudaMemcpy(&agent, out, sizeof agent, cudaMemcpyDeviceToHost),
                       "cudaMemcpy");
    aotx_catalog_test_check(agent < AOTX_SLOTS, "an agent of that role stands", applied,
                            failed);
    if (agent >= AOTX_SLOTS) {
        return;
    }
    /* One turn of that agent measures the system block of its role, which is what the
     * room of the turn after it comes from. */
    aotx_check_runtime(cudaMemcpy(text, turn, sizeof turn, cudaMemcpyHostToDevice),
                       "cudaMemcpy");
    aotx_catalog_test_turn<<<1, 1>>>(agent, text, (unsigned int)strlen(turn), 0u, tick,
                                     length);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    aotx_check_runtime(cudaMemcpy(name_on, "long_one", 9u, cudaMemcpyHostToDevice),
                       "cudaMemcpy");
    aotx_catalog_counts before = aotx_catalog_test_counts();
    aotx_catalog_test_ask_one<<<1, 1>>>(agent, name_on, 8u, out, 1ull);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_tool_step<<<1, AOTX_SLOTS>>>(1ull);
    aotx_catalog_test_cut<<<1, 1>>>(agent, room);
    aotx_catalog_test_result_one<<<1, 256u>>>(agent, length, bytes);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_check_runtime(cudaMemcpy(&had, room, sizeof had, cudaMemcpyDeviceToHost),
                       "cudaMemcpy");
    aotx_check_runtime(cudaMemcpy(&made, length, sizeof made, cudaMemcpyDeviceToHost),
                       "cudaMemcpy");
    aotx_check_runtime(cudaMemcpy(held, bytes, AOTX_TOOL_RESULT_BYTES,
                                  cudaMemcpyDeviceToHost), "cudaMemcpy");
    aotx_catalog_counts after = aotx_catalog_test_counts();
    held[(made < AOTX_TOOL_RESULT_BYTES) ? made : AOTX_TOOL_RESULT_BYTES - 1u] = '\0';
    aotx_catalog_test_check(had < (unsigned int)AOTX_SKILL_BYTES && made == had
                            && after.room_cut == before.room_cut + 1u,
                            "a body the room of a prompt cannot hold is cut to that room",
                            applied, failed);
    aotx_catalog_test_check(strstr(held, words) != NULL,
                            "the bytes that go in say that the result was cut", applied,
                            failed);

    /* The guard bites nothing on a body the room holds. */
    aotx_catalog_test_free_requests<<<1, AOTX_SLOTS>>>();
    aotx_catalog_test_spawn<<<1, 1>>>(broad, out, 1ull);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_check_runtime(cudaMemcpy(&agent, out, sizeof agent, cudaMemcpyDeviceToHost),
                       "cudaMemcpy");
    aotx_check_runtime(cudaMemcpy(name_on, "short_one", 10u, cudaMemcpyHostToDevice),
                       "cudaMemcpy");
    before = aotx_catalog_test_counts();
    aotx_catalog_test_ask_one<<<1, 1>>>(agent, name_on, 9u, out, 2ull);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_tool_step<<<1, AOTX_SLOTS>>>(2ull);
    aotx_catalog_test_cut<<<1, 1>>>(agent, room);
    aotx_catalog_test_result_one<<<1, 256u>>>(agent, length, bytes);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_check_runtime(cudaMemcpy(&made, length, sizeof made, cudaMemcpyDeviceToHost),
                       "cudaMemcpy");
    after = aotx_catalog_test_counts();
    aotx_catalog_test_check(after.room_cut == before.room_cut
                            && made == (unsigned int)strlen("Read the head of the table."),
                            "a body the room holds is not cut", applied, failed);
    printf("catalog: the room of a turn of that role is %u bytes; a body of %u was cut and "
           "said so, and a body of %u was not\n", had, (unsigned int)AOTX_SKILL_BYTES,
           made);
    aotx_catalog_test_free_requests<<<1, AOTX_SLOTS>>>();
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    cudaFree(out);
    cudaFree(room);
    cudaFree(length);
    cudaFree(tick);
    cudaFree(name_on);
    cudaFree(text);
    cudaFree(bytes);
    free(held);
}

#endif
