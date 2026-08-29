/* Purpose: Give the catalog check the remove case, the role case, the lists and skill_use.
 * Owns: The module texts of each case.
 * Threading: One host thread drives the cases one at a time.
 * Lifetime: One run of the test program.
 *
 * The file is a part of the catalog check. It reads the ring cases of that check, so it
 * comes after them in the same translation unit. */
#ifndef AOTX_TESTS_CATALOG_MORE_H
#define AOTX_TESTS_CATALOG_MORE_H

#include "catalog_ring.h"

/* The remove case: what goes, and what the catalog holds back. */
static void aotx_catalog_test_remove(aotx_pump *pump, aotx_seam_rings *rings,
                                     unsigned long long boot_id, unsigned int *applied,
                                     unsigned int *failed)
{
    aotx_test_module module;
    aotx_catalog_test_reset();
    aotx_catalog_test_free_requests<<<1, AOTX_SLOTS>>>();
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    aotx_test_module_text(&module, AOTX_MODULE_ROLE, "keeper",
                          "kind: role\nname: keeper\nmodel: language\n"
                          "tools: memory_recall\nbody: overlay.txt\n",
                          "You are a keeper. You hold the work.");
    aotx_test_import_feed(rings, &module, 51u, boot_id);
    aotx_test_module_free(&module);
    aotx_test_module_text(&module, AOTX_MODULE_TOOL, "helper",
                          "kind: tool\nname: helper\nside: device\narguments: text\n",
                          NULL);
    aotx_test_import_feed(rings, &module, 52u, boot_id);
    aotx_test_module_free(&module);
    aotx_catalog_test_settle(pump, rings);
    unsigned int role = aotx_test_catalog_entry("keeper", AOTX_MODULE_ROLE);
    unsigned int tool = aotx_test_catalog_entry("helper", AOTX_MODULE_TOOL);
    aotx_catalog_test_check(role < AOTX_MODULE_SLOTS && tool < AOTX_MODULE_SLOTS,
                            "the role and the tool of the remove case went in", applied,
                            failed);
    if (role >= AOTX_MODULE_SLOTS || tool >= AOTX_MODULE_SLOTS) {
        return;
    }

    aotx_catalog_counts before = aotx_catalog_test_counts();
    aotx_test_remove_feed(rings, "no_such_module", boot_id);
    aotx_test_remove_feed(rings, "fs_read", boot_id);
    aotx_catalog_test_settle(pump, rings);
    aotx_catalog_counts after = aotx_catalog_test_counts();
    aotx_catalog_test_check(after.gone == before.gone + 2u
                            && after.removed == before.removed,
                            "a name the catalog does not hold and a built-in tool are "
                            "held back", applied, failed);
    aotx_catalog_test_check(aotx_test_catalog_entry("fs_read", AOTX_MODULE_TOOL)
                            < AOTX_MODULE_SLOTS,
                            "the built-in tool is still in the catalog", applied, failed);

    /* A role an agent runs on does not go. */
    unsigned int *out = (unsigned int *)aotx_catalog_test_take(sizeof(unsigned int));
    aotx_catalog_test_spawn<<<1, 1>>>(role, out, 1ull);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    before = aotx_catalog_test_counts();
    aotx_test_remove_feed(rings, "keeper", boot_id);
    aotx_catalog_test_settle(pump, rings);
    after = aotx_catalog_test_counts();
    aotx_catalog_test_check(after.gone == before.gone + 1u
                            && aotx_test_catalog_entry("keeper", AOTX_MODULE_ROLE)
                               < AOTX_MODULE_SLOTS,
                            "a role an agent runs on is held back", applied, failed);

    /* A tool with a request in flight does not go. */
    unsigned int *made = (unsigned int *)aotx_catalog_test_take(AOTX_SLOTS
                                                                * sizeof(unsigned int));
    aotx_catalog_test_hold<<<1, 1>>>(0u, tool, 1ull, made);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    before = aotx_catalog_test_counts();
    aotx_test_remove_feed(rings, "helper", boot_id);
    aotx_catalog_test_settle(pump, rings);
    after = aotx_catalog_test_counts();
    aotx_catalog_test_check(after.gone == before.gone + 1u
                            && aotx_test_catalog_entry("helper", AOTX_MODULE_TOOL)
                               < AOTX_MODULE_SLOTS,
                            "a tool with a request in flight is held back", applied,
                            failed);

    /* The agent goes and the request goes, and the two modules go with the next line. */
    aotx_catalog_test_free_requests<<<1, AOTX_SLOTS>>>();
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    before = aotx_catalog_test_counts();
    aotx_test_remove_feed(rings, "keeper", boot_id);
    aotx_test_remove_feed(rings, "helper", boot_id);
    aotx_catalog_test_settle(pump, rings);
    after = aotx_catalog_test_counts();
    aotx_catalog_test_check(after.removed == before.removed + 2u
                            && aotx_test_catalog_entry("keeper", AOTX_MODULE_ROLE)
                               >= AOTX_MODULE_SLOTS
                            && aotx_test_catalog_entry("helper", AOTX_MODULE_TOOL)
                               >= AOTX_MODULE_SLOTS,
                            "the two modules go when nothing holds them", applied,
                            failed);
    aotx_catalog_state *state = aotx_test_catalog_read();
    aotx_catalog_test_check(state->frees == 1u,
                            "the runs of the modules that went joined in the free list",
                            applied, failed);
    printf("catalog: remove held back 4 names and took 2, and the free list holds %u "
           "run\n", state->frees);
    free(state);
    cudaFree(out);
    cudaFree(made);
}

/* The role case: the three role directories of the repository over the ring. */
static void aotx_catalog_test_roles(aotx_pump *pump, aotx_seam_rings *rings,
                                    unsigned long long boot_id, const char *modules,
                                    unsigned int *applied, unsigned int *failed)
{
    static const char *const names[3] = { "conductor", "verifier", "worker" };
    char dir[1024];
    aotx_catalog_test_reset();
    aotx_catalog_test_free_requests<<<1, AOTX_SLOTS>>>();
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    unsigned int read = 0u;
    unsigned int overlay[3] = { 0u, 0u, 0u };
    for (unsigned int i = 0u; i < 3u; ++i) {
        aotx_test_module module;
        snprintf(dir, sizeof dir, "%s/roles/%s", modules, names[i]);
        if (aotx_test_module_dir(&module, dir) != 0) {
            continue;
        }
        overlay[i] = module.body_len;
        aotx_test_import_feed(rings, &module, 61u + i, boot_id);
        aotx_test_module_free(&module);
        read += 1u;
    }
    aotx_catalog_test_settle(pump, rings);
    aotx_catalog_test_check(read == 3u, "the three role directories were read", applied,
                            failed);

    aotx_catalog_state *state = aotx_test_catalog_read();
    unsigned int found = 0u;
    unsigned int bytes = 0u;
    for (unsigned int i = 0u; i < 3u; ++i) {
        unsigned int at = aotx_test_catalog_entry(names[i], AOTX_MODULE_ROLE);
        if (at < AOTX_MODULE_SLOTS && state->entry[at].role.overlay.length == overlay[i]) {
            found += 1u;
            bytes += overlay[i];
        }
    }
    aotx_catalog_test_check(found == 3u,
                            "each role holds the overlay bytes of its directory", applied,
                            failed);
    unsigned int conductor = aotx_test_catalog_entry("conductor", AOTX_MODULE_ROLE);
    unsigned int verifier = aotx_test_catalog_entry("verifier", AOTX_MODULE_ROLE);
    aotx_catalog_test_check(state->conductor == conductor && state->verifier == verifier
                            && conductor < AOTX_MODULE_SLOTS,
                            "the catalog holds the entry of the console role and of the "
                            "role that judges", applied, failed);
    unsigned int recall = aotx_test_catalog_entry("memory_recall", AOTX_MODULE_TOOL);
    unsigned int use = aotx_test_catalog_entry("skill_use", AOTX_MODULE_TOOL);
    const aotx_catalog_role *row = &state->entry[conductor].role;
    aotx_catalog_test_check(((row->tools[recall >> 5] & (1u << (recall & 31u))) != 0u)
                            && ((row->tools[use >> 5] & (1u << (use & 31u))) != 0u),
                            "the role of the console names the built-in tools", applied,
                            failed);
    free(state);

    /* The agent of the console stands as soon as its role is installed. */
    aotx_pump_report report;
    aotx_catalog_test_ticks(pump, 4u);
    aotx_pump_read(&report);
    aotx_agent_table *table = (aotx_agent_table *)calloc(1, sizeof *table);
    aotx_check_runtime(cudaMemcpyFromSymbol(table, aotx_agents, sizeof *table),
                       "cudaMemcpyFromSymbol");
    aotx_catalog_test_check(report.console_agent == 1u
                            && table->agent[0].state != AOTX_AGENT_STATE_FREE
                            && table->agent[0].role == conductor,
                            "the agent of the console spawns when its role lands",
                            applied, failed);
    printf("catalog: the three roles came in with %u overlay bytes, and the agent of the "
           "console took slot 0 on entry %u\n", bytes, conductor);
    free(table);
}


/* The list case: the tool list and the skill list of a role, and the bound on the block. */
static void aotx_catalog_test_lists(aotx_pump *pump, aotx_seam_rings *rings,
                                    unsigned long long boot_id, unsigned int *applied,
                                    unsigned int *failed)
{
    aotx_test_module module;
    char manifest[1024];
    char name[64];
    aotx_catalog_test_reset();
    aotx_catalog_test_free_requests<<<1, AOTX_SLOTS>>>();
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    /* Twenty tools with a long description, and two skills. The role that allows all of
     * them asks for more than the bound of the block holds. */
    const unsigned int tools = 20u;
    for (unsigned int i = 0u; i < tools; ++i) {
        snprintf(name, sizeof name, "wide_%u", i);
        snprintf(manifest, sizeof manifest,
                 "kind: tool\nname: %s\nside: device\narguments: text\n"
                 "description: The tool of number %u, which has a description that is "
                 "long enough to fill the block of a prompt after a few of them.\n",
                 name, i);
        aotx_test_module_text(&module, AOTX_MODULE_TOOL, name, manifest, NULL);
        aotx_test_import_feed(rings, &module, 101u + i, boot_id);
        aotx_test_module_free(&module);
    }
    aotx_catalog_test_settle(pump, rings);
    for (unsigned int i = 0u; i < 2u; ++i) {
        snprintf(name, sizeof name, "note_%u", i);
        snprintf(manifest, sizeof manifest,
                 "kind: skill\nname: %s\ndescription: The skill of number %u.\n", name, i);
        aotx_test_module_text(&module, AOTX_MODULE_SKILL, name, manifest,
                              (i == 0u) ? "Count the rows of the table."
                                        : "Count the columns of the table.");
        aotx_test_import_feed(rings, &module, 131u + i, boot_id);
        aotx_test_module_free(&module);
    }
    aotx_catalog_test_settle(pump, rings);

    /* A role that allows two tools, and a role that allows every one of the twenty. */
    aotx_test_module_text(&module, AOTX_MODULE_ROLE, "narrow",
                          "kind: role\nname: narrow\nmodel: language\n"
                          "tools: wide_0,wide_1\nskills: note_0\nbody: overlay.txt\n",
                          "You are narrow. You call two tools.");
    aotx_test_import_feed(rings, &module, 141u, boot_id);
    aotx_test_module_free(&module);
    snprintf(manifest, sizeof manifest, "kind: role\nname: broad\nmodel: language\n"
             "tools: wide_0,wide_1,wide_2,wide_3,wide_4,wide_5,wide_6,wide_7,wide_8,"
             "wide_9,wide_10,wide_11,wide_12,wide_13,wide_14,wide_15,wide_16,wide_17,"
             "wide_18,wide_19\nskills: note_0,note_1\nbody: overlay.txt\n");
    aotx_test_module_text(&module, AOTX_MODULE_ROLE, "broad", manifest,
                          "You are broad. You call every tool.");
    aotx_test_import_feed(rings, &module, 142u, boot_id);
    aotx_test_module_free(&module);
    aotx_catalog_test_settle(pump, rings);

    unsigned int narrow = aotx_test_catalog_entry("narrow", AOTX_MODULE_ROLE);
    unsigned int broad = aotx_test_catalog_entry("broad", AOTX_MODULE_ROLE);
    aotx_catalog_test_check(narrow < AOTX_MODULE_SLOTS && broad < AOTX_MODULE_SLOTS,
                            "the two roles of the list case went in", applied, failed);
    if (narrow >= AOTX_MODULE_SLOTS || broad >= AOTX_MODULE_SLOTS) {
        return;
    }
    char *block = (char *)calloc(AOTX_SAY_BYTES + 1u, 1u);
    unsigned int length = 0u;

    aotx_catalog_counts before = aotx_catalog_test_counts();
    aotx_catalog_test_build_block<<<1, 1>>>(narrow);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_check_runtime(cudaMemcpyFromSymbol(block, aotx_catalog_test_block, AOTX_SAY_BYTES),
                       "cudaMemcpyFromSymbol");
    aotx_check_runtime(cudaMemcpyFromSymbol(&length, aotx_catalog_test_block_len,
                                            sizeof length), "cudaMemcpyFromSymbol");
    aotx_catalog_counts after = aotx_catalog_test_counts();
    aotx_catalog_test_check(strstr(block, "\"name\": \"wide_0\"") != NULL
                            && strstr(block, "\"name\": \"wide_1\"") != NULL
                            && strstr(block, "\"name\": \"wide_2\"") == NULL,
                            "the tool list holds the tools of the mask and no other",
                            applied, failed);
    aotx_catalog_test_check(strstr(block, "Skills you may ask for with skill_use")
                            != NULL && strstr(block, "note_0") != NULL,
                            "the skill list names the skills of the catalog", applied,
                            failed);
    aotx_catalog_test_check(strstr(block, "Count the rows of the table.") != NULL,
                            "the body of a skill of the role stands in the block", applied,
                            failed);
    aotx_catalog_test_check(strstr(block, "You are narrow.") != NULL,
                            "the overlay of the role stands in the block", applied,
                            failed);
    aotx_catalog_test_check(after.list_cut == before.list_cut,
                            "a role of two tools cuts nothing", applied, failed);
    unsigned int narrow_len = length;

    before = aotx_catalog_test_counts();
    aotx_catalog_test_build_block<<<1, 1>>>(broad);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_check_runtime(cudaMemcpyFromSymbol(block, aotx_catalog_test_block, AOTX_SAY_BYTES),
                       "cudaMemcpyFromSymbol");
    aotx_check_runtime(cudaMemcpyFromSymbol(&length, aotx_catalog_test_block_len,
                                            sizeof length), "cudaMemcpyFromSymbol");
    after = aotx_catalog_test_counts();
    const char *head = strstr(block, "<tools>");
    const char *tail = strstr(block, "</tools>");
    unsigned int span = (head != NULL && tail != NULL && tail > head)
                      ? (unsigned int)(tail - head) : 0u;
    aotx_catalog_test_check(span != 0u && span <= AOTX_CATALOG_LIST_BYTES,
                            "the tool list of a wide role stays inside the bound", applied,
                            failed);
    aotx_catalog_test_check(after.list_cut > before.list_cut,
                            "the catalog counts the tools that did not fit", applied,
                            failed);
    aotx_catalog_test_check(length < AOTX_SAY_BYTES,
                            "the block of a wide role fits the prompt table", applied,
                            failed);
    printf("catalog: the narrow role made %u bytes and the wide role %u, with %u tools "
           "cut at the bound of %u\n", narrow_len, length,
           after.list_cut - before.list_cut, (unsigned int)AOTX_CATALOG_LIST_BYTES);
    free(block);
}

/* The skill_use case: the body of a skill reaches the result of a request, and an unknown
 * name gives an error that names it. */
static void aotx_catalog_test_skill_use(aotx_pump *pump, aotx_seam_rings *rings,
                                        unsigned long long boot_id, unsigned int count,
                                        unsigned int *applied, unsigned int *failed)
{
    static const char body[] = "Take the rows of the table. Then take the columns.";
    aotx_test_module module;
    aotx_catalog_test_reset();
    aotx_catalog_test_free_requests<<<1, AOTX_SLOTS>>>();
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_test_module_text(&module, AOTX_MODULE_SKILL, "recipe",
                          "kind: skill\nname: recipe\ndescription: How to read a table.\n",
                          body);
    aotx_test_import_feed(rings, &module, 151u, boot_id);
    aotx_test_module_free(&module);
    aotx_catalog_test_settle(pump, rings);

    char *name = (char *)aotx_catalog_test_take(64u);
    unsigned int *made = (unsigned int *)aotx_catalog_test_take((size_t)AOTX_SLOTS
                                                                * sizeof(unsigned int));
    unsigned int *status = (unsigned int *)aotx_catalog_test_take((size_t)AOTX_SLOTS
                                                                  * sizeof(unsigned int));
    unsigned int *length = (unsigned int *)aotx_catalog_test_take((size_t)AOTX_SLOTS
                                                                  * sizeof(unsigned int));
    char *bytes = (char *)aotx_catalog_test_take((size_t)AOTX_SLOTS
                                                 * AOTX_TOOL_RESULT_BYTES);
    unsigned int *marks = (unsigned int *)calloc(AOTX_SLOTS, sizeof(unsigned int));
    unsigned int *lens = (unsigned int *)calloc(AOTX_SLOTS, sizeof(unsigned int));
    char *held = (char *)calloc((size_t)AOTX_SLOTS * AOTX_TOOL_RESULT_BYTES, 1u);

    aotx_check_runtime(cudaMemcpy(name, "recipe", 7u, cudaMemcpyHostToDevice),
                       "cudaMemcpy");
    aotx_catalog_test_ask<<<1, 1>>>(count, name, 6u, made, 1ull);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_tool_step<<<1, AOTX_SLOTS>>>(1ull);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_catalog_test_results<<<1, AOTX_SLOTS>>>(count, status, length, bytes);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_check_runtime(cudaMemcpy(marks, status, (size_t)count * sizeof(unsigned int),
                                  cudaMemcpyDeviceToHost), "cudaMemcpy");
    aotx_check_runtime(cudaMemcpy(lens, length, (size_t)count * sizeof(unsigned int),
                                  cudaMemcpyDeviceToHost), "cudaMemcpy");
    aotx_check_runtime(cudaMemcpy(held, bytes, (size_t)count * AOTX_TOOL_RESULT_BYTES,
                                  cudaMemcpyDeviceToHost), "cudaMemcpy");
    unsigned int good = 0u;
    for (unsigned int i = 0u; i < count; ++i) {
        if (marks[i] == AOTX_TOOL_OK && lens[i] == (unsigned int)strlen(body)
            && memcmp(held + (size_t)i * AOTX_TOOL_RESULT_BYTES, body, strlen(body)) == 0) {
            good += 1u;
        }
    }
    aotx_catalog_test_check(good == count,
                            "every skill_use call carries the body of the skill", applied,
                            failed);

    aotx_catalog_test_free_requests<<<1, AOTX_SLOTS>>>();
    aotx_check_runtime(cudaMemcpy(name, "no_recipe", 10u, cudaMemcpyHostToDevice),
                       "cudaMemcpy");
    aotx_catalog_test_ask<<<1, 1>>>(count, name, 9u, made, 2ull);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_tool_step<<<1, AOTX_SLOTS>>>(2ull);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_catalog_test_results<<<1, AOTX_SLOTS>>>(count, status, length, bytes);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_check_runtime(cudaMemcpy(marks, status, (size_t)count * sizeof(unsigned int),
                                  cudaMemcpyDeviceToHost), "cudaMemcpy");
    aotx_check_runtime(cudaMemcpy(held, bytes, (size_t)count * AOTX_TOOL_RESULT_BYTES,
                                  cudaMemcpyDeviceToHost), "cudaMemcpy");
    unsigned int named = 0u;
    for (unsigned int i = 0u; i < count; ++i) {
        const char *one = held + (size_t)i * AOTX_TOOL_RESULT_BYTES;
        if (marks[i] == AOTX_TOOL_ERROR && strncmp(one, "no_recipe", 9u) == 0) {
            named += 1u;
        }
    }
    aotx_catalog_test_check(named == count,
                            "a skill_use call of a name the catalog does not hold gives "
                            "an error that names it", applied, failed);
    printf("catalog: skill_use gave %u of %u bodies and named %u unknown skills at %u "
           "agents\n", good, count, named, count);
    aotx_catalog_test_free_requests<<<1, AOTX_SLOTS>>>();
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    cudaFree(name);
    cudaFree(made);
    cudaFree(status);
    cudaFree(length);
    cudaFree(bytes);
    free(marks);
    free(lens);
    free(held);
}


/* The tick case. A skill that goes in at one tick stands in a prompt two ticks later. No
 * host stands in the path but the read of the feeder. */
static void aotx_catalog_test_prompt_tick(aotx_pump *pump, aotx_seam_rings *rings,
                                          unsigned long long boot_id, const char *modules,
                                          unsigned int *applied, unsigned int *failed)
{
    static const char body[] = "Read the head of the table before the rows.";
    static const char *const names[3] = { "conductor", "verifier", "worker" };
    static const char message[] = "count the rows of the table";
    char dir[1024];
    aotx_test_module module;
    aotx_catalog_test_reset();
    aotx_catalog_test_free_requests<<<1, AOTX_SLOTS>>>();
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    for (unsigned int i = 0u; i < 3u; ++i) {
        snprintf(dir, sizeof dir, "%s/roles/%s", modules, names[i]);
        if (aotx_test_module_dir(&module, dir) != 0) {
            continue;
        }
        aotx_test_import_feed(rings, &module, 161u + i, boot_id);
        aotx_test_module_free(&module);
    }
    aotx_catalog_test_settle(pump, rings);
    aotx_catalog_test_ticks(pump, 2u);

    /* The skill goes in at the tick the entry names. */
    aotx_test_module_text(&module, AOTX_MODULE_SKILL, "table_head",
                          "kind: skill\nname: table_head\n"
                          "description: How to read the head of a table.\n", body);
    aotx_test_import_feed(rings, &module, 171u, boot_id);
    aotx_test_module_free(&module);
    aotx_catalog_test_settle(pump, rings);
    unsigned int skill = aotx_test_catalog_entry("table_head", AOTX_MODULE_SKILL);
    aotx_catalog_test_check(skill < AOTX_MODULE_SLOTS, "the skill of the tick case went in",
                            applied, failed);
    if (skill >= AOTX_MODULE_SLOTS) {
        return;
    }
    aotx_catalog_state *state = aotx_test_catalog_read();
    unsigned long long import_tick = state->entry[skill].tick;
    free(state);

    char *text = (char *)aotx_catalog_test_take(256u);
    unsigned long long *tick = (unsigned long long *)aotx_catalog_test_take(
        sizeof(unsigned long long));
    unsigned int *length = (unsigned int *)aotx_catalog_test_take(sizeof(unsigned int));
    unsigned char *bytes = (unsigned char *)aotx_catalog_test_take(AOTX_SAY_BYTES);
    unsigned char *held = (unsigned char *)calloc(AOTX_SAY_BYTES + 1u, 1u);
    unsigned long long at = 0ull;
    unsigned int made = 0u;
    aotx_check_runtime(cudaMemcpy(text, message, sizeof message, cudaMemcpyHostToDevice),
                       "cudaMemcpy");

    /* The prompt of the turn after the import names the skill in the skill list. */
    aotx_catalog_test_turn<<<1, 1>>>(0u, text, (unsigned int)strlen(message), 0u, tick,
                                     length);
    aotx_catalog_test_prompt<<<(AOTX_SAY_BYTES + 255u) / 256u, 256u>>>(0u, bytes);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_check_runtime(cudaMemcpy(&at, tick, sizeof at, cudaMemcpyDeviceToHost),
                       "cudaMemcpy");
    aotx_check_runtime(cudaMemcpy(&made, length, sizeof made, cudaMemcpyDeviceToHost),
                       "cudaMemcpy");
    aotx_check_runtime(cudaMemcpy(held, bytes, AOTX_SAY_BYTES, cudaMemcpyDeviceToHost),
                       "cudaMemcpy");
    unsigned long long list_tick = at;
    aotx_catalog_test_check(made != 0u
                            && strstr((const char *)held, "table_head") != NULL,
                            "the skill list of the prompt names a skill of this tick",
                            applied, failed);
    aotx_catalog_test_check(list_tick <= import_tick + 1ull,
                            "the skill stands in a prompt one tick after the import",
                            applied, failed);

    /* The body reaches the prompt of the turn after that, through skill_use. */
    unsigned int *made_ids = (unsigned int *)aotx_catalog_test_take(sizeof(unsigned int));
    aotx_check_runtime(cudaMemcpy(text, "table_head", 11u, cudaMemcpyHostToDevice),
                       "cudaMemcpy");
    aotx_catalog_test_ask<<<1, 1>>>(1u, text, 10u, made_ids, at);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_tool_step<<<1, AOTX_SLOTS>>>(at);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_pump_tick(pump);
    aotx_check_runtime(cudaMemcpy(text, message, sizeof message, cudaMemcpyHostToDevice),
                       "cudaMemcpy");
    aotx_catalog_test_turn<<<1, 1>>>(0u, text, (unsigned int)strlen(message), 1u, tick,
                                     length);
    aotx_catalog_test_prompt<<<(AOTX_SAY_BYTES + 255u) / 256u, 256u>>>(0u, bytes);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_check_runtime(cudaMemcpy(&at, tick, sizeof at, cudaMemcpyDeviceToHost),
                       "cudaMemcpy");
    aotx_check_runtime(cudaMemcpy(held, bytes, AOTX_SAY_BYTES, cudaMemcpyDeviceToHost),
                       "cudaMemcpy");
    aotx_catalog_test_check(strstr((const char *)held, body) != NULL,
                            "the body of the skill stands in the prompt bytes of the turn",
                            applied, failed);
    aotx_catalog_test_check(at <= import_tick + 2ull,
                            "the body of a skill of tick T stands in a prompt at T plus 2",
                            applied, failed);
    printf("catalog: the skill went in at tick %llu, its name stood in a prompt at tick "
           "%llu and its body at tick %llu\n", import_tick, list_tick, at);
    aotx_catalog_test_free_requests<<<1, AOTX_SLOTS>>>();
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    cudaFree(text);
    cudaFree(tick);
    cudaFree(length);
    cudaFree(bytes);
    cudaFree(made_ids);
    free(held);
}

#endif
