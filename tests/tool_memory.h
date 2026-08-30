/* Purpose: Check concurrent requests, empty recall and memory similarity.
 * Inputs: The tool test helpers and the live tick graph.
 * Outputs: Case counts and diagnostics for the parent check.
 * Lifetime: One run of tool_test.cu. */
#ifndef AOTX_TEST_TOOL_MEMORY_H
#define AOTX_TEST_TOOL_MEMORY_H

static void aotx_tool_test_case_wide(unsigned int *applied, unsigned int *failed)
{
    aotx_tool_test_clear<<<1, AOTX_SLOTS>>>(0u);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_tool_call *call =
        (aotx_tool_call *)calloc(AOTX_SLOTS, sizeof(aotx_tool_call));
    for (unsigned int i = 0u; i < AOTX_SLOTS; ++i) {
        call[i].tool = AOTX_TOOL_FS_READ;
        call[i].arg_len = (unsigned int)snprintf(call[i].arg, AOTX_TOOL_ARG_BYTES,
                                                 "notes/%s.txt",
                                                 aotx_tool_noun[i % AOTX_TOOL_CASE_GOOD]);
    }
    aotx_tool_call *on =
        (aotx_tool_call *)aotx_tool_test_take(AOTX_SLOTS * sizeof(aotx_tool_call));
    unsigned int *id =
        (unsigned int *)aotx_tool_test_take(AOTX_SLOTS * sizeof(unsigned int));
    aotx_check_runtime(cudaMemcpy(on, call, AOTX_SLOTS * sizeof(aotx_tool_call),
                                  cudaMemcpyHostToDevice), "cudaMemcpy");
    unsigned long long tick = 0ull;
    aotx_check_runtime(cudaMemcpyFromSymbol(&tick, aotx_time_tick, sizeof tick),
                       "cudaMemcpyFromSymbol");
    aotx_tool_test_open<<<1, 1>>>(on, 0u, AOTX_SLOTS, 1u, id, tick);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    unsigned int *ids = (unsigned int *)calloc(AOTX_SLOTS, sizeof(unsigned int));
    aotx_check_runtime(cudaMemcpy(ids, id, AOTX_SLOTS * sizeof(unsigned int),
                                  cudaMemcpyDeviceToHost), "cudaMemcpy");
    aotx_request_table *table = (aotx_request_table *)calloc(1, sizeof *table);
    aotx_check_runtime(cudaMemcpyFromSymbol(table, aotx_requests, sizeof *table),
                       "cudaMemcpyFromSymbol");
    unsigned int wrong = 0u;
    for (unsigned int i = 0u; i < AOTX_SLOTS; ++i) {
        if (ids[i] == 0u || table->slot[i].auth != AOTX_AUTH_PENDING) {
            wrong += 1u;
        }
        for (unsigned int j = 0u; j < i; ++j) {
            if (ids[i] == ids[j]) {
                wrong += 1u;
            }
        }
    }
    *applied += 1u;
    if (wrong != 0u || table->pending_auth != AOTX_SLOTS) {
        printf("tool: %u of %u wide requests are wrong and %u wait for the operator\n",
               wrong, AOTX_SLOTS, table->pending_auth);
        *failed += 1u;
    }
    printf("tool: %u requests opened at once with %u different numbers\n",
           AOTX_SLOTS, AOTX_SLOTS - wrong);
    free(call);
    free(ids);
    free(table);
    cudaFree(on);
    cudaFree(id);
}

/* Run ticks until every request of a run has its result, or the tick allowance runs out. */
static unsigned int aotx_tool_test_wait(aotx_pump *pump, unsigned int count,
                                        unsigned int ticks)
{
    unsigned int *done = (unsigned int *)calloc(AOTX_SLOTS, sizeof(unsigned int));
    unsigned int made = 0u;
    for (unsigned int t = 0u; t < ticks; ++t) {
        aotx_pump_tick(pump);
        aotx_check_runtime(cudaMemcpyFromSymbol(done, aotx_tool_done,
                                                AOTX_SLOTS * sizeof(unsigned int)),
                           "cudaMemcpyFromSymbol");
        made = 0u;
        for (unsigned int i = 0u; i < count; ++i) {
            made += (done[i] != 0u) ? 1u : 0u;
        }
        if (made >= count) {
            break;
        }
    }
    free(done);
    return made;
}

/* A recall from an empty store has its complete answer without an embedding pass. The
 * request also has an expired deadline, which proves that a ready device answer wins. */
static void aotx_tool_test_case_empty_recall(aotx_pump *pump, unsigned int *applied,
                                             unsigned int *failed)
{
    aotx_tool_call call;
    aotx_tool_call *on;
    unsigned int *id;
    unsigned long long tick = 0ull;
    aotx_request_table *table = (aotx_request_table *)calloc(1, sizeof *table);
    aotx_tool_counts before;
    aotx_tool_counts after;
    memset(&call, 0, sizeof call);
    call.tool = AOTX_TOOL_MEMORY_RECALL;
    memcpy(call.arg, "nothing", 7u);
    call.arg_len = 7u;
    aotx_tool_test_clear<<<1, AOTX_SLOTS>>>(1u);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    on = (aotx_tool_call *)aotx_tool_test_take(sizeof call);
    id = (unsigned int *)aotx_tool_test_take(sizeof(unsigned int));
    aotx_check_runtime(cudaMemcpy(on, &call, sizeof call, cudaMemcpyHostToDevice),
                       "cudaMemcpy");
    aotx_check_runtime(cudaMemcpyFromSymbol(&tick, aotx_time_tick, sizeof tick),
                       "cudaMemcpyFromSymbol");
    aotx_check_runtime(cudaMemcpyFromSymbol(&before, aotx_tool_count, sizeof before),
                       "cudaMemcpyFromSymbol");
    aotx_tool_test_open<<<1, 1>>>(on, 0u, 1u, 0u, id, tick);
    aotx_tool_test_expire<<<1, 1>>>(0u);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_pump_tick(pump);
    aotx_check_runtime(cudaMemcpyFromSymbol(table, aotx_requests, sizeof *table),
                       "cudaMemcpyFromSymbol");
    aotx_check_runtime(cudaMemcpyFromSymbol(&after, aotx_tool_count, sizeof after),
                       "cudaMemcpyFromSymbol");
    unsigned int done = 0u;
    aotx_check_runtime(cudaMemcpyFromSymbol(&done, aotx_tool_done, sizeof done),
                       "cudaMemcpyFromSymbol");
    *applied += 1u;
    if (done == 0u || table->slot[0].status != AOTX_TOOL_OK
        || table->slot[0].result_len != 0u || after.late != before.late) {
        printf("tool: empty recall done %u, status %u, bytes %u, late %u\n", done,
               table->slot[0].status, table->slot[0].result_len, after.late - before.late);
        *failed += 1u;
    }
    printf("tool: an empty recall gave %u bytes in one tick and no late result\n",
           table->slot[0].result_len);
    free(table);
    cudaFree(on);
    cudaFree(id);
}

/* The memory case: 64 notes go in with memory_write, then 64 queries come back with
 * memory_recall. The nearest note of a query is the note that names the same word. */
static void aotx_tool_test_case_memory(aotx_pump *pump, unsigned int count, int paraphrase,
                                       unsigned int *applied, unsigned int *failed,
                                       unsigned int *right_out)
{
    aotx_tool_call *call =
        (aotx_tool_call *)calloc(AOTX_SLOTS, sizeof(aotx_tool_call));
    aotx_tool_call *on =
        (aotx_tool_call *)aotx_tool_test_take(AOTX_SLOTS * sizeof(aotx_tool_call));
    unsigned int *id =
        (unsigned int *)aotx_tool_test_take(AOTX_SLOTS * sizeof(unsigned int));
    aotx_request_table *table = (aotx_request_table *)calloc(1, sizeof *table);
    unsigned long long tick = 0ull;

    /* The notes go in first. Every note names one word, so no two notes are alike. */
    aotx_tool_test_clear<<<1, AOTX_SLOTS>>>(1u);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    for (unsigned int i = 0u; i < count; ++i) {
        call[i].tool = AOTX_TOOL_MEMORY_WRITE;
        call[i].provenance = AOTX_PROV_COMPUTED;
        call[i].arg_len = aotx_tool_note_text(i, call[i].arg, AOTX_TOOL_ARG_BYTES);
    }
    aotx_check_runtime(cudaMemcpy(on, call, AOTX_SLOTS * sizeof(aotx_tool_call),
                                  cudaMemcpyHostToDevice), "cudaMemcpy");
    aotx_check_runtime(cudaMemcpyFromSymbol(&tick, aotx_time_tick, sizeof tick),
                       "cudaMemcpyFromSymbol");
    aotx_tool_test_open<<<1, 1>>>(on, 0u, count, 0u, id, tick);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    unsigned int made = aotx_tool_test_wait(pump, count, AOTX_TOOL_TEST_TICKS);
    unsigned int notes = 0u;
    aotx_check_runtime(cudaMemcpyFromSymbol(&notes, aotx_embed_notes, sizeof notes,
                                            offsetof(aotx_embed_store, count)),
                       "cudaMemcpyFromSymbol");
    *applied += 1u;
    if (made != count || notes != count) {
        printf("tool: %u of %u notes went in and the store holds %u\n", made, count,
               notes);
        *failed += 1u;
    }

    /* The queries come back next. Each one names the word of one note. */
    aotx_tool_test_clear<<<1, AOTX_SLOTS>>>(0u);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    memset(call, 0, AOTX_SLOTS * sizeof(aotx_tool_call));
    for (unsigned int i = 0u; i < count; ++i) {
        call[i].tool = AOTX_TOOL_MEMORY_RECALL;
        if (paraphrase == 2) {
            /* The negative arm: one text for every query. A search that took the query
             * into account cannot name a different note for each one. */
            call[i].arg_len = (unsigned int)snprintf(call[i].arg, AOTX_TOOL_ARG_BYTES,
                                                     "a row of the table");
        } else if (paraphrase == 1) {
            call[i].arg_len = aotx_tool_note_query(i, call[i].arg, AOTX_TOOL_ARG_BYTES);
        } else {
            call[i].arg_len = aotx_tool_note_text(i, call[i].arg, AOTX_TOOL_ARG_BYTES);
        }
    }
    aotx_check_runtime(cudaMemcpy(on, call, AOTX_SLOTS * sizeof(aotx_tool_call),
                                  cudaMemcpyHostToDevice), "cudaMemcpy");
    aotx_check_runtime(cudaMemcpyFromSymbol(&tick, aotx_time_tick, sizeof tick),
                       "cudaMemcpyFromSymbol");
    aotx_tool_test_open<<<1, 1>>>(on, 0u, count, 0u, id, tick);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    unsigned int back = aotx_tool_test_wait(pump, count, AOTX_TOOL_TEST_TICKS);
    aotx_check_runtime(cudaMemcpyFromSymbol(table, aotx_requests, sizeof *table),
                       "cudaMemcpyFromSymbol");

    unsigned int right = 0u;
    char want[AOTX_TOOL_ARG_BYTES];
    for (unsigned int i = 0u; i < count; ++i) {
        unsigned int span = aotx_tool_note_text(i, want, sizeof want);
        if (table->slot[i].result_len >= span
            && memcmp(table->slot[i].result, want, span) == 0) {
            right += 1u;
        } else if (right + 4u > i && i < 4u) {
            printf("tool: query %u gave %.*s and the note is %s\n", i,
                   (int)table->slot[i].result_len, table->slot[i].result, want);
        }
    }
    *applied += 1u;
    if (back != count) {
        printf("tool: %u of %u queries gave a result\n", back, count);
        *failed += 1u;
    }
    *right_out = right;
    const char *kind = (paraphrase == 2) ? "one text for every"
                     : ((paraphrase == 1) ? "paraphrase" : "same text");
    printf("tool: %u of %u %s queries named the right note first\n", right, count, kind);
    free(call);
    free(table);
    cudaFree(on);
    cudaFree(id);
}

#endif
