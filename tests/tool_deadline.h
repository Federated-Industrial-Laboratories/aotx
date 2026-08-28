/* Purpose: Check the deadline of a request that waits for the operator authorization.
 * Owns: The buffers of that case.
 * Threading: One host thread; the tool check calls the case one at a time.
 * Lifetime: The program.
 *
 * The file is a part of the tool check. It reads the helpers of that check, so it comes
 * after them in the same translation unit. */
#ifndef AOTX_TEST_TOOL_DEADLINE_H
#define AOTX_TEST_TOOL_DEADLINE_H

/* Put the deadline of a run of requests in the past. */
__global__ void aotx_tool_test_expire_many(unsigned int first, unsigned int count)
{
    unsigned int at = threadIdx.x;
    if (at < count && first + at < AOTX_REQUEST_SLOTS) {
        aotx_requests.slot[first + at].deadline = 0ull;
    }
}

/* Answer a run of pending authorizations, as the console does for each one. */
__global__ void aotx_tool_test_auth_many(const unsigned int *id, unsigned int count,
                                         unsigned int granted, unsigned long long tick)
{
    if (blockIdx.x != 0u || threadIdx.x != 0u) {
        return;
    }
    for (unsigned int i = 0u; i < count; ++i) {
        aotx_agent_authorize(id[i], granted, tick);
    }
}

/* The deadline of a request that waits for the operator. A tool which needs authorization
 * has no deadline while it waits. The request is still open and still in the list after
 * more than twice AOTX_TOOL_DEADLINE ticks. The deadline starts at the grant. A reply
 * inside it lands, and a reply after it finds a request that already failed.
 *
 * The check runs the real ticks of the pump, because the rule is about the passage of
 * ticks. It does not put a deadline in the past to make the wait short. */
static void aotx_tool_test_case_operator_deadline(aotx_pump *pump, aotx_seam_rings *rings,
                                                  unsigned long long boot_id,
                                                  unsigned int count, unsigned int *applied,
                                                  unsigned int *failed)
{
    aotx_tool_test_clear<<<1, AOTX_REQUEST_SLOTS>>>(0u);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_tool_call *call =
        (aotx_tool_call *)calloc(AOTX_REQUEST_SLOTS, sizeof(aotx_tool_call));
    for (unsigned int i = 0u; i < count; ++i) {
        call[i].tool = AOTX_TOOL_FS_READ;
        call[i].arg_len = (unsigned int)snprintf(call[i].arg, AOTX_TOOL_ARG_BYTES,
                                                 "notes/wait-%u.txt", i);
    }
    aotx_tool_call *on =
        (aotx_tool_call *)aotx_tool_test_take(AOTX_REQUEST_SLOTS * sizeof(aotx_tool_call));
    unsigned int *id =
        (unsigned int *)aotx_tool_test_take(AOTX_REQUEST_SLOTS * sizeof(unsigned int));
    aotx_check_runtime(cudaMemcpy(on, call, AOTX_REQUEST_SLOTS * sizeof(aotx_tool_call),
                                  cudaMemcpyHostToDevice), "cudaMemcpy");
    unsigned long long tick = 0ull;
    aotx_check_runtime(cudaMemcpyFromSymbol(&tick, aotx_time_tick, sizeof tick),
                       "cudaMemcpyFromSymbol");
    aotx_tool_test_open<<<1, 1>>>(on, 0u, count, 1u, id, tick);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    unsigned int *ids = (unsigned int *)calloc(AOTX_REQUEST_SLOTS, sizeof(unsigned int));
    aotx_check_runtime(cudaMemcpy(ids, id, AOTX_REQUEST_SLOTS * sizeof(unsigned int),
                                  cudaMemcpyDeviceToHost), "cudaMemcpy");
    aotx_request_table *table = (aotx_request_table *)calloc(1, sizeof *table);
    aotx_check_runtime(cudaMemcpyFromSymbol(table, aotx_requests, sizeof *table),
                       "cudaMemcpyFromSymbol");
    unsigned int without = 0u;
    for (unsigned int i = 0u; i < count; ++i) {
        without += (table->slot[i].deadline == AOTX_TOOL_NO_DEADLINE) ? 1u : 0u;
    }
    *applied += 1u;
    if (without != count) {
        printf("tool: %u of %u requests that wait for the operator hold no deadline\n",
               without, count);
        *failed += 1u;
    }

    /* The wait. The ticks are more than twice the deadline of a tool. A request that
     * took a deadline at its own tick would have failed by now. */
    aotx_tool_counts before;
    aotx_check_runtime(cudaMemcpyFromSymbol(&before, aotx_tool_count, sizeof before),
                       "cudaMemcpyFromSymbol");
    for (unsigned int t = 0u; t < 2u * AOTX_TOOL_DEADLINE + 16u; ++t) {
        aotx_pump_tick(pump);
    }
    aotx_check_runtime(cudaMemcpyFromSymbol(table, aotx_requests, sizeof *table),
                       "cudaMemcpyFromSymbol");
    aotx_tool_counts after;
    aotx_check_runtime(cudaMemcpyFromSymbol(&after, aotx_tool_count, sizeof after),
                       "cudaMemcpyFromSymbol");
    unsigned int *done = (unsigned int *)calloc(AOTX_REQUEST_SLOTS, sizeof(unsigned int));
    aotx_check_runtime(cudaMemcpyFromSymbol(done, aotx_tool_done,
                                            AOTX_REQUEST_SLOTS * sizeof(unsigned int)),
                       "cudaMemcpyFromSymbol");
    /* The list of the panel is every slot that holds a request and waits for the operator.
     * The check reads the table with that same test. */
    unsigned int listed = 0u;
    unsigned int waiting = 0u;
    for (unsigned int i = 0u; i < AOTX_REQUEST_SLOTS; ++i) {
        if (table->slot[i].request != 0u && table->slot[i].auth == AOTX_AUTH_PENDING) {
            listed += 1u;
        }
    }
    for (unsigned int i = 0u; i < count; ++i) {
        waiting += (table->slot[i].request == ids[i] && done[i] == 0u
                    && table->slot[i].status == AOTX_TOOL_OK) ? 1u : 0u;
    }
    *applied += 1u;
    if (waiting != count || listed != count || table->pending_auth != count
        || after.late != before.late) {
        printf("tool: after %u ticks %u of %u requests still wait, %u are listed, the "
               "table counts %u and %u reached a deadline\n",
               2u * AOTX_TOOL_DEADLINE + 16u, waiting, count, listed,
               table->pending_auth, after.late - before.late);
        *failed += 1u;
    }

    /* The grant starts the deadline. Every request takes it from the tick of the answer. */
    aotx_check_runtime(cudaMemcpyFromSymbol(&tick, aotx_time_tick, sizeof tick),
                       "cudaMemcpyFromSymbol");
    aotx_tool_test_auth_many<<<1, 1>>>(id, count, 1u, tick);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_check_runtime(cudaMemcpyFromSymbol(table, aotx_requests, sizeof *table),
                       "cudaMemcpyFromSymbol");
    unsigned int started = 0u;
    for (unsigned int i = 0u; i < count; ++i) {
        started += (table->slot[i].deadline
                    == tick + (unsigned long long)AOTX_TOOL_DEADLINE) ? 1u : 0u;
    }
    *applied += 1u;
    if (started != count || table->pending_auth != 0u) {
        printf("tool: %u of %u granted requests took a deadline at the tick of the grant "
               "and %u still wait\n", started, count, table->pending_auth);
        *failed += 1u;
    }

    /* A reply inside the deadline lands. */
    aotx_tool_reply_body *parts =
        (aotx_tool_reply_body *)calloc(AOTX_REQUEST_SLOTS, sizeof(aotx_tool_reply_body));
    for (unsigned int i = 0u; i < count; ++i) {
        parts[i].agent = i;
        parts[i].request = ids[i];
        parts[i].status = AOTX_TOOL_OK;
        parts[i].part = 0u;
        parts[i].parts = 1u;
        parts[i].len = 8u;
        memcpy(parts[i].bytes, "in-time.", 8u);
    }
    for (unsigned int at = 0u; at < count; at += 16u) {
        unsigned int run = ((count - at) < 16u) ? (count - at) : 16u;
        aotx_test_feed_replies(rings, parts + at, run, boot_id);
        for (unsigned int t = 0u; t < 4u; ++t) {
            aotx_pump_tick(pump);
        }
    }
    aotx_check_runtime(cudaMemcpyFromSymbol(table, aotx_requests, sizeof *table),
                       "cudaMemcpyFromSymbol");
    unsigned int landed = 0u;
    for (unsigned int i = 0u; i < count; ++i) {
        landed += (table->slot[i].status == AOTX_TOOL_OK && table->slot[i].result_len == 8u
                   && memcmp(table->slot[i].result, "in-time.", 8u) == 0) ? 1u : 0u;
    }
    *applied += 1u;
    if (landed != count) {
        printf("tool: %u of %u replies inside the deadline of the grant landed\n", landed,
               count);
        *failed += 1u;
    }

    /* A second round: the grant, then the deadline, then a reply that comes after it.
     * The deadline of the grant is put in the past. A second wait of 500 ticks would give
     * the check no more than it has. */
    aotx_tool_test_clear<<<1, AOTX_REQUEST_SLOTS>>>(0u);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_check_runtime(cudaMemcpyFromSymbol(&tick, aotx_time_tick, sizeof tick),
                       "cudaMemcpyFromSymbol");
    aotx_tool_test_open<<<1, 1>>>(on, 0u, count, 1u, id, tick);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_check_runtime(cudaMemcpy(ids, id, AOTX_REQUEST_SLOTS * sizeof(unsigned int),
                                  cudaMemcpyDeviceToHost), "cudaMemcpy");
    aotx_tool_test_auth_many<<<1, 1>>>(id, count, 1u, tick);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_tool_test_expire_many<<<1, AOTX_REQUEST_SLOTS>>>(0u, count);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    for (unsigned int t = 0u; t < 4u; ++t) {
        aotx_pump_tick(pump);
    }
    aotx_check_runtime(cudaMemcpyFromSymbol(table, aotx_requests, sizeof *table),
                       "cudaMemcpyFromSymbol");
    const char *why = "the tool gave no answer before its deadline";
    unsigned int span = (unsigned int)strlen(why);
    unsigned int late = 0u;
    for (unsigned int i = 0u; i < count; ++i) {
        late += (table->slot[i].status == AOTX_TOOL_LATE
                 && table->slot[i].result_len == span
                 && memcmp(table->slot[i].result, why, span) == 0) ? 1u : 0u;
    }
    *applied += 1u;
    if (late != count) {
        printf("tool: %u of %u requests failed with the late reason after the deadline of "
               "the grant\n", late, count);
        *failed += 1u;
    }

    /* The reply of the feeder arrives after that. It finds no request and a note on the
     * bus names it. */
    for (unsigned int i = 0u; i < count; ++i) {
        parts[i].request = ids[i];
        parts[i].len = 6u;
        memcpy(parts[i].bytes, "late..", 6u);
    }
    aotx_test_feed_replies(rings, parts, 1u, boot_id);
    for (unsigned int t = 0u; t < 6u; ++t) {
        aotx_pump_tick(pump);
    }
    char want[64];
    unsigned int head = (unsigned int)snprintf(want, sizeof want,
                                               "tool reply refused: request %u", ids[0]);
    aotx_bus_buffer *bus = (aotx_bus_buffer *)calloc(1, sizeof *bus);
    aotx_check_runtime(cudaMemcpyFromSymbol(bus, aotx_bus_lines, sizeof *bus),
                       "cudaMemcpyFromSymbol");
    unsigned int named = 0u;
    for (unsigned int i = 0u; i < AOTX_BUS_LINES; ++i) {
        if (bus->line[i].kind == AOTX_BUS_NOTE && bus->line[i].text_len >= head
            && memcmp(bus->line[i].text, want, head) == 0) {
            named += 1u;
        }
    }
    *applied += 1u;
    if (named == 0u) {
        printf("tool: no note names the reply that came after the deadline of request "
               "%u\n", ids[0]);
        *failed += 1u;
    }
    printf("tool: %u requests waited %u ticks with no deadline and stayed in the list, "
           "took a deadline at the grant, %u replies landed inside it, %u failed late "
           "after it and %u note names the reply that came too late\n",
           count, 2u * AOTX_TOOL_DEADLINE + 16u, landed, late, named);
    free(call);
    free(ids);
    free(done);
    free(parts);
    free(table);
    free(bus);
    cudaFree(on);
    cudaFree(id);
}

#endif
