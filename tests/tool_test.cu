/* Purpose: Check the tool path: the parser, the request table and the two memory tools.
 * Owns: The case tables of the check and the counts of the cases.
 * Launch shape: One thread for each case; the memory cases run the tick graph.
 * Lifetime: One run of the test program. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "boot/boot.cuh"
#include "bus/bus.cuh"
#include "boot/check.h"
#include "mem/mem.cuh"
#include "sched/sched.cuh"
#include "seam/seam.cuh"

#include "catalog_feed.h"

#include "seam_feed.h"
#include "tool_cases.h"

/* Ticks the memory cases may take before the check gives up on them. */
#define AOTX_TOOL_TEST_TICKS  400u

/* Notes the memory case writes and queries it makes. */
/* Notes the memory case writes at the wide batch. The request table holds one slot for
 * each agent and the noun list holds AOTX_TOOL_CASE_GOOD words, so the batch takes the
 * smaller of the two. */
#define AOTX_TOOL_TEST_NOTES  ((AOTX_TOOL_CASE_GOOD < AOTX_SLOTS) ? AOTX_TOOL_CASE_GOOD \
                                                                  : AOTX_SLOTS)

/* The parser over a batch of replies. One thread takes one reply. The kernel is the one
 * the spill gate reads for the parser, because the parser is a device function. */
__global__ void aotx_tool_scan(const unsigned char *text, const unsigned int *start,
                               const unsigned int *length, unsigned int count,
                               aotx_tool_call *call, int *found)
{
    unsigned int at = blockIdx.x * blockDim.x + threadIdx.x;
    if (at >= count) {
        return;
    }
    found[at] = aotx_tool_parse(text + start[at], length[at], &call[at]);
}

/* Open one request for each slot of a run, from one thread, as an agent step does. A call
 * the check builds by hand names its tool by number and carries one value. This step gives
 * that call the entry of the number and the pack of the value. A parsed call holds both
 * already, so the request writes the argument line a run writes. */
__global__ void aotx_tool_test_open(aotx_tool_call *call, unsigned int first,
                                    unsigned int count, unsigned int needs_auth,
                                    unsigned int *id, unsigned long long tick)
{
    if (blockIdx.x != 0u || threadIdx.x != 0u) {
        return;
    }
    for (unsigned int i = 0u; i < count; ++i) {
        if (call[i].tool != AOTX_TOOL_NONE) {
            call[i].entry = aotx_catalog_built_entry(call[i].tool);
        }
        if (call[i].values == 0u && call[i].arg_len != 0u
            && call[i].key < AOTX_CATALOG_ARGS) {
            for (unsigned int b = 0u; b < call[i].arg_len; ++b) {
                call[i].pack[b] = call[i].arg[b];
            }
            call[i].pack_len = call[i].arg_len;
            call[i].at[call[i].key] = 0u;
            call[i].length[call[i].key] = call[i].arg_len;
            call[i].values = 1u;
        }
        id[i] = aotx_tool_request(first + i, &call[i], needs_auth, tick);
    }
}

/* Apply a run of reply records, in order, as the apply of the tick does. */
__global__ void aotx_tool_test_reply(const aotx_tool_reply_body *body, unsigned int count,
                                     int *out)
{
    if (blockIdx.x != 0u || threadIdx.x != 0u) {
        return;
    }
    for (unsigned int i = 0u; i < count; ++i) {
        out[i] = aotx_tool_reply_apply(&body[i]);
    }
}

/* Answer one pending authorization, as the console does. */
__global__ void aotx_tool_test_auth(unsigned int request, unsigned int granted, int *out,
                                    unsigned long long tick)
{
    if (blockIdx.x == 0u && threadIdx.x == 0u) {
        *out = aotx_agent_authorize(request, granted, tick);
    }
}

/* Put the deadline of one request in the past, so the tool step fails it in the next tick.
 * The deadline of the table is 500 ticks, and a check that waits for it takes seconds. */
__global__ void aotx_tool_test_expire(unsigned int slot)
{
    if (blockIdx.x == 0u && threadIdx.x == 0u && slot < AOTX_SLOTS) {
        aotx_requests.slot[slot].deadline = 0ull;
    }
}

/* Give every request slot back and clear the note store. */
__global__ void aotx_tool_test_clear(unsigned int notes)
{
    unsigned int slot = threadIdx.x;
    if (slot >= AOTX_SLOTS) {
        return;
    }
    aotx_requests.slot[slot].request = 0u;
    aotx_requests.slot[slot].auth = AOTX_AUTH_NONE;
    aotx_requests.slot[slot].result_len = 0u;
    aotx_requests.slot[slot].parts = 0u;
    aotx_requests.slot[slot].parts_in = 0u;
    aotx_tool_done[slot] = 0u;
    aotx_tool_embed.state[slot] = AOTX_TOOL_EMBED_NONE;
    aotx_tool_embed.place[slot] = AOTX_SLOTS;
    if (slot == 0u) {
        aotx_requests.pending_auth = 0u;
        if (notes != 0u) {
            aotx_embed_notes.count = 0u;
            aotx_embed_notes.width = 0u;
            aotx_embed_notes.refused = 0u;
        }
    }
}

static void *aotx_tool_test_take(size_t bytes)
{
    void *at = 0;
    aotx_check_runtime(cudaMalloc(&at, bytes), "cudaMalloc");
    aotx_check_runtime(cudaMemset(at, 0, bytes), "cudaMemset");
    return at;
}

/* The batch of the parser check: 64 replies that hold a call and 16 that hold none. */
typedef struct aotx_tool_test_batch {
    char text[AOTX_TOOL_CASES * AOTX_TOOL_CASE_BYTES];
    unsigned int start[AOTX_TOOL_CASES];
    unsigned int length[AOTX_TOOL_CASES];
    unsigned int count;
} aotx_tool_test_batch;

static void aotx_tool_test_build(aotx_tool_test_batch *batch)
{
    memset(batch, 0, sizeof *batch);
    unsigned int at = 0u;
    for (unsigned int i = 0u; i < AOTX_TOOL_CASE_GOOD; ++i) {
        batch->start[i] = at;
        batch->length[i] = aotx_tool_good_case(i, batch->text + at, AOTX_TOOL_CASE_BYTES);
        at += AOTX_TOOL_CASE_BYTES;
    }
    for (unsigned int i = 0u; i < AOTX_TOOL_CASE_BAD; ++i) {
        unsigned int which = AOTX_TOOL_CASE_GOOD + i;
        batch->start[which] = at;
        unsigned int span = (unsigned int)strlen(aotx_tool_bad_case[i]);
        memcpy(batch->text + at, aotx_tool_bad_case[i], span);
        batch->length[which] = span;
        at += AOTX_TOOL_CASE_BYTES;
    }
    batch->count = AOTX_TOOL_CASES;
}

/* The parser case. A run takes the first count cases of the batch, so the check runs at
 * one case and at the whole batch. */
static void aotx_tool_test_case_parse(const aotx_tool_test_batch *batch, unsigned int count,
                                      unsigned int *applied, unsigned int *failed)
{
    unsigned char *text = (unsigned char *)aotx_tool_test_take(sizeof batch->text);
    unsigned int *start = (unsigned int *)aotx_tool_test_take(sizeof batch->start);
    unsigned int *length = (unsigned int *)aotx_tool_test_take(sizeof batch->length);
    aotx_tool_call *call =
        (aotx_tool_call *)aotx_tool_test_take(AOTX_TOOL_CASES * sizeof(aotx_tool_call));
    int *found = (int *)aotx_tool_test_take(AOTX_TOOL_CASES * sizeof(int));
    aotx_check_runtime(cudaMemcpy(text, batch->text, sizeof batch->text,
                                  cudaMemcpyHostToDevice), "cudaMemcpy");
    aotx_check_runtime(cudaMemcpy(start, batch->start, sizeof batch->start,
                                  cudaMemcpyHostToDevice), "cudaMemcpy");
    aotx_check_runtime(cudaMemcpy(length, batch->length, sizeof batch->length,
                                  cudaMemcpyHostToDevice), "cudaMemcpy");
    aotx_tool_scan<<<(count + 63u) / 64u, 64u>>>(text, start, length, count, call, found);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    aotx_tool_call *back =
        (aotx_tool_call *)calloc(AOTX_TOOL_CASES, sizeof(aotx_tool_call));
    int *marks = (int *)calloc(AOTX_TOOL_CASES, sizeof(int));
    aotx_check_runtime(cudaMemcpy(back, call, AOTX_TOOL_CASES * sizeof(aotx_tool_call),
                                  cudaMemcpyDeviceToHost), "cudaMemcpy");
    aotx_check_runtime(cudaMemcpy(marks, found, AOTX_TOOL_CASES * sizeof(int),
                                  cudaMemcpyDeviceToHost), "cudaMemcpy");
    unsigned int wrong = 0u;
    char want[AOTX_TOOL_CASE_BYTES];
    for (unsigned int i = 0u; i < count; ++i) {
        if (i < AOTX_TOOL_CASE_GOOD) {
            unsigned int span = aotx_tool_good_arg(i, want, sizeof want);
            unsigned int tool = aotx_tool_good_tool(i);
            if (marks[i] != 1 || back[i].tool != tool || back[i].arg_len != span
                || memcmp(back[i].arg, want, span) != 0) {
                if (wrong < 4u) {
                    printf("tool: case %u gave found %d tool %u arg %.*s and the case is "
                           "tool %u arg %s\n", i, marks[i], back[i].tool,
                           (int)back[i].arg_len, back[i].arg, tool, want);
                }
                wrong += 1u;
            }
            if (tool == AOTX_TOOL_MEMORY_WRITE && back[i].provenance != AOTX_PROV_COMPUTED) {
                wrong += 1u;
            }
        } else {
            unsigned int which = i - AOTX_TOOL_CASE_GOOD;
            if (marks[i] != 0 || back[i].tool != AOTX_TOOL_NONE) {
                printf("tool: the shape with %s was taken\n", aotx_tool_bad_why[which]);
                wrong += 1u;
            }
        }
    }
    *applied += 1u;
    if (wrong != 0u) {
        printf("tool: %u of %u parser cases are wrong\n", wrong, count);
        *failed += 1u;
    }
    printf("tool: parser at %u cases, %u wrong\n", count, wrong);
    free(back);
    free(marks);
    cudaFree(text);
    cudaFree(start);
    cudaFree(length);
    cudaFree(call);
    cudaFree(found);
}

/* The request case. It holds a host tool that waits for the operator and a reply in
 * parts. It also holds a reply for an id that no request holds, and a deadline.
 *
 * Every reply crosses the inbound ring and the apply node of the tick, as the answer of
 * the feeder does. No reply of this check is put in the table by a call of its own. */
static void aotx_tool_test_case_request(aotx_pump *pump, aotx_seam_rings *rings,
                                        unsigned long long boot_id, unsigned int *applied,
                                        unsigned int *failed)
{
    aotx_tool_test_clear<<<1, AOTX_SLOTS>>>(1u);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_tool_call *call = (aotx_tool_call *)calloc(4, sizeof(aotx_tool_call));
    for (unsigned int i = 0u; i < 4u; ++i) {
        call[i].tool = AOTX_TOOL_FS_READ;
        call[i].provenance = 0u;
        call[i].arg_len = (unsigned int)snprintf(call[i].arg, AOTX_TOOL_ARG_BYTES,
                                                 "notes/%u.txt", i);
    }
    aotx_tool_call *on = (aotx_tool_call *)aotx_tool_test_take(4 * sizeof(aotx_tool_call));
    unsigned int *id = (unsigned int *)aotx_tool_test_take(4 * sizeof(unsigned int));
    int *out = (int *)aotx_tool_test_take(8 * sizeof(int));
    aotx_check_runtime(cudaMemcpy(on, call, 4 * sizeof(aotx_tool_call),
                                  cudaMemcpyHostToDevice), "cudaMemcpy");
    unsigned long long tick = 0ull;
    aotx_check_runtime(cudaMemcpyFromSymbol(&tick, aotx_time_tick, sizeof tick),
                       "cudaMemcpyFromSymbol");
    aotx_tool_test_open<<<1, 1>>>(on, 0u, 4u, 1u, id, tick);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    unsigned int ids[4] = { 0u, 0u, 0u, 0u };
    aotx_check_runtime(cudaMemcpy(ids, id, sizeof ids, cudaMemcpyDeviceToHost),
                       "cudaMemcpy");
    unsigned int pending = 0u;
    aotx_check_runtime(cudaMemcpyFromSymbol(&pending, aotx_requests, sizeof pending,
                                            offsetof(aotx_request_table, pending_auth)),
                       "cudaMemcpyFromSymbol");
    *applied += 1u;
    if (ids[0] == 0u || ids[1] == 0u || ids[0] == ids[1] || pending != 4u) {
        printf("tool: the four requests are %u, %u, %u and %u with %u waiting for the "
               "operator\n", ids[0], ids[1], ids[2], ids[3], pending);
        *failed += 1u;
    }

    /* The operator grants the first and the third request and refuses the second. The
     * fourth is left waiting, so its deadline finds it in that state. */
    aotx_tool_test_auth<<<1, 1>>>(ids[0], 1u, out, tick);
    aotx_tool_test_auth<<<1, 1>>>(ids[1], 0u, out + 1, tick);
    aotx_tool_test_auth<<<1, 1>>>(9999u, 1u, out + 2, tick);
    aotx_tool_test_auth<<<1, 1>>>(ids[2], 1u, out + 7, tick);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    int answers[3] = { 0, 0, 0 };
    aotx_check_runtime(cudaMemcpy(answers, out, sizeof answers, cudaMemcpyDeviceToHost),
                       "cudaMemcpy");
    *applied += 1u;
    if (answers[0] != 0 || answers[1] != 0 || answers[2] != 1) {
        printf("tool: the answers gave %d, %d and %d and the list is 0, 0 and 1\n",
               answers[0], answers[1], answers[2]);
        *failed += 1u;
    }

    /* The feeder answers the first request in three parts of content and one part that
     * carries a reason. A part whose status is not ok is the last part of the reply. */
    const char *why = "the file is longer than the allowance";
    aotx_tool_reply_body *parts =
        (aotx_tool_reply_body *)calloc(6, sizeof(aotx_tool_reply_body));
    unsigned int made = 0u;
    for (unsigned int p = 0u; p < 3u; ++p) {
        parts[p].agent = 0u;
        parts[p].request = ids[0];
        parts[p].status = AOTX_TOOL_OK;
        parts[p].part = p;
        parts[p].parts = 3u;
        parts[p].len = AOTX_TOOL_REPLY_BYTES;
        for (unsigned int i = 0u; i < AOTX_TOOL_REPLY_BYTES; ++i) {
            parts[p].bytes[i] = (char)('a' + ((p * 7u + i) % 26u));
        }
        made += AOTX_TOOL_REPLY_BYTES;
    }
    parts[3] = parts[0];
    parts[3].part = 3u;
    parts[3].status = AOTX_TOOL_ERROR;
    parts[3].len = (unsigned int)strlen(why);
    memset(parts[3].bytes, 0, AOTX_TOOL_REPLY_BYTES);
    memcpy(parts[3].bytes, why, parts[3].len);

    /* The fifth record repeats a part of a reply whose result is in hand. The sixth names
     * a number that no request holds. */
    parts[4] = parts[0];
    parts[5] = parts[0];
    parts[5].request = 0xABCDEFu;
    aotx_test_feed_replies(rings, parts, 6u, boot_id);
    for (unsigned int t = 0u; t < 8u; ++t) {
        aotx_pump_tick(pump);
    }
    aotx_request_table *table = (aotx_request_table *)calloc(1, sizeof *table);
    aotx_check_runtime(cudaMemcpyFromSymbol(table, aotx_requests, sizeof *table),
                       "cudaMemcpyFromSymbol");
    unsigned int want = made + parts[3].len;
    int carried = (table->slot[0].result_len == want
                   && memcmp(table->slot[0].result + made, why, parts[3].len) == 0) ? 1 : 0;
    *applied += 1u;
    if (carried == 0 || table->slot[0].status != AOTX_TOOL_ERROR || table->refused != 2u) {
        printf("tool: the parts gave %u result bytes of %u, status %u and %u refused\n",
               table->slot[0].result_len, want, table->slot[0].status, table->refused);
        *failed += 1u;
    }

    /* The third request reaches its deadline. The deadline of the table is 500 ticks and a
     * check that waits for it takes seconds. The deadline of the slot is therefore put in
     * the past. The fourth still waits for the operator, so it holds no deadline. A
     * deadline in the past leaves it waiting. The count of the requests that wait keeps
     * it. */
    aotx_tool_test_expire<<<1, 1>>>(2u);
    aotx_tool_test_expire<<<1, 1>>>(3u);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    for (unsigned int t = 0u; t < 3u; ++t) {
        aotx_pump_tick(pump);
    }
    aotx_check_runtime(cudaMemcpyFromSymbol(table, aotx_requests, sizeof *table),
                       "cudaMemcpyFromSymbol");
    aotx_tool_counts counts;
    aotx_check_runtime(cudaMemcpyFromSymbol(&counts, aotx_tool_count, sizeof counts),
                       "cudaMemcpyFromSymbol");
    *applied += 1u;
    if (table->slot[1].status != AOTX_TOOL_REFUSED || counts.host_open != 4u
        || table->slot[2].status != AOTX_TOOL_LATE || counts.late != 1u
        || table->slot[3].request != ids[3] || table->slot[3].status != AOTX_TOOL_OK
        || table->pending_auth != 1u) {
        printf("tool: the refused request gave status %u, the late one gave %u, %u host "
               "requests opened, %u reached a deadline and %u still wait\n",
               table->slot[1].status, table->slot[2].status, counts.host_open, counts.late,
               table->pending_auth);
        *failed += 1u;
    }
    printf("tool: request %u granted with %u result bytes in 3 parts and a reason of %u "
           "bytes, request %u refused by the operator, request %u late, request %u still "
           "waits, %u replies refused, %u waiting\n", ids[0], made, parts[3].len, ids[1],
           ids[2], ids[3], table->refused, table->pending_auth);
    free(call);
    free(parts);
    free(table);
    cudaFree(on);
    cudaFree(id);
    cudaFree(out);
}

/* A file read result keeps its digest line in the bytes the next agent turn reads. */
static void aotx_tool_test_case_digest(unsigned int *applied, unsigned int *failed)
{
    static const char answer[] =
        "sha256: 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\n"
        "one line from the file\n";
    aotx_tool_test_clear<<<1, AOTX_SLOTS>>>(0u);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    aotx_tool_call call;
    memset(&call, 0, sizeof call);
    call.tool = AOTX_TOOL_FS_READ;
    call.arg_len = (unsigned int)snprintf(call.arg, AOTX_TOOL_ARG_BYTES,
                                          "notes/digest.txt");
    aotx_tool_call *on = (aotx_tool_call *)aotx_tool_test_take(sizeof call);
    unsigned int *id = (unsigned int *)aotx_tool_test_take(sizeof(unsigned int));
    int *result = (int *)aotx_tool_test_take(sizeof(int));
    aotx_check_runtime(cudaMemcpy(on, &call, sizeof call, cudaMemcpyHostToDevice),
                       "cudaMemcpy");
    unsigned long long tick = 0ull;
    aotx_check_runtime(cudaMemcpyFromSymbol(&tick, aotx_time_tick, sizeof tick),
                       "cudaMemcpyFromSymbol");
    aotx_tool_test_open<<<1, 1>>>(on, 0u, 1u, 0u, id, tick);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    unsigned int request = 0u;
    aotx_check_runtime(cudaMemcpy(&request, id, sizeof request, cudaMemcpyDeviceToHost),
                       "cudaMemcpy");
    aotx_tool_reply_body body;
    memset(&body, 0, sizeof body);
    body.agent = 0u;
    body.request = request;
    body.status = AOTX_TOOL_OK;
    body.parts = 1u;
    body.len = (unsigned int)sizeof answer - 1u;
    memcpy(body.bytes, answer, body.len);
    aotx_tool_reply_body *reply =
        (aotx_tool_reply_body *)aotx_tool_test_take(sizeof body);
    aotx_check_runtime(cudaMemcpy(reply, &body, sizeof body, cudaMemcpyHostToDevice),
                       "cudaMemcpy");
    aotx_tool_test_reply<<<1, 1>>>(reply, 1u, result);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    aotx_request_table *table = (aotx_request_table *)calloc(1, sizeof *table);
    aotx_check_runtime(cudaMemcpyFromSymbol(table, aotx_requests, sizeof *table),
                       "cudaMemcpyFromSymbol");
    *applied += 1u;
    if (request == 0u || table->slot[0].result_len != body.len
        || memcmp(table->slot[0].result, answer, body.len) != 0) {
        printf("tool: the file read digest line did not reach the result\n");
        *failed += 1u;
    } else {
        printf("tool: the file read result keeps its %u-byte digest line\n", body.len);
    }
    free(table);
    cudaFree(on);
    cudaFree(id);
    cudaFree(result);
    cudaFree(reply);
}

/* A reply for a request whose deadline passed is refused, and a note on the bus names the
 * request and the reason. */
static void aotx_tool_test_case_late_note(aotx_pump *pump, aotx_seam_rings *rings,
                                          unsigned long long boot_id, unsigned int *applied,
                                          unsigned int *failed)
{
    aotx_tool_test_clear<<<1, AOTX_SLOTS>>>(0u);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_tool_call *call = (aotx_tool_call *)calloc(1, sizeof(aotx_tool_call));
    call->tool = AOTX_TOOL_FS_READ;
    call->arg_len = (unsigned int)snprintf(call->arg, AOTX_TOOL_ARG_BYTES, "notes/late.txt");
    aotx_tool_call *on = (aotx_tool_call *)aotx_tool_test_take(sizeof(aotx_tool_call));
    unsigned int *id = (unsigned int *)aotx_tool_test_take(sizeof(unsigned int));
    aotx_check_runtime(cudaMemcpy(on, call, sizeof(aotx_tool_call), cudaMemcpyHostToDevice),
                       "cudaMemcpy");
    unsigned long long tick = 0ull;
    aotx_check_runtime(cudaMemcpyFromSymbol(&tick, aotx_time_tick, sizeof tick),
                       "cudaMemcpyFromSymbol");
    aotx_tool_test_open<<<1, 1>>>(on, 0u, 1u, 0u, id, tick);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    unsigned int one = 0u;
    aotx_check_runtime(cudaMemcpy(&one, id, sizeof one, cudaMemcpyDeviceToHost),
                       "cudaMemcpy");
    aotx_tool_test_expire<<<1, 1>>>(0u);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    for (unsigned int t = 0u; t < 3u; ++t) {
        aotx_pump_tick(pump);
    }
    aotx_tool_reply_body late;
    memset(&late, 0, sizeof late);
    late.agent = 0u;
    late.request = one;
    late.status = AOTX_TOOL_OK;
    late.part = 0u;
    late.parts = 1u;
    late.len = 4u;
    memcpy(late.bytes, "note", 4u);
    aotx_test_feed_replies(rings, &late, 1u, boot_id);
    for (unsigned int t = 0u; t < 6u; ++t) {
        aotx_pump_tick(pump);
    }

    char want[64];
    unsigned int span = (unsigned int)snprintf(want, sizeof want,
                                               "tool reply refused: request %u", one);
    aotx_bus_buffer *bus = (aotx_bus_buffer *)calloc(1, sizeof *bus);
    aotx_check_runtime(cudaMemcpyFromSymbol(bus, aotx_bus_lines, sizeof *bus),
                       "cudaMemcpyFromSymbol");
    unsigned int found = 0u;
    for (unsigned int i = 0u; i < AOTX_BUS_LINES; ++i) {
        if (bus->line[i].kind == AOTX_BUS_NOTE && bus->line[i].text_len >= span
            && memcmp(bus->line[i].text, want, span) == 0) {
            found += 1u;
        }
    }
    *applied += 1u;
    if (found == 0u) {
        printf("tool: no note names the refused reply of request %u\n", one);
        *failed += 1u;
    }
    printf("tool: a reply after the deadline of request %u was refused and %u note names "
           "it\n", one, found);
    free(call);
    free(bus);
    cudaFree(on);
    cudaFree(id);
}

/* A reply of 49 parts of content and one part that carries a reason. The content stops at
 * the content bound. Every part that does not fit whole is dropped and counted, and the
 * reason lands in the tail that is kept for it. */
static void aotx_tool_test_case_long_reply(aotx_pump *pump, aotx_seam_rings *rings,
                                           unsigned long long boot_id,
                                           unsigned int *applied, unsigned int *failed)
{
    aotx_tool_test_clear<<<1, AOTX_SLOTS>>>(0u);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_tool_counts before;
    aotx_check_runtime(cudaMemcpyFromSymbol(&before, aotx_tool_count, sizeof before),
                       "cudaMemcpyFromSymbol");
    aotx_tool_call *call = (aotx_tool_call *)calloc(1, sizeof(aotx_tool_call));
    call->tool = AOTX_TOOL_FS_READ;
    call->arg_len = (unsigned int)snprintf(call->arg, AOTX_TOOL_ARG_BYTES, "notes/long.txt");
    aotx_tool_call *on = (aotx_tool_call *)aotx_tool_test_take(sizeof(aotx_tool_call));
    unsigned int *id = (unsigned int *)aotx_tool_test_take(sizeof(unsigned int));
    aotx_check_runtime(cudaMemcpy(on, call, sizeof(aotx_tool_call), cudaMemcpyHostToDevice),
                       "cudaMemcpy");
    unsigned long long tick = 0ull;
    aotx_check_runtime(cudaMemcpyFromSymbol(&tick, aotx_time_tick, sizeof tick),
                       "cudaMemcpyFromSymbol");
    aotx_tool_test_open<<<1, 1>>>(on, 0u, 1u, 0u, id, tick);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    unsigned int one = 0u;
    aotx_check_runtime(cudaMemcpy(&one, id, sizeof one, cudaMemcpyDeviceToHost),
                       "cudaMemcpy");

    const char *why = "the file is longer than the allowance of the device";
    unsigned int count = 50u;
    aotx_tool_reply_body *parts =
        (aotx_tool_reply_body *)calloc(count, sizeof(aotx_tool_reply_body));
    unsigned int fits = 0u;
    for (unsigned int p = 0u; p < count - 1u; ++p) {
        parts[p].agent = 0u;
        parts[p].request = one;
        parts[p].status = AOTX_TOOL_OK;
        parts[p].part = p;
        parts[p].parts = count - 1u;
        parts[p].len = AOTX_TOOL_REPLY_BYTES;
        for (unsigned int i = 0u; i < AOTX_TOOL_REPLY_BYTES; ++i) {
            parts[p].bytes[i] = (char)('a' + ((p * 3u + i) % 26u));
        }
        if ((p + 1u) * AOTX_TOOL_REPLY_BYTES <= AOTX_TOOL_CONTENT_BYTES) {
            fits += 1u;
        }
    }
    parts[count - 1u] = parts[0];
    parts[count - 1u].part = count - 1u;
    parts[count - 1u].status = AOTX_TOOL_ERROR;
    parts[count - 1u].len = (unsigned int)strlen(why);
    memset(parts[count - 1u].bytes, 0, AOTX_TOOL_REPLY_BYTES);
    memcpy(parts[count - 1u].bytes, why, parts[count - 1u].len);

    /* The inbound ring takes a run of records at a time. The parts therefore go in
     * pieces, and the ticks between them let the apply take each piece. */
    for (unsigned int at = 0u; at < count; at += 16u) {
        unsigned int run = ((count - at) < 16u) ? (count - at) : 16u;
        aotx_test_feed_replies(rings, parts + at, run, boot_id);
        for (unsigned int t = 0u; t < 4u; ++t) {
            aotx_pump_tick(pump);
        }
    }
    aotx_request_table *table = (aotx_request_table *)calloc(1, sizeof *table);
    aotx_check_runtime(cudaMemcpyFromSymbol(table, aotx_requests, sizeof *table),
                       "cudaMemcpyFromSymbol");
    aotx_tool_counts after;
    aotx_check_runtime(cudaMemcpyFromSymbol(&after, aotx_tool_count, sizeof after),
                       "cudaMemcpyFromSymbol");
    unsigned int content = fits * AOTX_TOOL_REPLY_BYTES;
    unsigned int want = content + parts[count - 1u].len;
    int carried = (table->slot[0].result_len == want
                   && memcmp(table->slot[0].result + content, why,
                             parts[count - 1u].len) == 0) ? 1 : 0;
    *applied += 1u;
    if (carried == 0 || table->slot[0].result_len > AOTX_TOOL_RESULT_BYTES
        || table->slot[0].status != AOTX_TOOL_ERROR
        || after.dropped - before.dropped != (count - 1u) - fits) {
        printf("tool: the long reply gave %u result bytes of %u, status %u and %u parts "
               "dropped of %u\n", table->slot[0].result_len, want, table->slot[0].status,
               after.dropped - before.dropped, (count - 1u) - fits);
        *failed += 1u;
    }
    printf("tool: %u parts of content and a reason of %u bytes gave %u bytes of the %u the "
           "result holds, %u parts dropped, and the reason stands at %u\n", count - 1u,
           parts[count - 1u].len, table->slot[0].result_len, AOTX_TOOL_RESULT_BYTES,
           after.dropped - before.dropped, content);
    free(call);
    free(parts);
    free(table);
    cudaFree(on);
    cudaFree(id);
}

/* Every slot opens a host request and the feeder answers each one through the ring. The
 * result of a slot must hold the bytes of its own reply, so no reply reached the wrong
 * request. */
static void aotx_tool_test_case_replies(aotx_pump *pump, aotx_seam_rings *rings,
                                        unsigned long long boot_id, unsigned int count,
                                        unsigned int *applied, unsigned int *failed)
{
    aotx_tool_test_clear<<<1, AOTX_SLOTS>>>(0u);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_tool_call *call =
        (aotx_tool_call *)calloc(AOTX_SLOTS, sizeof(aotx_tool_call));
    for (unsigned int i = 0u; i < count; ++i) {
        call[i].tool = AOTX_TOOL_FS_READ;
        call[i].arg_len = (unsigned int)snprintf(call[i].arg, AOTX_TOOL_ARG_BYTES,
                                                 "notes/%s.txt", aotx_tool_noun[i % AOTX_TOOL_CASE_GOOD]);
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
    aotx_tool_test_open<<<1, 1>>>(on, 0u, count, 0u, id, tick);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    unsigned int *ids = (unsigned int *)calloc(AOTX_SLOTS, sizeof(unsigned int));
    aotx_check_runtime(cudaMemcpy(ids, id, AOTX_SLOTS * sizeof(unsigned int),
                                  cudaMemcpyDeviceToHost), "cudaMemcpy");

    aotx_tool_reply_body *parts =
        (aotx_tool_reply_body *)calloc(AOTX_SLOTS, sizeof(aotx_tool_reply_body));
    for (unsigned int i = 0u; i < count; ++i) {
        parts[i].agent = i;
        parts[i].request = ids[i];
        parts[i].status = AOTX_TOOL_OK;
        parts[i].part = 0u;
        parts[i].parts = 1u;
        parts[i].len = 32u;
        for (unsigned int b = 0u; b < 32u; ++b) {
            parts[i].bytes[b] = (char)('a' + ((i * 5u + b) % 26u));
        }
    }
    for (unsigned int at = 0u; at < count; at += 16u) {
        unsigned int run = ((count - at) < 16u) ? (count - at) : 16u;
        aotx_test_feed_replies(rings, parts + at, run, boot_id);
        for (unsigned int t = 0u; t < 4u; ++t) {
            aotx_pump_tick(pump);
        }
    }
    aotx_request_table *table = (aotx_request_table *)calloc(1, sizeof *table);
    aotx_check_runtime(cudaMemcpyFromSymbol(table, aotx_requests, sizeof *table),
                       "cudaMemcpyFromSymbol");
    unsigned int wrong = 0u;
    for (unsigned int i = 0u; i < count; ++i) {
        if (table->slot[i].result_len != 32u
            || memcmp(table->slot[i].result, parts[i].bytes, 32u) != 0
            || table->slot[i].status != AOTX_TOOL_OK) {
            wrong += 1u;
        }
    }
    *applied += 1u;
    if (wrong != 0u) {
        printf("tool: %u of %u replies through the ring reached the wrong request\n",
               wrong, count);
        *failed += 1u;
    }
    printf("tool: %u replies crossed the inbound ring and the apply, %u wrong\n", count,
           wrong);
    free(call);
    free(ids);
    free(parts);
    free(table);
    cudaFree(on);
    cudaFree(id);
}

#include "tool_memory.h"

#include "tool_deadline.h"
#include "tool_verdict.h"

int main(int argc, char **argv)
{
    const char *models = (argc > 1) ? argv[1] : "models";
    unsigned int applied = 0u;
    unsigned int failed = 0u;
    unsigned int skipped = 0u;
    char path[1024];

    CUdevice device;
    CUcontext context;
    aotx_mem_map map;
    aotx_seam_rings rings;
    aotx_pump pump;
    aotx_check_driver(cuInit(0), "cuInit");
    aotx_check_driver(cuDeviceGet(&device, 0), "cuDeviceGet");
    aotx_check_driver(cuDevicePrimaryCtxRetain(&context, device), "cuDevicePrimaryCtxRetain");
    aotx_check_driver(cuCtxSetCurrent(context), "cuCtxSetCurrent");
    unsigned long long boot_id = 0x700100ull;
    if (aotx_mem_reserve(&map) != 0 || aotx_seam_open(&rings, boot_id) != 0) {
        printf("tool: the map or the rings did not open\n");
        return 1;
    }
    aotx_seam_bind(&rings, map.ring, map.ring_bytes, boot_id);
    if (aotx_test_catalog_setup() != 0) {
        printf("tool: the catalog did not take the built-in tools and the three roles\n");
        return 1;
    }
    aotx_seam_bind_bulk(&rings, map.scratch, AOTX_BULK_STAGE_BYTES);
    aotx_seam_note_boot<<<1, 1>>>(0ull, 0ull);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    aotx_tool_test_batch *batch =
        (aotx_tool_test_batch *)calloc(1, sizeof *batch);
    aotx_tool_test_build(batch);
    aotx_tool_test_case_parse(batch, 1u, &applied, &failed);
    aotx_tool_test_case_parse(batch, AOTX_TOOL_CASES, &applied, &failed);

    snprintf(path, sizeof path, "%s/manifest.jsonl", models);
    if (access(path, R_OK) != 0) {
        printf("tool: %u cases, %u bad, 6 skipped, no model files in %s\n", applied,
               failed, models);
        return (failed == 0u) ? 0 : 1;
    }
    if (aotx_boot_models(models, "embedding", 0) != 0) {
        printf("tool: the model files did not load\n");
        return 1;
    }
    if (aotx_pump_build(&pump, 0ull, 1u) != 0 || pump.embed == 0u) {
        printf("tool: the tick graph did not take the embedding pass\n");
        return 1;
    }
    printf("tool: %u nodes in the tick graph, %u from the tool path, %u from the agent "
           "step\n", pump.nodes, pump.tool_nodes, pump.agent_nodes);

    aotx_tool_test_case_request(&pump, &rings, boot_id, &applied, &failed);
    aotx_tool_test_case_digest(&applied, &failed);
    aotx_tool_test_case_wide(&applied, &failed);
    aotx_tool_test_case_replies(&pump, &rings, boot_id, 1u, &applied, &failed);
    aotx_tool_test_case_replies(&pump, &rings, boot_id, AOTX_SLOTS, &applied,
                                &failed);
    aotx_tool_test_case_long_reply(&pump, &rings, boot_id, &applied, &failed);
    aotx_tool_test_case_late_note(&pump, &rings, boot_id, &applied, &failed);
    aotx_tool_test_case_operator_deadline(&pump, &rings, boot_id, 1u, &applied, &failed);
    aotx_tool_test_case_operator_deadline(&pump, &rings, boot_id, AOTX_SLOTS,
                                          &applied, &failed);
    aotx_tool_test_case_verdict_record(&pump, 1u, &applied, &failed);
    aotx_tool_test_case_verdict_record(&pump, AOTX_SLOTS, &applied, &failed);
    aotx_tool_test_case_replay_flag(&applied, &failed);
    aotx_tool_test_case_empty_recall(&pump, &applied, &failed);
    unsigned int right_one = 0u;
    unsigned int right_all = 0u;
    unsigned int right_near = 0u;
    aotx_tool_test_case_memory(&pump, 1u, 0, &applied, &failed, &right_one);
    aotx_tool_test_case_memory(&pump, AOTX_TOOL_TEST_NOTES, 0, &applied, &failed,
                               &right_all);
    aotx_tool_test_case_memory(&pump, AOTX_TOOL_TEST_NOTES, 1, &applied, &failed,
                               &right_near);
    unsigned int right_flat = 0u;
    aotx_tool_test_case_memory(&pump, AOTX_TOOL_TEST_NOTES, 2, &applied, &failed,
                               &right_flat);
    applied += 1u;
    if (right_one != 1u || right_all != AOTX_TOOL_TEST_NOTES) {
        printf("tool: the same text found %u of 1 and %u of %u notes\n", right_one,
               right_all, AOTX_TOOL_TEST_NOTES);
        failed += 1u;
    }
    applied += 1u;
    if (right_near * 2u < AOTX_TOOL_TEST_NOTES) {
        printf("tool: a paraphrase found %u of %u notes, which is under half\n",
               right_near, AOTX_TOOL_TEST_NOTES);
        failed += 1u;
    }
    /* The negative arm of the search. One text for every query cannot name a different
     * note for each query. At most one query therefore finds the note of its number. */
    applied += 1u;
    if (right_flat > 1u) {
        printf("tool: one text for every query found %u notes and the arm allows 1\n",
               right_flat);
        failed += 1u;
    }
    aotx_pump_close(&pump);
    printf("tool: %u cases applied, %u failed, %u skipped\n", applied, failed, skipped);
    return (failed == 0u) ? 0 : 1;
}
