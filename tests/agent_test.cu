#define _GNU_SOURCE
/* Purpose: Check the agent record, the turn loop, the agenda engine and the manifest.
 * Owns: The counts of the cases and the reply texts the fixed arms give.
 * Launch shape: One thread for each agent; the checks run the tick graph.
 * Lifetime: One run of the test program. */
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "agent/agent_state.cuh"
#include "agent/overlays.cuh"
#include "boot/boot.cuh"
#include "boot/check.h"
#include "cli/prompt.cuh"
#include "model/decode_state.cuh"
#include "mem/mem.cuh"
#include "sched/sched.cuh"
#include "seam/seam.cuh"
#include "tool/tool_state.cuh"

#include "agent_drain.h"
#include "agent_kernels.h"
#include "agent_prefix.h"

/* Ticks a fixed arm may take before the check gives up on it. */
#define AOTX_AGENT_TEST_TICKS   600u

/* Ticks a turn on the real model may take. A reply of 128 tokens at one sequence takes
 * about 2.5 seconds, and a tick is 10 milliseconds or the pace of the model. */
#define AOTX_AGENT_TEST_LONG    6000u

/* Agents of the run of the real model at the small batch and at the wide batch. */
#define AOTX_AGENT_TEST_FEW     1u
#define AOTX_AGENT_TEST_MANY    16u

/* Agents of the wide form of the fixed arms. The table holds a conductor, this many
 * workers and this many verifiers, which is every slot of the batch. */
#define AOTX_AGENT_TEST_WIDE    ((AOTX_AGENT_SLOTS - 1u) / 2u)

/* Launches of the rate case. */
#define AOTX_AGENT_TEST_RATE    1000u

/* The table case: spawn, message and task open at one agent and at every agent. */
static void aotx_agent_test_case_table(unsigned int count, unsigned int *applied,
                                       unsigned int *failed)
{
    aotx_agent_test_clear<<<1, AOTX_AGENT_SLOTS>>>();
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    unsigned int *out =
        (unsigned int *)aotx_agent_test_take(AOTX_AGENT_SLOTS * sizeof(unsigned int));
    int *marks = (int *)aotx_agent_test_take(AOTX_AGENT_SLOTS * sizeof(int));
    unsigned long long tick = 0ull;
    aotx_check_runtime(cudaMemcpyFromSymbol(&tick, aotx_time_tick, sizeof tick),
                       "cudaMemcpyFromSymbol");

    aotx_agent_test_spawn<<<1, 1>>>(AOTX_ROLE_CONDUCTOR, 1u, out, tick);
    aotx_agent_test_spawn<<<1, 1>>>(AOTX_ROLE_WORKER, count - 1u, out + 1u, tick);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    unsigned int *slots = (unsigned int *)calloc(AOTX_AGENT_SLOTS, sizeof(unsigned int));
    aotx_check_runtime(cudaMemcpy(slots, out, AOTX_AGENT_SLOTS * sizeof(unsigned int),
                                  cudaMemcpyDeviceToHost), "cudaMemcpy");
    aotx_agent_table *table = (aotx_agent_table *)calloc(1, sizeof *table);
    aotx_agent_test_read(table);
    unsigned int wrong = 0u;
    for (unsigned int i = 0u; i < count; ++i) {
        unsigned int want = (i == 0u) ? AOTX_ROLE_CONDUCTOR : AOTX_ROLE_WORKER;
        if (slots[i] != i || table->agent[i].state != AOTX_AGENT_STATE_IDLE
            || table->agent[i].role != want) {
            wrong += 1u;
        }
    }
    *applied += 1u;
    if (wrong != 0u || table->live != count) {
        printf("agent: %u of %u spawns are wrong and %u agents are live\n", wrong, count,
               table->live);
        *failed += 1u;
    }

    /* A second conductor takes no slot, because agent 0 is the conductor. */
    aotx_agent_test_spawn<<<1, 1>>>(AOTX_ROLE_CONDUCTOR, 1u, out + AOTX_AGENT_SLOTS - 1u,
                                    tick);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_check_runtime(cudaMemcpy(slots, out, AOTX_AGENT_SLOTS * sizeof(unsigned int),
                                  cudaMemcpyDeviceToHost), "cudaMemcpy");
    *applied += 1u;
    if (slots[AOTX_AGENT_SLOTS - 1u] != ~0u) {
        printf("agent: a second conductor took slot %u\n", slots[AOTX_AGENT_SLOTS - 1u]);
        *failed += 1u;
    }

    aotx_agent_test_text lines = aotx_agent_test_lines("count the rows of table %u", count);
    aotx_agent_test_message<<<1, 1>>>(lines.bytes, lines.start, lines.length, 0u, count,
                                      marks, tick);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    int *answers = (int *)calloc(AOTX_AGENT_SLOTS, sizeof(int));
    aotx_check_runtime(cudaMemcpy(answers, marks, AOTX_AGENT_SLOTS * sizeof(int),
                                  cudaMemcpyDeviceToHost), "cudaMemcpy");
    unsigned int refused = 0u;
    for (unsigned int i = 0u; i < count; ++i) {
        refused += (answers[i] != 0) ? 1u : 0u;
    }
    *applied += 1u;
    if (refused != 0u) {
        printf("agent: %u of %u messages were refused\n", refused, count);
        *failed += 1u;
    }

    /* A task for a named agent, and a task for the first idle agent of a role. */
    aotx_agent_test_task<<<1, 1>>>(lines.bytes, lines.start, lines.length, 0u,
                                   AOTX_ROLE_WORKER, count, AOTX_VERIFY_NONE, out, tick);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_check_runtime(cudaMemcpy(slots, out, AOTX_AGENT_SLOTS * sizeof(unsigned int),
                                  cudaMemcpyDeviceToHost), "cudaMemcpy");
    aotx_agent_test_read(table);
    wrong = 0u;
    for (unsigned int i = 0u; i < count; ++i) {
        if (slots[i] != i || table->task[i].text_len == 0u
            || table->task[i].state != AOTX_TASK_PENDING) {
            wrong += 1u;
        }
    }
    *applied += 1u;
    if (wrong != 0u || table->tasks != count) {
        printf("agent: %u of %u task opens are wrong and the table holds %u\n", wrong,
               count, table->tasks);
        *failed += 1u;
    }
    printf("agent: %u agents spawned, %u messages taken, %u tasks opened\n", count, count,
           count);
    free(slots);
    free(answers);
    free(table);
    aotx_agent_test_free(&lines);
    cudaFree(out);
    cudaFree(marks);
}

/* Put a run of request numbers on the device, so a batch call reads them there. */
static void aotx_agent_check_ids(unsigned int *on, const unsigned int *ids,
                                 unsigned int count)
{
    aotx_check_runtime(cudaMemcpy(on, ids, (size_t)count * sizeof(unsigned int),
                                  cudaMemcpyHostToDevice), "cudaMemcpy");
}

/* The fixed arm of the turn loop. Every reply is given, so the loop runs with no model.
 * It runs the same way in every run. The steps are a call to memory_write, then the
 * result, then the answer, then the verdict of a verifier. The arm runs at one worker and
 * at the batch, where a conductor, count workers and count verifiers fill the table. */
static void aotx_agent_test_case_loop(aotx_pump *pump, aotx_agent_test_drain *drain,
                                      unsigned int count, unsigned int *applied,
                                      unsigned int *failed)
{
    unsigned int first_manifests = drain->manifests;
    unsigned int first_handoffs = drain->handoffs;
    unsigned int first_findings = drain->findings;
    unsigned int notes_before = 0u;
    aotx_check_runtime(cudaMemcpyFromSymbol(&notes_before, aotx_embed_notes,
                                            sizeof notes_before,
                                            offsetof(aotx_embed_store, count)),
                       "cudaMemcpyFromSymbol");
    aotx_agent_test_clear<<<1, AOTX_AGENT_SLOTS>>>();
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    unsigned int *out =
        (unsigned int *)aotx_agent_test_take(AOTX_AGENT_SLOTS * sizeof(unsigned int));
    unsigned long long tick = 0ull;
    aotx_check_runtime(cudaMemcpyFromSymbol(&tick, aotx_time_tick, sizeof tick),
                       "cudaMemcpyFromSymbol");
    aotx_agent_test_spawn<<<1, 1>>>(AOTX_ROLE_CONDUCTOR, 1u, out, tick);
    aotx_agent_test_spawn<<<1, 1>>>(AOTX_ROLE_WORKER, count, out + 1u, tick);
    aotx_agent_test_spawn<<<1, 1>>>(AOTX_ROLE_VERIFIER, count, out + 1u + count, tick);
    aotx_agent_test_text lines =
        aotx_agent_test_lines("put the row of item %u in memory", count);
    aotx_agent_test_task<<<1, 1>>>(lines.bytes, lines.start, lines.length, ~0u,
                                   AOTX_ROLE_WORKER, count, AOTX_VERIFY_SIBLING, out, tick);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    /* The agenda gives a task to each worker, and each worker writes its prompt. */
    unsigned int ready = aotx_agent_test_turn_many(pump, 1u, count,
                                                   AOTX_AGENT_TEST_TICKS);
    aotx_agent_table *table = (aotx_agent_table *)calloc(1, sizeof *table);
    aotx_agent_test_read(table);
    unsigned int running = 0u;
    for (unsigned int i = 0u; i < count; ++i) {
        running += (table->task[i].state == AOTX_TASK_RUNNING
                    && table->task[i].agent >= 1u && table->task[i].agent <= count) ? 1u : 0u;
    }
    *applied += 1u;
    if (ready != count || running != count) {
        printf("agent: the agenda started %u of %u turns and %u tasks run\n", ready, count,
               running);
        *failed += 1u;
    }

    /* Turn one: every reply calls memory_write with a text of its own. */
    aotx_agent_test_text calls = aotx_agent_test_lines(
        "<tool_call>\n{\"name\": \"memory_write\", \"arguments\": {\"provenance\": "
        "\"computed\", \"text\": \"item %u stands in the table\"}}\n</tool_call>", count);
    unsigned int took = aotx_agent_test_drive(pump, 1u, count, &calls,
                                              AOTX_AGENT_STATE_TOOL,
                                              AOTX_AGENT_TEST_TICKS);
    aotx_agent_test_read(table);
    unsigned int called = 0u;
    for (unsigned int i = 1u; i <= count; ++i) {
        called += (table->agent[i].tool == AOTX_TOOL_MEMORY_WRITE
                   && table->agent[i].request != 0u) ? 1u : 0u;
    }
    *applied += 1u;
    if (took != count || called != count) {
        printf("agent: %u of %u agents hold a memory_write request\n", called, count);
        *failed += 1u;
    }

    /* The tool step writes the findings and gives the results. The ticks run until memory
     * holds a note for every call, and no state is read. An agent that took its result
     * starts its next turn, and the driver of that turn gives it the next reply. */
    unsigned int notes = notes_before;
    for (unsigned int t = 0u; t < AOTX_AGENT_TEST_TICKS; ++t) {
        aotx_pump_tick(pump);
        aotx_check_runtime(cudaMemcpyFromSymbol(&notes, aotx_embed_notes, sizeof notes,
                                                offsetof(aotx_embed_store, count)),
                           "cudaMemcpyFromSymbol");
        if (notes - notes_before >= count) {
            break;
        }
    }
    aotx_embed_store *store = (aotx_embed_store *)calloc(1, sizeof *store);
    aotx_check_runtime(cudaMemcpyFromSymbol(store, aotx_embed_notes, sizeof *store),
                       "cudaMemcpyFromSymbol");
    aotx_tool_counts tools;
    aotx_check_runtime(cudaMemcpyFromSymbol(&tools, aotx_tool_count, sizeof tools),
                       "cudaMemcpyFromSymbol");
    *applied += 1u;
    if (notes - notes_before != count) {
        printf("agent: memory gained %u notes of %u, %u writes were made and %u notes "
               "were refused\n", notes - notes_before, count, tools.written,
               store->refused);
        *failed += 1u;
    }
    free(store);

    /* Turn two: no reply holds a call, so every task ends and goes to a verifier. The
     * workers end at different ticks, so the verifiers start at different ticks as well.
     * Both ranges therefore take their reply in the same loop, and the loop ends when
     * every task holds a verdict. */
    unsigned int marked = drain->tasks;
    aotx_agent_test_text answers =
        aotx_agent_test_lines("item %u is in memory now.", count);
    aotx_agent_test_text words = aotx_agent_test_lines("uphold", count);
    unsigned int upheld = 0u;
    for (unsigned int t = 0u; t < AOTX_AGENT_TEST_TICKS; ++t) {
        aotx_agent_test_force_ready<<<1, AOTX_AGENT_SLOTS>>>(answers.bytes, answers.start,
                                                             answers.length, 1u, count);
        aotx_agent_test_force_ready<<<1, AOTX_AGENT_SLOTS>>>(words.bytes, words.start,
                                                             words.length, 1u + count,
                                                             count);
        aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        aotx_pump_tick(pump);
        aotx_agent_test_read(table);
        upheld = 0u;
        for (unsigned int i = 0u; i < count; ++i) {
            upheld += (table->task[i].state == AOTX_TASK_DONE
                       && table->agent[1u + count + i].verdict == AOTX_VERDICT_UPHOLD)
                    ? 1u : 0u;
        }
        if (upheld >= count) {
            break;
        }
    }
    *applied += 1u;
    if (upheld != count) {
        printf("agent: %u of %u tasks are done with the verdict uphold\n", upheld, count);
        *failed += 1u;
    }

    /* Every task went through the state that waits for a verdict. The records hold that
     * state, and a state that a reader saw between two ticks would not. */
    usleep(250000);
    unsigned int waited = 0u;
    for (unsigned int i = marked; i < drain->tasks && i < AOTX_AGENT_TEST_KEEP; ++i) {
        waited += (drain->task[i].state == AOTX_TASK_VERIFYING) ? 1u : 0u;
    }
    *applied += 1u;
    if (waited < count) {
        printf("agent: %u of %u tasks have a record of the state that waits for a "
               "verdict\n", waited, count);
        *failed += 1u;
    }

    /* The records of the loop: one manifest for each turn, one finding for each task, and
     * a handoff from each worker and from each verifier. */
    usleep(250000);
    unsigned int manifests = drain->manifests - first_manifests;
    unsigned int handoffs = drain->handoffs - first_handoffs;
    unsigned int findings = drain->findings - first_findings;
    *applied += 1u;
    if (manifests != 3u * count || handoffs != 2u * count || findings != count) {
        printf("agent: the loop wrote %u manifests, %u handoffs and %u findings and the "
               "list is %u, %u and %u\n", manifests, handoffs, findings, 3u * count,
               2u * count, count);
        *failed += 1u;
    }
    const aotx_manifest_body *first = &drain->manifest[first_manifests];
    *applied += 1u;
    if (first->turn != 1u || first->input_hash == 0ull || first->output_hash == 0ull
        || first->finish != AOTX_TURN_TOOL || first->tool != AOTX_TOOL_MEMORY_WRITE
        || first->request == 0u) {
        printf("agent: the first manifest is agent %u turn %u finish %u tool %u request "
               "%u\n", first->agent, first->turn, first->finish, first->tool,
               first->request);
        *failed += 1u;
    }
    printf("agent: the fixed loop ran %u turns over %u tasks, wrote %u manifests, %u "
           "findings and %u handoffs, %u tasks waited for a verdict, and every verdict is "
           "uphold\n", 3u * count, count, manifests, findings, handoffs, waited);
    free(table);
    aotx_agent_test_free(&lines);
    aotx_agent_test_free(&calls);
    aotx_agent_test_free(&answers);
    aotx_agent_test_free(&words);
    cudaFree(out);
}
static void aotx_agent_test_case_budget(aotx_pump *pump, unsigned int *applied,
                                        unsigned int *failed)
{
    aotx_agent_test_clear<<<1, AOTX_AGENT_SLOTS>>>();
    aotx_agent_test_budget<<<1, 1>>>(AOTX_ROLE_WORKER, 2u);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    unsigned int *out = (unsigned int *)aotx_agent_test_take(4 * sizeof(unsigned int));
    unsigned long long tick = 0ull;
    aotx_check_runtime(cudaMemcpyFromSymbol(&tick, aotx_time_tick, sizeof tick),
                       "cudaMemcpyFromSymbol");
    aotx_agent_test_spawn<<<1, 1>>>(AOTX_ROLE_WORKER, 1u, out, tick);
    aotx_agent_test_text lines = aotx_agent_test_lines("look up row %u again and again",
                                                       1u);
    aotx_agent_test_task<<<1, 1>>>(lines.bytes, lines.start, lines.length, ~0u,
                                   AOTX_ROLE_WORKER, 1u, AOTX_VERIFY_NONE, out + 1u, tick);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    unsigned int span = 0u;
    unsigned char *call = aotx_agent_test_bytes(
        "<tool_call>\n{\"name\": \"memory_recall\", \"arguments\": "
        "{\"text\": \"the row of the otter\"}}\n</tool_call>", &span);

    unsigned int turns = 0u;
    aotx_agent_table *table = (aotx_agent_table *)calloc(1, sizeof *table);
    for (unsigned int round = 0u; round < 6u; ++round) {
        if (aotx_agent_test_turn(pump, 1u, AOTX_AGENT_TEST_TICKS) == 0u) {
            break;
        }
        aotx_agent_test_force<<<1, 1>>>(1u, call, span);
        aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        turns += 1u;
        aotx_pump_tick(pump);
        aotx_agent_test_read(table);
        if (table->task[0].state == AOTX_TASK_DONE
            || table->task[0].state == AOTX_TASK_FAILED) {
            break;
        }
    }
    aotx_agent_test_read(table);
    *applied += 1u;
    if (table->task[0].state != AOTX_TASK_FAILED || turns != 2u) {
        printf("agent: the budget arm ran %u turns and the task is state %u, and the "
               "budget of 2 turns gives 2 turns and a failed task\n", turns,
               table->task[0].state);
        *failed += 1u;
    } else {
        printf("agent: the budget of 2 turns ran out after %u turns and the task "
               "failed\n", turns);
    }
    aotx_agent_test_budget<<<1, 1>>>(AOTX_ROLE_WORKER, AOTX_AGENT_BUDGET);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    free(table);
    aotx_agent_test_free(&lines);
    cudaFree(out);
    cudaFree(call);
}

/* The host tool arm. Each request waits for the operator and the operator grants it. The
 * feeder answers through the inbound ring and each agent takes its own bytes into its next
 * prompt. The late form lets every deadline pass instead. */
static void aotx_agent_test_case_host(aotx_pump *pump, aotx_agent_test_drain *drain,
                                      aotx_seam_rings *rings, unsigned long long boot_id,
                                      unsigned int count, int late, unsigned int *applied,
                                      unsigned int *failed)
{
    unsigned int first_requests = drain->requests;
    aotx_agent_test_clear<<<1, AOTX_AGENT_SLOTS>>>();
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    unsigned int *out =
        (unsigned int *)aotx_agent_test_take(AOTX_AGENT_SLOTS * sizeof(unsigned int));
    unsigned int *id =
        (unsigned int *)aotx_agent_test_take(AOTX_AGENT_SLOTS * sizeof(unsigned int));
    unsigned long long tick = 0ull;
    aotx_check_runtime(cudaMemcpyFromSymbol(&tick, aotx_time_tick, sizeof tick),
                       "cudaMemcpyFromSymbol");
    aotx_agent_test_spawn<<<1, 1>>>(AOTX_ROLE_WORKER, count, out, tick);
    aotx_agent_test_text lines =
        aotx_agent_test_lines("read the note file of item %u", count);
    aotx_agent_test_task<<<1, 1>>>(lines.bytes, lines.start, lines.length, ~0u,
                                   AOTX_ROLE_WORKER, count, AOTX_VERIFY_NONE, out, tick);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_agent_test_turn_many(pump, 1u, count, AOTX_AGENT_TEST_TICKS);
    aotx_agent_test_text calls = aotx_agent_test_lines(
        "<tool_call>\n{\"name\": \"fs_read\", \"arguments\": "
        "{\"path\": \"notes/%u.txt\"}}\n</tool_call>", count);
    aotx_agent_test_force_many<<<1, AOTX_AGENT_SLOTS>>>(calls.bytes, calls.start,
                                                        calls.length, 1u, count);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    unsigned int took = aotx_agent_test_state_many(pump, 1u, count, AOTX_AGENT_STATE_TOOL,
                                                   AOTX_AGENT_TEST_TICKS);
    aotx_request_table *requests = (aotx_request_table *)calloc(1, sizeof *requests);
    aotx_check_runtime(cudaMemcpyFromSymbol(requests, aotx_requests, sizeof *requests),
                       "cudaMemcpyFromSymbol");
    unsigned int *ids = (unsigned int *)calloc(AOTX_AGENT_SLOTS, sizeof(unsigned int));
    unsigned int waiting = 0u;
    for (unsigned int i = 0u; i < count; ++i) {
        ids[i] = requests->slot[1u + i].request;
        waiting += (requests->slot[1u + i].auth == AOTX_AUTH_PENDING) ? 1u : 0u;
    }
    usleep(200000);
    *applied += 1u;
    if (took != count || waiting != count || requests->pending_auth != count
        || drain->requests - first_requests < count) {
        printf("agent: %u of %u host requests wait, the table counts %u and %u records "
               "went in\n", waiting, count, requests->pending_auth,
               drain->requests - first_requests);
        *failed += 1u;
    }

    if (late != 0) {
        /* The deadline arm: no answer arrives, so the tool step fails every call and each
         * agent starts its next turn with the reason. */
        aotx_agent_test_expire_many<<<1, AOTX_AGENT_SLOTS>>>(1u, count);
        aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        unsigned int ready = aotx_agent_test_turn_many(pump, 1u, count,
                                                       AOTX_AGENT_TEST_TICKS);
        aotx_check_runtime(cudaMemcpyFromSymbol(requests, aotx_requests, sizeof *requests),
                           "cudaMemcpyFromSymbol");
        unsigned int gone = 0u;
        for (unsigned int i = 0u; i < count; ++i) {
            gone += (requests->slot[1u + i].status == AOTX_TOOL_LATE) ? 1u : 0u;
        }
        *applied += 1u;
        if (ready != count || gone != count || requests->pending_auth != 0u) {
            printf("agent: %u of %u deadlines gave the late status, %u agents went on and "
                   "%u still wait\n", gone, count, ready, requests->pending_auth);
            *failed += 1u;
        }
        printf("agent: %u requests of the deadline arm failed with the late status and "
               "the count that waits is %u\n", gone, requests->pending_auth);
    } else {
        aotx_agent_check_ids(id, ids, count);
        aotx_agent_test_auth_many<<<1, 1>>>(id, count, 1u, tick);
        aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        aotx_tool_reply_body *parts =
            (aotx_tool_reply_body *)calloc(2u * count, sizeof(aotx_tool_reply_body));
        for (unsigned int i = 0u; i < count; ++i) {
            for (unsigned int p = 0u; p < 2u; ++p) {
                aotx_tool_reply_body *one = &parts[2u * i + p];
                one->agent = 1u + i;
                one->request = ids[i];
                one->status = AOTX_TOOL_OK;
                one->part = p;
                one->parts = 2u;
                one->len = AOTX_TOOL_REPLY_BYTES;
                for (unsigned int b = 0u; b < AOTX_TOOL_REPLY_BYTES; ++b) {
                    one->bytes[b] = (char)('a' + ((i * 3u + p * 5u + b) % 26u));
                }
            }
        }
        for (unsigned int at = 0u; at < 2u * count; at += 16u) {
            unsigned int run = ((2u * count - at) < 16u) ? (2u * count - at) : 16u;
            aotx_test_feed_replies(rings, parts + at, run, boot_id);
            for (unsigned int t = 0u; t < 4u; ++t) {
                aotx_pump_tick(pump);
            }
        }
        unsigned int ready = aotx_agent_test_turn_many(pump, 1u, count,
                                                       AOTX_AGENT_TEST_TICKS);
        aotx_say_state *say = (aotx_say_state *)calloc(1, sizeof *say);
        aotx_check_runtime(cudaMemcpyFromSymbol(say, aotx_say, sizeof *say),
                           "cudaMemcpyFromSymbol");
        unsigned int carried = 0u;
        for (unsigned int i = 0u; i < count; ++i) {
            carried += (memmem(say->prompt[1u + i], AOTX_SAY_BYTES,
                               parts[2u * i + 1u].bytes, AOTX_TOOL_REPLY_BYTES) != NULL)
                     ? 1u : 0u;
        }
        /* The prompt of the next turn is the proof and not the state of the agent. A
         * state is read between two ticks, and an agent may have gone on by then. */
        *applied += 1u;
        if (carried != count) {
            printf("agent: %u of %u agents took the reply and %u prompts carry the "
                   "bytes\n", ready, count, carried);
            *failed += 1u;
        }
        printf("agent: %u host tools gave %u bytes each in 2 parts over the inbound ring, "
               "and %u prompts carry them\n", count, 2u * AOTX_TOOL_REPLY_BYTES, carried);
        free(parts);
        free(say);
    }
    free(requests);
    free(ids);
    aotx_agent_test_free(&lines);
    aotx_agent_test_free(&calls);
    cudaFree(out);
    cudaFree(id);
}

/* The authorization arm across a replay. A request that waits for the operator is still
 * waiting when the replay ends. It takes a new deadline and a new record from that tick,
 * and the operator answers it by the number it had before.
 *
 * This is the device form of the kill and restore case, over the ring replay path. The
 * whole system form of the same case is the third scenario of tests/replay_test.sh. */
static void aotx_agent_test_case_authorize(aotx_pump *pump, aotx_agent_test_drain *drain,
                                           aotx_seam_rings *rings,
                                           unsigned long long boot_id, unsigned int count,
                                           unsigned int *applied, unsigned int *failed)
{
    aotx_agent_test_clear<<<1, AOTX_AGENT_SLOTS>>>();
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    unsigned int *out =
        (unsigned int *)aotx_agent_test_take(AOTX_AGENT_SLOTS * sizeof(unsigned int));
    unsigned int *id =
        (unsigned int *)aotx_agent_test_take(AOTX_AGENT_SLOTS * sizeof(unsigned int));
    unsigned long long tick = 0ull;
    aotx_check_runtime(cudaMemcpyFromSymbol(&tick, aotx_time_tick, sizeof tick),
                       "cudaMemcpyFromSymbol");
    aotx_agent_test_spawn<<<1, 1>>>(AOTX_ROLE_WORKER, count, out, tick);
    aotx_agent_test_text lines =
        aotx_agent_test_lines("read the file of item %u with the fs_read tool", count);
    aotx_agent_test_task<<<1, 1>>>(lines.bytes, lines.start, lines.length, ~0u,
                                   AOTX_ROLE_WORKER, count, AOTX_VERIFY_NONE, out, tick);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_agent_test_turn_many(pump, 1u, count, AOTX_AGENT_TEST_TICKS);
    aotx_agent_test_text calls = aotx_agent_test_lines(
        "<tool_call>\n{\"name\": \"fs_read\", \"arguments\": "
        "{\"path\": \"one%u.txt\"}}\n</tool_call>", count);
    aotx_agent_test_force_many<<<1, AOTX_AGENT_SLOTS>>>(calls.bytes, calls.start,
                                                        calls.length, 1u, count);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_agent_test_state_many(pump, 1u, count, AOTX_AGENT_STATE_TOOL,
                               AOTX_AGENT_TEST_TICKS);
    aotx_request_table *requests = (aotx_request_table *)calloc(1, sizeof *requests);
    aotx_check_runtime(cudaMemcpyFromSymbol(requests, aotx_requests, sizeof *requests),
                       "cudaMemcpyFromSymbol");
    unsigned int *ids = (unsigned int *)calloc(AOTX_AGENT_SLOTS, sizeof(unsigned int));
    for (unsigned int i = 0u; i < count; ++i) {
        ids[i] = requests->slot[1u + i].request;
    }
    unsigned int waiting = requests->pending_auth;

    /* The kill: the deadline of every request goes in the past. A step that let a deadline
     * pass while a replay ran would fail them in the first tick of the replay. */
    aotx_agent_test_expire_many<<<1, AOTX_AGENT_SLOTS>>>(1u, count);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_seam_set_replaying(1);
    for (unsigned int t = 0u; t < 40u; ++t) {
        aotx_pump_tick(pump);
    }
    unsigned int *done = (unsigned int *)calloc(AOTX_REQUEST_SLOTS, sizeof(unsigned int));
    aotx_check_runtime(cudaMemcpyFromSymbol(requests, aotx_requests, sizeof *requests),
                       "cudaMemcpyFromSymbol");
    aotx_check_runtime(cudaMemcpyFromSymbol(done, aotx_tool_done,
                                            AOTX_REQUEST_SLOTS * sizeof(unsigned int)),
                       "cudaMemcpyFromSymbol");
    unsigned int held = 0u;
    for (unsigned int i = 0u; i < count; ++i) {
        held += (requests->slot[1u + i].auth == AOTX_AUTH_PENDING && done[1u + i] == 0u
                 && requests->slot[1u + i].status == AOTX_TOOL_OK) ? 1u : 0u;
    }
    *applied += 1u;
    if (waiting != count || held != count) {
        printf("agent: %u of %u requests still wait in the replay and %u waited before\n",
               held, count, waiting);
        *failed += 1u;
    }

    /* The replay ends. Every request waits again, with a new deadline and a new record. */
    unsigned int records_held = drain->requests;
    aotx_seam_set_replaying(0);
    for (unsigned int t = 0u; t < 3u; ++t) {
        aotx_pump_tick(pump);
    }
    aotx_check_runtime(cudaMemcpyFromSymbol(&tick, aotx_time_tick, sizeof tick),
                       "cudaMemcpyFromSymbol");
    aotx_check_runtime(cudaMemcpyFromSymbol(requests, aotx_requests, sizeof *requests),
                       "cudaMemcpyFromSymbol");
    usleep(250000);
    unsigned int again = 0u;
    for (unsigned int i = 0u; i < count; ++i) {
        again += (requests->slot[1u + i].request == ids[i]
                  && requests->slot[1u + i].auth == AOTX_AUTH_PENDING
                  && requests->slot[1u + i].deadline > tick) ? 1u : 0u;
    }
    *applied += 1u;
    if (again != count || requests->pending_auth != count
        || drain->requests - records_held < count) {
        printf("agent: %u of %u requests are presented again, %u wait and %u records went "
               "in at the end of the replay\n", again, count, requests->pending_auth,
               drain->requests - records_held);
        *failed += 1u;
    }

    /* The operator answers by the numbers the requests had, and the replies reach them
     * over the inbound ring under those same numbers. */
    aotx_agent_check_ids(id, ids, count);
    aotx_agent_test_auth_many<<<1, 1>>>(id, count, 1u, tick);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_tool_reply_body *parts =
        (aotx_tool_reply_body *)calloc(2u * count, sizeof(aotx_tool_reply_body));
    for (unsigned int i = 0u; i < count; ++i) {
        for (unsigned int p = 0u; p < 2u; ++p) {
            aotx_tool_reply_body *one = &parts[2u * i + p];
            one->agent = 1u + i;
            one->request = ids[i];
            one->status = AOTX_TOOL_OK;
            one->part = p;
            one->parts = 2u;
            one->len = AOTX_TOOL_REPLY_BYTES;
            for (unsigned int b = 0u; b < AOTX_TOOL_REPLY_BYTES; ++b) {
                one->bytes[b] = (char)('a' + ((i * 7u + p * 11u + b) % 26u));
            }
        }
    }
    for (unsigned int at = 0u; at < 2u * count; at += 16u) {
        unsigned int run = ((2u * count - at) < 16u) ? (2u * count - at) : 16u;
        aotx_test_feed_replies(rings, parts + at, run, boot_id);
        for (unsigned int t = 0u; t < 4u; ++t) {
            aotx_pump_tick(pump);
        }
    }
    unsigned int ready = aotx_agent_test_turn_many(pump, 1u, count,
                                                   AOTX_AGENT_TEST_TICKS);
    aotx_say_state *say = (aotx_say_state *)calloc(1, sizeof *say);
    aotx_check_runtime(cudaMemcpyFromSymbol(say, aotx_say, sizeof *say),
                       "cudaMemcpyFromSymbol");
    unsigned int carried = 0u;
    for (unsigned int i = 0u; i < count; ++i) {
        carried += (memmem(say->prompt[1u + i], AOTX_SAY_BYTES, parts[2u * i + 1u].bytes,
                           AOTX_TOOL_REPLY_BYTES) != NULL) ? 1u : 0u;
    }
    /* The prompt of the next turn is the proof. The state of an agent is read between two
     * ticks, and an agent that answered already left the turn it started. */
    *applied += 1u;
    if (carried != count) {
        printf("agent: after the answer %u of %u turns started and %u prompts carry the "
               "bytes\n", ready, count, carried);
        *failed += 1u;
    }
    printf("agent: %u requests waited over 40 replay ticks with the deadline in the past, "
           "were presented again at tick %llu with %u records, and the answers by those "
           "numbers reached %u prompts\n", count, tick, drain->requests - records_held,
           carried);
    free(requests);
    free(ids);
    free(done);
    free(parts);
    free(say);
    aotx_agent_test_free(&lines);
    aotx_agent_test_free(&calls);
    cudaFree(out);
    cudaFree(id);
}
static void aotx_agent_test_case_model(aotx_pump *pump, aotx_agent_test_drain *drain,
                                       unsigned int count, unsigned int *applied,
                                       unsigned int *failed)
{
    unsigned int first_manifests = drain->manifests;
    aotx_agent_test_clear<<<1, AOTX_AGENT_SLOTS>>>();
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    unsigned int *out =
        (unsigned int *)aotx_agent_test_take(AOTX_AGENT_SLOTS * sizeof(unsigned int));
    unsigned long long tick = 0ull;
    aotx_check_runtime(cudaMemcpyFromSymbol(&tick, aotx_time_tick, sizeof tick),
                       "cudaMemcpyFromSymbol");
    aotx_agent_test_spawn<<<1, 1>>>(AOTX_ROLE_CONDUCTOR, 1u, out, tick);
    aotx_agent_test_spawn<<<1, 1>>>(AOTX_ROLE_WORKER, count, out + 1u, tick);
    aotx_agent_test_text lines = aotx_agent_test_lines(
        "Put this note in memory with the memory_write tool, with provenance computed: "
        "the item of row %u is the otter. Then say that it is done.", count);
    aotx_agent_test_task<<<1, 1>>>(lines.bytes, lines.start, lines.length, ~0u,
                                   AOTX_ROLE_WORKER, count, AOTX_VERIFY_NONE, out, tick);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    aotx_agent_table *table = (aotx_agent_table *)calloc(1, sizeof *table);
    double began = aotx_agent_test_now();
    unsigned int ended = 0u;
    unsigned int ticks = 0u;
    for (unsigned int t = 0u; t < AOTX_AGENT_TEST_LONG; ++t) {
        aotx_pump_tick(pump);
        ticks += 1u;
        if ((t % 16u) != 0u) {
            continue;
        }
        aotx_agent_test_read(table);
        ended = 0u;
        for (unsigned int i = 0u; i < count; ++i) {
            unsigned int state = table->task[i].state;
            ended += (state == AOTX_TASK_DONE || state == AOTX_TASK_FAILED) ? 1u : 0u;
        }
        if (ended >= count) {
            break;
        }
    }
    double spent = aotx_agent_test_now() - began;
    aotx_agent_test_read(table);
    usleep(200000);
    unsigned int manifests = drain->manifests - first_manifests;
    unsigned int calls = 0u;
    for (unsigned int i = first_manifests; i < drain->manifests
         && i < AOTX_AGENT_TEST_KEEP; ++i) {
        calls += (drain->manifest[i].tool == AOTX_TOOL_MEMORY_WRITE) ? 1u : 0u;
    }
    *applied += 1u;
    if (ended < count || manifests < count) {
        printf("agent: %u of %u tasks ended in %u ticks with %u manifests\n", ended, count,
               ticks, manifests);
        *failed += 1u;
    }
    printf("agent: %u tasks on the real model ended in %u ticks and %.1f s, %u turns, %u "
           "of them a memory_write call\n", ended, ticks, spent, manifests, calls);
    printf("agent: task 0 is state %u and its result is: %.*s\n", table->task[0].state,
           (int)table->task[0].result_len, table->task[0].result);
    free(table);
    aotx_agent_test_free(&lines);
    cudaFree(out);
}

/* The replay case. A replayed line gives an agent a message while the replay runs. The
 * step runs in a replay, so the message goes in a prompt then and not after. One sequence
 * opens for that message and no second one opens when the replay ends. */
static void aotx_agent_test_case_replay(aotx_pump *pump, unsigned int count,
                                        unsigned int *applied, unsigned int *failed)
{
    aotx_agent_test_clear<<<1, AOTX_AGENT_SLOTS>>>();
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    unsigned int *out =
        (unsigned int *)aotx_agent_test_take(AOTX_AGENT_SLOTS * sizeof(unsigned int));
    int *marks = (int *)aotx_agent_test_take(AOTX_AGENT_SLOTS * sizeof(int));
    unsigned long long tick = 0ull;
    aotx_check_runtime(cudaMemcpyFromSymbol(&tick, aotx_time_tick, sizeof tick),
                       "cudaMemcpyFromSymbol");
    aotx_agent_test_spawn<<<1, 1>>>(AOTX_ROLE_CONDUCTOR, 1u, out, tick);
    if (count > 1u) {
        aotx_agent_test_spawn<<<1, 1>>>(AOTX_ROLE_WORKER, count - 1u, out + 1u, tick);
    }
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    aotx_seam_set_replaying(1);
    aotx_agent_test_text lines =
        aotx_agent_test_lines("count the rows of table %u", count);
    aotx_agent_test_message<<<1, 1>>>(lines.bytes, lines.start, lines.length, 0u, count,
                                      marks, tick);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    unsigned int took = aotx_agent_test_turn_many(pump, 0u, count, AOTX_AGENT_TEST_TICKS);
    aotx_agent_work *gear =
        (aotx_agent_work *)calloc(AOTX_AGENT_SLOTS, sizeof(aotx_agent_work));
    aotx_check_runtime(cudaMemcpyFromSymbol(gear, aotx_agent_gear,
                                            AOTX_AGENT_SLOTS * sizeof(aotx_agent_work)),
                       "cudaMemcpyFromSymbol");
    unsigned int opened = 0u;
    unsigned int held = 0u;
    for (unsigned int i = 0u; i < count; ++i) {
        opened += (gear[i].opens == 1u) ? 1u : 0u;
        held += gear[i].has_message;
    }
    *applied += 1u;
    if (took != count || opened != count || held != 0u) {
        printf("agent: %u of %u replayed messages took a turn, %u opened one sequence and "
               "%u are still in hand\n", took, count, opened, held);
        *failed += 1u;
    }

    aotx_seam_set_replaying(0);
    for (unsigned int t = 0u; t < 30u; ++t) {
        aotx_pump_tick(pump);
    }
    aotx_check_runtime(cudaMemcpyFromSymbol(gear, aotx_agent_gear,
                                            AOTX_AGENT_SLOTS * sizeof(aotx_agent_work)),
                       "cudaMemcpyFromSymbol");
    unsigned int after = 0u;
    for (unsigned int i = 0u; i < count; ++i) {
        after += (gear[i].opens == 1u) ? 1u : 0u;
    }
    *applied += 1u;
    if (after != count) {
        printf("agent: %u of %u agents kept one prompt after the replay ended\n", after,
               count);
        *failed += 1u;
    }
    printf("agent: %u replayed messages opened %u sequences in the replay and %u after it "
           "ended\n", count, opened, count - after);
    aotx_agent_test_clear<<<1, AOTX_AGENT_SLOTS>>>();
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    free(gear);
    aotx_agent_test_free(&lines);
    cudaFree(out);
    cudaFree(marks);
}
/* The rate case: what one launch of the agent step costs at 64 live agents. */
static void aotx_agent_test_case_rate(unsigned int *applied)
{
    aotx_agent_test_clear<<<1, AOTX_AGENT_SLOTS>>>();
    unsigned int *out =
        (unsigned int *)aotx_agent_test_take(AOTX_AGENT_SLOTS * sizeof(unsigned int));
    aotx_agent_test_spawn<<<1, 1>>>(AOTX_ROLE_CONDUCTOR, 1u, out, 1ull);
    aotx_agent_test_spawn<<<1, 1>>>(AOTX_ROLE_WORKER, AOTX_AGENT_SLOTS - 1u, out + 1u,
                                    1ull);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    cudaEvent_t from;
    cudaEvent_t to;
    aotx_check_runtime(cudaEventCreate(&from), "cudaEventCreate");
    aotx_check_runtime(cudaEventCreate(&to), "cudaEventCreate");
    aotx_agent_step<<<1, AOTX_AGENT_SLOTS>>>(1ull);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_check_runtime(cudaEventRecord(from), "cudaEventRecord");
    for (unsigned int i = 0u; i < AOTX_AGENT_TEST_RATE; ++i) {
        aotx_agent_step<<<1, AOTX_AGENT_SLOTS>>>(1ull);
    }
    aotx_check_runtime(cudaEventRecord(to), "cudaEventRecord");
    aotx_check_runtime(cudaEventSynchronize(to), "cudaEventSynchronize");
    float ms = 0.0f;
    aotx_check_runtime(cudaEventElapsedTime(&ms, from, to), "cudaEventElapsedTime");
    printf("agent: the agent step at %u agents costs %.1f us for each tick\n",
           AOTX_AGENT_SLOTS, 1000.0 * (double)ms / (double)AOTX_AGENT_TEST_RATE);
    *applied += 1u;
    cudaEventDestroy(from);
    cudaEventDestroy(to);
    cudaFree(out);
}

int main(int argc, char **argv)
{
    const char *models = (argc > 1) ? argv[1] : "models";
    unsigned int applied = 0u;
    unsigned int failed = 0u;
    unsigned int skipped = 0u;
    char path[1024];

    /* The sanitizer gate sets AOTX_SANITIZER. The sanitizer makes every kernel far slower
     * than the watchdog allows, and it takes memory of the card. The run then loads the
     * embedding role alone and leaves out the two arms of the language model. The arms
     * that give every reply still run, and they run at 64 agents. */
    const char *sanitizer = getenv("AOTX_SANITIZER");
    int lowered = (sanitizer != NULL) ? 1 : 0;
    const char *roles = (lowered != 0) ? "embedding" : "embedding,language";
    if (lowered != 0) {
        printf("agent: AOTX_SANITIZER is %s: the embedding role alone and no arm of the "
               "language model\n", sanitizer);
    }

    CUdevice device;
    CUcontext context;
    aotx_mem_map map;
    aotx_seam_rings rings;
    aotx_pump pump;
    aotx_check_driver(cuInit(0), "cuInit");
    aotx_check_driver(cuDeviceGet(&device, 0), "cuDeviceGet");
    aotx_check_driver(cuDevicePrimaryCtxRetain(&context, device), "cuDevicePrimaryCtxRetain");
    aotx_check_driver(cuCtxSetCurrent(context), "cuCtxSetCurrent");
    unsigned long long boot_id = 0xA6E17ull;
    if (aotx_mem_reserve(&map) != 0 || aotx_seam_open(&rings, boot_id) != 0) {
        printf("agent: the map or the rings did not open\n");
        return 1;
    }
    aotx_seam_bind(&rings, map.ring, map.ring_bytes, boot_id);
    aotx_seam_bind_bulk(&rings, map.scratch, AOTX_BULK_STAGE_BYTES);
    aotx_seam_note_boot<<<1, 1>>>(0ull, 0ull);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    aotx_agent_test_case_table(1u, &applied, &failed);
    aotx_agent_test_case_table(AOTX_AGENT_SLOTS, &applied, &failed);
    aotx_agent_test_case_rate(&applied);

    snprintf(path, sizeof path, "%s/manifest.jsonl", models);
    if (access(path, R_OK) != 0) {
        printf("agent: %u cases, %u bad, 12 skipped, no model files in %s\n", applied,
               failed, models);
        return (failed == 0u) ? 0 : 1;
    }
    double began = aotx_agent_test_now();
    if (aotx_boot_models(models, roles, 0) != 0) {
        printf("agent: the model files did not load\n");
        return 1;
    }
    printf("agent: the roles %s loaded in %.1f s\n", roles,
           aotx_agent_test_now() - began);

    aotx_agent_test_drain *drain = (aotx_agent_test_drain *)calloc(1, sizeof *drain);
    drain->map = rings.host_map;
    drain->data = rings.host_map + sizeof(aotx_host_ring_preamble);
    drain->data_bytes = AOTX_HOST_RING_DATA_BYTES;
    drain->mask = AOTX_HOST_RING_DATA_BYTES - 1ull;
    drain->boot_id = boot_id;
    pthread_t thread;
    pthread_create(&thread, NULL, aotx_agent_test_reader, drain);

    if (aotx_pump_build(&pump, 0ull, 1u) != 0 || pump.embed == 0u
        || (lowered == 0 && pump.decode == 0u)) {
        printf("agent: the tick graph did not take the decode and the embedding pass\n");
        return 1;
    }
    printf("agent: %u nodes and %u edges in the tick graph: %u say, %u decode, %u tool, "
           "%u agent, %u reply\n", pump.nodes, pump.nodes - 1u, pump.say_nodes,
           pump.decode_nodes, pump.tool_nodes, pump.agent_nodes, pump.reply_nodes);

    aotx_agent_test_case_loop(&pump, drain, 1u, &applied, &failed);
    aotx_agent_test_case_loop(&pump, drain, AOTX_AGENT_TEST_WIDE, &applied, &failed);
    aotx_agent_test_case_budget(&pump, &applied, &failed);
    aotx_agent_test_case_host(&pump, drain, &rings, boot_id, 1u, 0, &applied, &failed);
    aotx_agent_test_case_host(&pump, drain, &rings, boot_id, AOTX_AGENT_SLOTS - 1u, 0,
                              &applied, &failed);
    aotx_agent_test_case_host(&pump, drain, &rings, boot_id, 1u, 1, &applied, &failed);
    aotx_agent_test_case_host(&pump, drain, &rings, boot_id, AOTX_AGENT_SLOTS - 1u, 1,
                              &applied, &failed);
    aotx_agent_test_case_prefix(&pump, &rings, boot_id, 1u, &applied, &failed);
    aotx_agent_test_case_prefix(&pump, &rings, boot_id, AOTX_SEQ_SLOTS,
                                &applied, &failed);
    aotx_agent_test_case_replay(&pump, 1u, &applied, &failed);
    aotx_agent_test_case_replay(&pump, AOTX_AGENT_SLOTS, &applied, &failed);
    aotx_agent_test_case_authorize(&pump, drain, &rings, boot_id, 1u, &applied, &failed);
    aotx_agent_test_case_authorize(&pump, drain, &rings, boot_id, AOTX_AGENT_SLOTS - 1u,
                                   &applied, &failed);
    if (lowered == 0) {
        aotx_agent_test_case_model(&pump, drain, AOTX_AGENT_TEST_FEW, &applied, &failed);
        aotx_agent_test_case_model(&pump, drain, AOTX_AGENT_TEST_MANY, &applied, &failed);
    } else {
        skipped += 2u;
    }

    aotx_pump_close(&pump);
    drain->stop = 1;
    pthread_join(thread, NULL);
    applied += 1u;
    if (drain->bad != 0ull || drain->lost != 0u) {
        printf("agent: %llu blocks bad and %u records lost\n", drain->bad, drain->lost);
        failed += 1u;
    }
    printf("agent: %u manifest, %u task, %u agent and %u request records over %llu "
           "blocks\n", drain->manifests, drain->tasks, drain->agents, drain->requests,
           drain->blocks);
    printf("agent: %u cases applied, %u failed, %u skipped\n", applied, failed, skipped);
    return (failed == 0u) ? 0 : 1;
}
