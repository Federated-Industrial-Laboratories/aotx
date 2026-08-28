/* Purpose: Check the open of a line whose prompt tokens cross a batch of the apply.
 * Owns: The token bodies and the identity table of the case.
 * Threading: One host thread feeds the ring while the pump makes ticks.
 * Lifetime: One run of the test program. */
#ifndef AOTX_TESTS_AGENT_PREFIX_H
#define AOTX_TESTS_AGENT_PREFIX_H

#include "agent_kernels.h"
#include "seam_feed.h"

/* Prompt tokens, sampled tokens and the batch boundary of the prefix case. */
#define AOTX_AGENT_TEST_PROMPT   40u
#define AOTX_AGENT_TEST_SPLIT    20u
#define AOTX_AGENT_TEST_SAMPLED   4u
#define AOTX_AGENT_TEST_LIMIT    16u

/* The prefix case. A replay gives the tokens of a line in the batches of the apply, and
 * the records of one line may cross a batch. The open of that line therefore meets a slot
 * that holds the first part of its prompt. The open takes that slot, and the records that
 * follow confirm the rest at their own positions.
 *
 * The case puts the boundary inside the prompt. The first half of every prompt goes in,
 * then the open, then the second half and the sampled tokens. Every token of every slot
 * is its own value, so a token that reached the wrong place is a defect. */
static void aotx_agent_test_case_prefix(aotx_pump *pump, aotx_seam_rings *rings,
                                        unsigned long long boot_id, unsigned int slots,
                                        unsigned int *applied, unsigned int *failed)
{
    const unsigned int whole = AOTX_AGENT_TEST_PROMPT + AOTX_AGENT_TEST_SAMPLED;
    aotx_agent_test_clear<<<1, AOTX_AGENT_SLOTS>>>();
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    unsigned int marks[2] = { 0u, 0u };
    aotx_check_runtime(cudaMemcpyFromSymbol(marks, aotx_seqs, sizeof marks,
                                            offsetof(aotx_seq_table, live)),
                       "cudaMemcpyFromSymbol");
    unsigned int refused_before = marks[1];

    /* The token of a place is its own value, so no two slots and no two places agree. */
    int *ids = (int *)calloc((size_t)AOTX_SEQ_SLOTS * whole, sizeof(int));
    for (unsigned int s = 0u; s < slots; ++s) {
        for (unsigned int i = 0u; i < whole; ++i) {
            ids[(size_t)s * whole + i] = (int)(1000u + s * whole + i);
        }
    }
    int *on = (int *)aotx_agent_test_take((size_t)AOTX_SEQ_SLOTS * whole * sizeof(int));
    aotx_check_runtime(cudaMemcpy(on, ids, (size_t)AOTX_SEQ_SLOTS * whole * sizeof(int),
                                  cudaMemcpyHostToDevice), "cudaMemcpy");
    unsigned int *bad = (unsigned int *)aotx_agent_test_take(sizeof(unsigned int));

    /* A replay makes no draw, so the sequence of a slot moves by its records alone. */
    aotx_seam_set_replaying(1);
    aotx_token_body *body =
        (aotx_token_body *)calloc((size_t)AOTX_SEQ_SLOTS * whole, sizeof(aotx_token_body));
    unsigned int made = 0u;
    for (unsigned int s = 0u; s < slots; ++s) {
        for (unsigned int i = 0u; i < AOTX_AGENT_TEST_SPLIT; ++i) {
            aotx_token_body *one = &body[made++];
            one->slot = s;
            one->token = (unsigned int)ids[(size_t)s * whole + i];
            one->position = i;
            one->flags = AOTX_TOKEN_PROMPT;
            one->seed = 0x5EEDull + s;
            one->draw = 0ull;
            one->role = AOTX_MODEL_LANGUAGE;
        }
    }
    for (unsigned int at = 0u; at < made; at += 128u) {
        unsigned int run = ((made - at) < 128u) ? (made - at) : 128u;
        aotx_test_feed_records(rings, AOTX_REC_TOKEN, AOTX_CLASS_A, AOTX_WRITER_RESTORE,
                               AOTX_FLAG_REPLAYED, body + at,
                               (unsigned int)sizeof(aotx_token_body), run, boot_id);
        for (unsigned int t = 0u; t < 2u; ++t) {
            aotx_pump_tick(pump);
        }
    }

    /* Every slot now holds the first half of its prompt. The open of the line arrives. */
    aotx_seq_table *table = (aotx_seq_table *)calloc(1, sizeof *table);
    aotx_check_runtime(cudaMemcpyFromSymbol(table, aotx_seqs, sizeof *table),
                       "cudaMemcpyFromSymbol");
    unsigned int part = 0u;
    for (unsigned int s = 0u; s < slots; ++s) {
        part += (table->slot[s].prompt == AOTX_AGENT_TEST_SPLIT) ? 1u : 0u;
    }
    aotx_agent_test_open_many<<<1, 1>>>(on, whole, slots, AOTX_AGENT_TEST_PROMPT,
                                        AOTX_MODEL_LANGUAGE, AOTX_AGENT_TEST_LIMIT, bad);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    unsigned int refused_open = 0u;
    aotx_check_runtime(cudaMemcpy(&refused_open, bad, sizeof refused_open,
                                  cudaMemcpyDeviceToHost), "cudaMemcpy");
    *applied += 1u;
    if (part != slots || refused_open != 0u) {
        printf("agent: %u of %u slots held the first half of the prompt and %u opens were "
               "refused\n", part, slots, refused_open);
        *failed += 1u;
    }

    /* The rest of the prompt and the tokens of the reply follow. */
    made = 0u;
    for (unsigned int s = 0u; s < slots; ++s) {
        for (unsigned int i = AOTX_AGENT_TEST_SPLIT; i < whole; ++i) {
            aotx_token_body *one = &body[made++];
            one->slot = s;
            one->token = (unsigned int)ids[(size_t)s * whole + i];
            one->position = i;
            one->flags = (i < AOTX_AGENT_TEST_PROMPT) ? AOTX_TOKEN_PROMPT
                                                      : AOTX_TOKEN_SAMPLED;
            if (i + 1u == whole) {
                one->flags |= AOTX_TOKEN_LAST;
            }
            one->seed = 0x5EEDull + s;
            one->draw = (i < AOTX_AGENT_TEST_PROMPT) ? 0ull : (i - AOTX_AGENT_TEST_PROMPT);
            one->role = AOTX_MODEL_LANGUAGE;
        }
    }
    for (unsigned int at = 0u; at < made; at += 128u) {
        unsigned int run = ((made - at) < 128u) ? (made - at) : 128u;
        aotx_test_feed_records(rings, AOTX_REC_TOKEN, AOTX_CLASS_A, AOTX_WRITER_RESTORE,
                               AOTX_FLAG_REPLAYED, body + at,
                               (unsigned int)sizeof(aotx_token_body), run, boot_id);
        for (unsigned int t = 0u; t < 2u; ++t) {
            aotx_pump_tick(pump);
        }
    }
    aotx_seam_set_replaying(0);
    aotx_check_runtime(cudaMemcpyFromSymbol(table, aotx_seqs, sizeof *table),
                       "cudaMemcpyFromSymbol");
    aotx_check_runtime(cudaMemcpyFromSymbol(marks, aotx_seqs, sizeof marks,
                                            offsetof(aotx_seq_table, live)),
                       "cudaMemcpyFromSymbol");
    unsigned int whole_slots = 0u;
    unsigned int wrong = 0u;
    for (unsigned int s = 0u; s < slots; ++s) {
        const aotx_seq *seq = &table->slot[s];
        /* The commit of the tick after the last token gives the pages of the slot back
         * and the slot is free again. A sequence that ended is done or free. */
        if (seq->prompt == AOTX_AGENT_TEST_PROMPT
            && seq->sampled == AOTX_AGENT_TEST_SAMPLED
            && (seq->state == AOTX_SEQ_STATE_DONE
                || seq->state == AOTX_SEQ_STATE_FREE)) {
            whole_slots += 1u;
        }
        for (unsigned int i = 0u; i < whole; ++i) {
            if (table->tokens[s][i] != ids[(size_t)s * whole + i]) {
                wrong += 1u;
            }
        }
    }
    *applied += 1u;
    if (whole_slots != slots || wrong != 0u || marks[1] != refused_before) {
        printf("agent: %u of %u slots hold the whole prompt and reply, %u tokens differ "
               "and the table refused %u opens\n", whole_slots, slots, wrong,
               marks[1] - refused_before);
        *failed += 1u;
    }
    printf("agent: a prompt of %u tokens over a batch boundary at %u opened %u slots, "
           "refused %u, and every one of the %u tokens of every slot stands in its place\n",
           AOTX_AGENT_TEST_PROMPT, AOTX_AGENT_TEST_SPLIT, slots,
           marks[1] - refused_before, whole);
    aotx_agent_test_clear<<<1, AOTX_AGENT_SLOTS>>>();
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    free(ids);
    free(body);
    free(table);
    cudaFree(on);
    cudaFree(bad);
}


#endif
