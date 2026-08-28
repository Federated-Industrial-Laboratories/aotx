/* Purpose: Cross the seam for a decode check: read the host ring and write the inbound ring.
 * Owns: The cursor of the test consumer, the arrays it fills, and the prompt list.
 * Threading: One host thread reads the ring while the pump makes ticks.
 * Lifetime: One run of the test program. */
#ifndef AOTX_TESTS_DECODE_DRAIN_H
#define AOTX_TESTS_DECODE_DRAIN_H

#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <stdlib.h>

#include "boot/check.h"
#include "model/decode_state.cuh"
#include "model/graph_host.h"
#include "seam/seam.cuh"

/* Token records the consumer keeps. A run of 64 sequences of 2,048 tokens is under this
 * bound, and a run that goes over it counts the records it lost. */
#define AOTX_DECODE_TEST_RECORDS 300000u

typedef struct aotx_decode_test_token {
    unsigned int slot;
    unsigned int token;
    unsigned int position;
    unsigned int flags;
    unsigned long long seed;
    unsigned long long draw;
} aotx_decode_test_token;

typedef struct aotx_decode_test_drain {
    const unsigned char *map;
    const unsigned char *data;
    unsigned long long data_bytes;
    unsigned long long mask;
    unsigned long long boot_id;
    volatile int stop;
    unsigned long long cursor;
    unsigned long long blocks;
    unsigned long long bad;
    unsigned int taken;                /* token records kept */
    unsigned int lost;                 /* token records the array had no room for */
    unsigned int events;               /* sequence records seen */
    unsigned int released;             /* released events seen */
    unsigned int ticks;                /* statistics records seen */
    unsigned long long tick_ns_sum;
    unsigned long long tick_ns_worst;
    aotx_decode_test_token *token;
} aotx_decode_test_drain;

static double aotx_decode_test_now(void)
{
    struct timespec at;
    clock_gettime(CLOCK_MONOTONIC, &at);
    return (double)at.tv_sec + 1e-9 * (double)at.tv_nsec;
}

static unsigned long long aotx_decode_test_load(const void *at)
{
    return __atomic_load_n((const unsigned long long *)at, __ATOMIC_ACQUIRE);
}

/* Take one block and keep the token records and the tick times it holds. */
static int aotx_decode_test_block(aotx_decode_test_drain *state)
{
    const unsigned char *at = state->data + (state->cursor & state->mask);
    const aotx_block_header *block = (const aotx_block_header *)at;
    unsigned long long first = aotx_decode_test_load(&block->block_seq);
    if (first == 0ull) {
        return 0;
    }
    aotx_block_header header = *block;
    if (header.magic != AOTX_BLOCK_MAGIC || header.boot_id != state->boot_id
        || header.byte_len < AOTX_BLOCK_HEADER_BYTES
        || (unsigned long long)header.byte_len
           > state->data_bytes - (state->cursor & state->mask)) {
        state->bad += 1ull;
        return -1;
    }
    for (unsigned int i = 0u; i < header.record_count; ++i) {
        const aotx_record_header *record =
            (const aotx_record_header *)(at + AOTX_BLOCK_HEADER_BYTES
                                         + (size_t)i * AOTX_SLOT_BYTES);
        const unsigned char *body = (const unsigned char *)record + AOTX_HEADER_BYTES;
        if (record->type == AOTX_REC_TOKEN) {
            const aotx_token_body *one = (const aotx_token_body *)body;
            if (state->taken < AOTX_DECODE_TEST_RECORDS) {
                aotx_decode_test_token *keep = &state->token[state->taken];
                keep->slot = one->slot;
                keep->token = one->token;
                keep->position = one->position;
                keep->flags = one->flags;
                keep->seed = one->seed;
                keep->draw = one->draw;
                state->taken += 1u;
            } else {
                state->lost += 1u;
            }
        } else if (record->type == AOTX_REC_SEQUENCE) {
            const aotx_sequence_body *one = (const aotx_sequence_body *)body;
            state->events += 1u;
            if (one->event == AOTX_SEQ_RELEASED) {
                state->released += 1u;
            }
        } else if (record->type == AOTX_REC_STATS) {
            const aotx_stats_body *one = (const aotx_stats_body *)body;
            state->ticks += 1u;
            state->tick_ns_sum += one->tick_ns;
            if (one->tick_ns > state->tick_ns_worst) {
                state->tick_ns_worst = one->tick_ns;
            }
        }
    }
    if (aotx_decode_test_load(&block->block_seq) != first) {
        return 0;
    }
    state->blocks += 1ull;
    state->cursor += header.byte_len;
    return 1;
}

static void *aotx_decode_test_reader(void *argument)
{
    aotx_decode_test_drain *state = (aotx_decode_test_drain *)argument;
    aotx_host_ring_preamble *preamble = (aotx_host_ring_preamble *)state->map;
    while (state->stop == 0) {
        unsigned long long head = aotx_decode_test_load(&preamble->head);
        while (state->cursor < head) {
            if (aotx_decode_test_block(state) <= 0) {
                break;
            }
        }
        __atomic_store_n(&preamble->cursor, state->cursor, __ATOMIC_RELEASE);
        usleep(200);
    }
    return NULL;
}

/* The prompts of the check: one row of the golden list for each sequence. */
typedef struct aotx_decode_test_prompt {
    int ids[AOTX_SEQ_SLOTS * 64u];
    unsigned int start[AOTX_SEQ_SLOTS];
    unsigned int count[AOTX_SEQ_SLOTS];
    unsigned int rows;
} aotx_decode_test_prompt;

/* Read the golden token list and keep the first rows of it, at most 64 tokens each. */
static int aotx_decode_test_prompts(const char *path, aotx_decode_test_prompt *out)
{
    FILE *file = fopen(path, "rb");
    char line[65536];
    if (file == NULL) {
        return 1;
    }
    memset(out, 0, sizeof *out);
    unsigned int at = 0u;
    while (out->rows < AOTX_SEQ_SLOTS && fgets(line, sizeof line, file) != NULL) {
        if (line[0] == '#' || line[0] == '\n') {
            continue;
        }
        unsigned int count = 0u;
        const char *walk = line;
        out->start[out->rows] = at;
        while (*walk != '\0' && count < 64u) {
            char *end = NULL;
            long value = strtol(walk, &end, 10);
            if (end == walk) {
                break;
            }
            out->ids[at] = (int)value;
            at += 1u;
            count += 1u;
            walk = (*end == ',') ? (end + 1) : end;
        }
        if (count < 2u) {
            at = out->start[out->rows];
            continue;
        }
        out->count[out->rows] = count;
        out->rows += 1u;
    }
    fclose(file);
    return (out->rows >= AOTX_SEQ_SLOTS) ? 0 : 1;
}

/* Bytes of the token lists of every slot in the sequence table. */
#define AOTX_DECODE_TEST_LIST_BYTES \
    ((size_t)AOTX_SEQ_SLOTS * AOTX_SEQ_MAX_TOKENS * sizeof(int))

/* The token lists of every slot, from the sequence table of the device. */
static void aotx_decode_test_tokens(int *out)
{
    aotx_check_runtime(cudaMemcpyFromSymbol(out, aotx_seqs, AOTX_DECODE_TEST_LIST_BYTES,
                                            offsetof(aotx_seq_table, tokens),
                                            cudaMemcpyDeviceToHost),
                       "cudaMemcpyFromSymbol");
}

/* Read the state hash of the device, or put a value back in it. */
static void aotx_decode_test_hash(unsigned long long *value, int write)
{
    size_t at = offsetof(aotx_seam_state, apply)
              + offsetof(aotx_seam_apply_state, state_hash);
    if (write != 0) {
        aotx_check_runtime(cudaMemcpyToSymbol(aotx_seam, value, sizeof *value, at),
                           "cudaMemcpyToSymbol");
        return;
    }
    aotx_check_runtime(cudaMemcpyFromSymbol(value, aotx_seam, sizeof *value, at),
                       "cudaMemcpyFromSymbol");
}

/* The logits of the last row of the first sequence, from the buffer of the head. */
static float *aotx_decode_test_logits(unsigned int role, unsigned int vocab)
{
    aotx_model_hold *hold = aotx_model_hold_of(role);
    float *row = (float *)malloc((size_t)vocab * sizeof(float));
    aotx_check_runtime(cudaMemcpy(row, hold->work.head, (size_t)vocab * sizeof(float),
                                  cudaMemcpyDeviceToHost), "cudaMemcpy");
    return row;
}

/* The largest value of a row, and the place it stands in. */
static unsigned int aotx_decode_test_argmax(const float *row, unsigned int vocab)
{
    unsigned int at = 0u;
    for (unsigned int i = 1u; i < vocab; ++i) {
        if (row[i] > row[at]) {
            at = i;
        }
    }
    return at;
}

/* Put a run of token records in the inbound ring, as the restore program does. The device
 * applies them at the tick that follows, up to AOTX_INBOUND_MAX_TICK of them in one tick. */
static void aotx_decode_test_feed(aotx_seam_rings *rings, const aotx_token_body *body,
                                  unsigned int count, unsigned long long boot_id)
{
    aotx_inbound_preamble *preamble = (aotx_inbound_preamble *)rings->inbound_map;
    unsigned char *slots = rings->inbound_map + preamble->preamble_bytes;
    unsigned long long mask = preamble->slot_count - 1ull;
    unsigned long long head = preamble->head;
    for (unsigned int i = 0u; i < count; ++i) {
        aotx_record_header *header =
            (aotx_record_header *)(slots + ((head + i) & mask) * AOTX_SLOT_BYTES);
        __atomic_store_n(&header->seq, 0ull, __ATOMIC_RELEASE);
        header->magic = AOTX_WIRE_MAGIC;
        header->layout = (unsigned short)AOTX_WIRE_LAYOUT;
        header->header_bytes = (unsigned short)AOTX_HEADER_BYTES;
        header->boot_id = boot_id;
        header->tick = 0ull;
        header->globaltimer = 0ull;
        header->writer = AOTX_WRITER_RESTORE;
        header->cls = (unsigned char)AOTX_CLASS_A;
        header->type = (unsigned char)AOTX_REC_TOKEN;
        header->flags = (unsigned short)AOTX_FLAG_REPLAYED;
        header->body_len = (unsigned int)sizeof *body;
        memcpy((unsigned char *)header + AOTX_HEADER_BYTES, &body[i], sizeof *body);
        __atomic_store_n(&header->seq, head + i + 1ull, __ATOMIC_RELEASE);
    }
    __atomic_store_n(&preamble->head, head + count, __ATOMIC_RELEASE);
}

/* The token records of one slot, in the order the journal holds them. The check reads the
 * positions, the flags and the identities of the tokens. */
static unsigned int aotx_decode_test_records(aotx_decode_test_drain *drain,
                                             unsigned int from, unsigned int slot,
                                             unsigned int prompt, unsigned int sampled,
                                             unsigned int *bad)
{
    unsigned int want = 0u;
    unsigned int last = 0u;
    unsigned int seen = 0u;
    for (unsigned int i = from; i < drain->taken; ++i) {
        const aotx_decode_test_token *one = &drain->token[i];
        if (one->slot != slot) {
            continue;
        }
        if (one->position != want) {
            *bad += 1u;
            return seen;
        }
        unsigned int mark = (want < prompt) ? AOTX_TOKEN_PROMPT : AOTX_TOKEN_SAMPLED;
        if ((one->flags & mark) == 0u) {
            *bad += 1u;
            return seen;
        }
        if ((one->flags & AOTX_TOKEN_LAST) != 0u) {
            last += 1u;
        }
        want += 1u;
        seen += 1u;
    }
    if (want != prompt + sampled || last != 1u) {
        *bad += 1u;
    }
    return seen;
}

/* Build the token records of a run of slots from what the consumer kept, in slot order and
 * in position order. The count of the last tokens to leave out is the caller's. */
static unsigned int aotx_decode_test_bodies(aotx_decode_test_drain *drain,
                                            unsigned int mark, unsigned int seqs,
                                            unsigned int role, unsigned int leave,
                                            const unsigned int *list,
                                            aotx_token_body *body)
{
    unsigned int count = 0u;
    for (unsigned int s = 0u; s < seqs; ++s) {
        unsigned int want = (list[s] > leave) ? (list[s] - leave) : 0u;
        unsigned int at = 0u;
        for (unsigned int i = mark; i < drain->taken && at < want; ++i) {
            const aotx_decode_test_token *one = &drain->token[i];
            if (one->slot != s || one->position != at) {
                continue;
            }
            body[count].slot = s;
            body[count].token = one->token;
            body[count].position = one->position;
            body[count].flags = one->flags & (unsigned int)(AOTX_TOKEN_PROMPT
                                                            | AOTX_TOKEN_SAMPLED);
            body[count].seed = one->seed;
            body[count].draw = one->draw;
            body[count].role = role;
            body[count].reserved = 0u;
            count += 1u;
            at += 1u;
        }
    }
    return count;
}

#endif
