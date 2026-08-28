/* Purpose: Read the records an agent check makes from the host ring.
 * Owns: The cursor of the test consumer and the record counts it keeps.
 * Threading: One host thread reads the ring while the pump makes ticks.
 * Lifetime: One run of the test program.
 */
#ifndef AOTX_TESTS_AGENT_DRAIN_H
#define AOTX_TESTS_AGENT_DRAIN_H

#include <string.h>
#include <unistd.h>

#include "agent/agent.cuh"
#include "tool/tool.cuh"
#include "seam/seam.cuh"

/* Manifest records and task records the consumer keeps. */
#define AOTX_AGENT_TEST_KEEP  4096u

typedef struct aotx_agent_test_drain {
    const unsigned char *map;
    const unsigned char *data;
    unsigned long long data_bytes;
    unsigned long long mask;
    unsigned long long boot_id;
    volatile int stop;
    unsigned long long cursor;
    unsigned long long blocks;
    unsigned long long bad;
    unsigned int manifests;
    unsigned int tasks;
    unsigned int agents;
    unsigned int requests;
    unsigned int handoffs;
    unsigned int findings;
    unsigned int lost;
    aotx_tool_request_body request;   /* the last tool request record of the run */
    aotx_manifest_body manifest[AOTX_AGENT_TEST_KEEP];
    aotx_task_body task[AOTX_AGENT_TEST_KEEP];
} aotx_agent_test_drain;

static unsigned long long aotx_agent_test_load(const void *at)
{
    return __atomic_load_n((const unsigned long long *)at, __ATOMIC_ACQUIRE);
}

static int aotx_agent_test_block(aotx_agent_test_drain *state)
{
    const unsigned char *at = state->data + (state->cursor & state->mask);
    const aotx_block_header *block = (const aotx_block_header *)at;
    unsigned long long first = aotx_agent_test_load(&block->block_seq);
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
        if (record->type == AOTX_REC_MANIFEST) {
            if (state->manifests < AOTX_AGENT_TEST_KEEP) {
                memcpy(&state->manifest[state->manifests], body,
                       sizeof(aotx_manifest_body));
            } else {
                state->lost += 1u;
            }
            state->manifests += 1u;
        } else if (record->type == AOTX_REC_TASK) {
            if (state->tasks < AOTX_AGENT_TEST_KEEP) {
                memcpy(&state->task[state->tasks], body, sizeof(aotx_task_body));
            } else {
                state->lost += 1u;
            }
            state->tasks += 1u;
        } else if (record->type == AOTX_REC_AGENT) {
            state->agents += 1u;
        } else if (record->type == AOTX_REC_TOOL_REQUEST) {
            memcpy(&state->request, body, sizeof(aotx_tool_request_body));
            state->requests += 1u;
        } else if (record->type == AOTX_REC_BUS) {
            const aotx_bus_body *one = (const aotx_bus_body *)body;
            if (one->kind == AOTX_BUS_HANDOFF) {
                state->handoffs += 1u;
            } else if (one->kind == AOTX_BUS_FINDING) {
                state->findings += 1u;
            }
        }
    }
    if (aotx_agent_test_load(&block->block_seq) != first) {
        return 0;
    }
    state->blocks += 1ull;
    state->cursor += header.byte_len;
    return 1;
}

static void *aotx_agent_test_reader(void *argument)
{
    aotx_agent_test_drain *state = (aotx_agent_test_drain *)argument;
    aotx_host_ring_preamble *preamble = (aotx_host_ring_preamble *)state->map;
    while (state->stop == 0) {
        unsigned long long head = aotx_agent_test_load(&preamble->head);
        while (state->cursor < head) {
            if (aotx_agent_test_block(state) <= 0) {
                break;
            }
        }
        __atomic_store_n(&preamble->cursor, state->cursor, __ATOMIC_RELEASE);
        usleep(200);
    }
    return NULL;
}

#endif
