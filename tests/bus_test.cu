/* Purpose: Check the bus: the writer stamp, the writer count, the refusals and the cursors.
 * Owns: The test fixtures and the counts of the cases.
 * Launch shape: One thread for each message; one block for each writer.
 * Lifetime: One run of the test program. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "boot/check.h"
#include "bus/bus.cuh"
#include "mem/mem.cuh"
#include "sched/sched.cuh"
#include "seam/seam.cuh"

#define AOTX_TEST_MESSAGES 64u
#define AOTX_TEST_LIST     32u
#define AOTX_TEST_REFUSALS 10u

/* Records written after a message, to show that a message stays while the ring holds it. */
#define AOTX_TEST_FAR      8192ull

/* The text of one message names its writer and its position, so no two are the same and a
 * wrong index cannot pass. */
__device__ __host__ unsigned int aotx_bus_test_text(char *out, unsigned int writer,
                                                    unsigned int index)
{
    const char *head = "writer ";
    unsigned int at = 0u;
    for (unsigned int i = 0u; head[i] != 0; ++i) {
        out[at++] = head[i];
    }
    unsigned int values[2] = { writer, index };
    for (unsigned int v = 0u; v < 2u; ++v) {
        unsigned int value = values[v];
        char digits[12];
        unsigned int count = 0u;
        do {
            digits[count++] = (char)('0' + (value % 10u));
            value /= 10u;
        } while (value != 0u);
        while (count != 0u) {
            out[at++] = digits[--count];
        }
        if (v == 0u) {
            const char *mid = " message ";
            for (unsigned int i = 0u; mid[i] != 0; ++i) {
                out[at++] = mid[i];
            }
        }
    }
    return at;
}

/* One block for each writer, one thread for each message. Every thread appends at once, so
 * the count of a writer comes from the table and not from an order the test made. */
__global__ void aotx_bus_test_write(unsigned int messages)
{
    if (threadIdx.x >= messages) {
        return;
    }
    char text[64];
    unsigned int writer = AOTX_WRITER_AGENT_BASE + blockIdx.x;
    unsigned int length = aotx_bus_test_text(text, blockIdx.x, threadIdx.x);
    unsigned int kind = (threadIdx.x % 2u == 0u) ? AOTX_BUS_FINDING : AOTX_BUS_NOTE;
    unsigned int provenance = (kind == AOTX_BUS_FINDING)
                            ? (threadIdx.x % 4u) / 2u + AOTX_PROV_COMPUTED : 0u;
    aotx_bus_append(writer, kind, provenance, text, length, 0ull, 0ull, 0.0f,
                    aotx_time_tick);
}

/* Every call here breaks one rule, so every call must write nothing and give back zero. */
__global__ void aotx_bus_test_refuse(unsigned long long *results)
{
    const char *text = "refused";
    results[0] = aotx_bus_append(AOTX_WRITER_CONSOLE, AOTX_BUS_FINDING, 0u, text, 7u,
                                 0ull, 0ull, 0.0f, 1ull);
    results[1] = aotx_bus_append(AOTX_WRITER_CONSOLE, AOTX_BUS_FINDING, 5u, text, 7u,
                                 0ull, 0ull, 0.0f, 1ull);
    results[2] = aotx_bus_append(AOTX_WRITER_CONSOLE, AOTX_BUS_NOTE, AOTX_PROV_COMPUTED,
                                 text, 7u, 0ull, 0ull, 0.0f, 1ull);
    results[3] = aotx_bus_append(AOTX_BUS_WRITER_MAX, AOTX_BUS_NOTE, 0u, text, 7u,
                                 0ull, 0ull, 0.0f, 1ull);
    results[4] = aotx_bus_append(AOTX_BUS_WRITER_MAX + 4096u, AOTX_BUS_NOTE, 0u, text, 7u,
                                 0ull, 0ull, 0.0f, 1ull);
    results[5] = aotx_bus_append(AOTX_WRITER_CONSOLE, 0u, 0u, text, 7u, 0ull, 0ull, 0.0f,
                                 1ull);
    results[6] = aotx_bus_append(AOTX_WRITER_CONSOLE, AOTX_BUS_NOTE + 1u, 0u, text, 7u,
                                 0ull, 0ull, 0.0f, 1ull);
    results[7] = aotx_bus_append(AOTX_WRITER_CONSOLE, AOTX_BUS_RANK, AOTX_PROV_FETCHED,
                                 text, 7u, 0ull, 0ull, 0.5f, 1ull);
    /* A writer between the system set and the agent slots has no name on the disk side. */
    results[8] = aotx_bus_append(AOTX_WRITER_CONSOLE + 1u, AOTX_BUS_NOTE, 0u, text, 7u,
                                 0ull, 0ull, 0.0f, 1ull);
    results[9] = aotx_bus_append(AOTX_WRITER_AGENT_BASE - 1u, AOTX_BUS_NOTE, 0u, text, 7u,
                                 0ull, 0ull, 0.0f, 1ull);
}

/* One message of a kind that no other case writes, so the scan can find it alone. */
__global__ void aotx_bus_test_far(unsigned long long *seq)
{
    *seq = aotx_bus_append(AOTX_WRITER_CONSOLE, AOTX_BUS_QUESTION, 0u, "far back", 8u,
                           0ull, 0ull, 0.0f, 11ull);
}

/* A message with a text longer than the body holds keeps the bytes that fit. */
__global__ void aotx_bus_test_long(unsigned long long *seq)
{
    char text[AOTX_BUS_TEXT_BYTES + 32u];
    for (unsigned int i = 0u; i < AOTX_BUS_TEXT_BYTES + 32u; ++i) {
        text[i] = (char)('a' + (i % 26u));
    }
    *seq = aotx_bus_append(AOTX_WRITER_CONSOLE, AOTX_BUS_NOTE, 0u, text,
                           AOTX_BUS_TEXT_BYTES + 32u, 0ull, 0ull, 0.0f, 7ull);
}

/* One rank message that names another message, so the cursor filter has two kinds to sort. */
__global__ void aotx_bus_test_rank(unsigned long long re_seq, unsigned long long *seq)
{
    *seq = aotx_bus_append(AOTX_WRITER_CONSOLE, AOTX_BUS_RANK, 0u, "rank", 4u, re_seq,
                           0ull, 1.75f, 9ull);
}

__global__ void aotx_bus_test_recent(unsigned int mask, unsigned int max,
                                     unsigned long long *seqs, unsigned int *count)
{
    *count = aotx_bus_recent(mask, max, seqs);
}

/* The body of a sequence that a later record wrote over is not the body of that sequence. */
__global__ void aotx_bus_test_body_of(unsigned long long seq, unsigned long long *found,
                                      unsigned int *kind)
{
    const aotx_bus_body *body = aotx_bus_body_of(seq);
    *found = (body == 0) ? 0ull : 1ull;
    *kind = (body == 0) ? 0u : body->kind;
}

/* Fill the ring with records that carry nothing, so every earlier slot is written over. */
__global__ void aotx_bus_test_fill(unsigned long long count)
{
    unsigned long long lane = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = (unsigned long long)(gridDim.x * blockDim.x);
    for (unsigned long long i = lane; i < count; i += stride) {
        aotx_seam_pad(aotx_seam_claim(1u));
    }
}

static void aotx_bus_test_state(aotx_bus_state *state)
{
    aotx_check_runtime(cudaMemcpyFromSymbol(state, aotx_bus, sizeof *state),
                       "cudaMemcpyFromSymbol");
}

static unsigned long long aotx_bus_test_refused(void)
{
    aotx_bus_state state;
    aotx_bus_test_state(&state);
    return state.refused;
}

static unsigned long long aotx_bus_test_tail(void)
{
    aotx_seam_state state;
    aotx_check_runtime(cudaMemcpyFromSymbol(&state, aotx_seam, sizeof state),
                       "cudaMemcpyFromSymbol");
    return state.dev.tail;
}

/* One pass at a writer count: append, then read the ring back and check every message. */
static unsigned int aotx_bus_test_pass(unsigned char *ring, const aotx_mem_map *map,
                                       unsigned int writers, unsigned int *failed)
{
    unsigned long long slots = map->ring_bytes / AOTX_SLOT_BYTES;
    unsigned long long from = aotx_bus_test_tail();

    /* A writer keeps its count for the whole run. The count this pass must give starts at
     * the count before it plus one. */
    aotx_bus_state before;
    aotx_bus_test_state(&before);
    aotx_bus_test_write<<<writers, AOTX_TEST_MESSAGES>>>(AOTX_TEST_MESSAGES);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    unsigned long long to = aotx_bus_test_tail();
    aotx_check_runtime(cudaMemcpy(ring, (const void *)map->ring, (size_t)map->ring_bytes,
                                  cudaMemcpyDeviceToHost), "cudaMemcpy");
    unsigned int applied = 0u;

    unsigned int *seen = (unsigned int *)calloc((size_t)writers * AOTX_TEST_MESSAGES,
                                                sizeof(unsigned int));
    unsigned int found = 0u;
    unsigned int wrong_writer = 0u;
    unsigned int wrong_text = 0u;
    unsigned int wrong_length = 0u;
    for (unsigned long long seq = from + 1ull; seq <= to; ++seq) {
        const aotx_record_header *record = (const aotx_record_header *)
            (ring + ((seq - 1ull) & (slots - 1ull)) * AOTX_SLOT_BYTES);
        if (record->seq != seq || record->type != AOTX_REC_BUS) {
            continue;
        }
        const aotx_bus_body *body =
            (const aotx_bus_body *)((const unsigned char *)record + AOTX_HEADER_BYTES);
        unsigned int writer = record->writer - AOTX_WRITER_AGENT_BASE;
        if (record->writer < AOTX_WRITER_AGENT_BASE || writer >= writers) {
            wrong_writer += 1u;
            continue;
        }
        char want[64];
        unsigned int length = 0u;
        unsigned int index = 0u;
        for (index = 0u; index < AOTX_TEST_MESSAGES; ++index) {
            length = aotx_bus_test_text(want, writer, index);
            if (length == body->text_len && memcmp(want, body->text, length) == 0) {
                break;
            }
        }
        if (index == AOTX_TEST_MESSAGES) {
            wrong_text += 1u;
            continue;
        }
        if (record->body_len != 32u + body->text_len) {
            wrong_length += 1u;
        }
        unsigned int base = before.writer_seq[AOTX_WRITER_AGENT_BASE + writer];
        if (body->writer_seq > base && body->writer_seq <= base + AOTX_TEST_MESSAGES) {
            seen[(size_t)writer * AOTX_TEST_MESSAGES + body->writer_seq - base - 1u] += 1u;
        }
        found += 1u;
    }

    applied += 4u;
    if (found != writers * AOTX_TEST_MESSAGES) {
        printf("bus: %u messages of %u reached the ring at %u writers\n",
               found, writers * AOTX_TEST_MESSAGES, writers);
        *failed += 1u;
    }
    if (wrong_writer != 0u || wrong_text != 0u) {
        printf("bus: %u messages carry a wrong writer and %u a wrong text at %u writers\n",
               wrong_writer, wrong_text, writers);
        *failed += 1u;
    }
    if (wrong_length != 0u) {
        printf("bus: %u messages carry a body length that the text does not give\n",
               wrong_length);
        *failed += 1u;
    }
    unsigned int gaps = 0u;
    for (unsigned int w = 0u; w < writers; ++w) {
        for (unsigned int m = 0u; m < AOTX_TEST_MESSAGES; ++m) {
            if (seen[(size_t)w * AOTX_TEST_MESSAGES + m] != 1u) {
                gaps += 1u;
            }
        }
    }
    if (gaps != 0u) {
        printf("bus: %u of the writer counts at %u writers are not one to %u\n",
               gaps, writers, AOTX_TEST_MESSAGES);
        *failed += 1u;
    }
    free(seen);
    return applied;
}

int main(void)
{
    CUdevice device;
    CUcontext context;
    aotx_mem_map map;
    aotx_seam_rings rings;
    unsigned int applied = 0u;
    unsigned int failed = 0u;
    aotx_check_driver(cuInit(0), "cuInit");
    aotx_check_driver(cuDeviceGet(&device, 0), "cuDeviceGet");
    aotx_check_driver(cuDevicePrimaryCtxRetain(&context, device), "cuDevicePrimaryCtxRetain");
    aotx_check_driver(cuCtxSetCurrent(context), "cuCtxSetCurrent");

    unsigned long long boot_id = 0xB05B05ull;
    if (aotx_mem_reserve(&map) != 0 || aotx_seam_open(&rings, boot_id) != 0) {
        printf("bus: the map or the rings did not open\n");
        return 1;
    }
    aotx_seam_bind(&rings, map.ring, map.ring_bytes, boot_id);
    unsigned char *ring = (unsigned char *)malloc((size_t)map.ring_bytes);

    /* Case set 1: one writer, then AOTX_SLOTS writers, all appending at once. The table
     * of the bus holds one entry for each slot, so the profile gives the count. */
    const unsigned int writers[2] = { 1u, AOTX_SLOTS };
    for (unsigned int w = 0u; w < 2u; ++w) {
        applied += aotx_bus_test_pass(ring, &map, writers[w], &failed);
    }

    /* Case set 2: every refusal. Nothing is written and the counter counts each one. */
    {
        unsigned long long *results = 0;
        unsigned long long got[AOTX_TEST_REFUSALS];
        aotx_check_runtime(cudaMalloc(&results, sizeof got), "cudaMalloc");
        unsigned long long before = aotx_bus_test_refused();
        unsigned long long tail = aotx_bus_test_tail();
        aotx_bus_test_refuse<<<1, 1>>>(results);
        aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        aotx_check_runtime(cudaMemcpy(got, results, sizeof got, cudaMemcpyDeviceToHost),
                           "cudaMemcpy");
        unsigned int wrote = 0u;
        for (unsigned int i = 0u; i < AOTX_TEST_REFUSALS; ++i) {
            if (got[i] != 0ull) {
                wrote += 1u;
            }
        }
        applied += 3u;
        if (wrote != 0u) {
            printf("bus: %u of %u refused messages gave back a sequence\n",
                   wrote, AOTX_TEST_REFUSALS);
            failed += 1u;
        }
        if (aotx_bus_test_refused() != before + AOTX_TEST_REFUSALS) {
            printf("bus: the refusal counter moved by %llu and not by %u\n",
                   aotx_bus_test_refused() - before, AOTX_TEST_REFUSALS);
            failed += 1u;
        }
        if (aotx_bus_test_tail() != tail) {
            printf("bus: a refused message took a record sequence\n");
            failed += 1u;
        }
        cudaFree(results);
    }

    /* Case set 3: the cursors. The list is newest first and holds only the kinds asked for. */
    {
        unsigned long long *device_seqs = 0;
        unsigned int *device_count = 0;
        unsigned long long *device_seq = 0;
        unsigned long long seqs[AOTX_TEST_LIST];
        unsigned long long note_seq = 0ull;
        unsigned long long rank_seq = 0ull;
        unsigned int count = 0u;
        aotx_check_runtime(cudaMalloc(&device_seqs, sizeof seqs), "cudaMalloc");
        aotx_check_runtime(cudaMalloc(&device_count, sizeof count), "cudaMalloc");
        aotx_check_runtime(cudaMalloc(&device_seq, sizeof note_seq), "cudaMalloc");
        aotx_bus_test_long<<<1, 1>>>(device_seq);
        aotx_check_runtime(cudaMemcpy(&note_seq, device_seq, sizeof note_seq,
                                      cudaMemcpyDeviceToHost), "cudaMemcpy");
        aotx_bus_test_rank<<<1, 1>>>(note_seq, device_seq);
        aotx_check_runtime(cudaMemcpy(&rank_seq, device_seq, sizeof rank_seq,
                                      cudaMemcpyDeviceToHost), "cudaMemcpy");

        aotx_bus_test_recent<<<1, 1>>>(1u << AOTX_BUS_RANK, AOTX_TEST_LIST, device_seqs,
                                       device_count);
        aotx_check_runtime(cudaMemcpy(&count, device_count, sizeof count,
                                      cudaMemcpyDeviceToHost), "cudaMemcpy");
        aotx_check_runtime(cudaMemcpy(seqs, device_seqs, sizeof seqs,
                                      cudaMemcpyDeviceToHost), "cudaMemcpy");
        applied += 2u;
        if (count != 1u || seqs[0] != rank_seq) {
            printf("bus: the rank filter gave %u entries and the first is %llu of %llu\n",
                   count, count > 0u ? seqs[0] : 0ull, rank_seq);
            failed += 1u;
        }
        unsigned int mask = (1u << AOTX_BUS_FINDING) | (1u << AOTX_BUS_NOTE)
                          | (1u << AOTX_BUS_RANK);
        aotx_bus_test_recent<<<1, 1>>>(mask, AOTX_TEST_LIST, device_seqs, device_count);
        aotx_check_runtime(cudaMemcpy(&count, device_count, sizeof count,
                                      cudaMemcpyDeviceToHost), "cudaMemcpy");
        aotx_check_runtime(cudaMemcpy(seqs, device_seqs, sizeof seqs,
                                      cudaMemcpyDeviceToHost), "cudaMemcpy");
        unsigned int order = 0u;
        for (unsigned int i = 1u; i < count; ++i) {
            if (seqs[i] >= seqs[i - 1u]) {
                order += 1u;
            }
        }
        applied += 3u;
        if (count != AOTX_TEST_LIST) {
            printf("bus: the list gave %u entries and %u were asked for\n",
                   count, AOTX_TEST_LIST);
            failed += 1u;
        }
        if (order != 0u) {
            printf("bus: %u entries of the list are not newest first\n", order);
            failed += 1u;
        }
        if (count > 0u && seqs[0] != rank_seq) {
            printf("bus: the newest entry is %llu and the last message is %llu\n",
                   seqs[0], rank_seq);
            failed += 1u;
        }

        /* The long text kept the bytes that fit, and the record states that length. */
        aotx_check_runtime(cudaMemcpy(ring, (const void *)map.ring, (size_t)map.ring_bytes,
                                      cudaMemcpyDeviceToHost), "cudaMemcpy");
        unsigned long long slots = map.ring_bytes / AOTX_SLOT_BYTES;
        const aotx_record_header *record = (const aotx_record_header *)
            (ring + ((note_seq - 1ull) & (slots - 1ull)) * AOTX_SLOT_BYTES);
        const aotx_bus_body *body =
            (const aotx_bus_body *)((const unsigned char *)record + AOTX_HEADER_BYTES);
        applied += 3u;
        if (body->text_len != AOTX_BUS_TEXT_BYTES) {
            printf("bus: the long text gave a length of %u\n", body->text_len);
            failed += 1u;
        }
        if (body->text[0] != 'a' || body->text[AOTX_BUS_TEXT_BYTES - 1u]
            != (char)('a' + ((AOTX_BUS_TEXT_BYTES - 1u) % 26u))) {
            printf("bus: the long text did not keep the bytes that fit\n");
            failed += 1u;
        }
        record = (const aotx_record_header *)
            (ring + ((rank_seq - 1ull) & (slots - 1ull)) * AOTX_SLOT_BYTES);
        body = (const aotx_bus_body *)((const unsigned char *)record + AOTX_HEADER_BYTES);
        if (body->re_seq != note_seq || body->score != 1.0f) {
            printf("bus: the rank names %llu and carries the score %f\n",
                   (unsigned long long)body->re_seq, (double)body->score);
            failed += 1u;
        }

        /* Case set 4: a sequence whose slot a later record took gives back no body. */
        unsigned long long *device_found = 0;
        unsigned int *device_kind = 0;
        unsigned long long found = 0ull;
        unsigned int kind = 0u;
        aotx_check_runtime(cudaMalloc(&device_found, sizeof found), "cudaMalloc");
        aotx_check_runtime(cudaMalloc(&device_kind, sizeof kind), "cudaMalloc");
        aotx_bus_test_body_of<<<1, 1>>>(rank_seq, device_found, device_kind);
        aotx_check_runtime(cudaMemcpy(&found, device_found, sizeof found,
                                      cudaMemcpyDeviceToHost), "cudaMemcpy");
        aotx_check_runtime(cudaMemcpy(&kind, device_kind, sizeof kind,
                                      cudaMemcpyDeviceToHost), "cudaMemcpy");
        applied += 2u;
        if (found != 1ull || kind != AOTX_BUS_RANK) {
            printf("bus: the body of the last rank was not found\n");
            failed += 1u;
        }
        aotx_bus_test_fill<<<64, 256>>>(slots + 1024ull);
        aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        aotx_bus_test_body_of<<<1, 1>>>(rank_seq, device_found, device_kind);
        aotx_check_runtime(cudaMemcpy(&found, device_found, sizeof found,
                                      cudaMemcpyDeviceToHost), "cudaMemcpy");
        if (found != 0ull) {
            printf("bus: the body of a sequence that was written over was given back\n");
            failed += 1u;
        }
        cudaFree(device_seqs);
        cudaFree(device_count);
        cudaFree(device_seq);
        cudaFree(device_found);
        cudaFree(device_kind);
    }

    /* Case set 5: a message stays for as long as the ring holds its slot. The scan covers
     * the whole ring, so a message that 8,192 later records did not reach is still found. */
    {
        unsigned long long *device_seq = 0;
        unsigned long long *device_seqs = 0;
        unsigned int *device_count = 0;
        unsigned long long far_seq = 0ull;
        unsigned long long seqs[AOTX_TEST_LIST];
        unsigned int count = 0u;
        unsigned int kind = 0u;
        unsigned long long found = 0ull;
        unsigned long long *device_found = 0;
        unsigned int *device_kind = 0;
        aotx_check_runtime(cudaMalloc(&device_seq, sizeof far_seq), "cudaMalloc");
        aotx_check_runtime(cudaMalloc(&device_seqs, sizeof seqs), "cudaMalloc");
        aotx_check_runtime(cudaMalloc(&device_count, sizeof count), "cudaMalloc");
        aotx_check_runtime(cudaMalloc(&device_found, sizeof found), "cudaMalloc");
        aotx_check_runtime(cudaMalloc(&device_kind, sizeof kind), "cudaMalloc");
        aotx_bus_test_far<<<1, 1>>>(device_seq);
        aotx_check_runtime(cudaMemcpy(&far_seq, device_seq, sizeof far_seq,
                                      cudaMemcpyDeviceToHost), "cudaMemcpy");
        aotx_bus_test_fill<<<64, 256>>>(AOTX_TEST_FAR);
        aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        aotx_bus_test_recent<<<1, 1>>>(1u << AOTX_BUS_QUESTION, AOTX_TEST_LIST,
                                       device_seqs, device_count);
        aotx_check_runtime(cudaMemcpy(&count, device_count, sizeof count,
                                      cudaMemcpyDeviceToHost), "cudaMemcpy");
        aotx_check_runtime(cudaMemcpy(seqs, device_seqs, sizeof seqs,
                                      cudaMemcpyDeviceToHost), "cudaMemcpy");
        aotx_bus_test_body_of<<<1, 1>>>(far_seq, device_found, device_kind);
        aotx_check_runtime(cudaMemcpy(&found, device_found, sizeof found,
                                      cudaMemcpyDeviceToHost), "cudaMemcpy");
        aotx_check_runtime(cudaMemcpy(&kind, device_kind, sizeof kind,
                                      cudaMemcpyDeviceToHost), "cudaMemcpy");
        applied += 2u;
        if (count != 1u || seqs[0] != far_seq) {
            printf("bus: the scan gave %u entries for a message %llu records back\n",
                   count, AOTX_TEST_FAR);
            failed += 1u;
        }
        if (found != 1ull || kind != AOTX_BUS_QUESTION) {
            printf("bus: the body of a message %llu records back was not found\n",
                   AOTX_TEST_FAR);
            failed += 1u;
        }
        cudaFree(device_seq);
        cudaFree(device_seqs);
        cudaFree(device_count);
        cudaFree(device_found);
        cudaFree(device_kind);
    }

    free(ring);
    aotx_seam_finish(&rings);
    aotx_seam_close(&rings);
    aotx_mem_release(&map);
    printf("bus: %u cases applied, %u failed\n", applied, failed);
    return failed == 0u ? 0 : 1;
}
