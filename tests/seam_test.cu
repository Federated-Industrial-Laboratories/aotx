/* Purpose: Check the seam: the rate, the sequences, a held tick, and the apply.
 * Owns: The test consumer, the test fixtures and the counts of the cases.
 * Launch shape: The tick graph supplies the kernels; the consumer is a host thread.
 * Lifetime: One run of the test program. */
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "boot/boot.cuh"
#include "cli/cli.cuh"
#include "boot/check.h"
#include "mem/mem.cuh"
#include "sched/sched.cuh"
#include "seam/seam.cuh"

#define AOTX_TEST_KEEP     4096u
#define AOTX_TEST_RECENT   64u
#define AOTX_TEST_TYPES    16u
#define AOTX_TEST_LINE     64u

/* What the consumer of the host ring holds. The consumer is the drain of this test: it
 * accepts blocks by the double load rule and moves the cursor. */
typedef struct aotx_test_consumer {
    const unsigned char *map;
    const unsigned char *data;
    unsigned long long data_bytes;
    unsigned long long mask;
    volatile int paused;
    volatile int stop;
    volatile unsigned long long cursor;
    unsigned long long expect_block;
    unsigned long long expect_seq;
    unsigned long long boot_id;
    unsigned long long blocks;
    unsigned long long pads;
    unsigned long long records;
    unsigned long long by_type[AOTX_TEST_TYPES];
    unsigned long long gaps;
    unsigned long long bad;
    unsigned long long retries;
    unsigned long long last_hash;
    unsigned long long last_applied;
    unsigned int recent_at;
    unsigned long long recent_block[AOTX_TEST_RECENT];
    unsigned int recent_type[AOTX_TEST_RECENT][AOTX_TEST_TYPES];
    int capture;
    unsigned int kept;
    unsigned char *keep;
} aotx_test_consumer;

static long long aotx_test_now_ns(void)
{
    struct timespec at;
    clock_gettime(CLOCK_MONOTONIC, &at);
    return (long long)at.tv_sec * 1000000000ll + (long long)at.tv_nsec;
}

static unsigned long long aotx_test_load(const void *at)
{
    return __atomic_load_n((const unsigned long long *)at, __ATOMIC_ACQUIRE);
}

static void aotx_test_store(void *at, unsigned long long value)
{
    __atomic_store_n((unsigned long long *)at, value, __ATOMIC_RELEASE);
}

/* One block: check the header, check every record, then accept the block. The counts go in
 * only after the second load of the block sequence agrees with the first. */
static int aotx_test_block(aotx_test_consumer *state)
{
    const unsigned char *at = state->data + (state->cursor & state->mask);
    const aotx_block_header *block = (const aotx_block_header *)at;
    unsigned long long first = aotx_test_load(&block->block_seq);
    if (first == 0ull) {
        state->retries += 1ull;
        return 0;
    }
    aotx_block_header header = *block;
    if (header.magic != AOTX_BLOCK_MAGIC || header.layout != AOTX_WIRE_LAYOUT
        || header.boot_id != state->boot_id
        || header.byte_len < AOTX_BLOCK_HEADER_BYTES
        || (unsigned long long)header.byte_len
           > state->data_bytes - (state->cursor & state->mask)) {
        state->bad += 1ull;
        return -1;
    }
    /* The block rules the drain holds to. A record block is a header and whole records.
     * A pad block reaches the end of the data area, so the next block starts at zero. */
    unsigned long long offset = state->cursor & state->mask;
    if (header.kind == AOTX_BLOCK_PAD) {
        if (offset + header.byte_len != state->data_bytes || header.record_count != 0u) {
            state->bad += 1ull;
        }
    } else if ((unsigned long long)header.byte_len
               != AOTX_BLOCK_HEADER_BYTES
                  + (unsigned long long)header.record_count * AOTX_SLOT_BYTES) {
        state->bad += 1ull;
    }
    if ((offset & 7ull) != 0ull) {
        state->bad += 1ull;
    }

    unsigned int counted[AOTX_TEST_TYPES];
    memset(counted, 0, sizeof counted);
    unsigned long long next = state->expect_seq;
    unsigned long long bad = 0ull;
    unsigned long long gaps = 0ull;
    for (unsigned int i = 0u; i < header.record_count; ++i) {
        const aotx_record_header *record =
            (const aotx_record_header *)(at + AOTX_BLOCK_HEADER_BYTES
                                         + (size_t)i * AOTX_SLOT_BYTES);
        if (record->magic != AOTX_WIRE_MAGIC || record->layout != AOTX_WIRE_LAYOUT
            || record->boot_id != state->boot_id
            || record->body_len > AOTX_BODY_BYTES) {
            bad += 1ull;
            continue;
        }
        if (record->seq != next) {
            gaps += 1ull;
        }
        next = record->seq + 1ull;
        if (record->type < AOTX_TEST_TYPES) {
            counted[record->type] += 1u;
        }
        const unsigned long long *body =
            (const unsigned long long *)((const unsigned char *)record + AOTX_HEADER_BYTES);
        if (record->type == AOTX_REC_NOTE && body[1] != record->seq) {
            bad += 1ull;
        }
        if (record->type == AOTX_REC_TICK_COMMIT) {
            state->last_hash = body[0];
            state->last_applied = body[1];
        }
    }
    if (aotx_test_load(&block->block_seq) != first) {
        state->retries += 1ull;
        return 0;
    }
    if (first != state->expect_block) {
        state->gaps += 1ull;
    }
    state->expect_block = first + 1ull;
    state->expect_seq = next;
    state->gaps += gaps;
    state->bad += bad;
    state->blocks += 1ull;
    if (header.kind == AOTX_BLOCK_PAD) {
        state->pads += 1ull;
    }
    state->records += header.record_count;
    for (unsigned int t = 0u; t < AOTX_TEST_TYPES; ++t) {
        state->by_type[t] += counted[t];
        state->recent_type[state->recent_at][t] = counted[t];
    }
    state->recent_block[state->recent_at] = first;
    state->recent_at = (state->recent_at + 1u) % AOTX_TEST_RECENT;
    if (state->capture != 0) {
        for (unsigned int i = 0u; i < header.record_count && state->kept < AOTX_TEST_KEEP; ++i) {
            memcpy(state->keep + (size_t)state->kept * AOTX_SLOT_BYTES,
                   at + AOTX_BLOCK_HEADER_BYTES + (size_t)i * AOTX_SLOT_BYTES,
                   AOTX_SLOT_BYTES);
            state->kept += 1u;
        }
    }
    state->cursor += header.byte_len;
    return 1;
}

static void *aotx_test_drain(void *argument)
{
    aotx_test_consumer *state = (aotx_test_consumer *)argument;
    const aotx_host_ring_preamble *preamble = (const aotx_host_ring_preamble *)state->map;
    while (state->stop == 0) {
        if (state->paused != 0) {
            usleep(200);
            continue;
        }
        unsigned long long head = aotx_test_load(&preamble->head);
        int moved = 0;
        while (state->cursor < head) {
            int step = aotx_test_block(state);
            if (step <= 0) {
                break;
            }
            moved = 1;
        }
        if (moved != 0) {
            aotx_test_store((void *)&preamble->cursor, state->cursor);
        } else {
            usleep(100);
        }
    }
    return NULL;
}

/* Wait until the consumer has taken every block the device published. */
static void aotx_test_settle(aotx_test_consumer *state, const aotx_seam_rings *rings)
{
    const aotx_host_ring_preamble *preamble =
        (const aotx_host_ring_preamble *)rings->host_map;
    for (unsigned int i = 0u; i < 20000u; ++i) {
        if (state->cursor >= aotx_test_load(&preamble->head)) {
            return;
        }
        usleep(200);
    }
}

/* Write one record into the inbound ring, as the feeder and the replay do. */
static void aotx_test_put(const aotx_seam_rings *rings, unsigned long long boot_id,
                          unsigned long long index, unsigned int writer, unsigned int cls,
                          unsigned int type, unsigned int flags, const void *body,
                          unsigned int length)
{
    aotx_inbound_preamble *preamble = (aotx_inbound_preamble *)rings->inbound_map;
    unsigned char *slots = rings->inbound_map + sizeof(aotx_inbound_preamble);
    unsigned char *at = slots + (index & (AOTX_INBOUND_SLOTS - 1ull)) * AOTX_SLOT_BYTES;
    aotx_record_header *record = (aotx_record_header *)at;
    memset(at, 0, AOTX_SLOT_BYTES);
    record->magic = AOTX_WIRE_MAGIC;
    record->layout = (unsigned short)AOTX_WIRE_LAYOUT;
    record->header_bytes = (unsigned short)AOTX_HEADER_BYTES;
    record->boot_id = boot_id;
    record->tick = 0ull;
    record->globaltimer = 0ull;
    record->writer = writer;
    record->cls = (unsigned char)cls;
    record->type = (unsigned char)type;
    record->flags = (unsigned short)flags;
    record->body_len = length;
    memcpy(at + AOTX_HEADER_BYTES, body, length);
    __atomic_store_n(&record->seq, index + 1ull, __ATOMIC_RELEASE);
    aotx_test_store(&preamble->head, index + 1ull);
}

static void aotx_test_feed(const aotx_seam_rings *rings, unsigned long long boot_id,
                           unsigned long long index, const char *line, unsigned int length)
{
    aotx_test_put(rings, boot_id, index, AOTX_WRITER_FEEDER, AOTX_CLASS_A,
                  AOTX_REC_INPUT_LINE, 0u, line, length);
}

static unsigned long long aotx_test_fold(unsigned long long hash, const char *bytes,
                                         unsigned int length)
{
    for (unsigned int i = 0u; i < length; ++i) {
        hash ^= (unsigned long long)(unsigned char)bytes[i];
        hash *= AOTX_FNV_PRIME;
    }
    return hash;
}

int main(int argc, char **argv)
{
    unsigned long long workload = 12000ull;
    double seconds = 5.0;
    unsigned int applied = 0u;
    unsigned int failed = 0u;
    for (int i = 1; i < argc - 1; ++i) {
        if (strcmp(argv[i], "--workload") == 0) {
            workload = strtoull(argv[i + 1], NULL, 10);
        }
        if (strcmp(argv[i], "--seconds") == 0) {
            seconds = strtod(argv[i + 1], NULL);
        }
    }

    CUdevice device;
    CUcontext context;
    aotx_mem_map map;
    aotx_seam_rings rings;
    aotx_pump pump;
    aotx_pump_report report;
    aotx_check_driver(cuInit(0), "cuInit");
    aotx_check_driver(cuDeviceGet(&device, 0), "cuDeviceGet");
    aotx_check_driver(cuDevicePrimaryCtxRetain(&context, device), "cuDevicePrimaryCtxRetain");
    aotx_check_driver(cuCtxSetCurrent(context), "cuCtxSetCurrent");

    unsigned long long boot_id = 0x5EA11234ABCDull;
    if (aotx_mem_reserve(&map) != 0 || aotx_seam_open(&rings, boot_id) != 0) {
        printf("seam: the map or the rings did not open\n");
        return 1;
    }
    aotx_seam_bind(&rings, map.ring, map.ring_bytes, boot_id);
    aotx_seam_note_boot<<<1, 1>>>(0ull, 0ull);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    if (aotx_pump_build(&pump, 0ull, 1u) != 0) {
        printf("seam: the tick graph did not build\n");
        return 1;
    }

    aotx_test_consumer *state = (aotx_test_consumer *)calloc(1, sizeof *state);
    state->keep = (unsigned char *)malloc((size_t)AOTX_TEST_KEEP * AOTX_SLOT_BYTES);
    state->map = rings.host_map;
    state->data = rings.host_map + sizeof(aotx_host_ring_preamble);
    state->data_bytes = AOTX_HOST_RING_DATA_BYTES;
    state->mask = AOTX_HOST_RING_DATA_BYTES - 1ull;
    state->boot_id = boot_id;
    state->expect_block = 1ull;
    state->expect_seq = 1ull;
    pthread_t thread;
    pthread_create(&thread, NULL, aotx_test_drain, state);

    /* Case set 1: the rate, at one producer block and at 64. */
    const unsigned int producers[2] = { 1u, 64u };
    for (unsigned int p = 0u; p < 2u; ++p) {
        aotx_pump_set(&pump, workload, producers[p]);
        aotx_pump_read(&report);
        unsigned long long from = report.records;
        unsigned long long gaps = state->gaps + state->bad;
        long long started = aotx_test_now_ns();
        long long spent = 0ll;
        while ((double)spent < seconds * 1e9) {
            aotx_pump_tick(&pump);
            aotx_pump_pace(&pump);
            spent = aotx_test_now_ns() - started;
        }
        aotx_pump_read(&report);
        unsigned long long made = report.records - from;
        aotx_test_settle(state, &rings);
        double rate = (double)made * 1e9 / (double)spent;
        printf("seam: %u producer blocks made %llu records in %lld ms, %.0f a second\n",
               producers[p], made, spent / 1000000ll, rate);
        applied += 3u;
        if (rate < 1000000.0) {
            printf("seam: the rate at %u blocks is under one million a second\n",
                   producers[p]);
            failed += 1u;
        }
        if (state->gaps + state->bad != gaps) {
            printf("seam: %llu gaps and %llu bad records at %u blocks\n",
                   state->gaps, state->bad, producers[p]);
            failed += 1u;
        }
        if (report.held != 0ull) {
            printf("seam: %llu ticks were held at %u blocks\n", report.held, producers[p]);
            failed += 1u;
        }
    }

    /* Case set 2: a held tick, at one producer block and at 64. The consumer stops, so the
     * host ring fills and the tick start finds no room. A hold writes one stall record when
     * it starts and one when it ends, and no commit record while it lasts. */
    for (unsigned int p = 0u; p < 2u; ++p) {
        aotx_pump_read(&report);
        unsigned long long tick0 = report.tick;
        unsigned long long held0 = report.held;
        unsigned long long stalls0 = state->by_type[AOTX_REC_STALL];
        unsigned long long commits0 = state->by_type[AOTX_REC_TICK_COMMIT];
        unsigned long long notes0 = state->by_type[AOTX_REC_NOTE];
        state->paused = 1;
        aotx_pump_set(&pump, workload, producers[p]);
        for (unsigned int i = 0u; i < 200u; ++i) {
            aotx_pump_tick(&pump);
            aotx_pump_read(&report);
            if (report.held > held0) {
                break;
            }
        }
        applied += 1u;
        if (report.held <= held0) {
            printf("seam: no tick was held at %u blocks with the consumer stopped\n",
                   producers[p]);
            failed += 1u;
        }
        /* A hold that lasts writes nothing more: the ring tail and the last commit
         * sequence stand still while the hold goes on. */
        unsigned long long tail_held = report.tail;
        unsigned long long records_held = report.records;
        for (unsigned int i = 0u; i < 5u; ++i) {
            aotx_pump_tick(&pump);
        }
        aotx_pump_read(&report);
        applied += 3u;
        if (report.tail != tail_held) {
            printf("seam: the ring tail moved by %llu while the hold lasted at %u blocks\n",
                   report.tail - tail_held, producers[p]);
            failed += 1u;
        }
        if (report.records != records_held) {
            printf("seam: a commit record was written while the hold lasted at %u blocks\n",
                   producers[p]);
            failed += 1u;
        }
        if (report.held < held0 + 6ull) {
            printf("seam: only %llu ticks were held at %u blocks\n",
                   report.held - held0, producers[p]);
            failed += 1u;
        }
        /* The consumer starts again, and the next tick ends the hold and flows. */
        state->paused = 0;
        aotx_test_settle(state, &rings);
        unsigned long long ended = report.held;
        aotx_pump_tick(&pump);
        aotx_test_settle(state, &rings);
        aotx_pump_read(&report);
        applied += 4u;
        if (report.held != ended) {
            printf("seam: the tick after the hold at %u blocks was held too\n",
                   producers[p]);
            failed += 1u;
        }
        if (state->by_type[AOTX_REC_NOTE] <= notes0) {
            printf("seam: the tick after the hold at %u blocks made no load records\n",
                   producers[p]);
            failed += 1u;
        }
        unsigned long long stalls = state->by_type[AOTX_REC_STALL] - stalls0;
        if (stalls != 2ull) {
            printf("seam: the hold at %u blocks wrote %llu stall records, not 2\n",
                   producers[p], stalls);
            failed += 1u;
        }
        /* Every complete tick of the run wrote one commit record, and no held tick did. */
        unsigned long long ticks = report.tick - tick0;
        unsigned long long holds = report.held - held0;
        unsigned long long commits = state->by_type[AOTX_REC_TICK_COMMIT] - commits0;
        if (commits != ticks - holds) {
            printf("seam: %llu commit records for %llu complete ticks at %u blocks\n",
                   commits, ticks - holds, producers[p]);
            failed += 1u;
        }
    }

    /* Case set 3: the apply, at one input line and at 64. */
    aotx_pump_set(&pump, 0ull, 1u);
    const unsigned int batches[2] = { 1u, 64u };
    unsigned long long fed = 0ull;
    for (unsigned int b = 0u; b < 2u; ++b) {
        char lines[64][AOTX_TEST_LINE];
        unsigned int lengths[64];
        aotx_pump_read(&report);
        unsigned long long want = report.state_hash;
        unsigned long long before = report.applied;
        state->capture = 1;
        state->kept = 0u;
        for (unsigned int i = 0u; i < batches[b]; ++i) {
            lengths[i] = (unsigned int)snprintf(lines[i], AOTX_TEST_LINE,
                                                "batch %u line %u value %u",
                                                b, i, i * 37u + 11u);
            aotx_test_feed(&rings, boot_id, fed + i, lines[i], lengths[i]);
            want = aotx_test_fold(want, lines[i], lengths[i]);
        }
        fed += batches[b];
        aotx_pump_tick(&pump);
        aotx_test_settle(state, &rings);
        aotx_pump_read(&report);
        applied += 3u;
        if (report.applied != before + batches[b]) {
            printf("seam: %llu lines were applied and %u were fed\n",
                   report.applied - before, batches[b]);
            failed += 1u;
        }
        if (report.state_hash != want) {
            printf("seam: the state hash is %llx and the host made %llx\n",
                   report.state_hash, want);
            failed += 1u;
        }
        if (report.rejected != 0ull) {
            printf("seam: %llu inbound slots were refused\n", report.rejected);
            failed += 1u;
        }
        /* The echo of every line reached the host ring, with its line intact. */
        unsigned int seen = 0u;
        unsigned int again = 0u;
        for (unsigned int k = 0u; k < state->kept; ++k) {
            const aotx_record_header *record =
                (const aotx_record_header *)(state->keep + (size_t)k * AOTX_SLOT_BYTES);
            const char *body = (const char *)record + AOTX_HEADER_BYTES;
            if (record->type == AOTX_REC_CONSOLE && record->body_len >= 2u
                && body[0] == '>' && body[1] == ' ') {
                for (unsigned int i = 0u; i < batches[b]; ++i) {
                    if (record->body_len == lengths[i] + 2u
                        && memcmp(body + 2, lines[i], lengths[i]) == 0) {
                        seen += 1u;
                    }
                }
            }
            if (record->type == AOTX_REC_INPUT_LINE && record->writer == AOTX_WRITER_FEEDER
                && record->cls == AOTX_CLASS_A) {
                again += 1u;
            }
        }
        applied += 2u;
        if (seen != batches[b]) {
            printf("seam: %u echoes of %u lines reached the host ring\n", seen, batches[b]);
            failed += 1u;
        }
        if (again != batches[b]) {
            printf("seam: %u lines of %u were put in the journal again\n",
                   again, batches[b]);
            failed += 1u;
        }
        state->capture = 0;
    }

    /* Case set 4: the restore report crosses the inbound ring. The device puts its own hash
     * in the record that goes in the journal, and folds nothing. */
    {
        aotx_restore_body sent;
        aotx_pump_read(&report);
        unsigned long long want_hash = report.state_hash;
        unsigned long long before = report.applied;
        unsigned long long consoles = state->by_type[AOTX_REC_CONSOLE];
        sent.restored_boot_id = 0x1122334455667788ull;
        sent.last_tick = 4242ull;
        sent.replayed_count = 65ull;
        sent.state_hash = 0xDEADBEEFDEADBEEFull;
        state->capture = 1;
        state->kept = 0u;
        aotx_test_put(&rings, boot_id, fed, AOTX_WRITER_RESTORE, AOTX_CLASS_B,
                      AOTX_REC_RESTORE, 0u, &sent, (unsigned int)sizeof sent);
        fed += 1ull;
        aotx_pump_tick(&pump);
        aotx_test_settle(state, &rings);
        aotx_pump_read(&report);
        unsigned int found = 0u;
        for (unsigned int k = 0u; k < state->kept; ++k) {
            const aotx_record_header *record =
                (const aotx_record_header *)(state->keep + (size_t)k * AOTX_SLOT_BYTES);
            const aotx_restore_body *got =
                (const aotx_restore_body *)((const unsigned char *)record + AOTX_HEADER_BYTES);
            if (record->type != AOTX_REC_RESTORE) {
                continue;
            }
            found += 1u;
            applied += 4u;
            if (record->cls != AOTX_CLASS_B || record->writer != AOTX_WRITER_RESTORE) {
                printf("seam: the restore record carries class %u and writer %u\n",
                       record->cls, record->writer);
                failed += 1u;
            }
            if (got->restored_boot_id != sent.restored_boot_id
                || got->last_tick != sent.last_tick
                || got->replayed_count != sent.replayed_count) {
                printf("seam: the restore record body did not survive the seam\n");
                failed += 1u;
            }
            if (got->state_hash != want_hash) {
                printf("seam: the restore record carries hash %llx and the device holds %llx\n",
                       (unsigned long long)got->state_hash, want_hash);
                failed += 1u;
            }
            if (got->state_hash == sent.state_hash) {
                printf("seam: the restore record kept the hash the disk side sent\n");
                failed += 1u;
            }
        }
        applied += 4u;
        if (found != 1u) {
            printf("seam: %u restore records reached the host ring, not 1\n", found);
            failed += 1u;
        }
        if (report.applied != before) {
            printf("seam: the restore record was counted as an applied input\n");
            failed += 1u;
        }
        if (report.state_hash != want_hash) {
            printf("seam: the restore record changed the state hash\n");
            failed += 1u;
        }
        if (state->by_type[AOTX_REC_CONSOLE] != consoles) {
            printf("seam: the restore record made an echo\n");
            failed += 1u;
        }
        state->capture = 0;
    }

    /* Case set 5: key events. A key is a class A input like a line. A key folds into the
     * state hash and goes in the journal again. A key has no echo, because the line editor
     * shows the line it builds. The command layer takes every key in slot order, so the
     * line it makes from them is the line that was typed. */
    for (unsigned int b = 0u; b < 2u; ++b) {
        aotx_key_body keys[65];
        char typed[65];
        unsigned int count = batches[b];
        aotx_pump_read(&report);
        unsigned long long want = report.state_hash;
        unsigned long long before = report.applied;
        state->capture = 1;
        state->kept = 0u;
        for (unsigned int i = 0u; i < count; ++i) {
            typed[i] = (char)('a' + (i % 26u));
            keys[i].key = 0u;                       /* a character event carries no key code */
            keys[i].codepoint = (unsigned int)(unsigned char)typed[i];
            keys[i].action = AOTX_CLI_PRESS;
            keys[i].mods = 0u;
        }
        keys[count].key = AOTX_CLI_KEY_ENTER;
        keys[count].codepoint = 0u;
        keys[count].action = AOTX_CLI_PRESS;
        keys[count].mods = 0u;
        for (unsigned int i = 0u; i <= count; ++i) {
            aotx_test_put(&rings, boot_id, fed + i, AOTX_WRITER_FEEDER, AOTX_CLASS_A,
                          AOTX_REC_KEY, 0u, &keys[i], (unsigned int)sizeof keys[i]);
            want = aotx_test_fold(want, (const char *)&keys[i],
                                  (unsigned int)sizeof keys[i]);
        }
        fed += count + 1u;
        aotx_pump_tick(&pump);
        aotx_test_settle(state, &rings);
        aotx_pump_read(&report);

        unsigned int again = 0u;
        unsigned int echoes = 0u;
        unsigned int commands = 0u;
        for (unsigned int k = 0u; k < state->kept; ++k) {
            const aotx_record_header *record =
                (const aotx_record_header *)(state->keep + (size_t)k * AOTX_SLOT_BYTES);
            const char *body = (const char *)record + AOTX_HEADER_BYTES;
            if (record->type == AOTX_REC_KEY && record->cls == AOTX_CLASS_A
                && record->writer == AOTX_WRITER_FEEDER) {
                again += 1u;
            }
            if (record->type == AOTX_REC_CONSOLE && record->body_len >= 2u
                && body[0] == '>' && body[1] == ' ') {
                echoes += 1u;
            }
            if (record->type == AOTX_REC_COMMAND && record->body_len == count
                && memcmp(body, typed, count) == 0) {
                commands += 1u;
            }
        }
        applied += 5u;
        if (report.applied != before + count + 1u) {
            printf("seam: %llu keys were applied and %u were fed\n",
                   report.applied - before, count + 1u);
            failed += 1u;
        }
        if (report.state_hash != want) {
            printf("seam: the state hash after %u keys is %llx and the host made %llx\n",
                   count + 1u, report.state_hash, want);
            failed += 1u;
        }
        if (again != count + 1u) {
            printf("seam: %u keys of %u were put in the journal again\n",
                   again, count + 1u);
            failed += 1u;
        }
        if (echoes != 0u) {
            printf("seam: %u echoes were made for %u keys\n", echoes, count + 1u);
            failed += 1u;
        }
        if (commands != 1u) {
            printf("seam: %u command records hold the %u keys that were typed\n",
                   commands, count);
            failed += 1u;
        }
        state->capture = 0;
    }

    /* Case set 6: a replayed key and a replayed line. Both go in the journal again with
     * the restore as the writer. Neither makes an echo. The command layer takes both,
     * because the device makes the command again from the input. */
    {
        aotx_key_body key;
        const char *line = "note replayed line";
        unsigned int length = 18u;
        aotx_pump_read(&report);
        unsigned long long before = report.applied;
        key.key = 0u;
        key.codepoint = (unsigned int)(unsigned char)'z';
        key.action = AOTX_CLI_PRESS;
        key.mods = 0u;
        state->capture = 1;
        state->kept = 0u;
        aotx_test_put(&rings, boot_id, fed, AOTX_WRITER_RESTORE, AOTX_CLASS_A,
                      AOTX_REC_KEY, AOTX_FLAG_REPLAYED, &key, (unsigned int)sizeof key);
        aotx_test_put(&rings, boot_id, fed + 1ull, AOTX_WRITER_RESTORE, AOTX_CLASS_A,
                      AOTX_REC_INPUT_LINE, AOTX_FLAG_REPLAYED, line, length);
        fed += 2ull;
        aotx_pump_tick(&pump);
        aotx_test_settle(state, &rings);
        aotx_pump_read(&report);

        unsigned int restored = 0u;
        unsigned int echoes = 0u;
        unsigned int commands = 0u;
        for (unsigned int k = 0u; k < state->kept; ++k) {
            const aotx_record_header *record =
                (const aotx_record_header *)(state->keep + (size_t)k * AOTX_SLOT_BYTES);
            const char *body = (const char *)record + AOTX_HEADER_BYTES;
            if ((record->type == AOTX_REC_KEY || record->type == AOTX_REC_INPUT_LINE)
                && record->writer == AOTX_WRITER_RESTORE
                && (record->flags & AOTX_FLAG_REPLAYED) != 0u) {
                restored += 1u;
            }
            if (record->type == AOTX_REC_CONSOLE && record->body_len == length + 2u
                && body[0] == '>' && body[1] == ' '
                && memcmp(body + 2, line, length) == 0) {
                echoes += 1u;
            }
            if (record->type == AOTX_REC_COMMAND && record->body_len == length
                && memcmp(body, line, length) == 0) {
                commands += 1u;
            }
        }
        applied += 4u;
        if (restored != 2u) {
            printf("seam: %u replayed inputs of 2 went in the journal again\n", restored);
            failed += 1u;
        }
        if (echoes != 0u) {
            printf("seam: a replayed input made an echo\n");
            failed += 1u;
        }
        if (commands != 1u) {
            printf("seam: %u command records hold the replayed line\n", commands);
            failed += 1u;
        }
        if (report.applied != before + 2ull) {
            printf("seam: %llu replayed inputs of 2 were applied\n",
                   report.applied - before);
            failed += 1u;
        }
        state->capture = 0;
    }

    state->stop = 1;
    pthread_join(thread, NULL);
    aotx_pump_read(&report);
    printf("seam: blocks %llu pads %llu records %llu retries %llu gaps %llu bad %llu\n",
           state->blocks, state->pads, state->records, state->retries,
           state->gaps, state->bad);
    printf("seam: notes %llu commits %llu stalls %llu console %llu input %llu\n",
           state->by_type[AOTX_REC_NOTE], state->by_type[AOTX_REC_TICK_COMMIT],
           state->by_type[AOTX_REC_STALL], state->by_type[AOTX_REC_CONSOLE],
           state->by_type[AOTX_REC_INPUT_LINE]);
    applied += 2u;
    if (state->gaps != 0ull || state->bad != 0ull) {
        printf("seam: the sequences are not contiguous\n");
        failed += 1u;
    }
    if (report.overrun != 0ull) {
        printf("seam: the flush dropped %llu runs of records\n", report.overrun);
        failed += 1u;
    }

    aotx_seam_finish(&rings);
    aotx_pump_close(&pump);
    aotx_seam_close(&rings);
    aotx_mem_release(&map);
    free(state->keep);
    free(state);
    printf("seam: %u cases applied, %u failed\n", applied, failed);
    return failed == 0u ? 0 : 1;
}
