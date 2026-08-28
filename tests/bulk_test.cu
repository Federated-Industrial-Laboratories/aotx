/* Purpose: Check the bulk channel: staging, the handle, the block, and the refusal.
 * Owns: The test consumer of the bulk ring, the fixtures and the counts of the cases.
 * Launch shape: One block for each payload; the consumer is a host thread.
 * Lifetime: One run of the test program. */
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "boot/check.h"
#include "mem/mem.cuh"
#include "sched/sched.cuh"
#include "seam/seam.cuh"

#define AOTX_TEST_PAYLOADS 64u
#define AOTX_TEST_BLOCKS   4096u
#define AOTX_TEST_THREADS  256u

/* The length of one payload. Every length differs, so a wrong index cannot pass. The set
 * covers one byte, the middle, and one megabyte. */
__device__ __host__ static unsigned long long aotx_bulk_test_length(unsigned int index)
{
    if (index == 0u) {
        return 1ull;
    }
    if (index == AOTX_TEST_PAYLOADS - 1u) {
        return 1048576ull;
    }
    return 1ull + (unsigned long long)index * 1021ull;
}

/* The content of a payload comes from its length, which names it. */
__device__ __host__ static unsigned char aotx_bulk_test_byte(unsigned long long length,
                                                             unsigned long long at)
{
    return (unsigned char)((length * 31ull + at * 7ull + 13ull) & 0xFFull);
}

static unsigned long long aotx_bulk_test_fold(unsigned long long hash,
                                              const unsigned char *bytes,
                                              unsigned long long count)
{
    for (unsigned long long i = 0ull; i < count; ++i) {
        hash ^= (unsigned long long)bytes[i];
        hash *= AOTX_FNV_PRIME;
    }
    return hash;
}

/* One block for each payload. The first thread claims the staging bytes, every thread
 * fills them, and the first thread writes the record that names the payload. */
__global__ void aotx_bulk_test_put(unsigned int payloads, unsigned long long *handles)
{
    __shared__ unsigned char *shared_at;
    __shared__ unsigned long long shared_length;
    if (blockIdx.x >= payloads) {
        return;
    }
    if (threadIdx.x == 0u) {
        shared_length = aotx_bulk_test_length(blockIdx.x);
        shared_at = (unsigned char *)aotx_bulk_stage(AOTX_BULK_KIND_TEXT, shared_length);
    }
    __syncthreads();
    unsigned char *at = shared_at;
    unsigned long long length = shared_length;
    if (at == 0) {
        if (threadIdx.x == 0u) {
            handles[blockIdx.x] = 0ull;
        }
        return;
    }
    for (unsigned long long b = threadIdx.x; b < length; b += AOTX_TEST_THREADS) {
        at[b] = aotx_bulk_test_byte(length, b);
    }
    __syncthreads();
    if (threadIdx.x == 0u) {
        handles[blockIdx.x] = aotx_bulk_commit(at, AOTX_BULK_KIND_TEXT, length,
                                               aotx_time_tick);
    }
}

/* One thread for each payload of the flood. The content does not matter here: the case is
 * the room in the ring, and a payload that finds none must be refused. */
__global__ void aotx_bulk_test_flood(unsigned int payloads, unsigned long long length,
                                     unsigned long long *handles)
{
    unsigned int at = blockIdx.x * blockDim.x + threadIdx.x;
    if (at >= payloads) {
        return;
    }
    unsigned char *staged = (unsigned char *)aotx_bulk_stage(AOTX_BULK_KIND_TEXT, length);
    handles[at] = (staged == 0) ? 0ull
                : aotx_bulk_commit(staged, AOTX_BULK_KIND_TEXT, length, aotx_time_tick);
}

/* The pointer that one tick staged, kept for the tick that follows. */
__device__ unsigned char *aotx_bulk_test_kept = 0;

__global__ void aotx_bulk_test_hold(unsigned long long length)
{
    aotx_bulk_test_kept = (unsigned char *)aotx_bulk_stage(AOTX_BULK_KIND_TEXT, length);
}

/* A commit of a pointer that an earlier tick staged. The entry it names belongs to that
 * tick, so the record would carry a handle whose payload no block holds. */
__global__ void aotx_bulk_test_late(unsigned long long length, unsigned long long *handle)
{
    *handle = aotx_bulk_commit(aotx_bulk_test_kept, AOTX_BULK_KIND_TEXT, length,
                               aotx_time_tick);
}

/* What the consumer of the bulk ring holds. The consumer accepts a block by the double
 * load rule: it loads the sequence, reads the block, and loads the sequence again. */
typedef struct aotx_bulk_test_seen {
    unsigned long long block_seq;
    unsigned long long handle;
    unsigned long long byte_len;
    unsigned long long hash;
} aotx_bulk_test_seen;

typedef struct aotx_bulk_test_consumer {
    const unsigned char *map;
    const unsigned char *data;
    unsigned long long data_bytes;
    unsigned long long mask;
    unsigned long long boot_id;
    volatile int stop;
    volatile int paused;
    unsigned long long cursor;
    unsigned long long expect_block;
    unsigned long long blocks;
    unsigned long long pads;
    unsigned long long bad;
    unsigned long long gaps;
    unsigned long long tails;
    unsigned long long retries;
    unsigned int count;
    aotx_bulk_test_seen seen[AOTX_TEST_BLOCKS];
} aotx_bulk_test_consumer;

static unsigned long long aotx_bulk_test_load(const void *at)
{
    return __atomic_load_n((const unsigned long long *)at, __ATOMIC_ACQUIRE);
}

static int aotx_bulk_test_block(aotx_bulk_test_consumer *state)
{
    const unsigned char *at = state->data + (state->cursor & state->mask);
    const aotx_block_header *block = (const aotx_block_header *)at;
    unsigned long long first = aotx_bulk_test_load(&block->block_seq);
    if (first == 0ull) {
        state->retries += 1ull;
        return 0;
    }
    aotx_block_header header = *block;
    unsigned long long offset = state->cursor & state->mask;
    if (header.magic != AOTX_BLOCK_MAGIC || header.layout != AOTX_WIRE_LAYOUT
        || header.boot_id != state->boot_id || header.record_count != 0u
        || header.byte_len < AOTX_BLOCK_HEADER_BYTES
        || (unsigned long long)header.byte_len > state->data_bytes - offset
        || (offset & 7ull) != 0ull) {
        state->bad += 1ull;
        return -1;
    }
    if (header.kind == AOTX_BLOCK_PAD) {
        if (offset + header.byte_len != state->data_bytes) {
            state->bad += 1ull;
        }
    } else if (header.kind != AOTX_BLOCK_BULK) {
        state->bad += 1ull;
    }
    /* The space left after a block reaches the end of the data area or holds a block
     * header. A tail of fewer bytes than a header has no place for a pad block. */
    unsigned long long left = state->data_bytes - offset - (unsigned long long)header.byte_len;
    if (left != 0ull && left < AOTX_BLOCK_HEADER_BYTES) {
        state->bad += 1ull;
        state->tails += 1ull;
    }
    unsigned long long hash = AOTX_FNV_BASIS;
    if (header.kind == AOTX_BLOCK_BULK) {
        hash = aotx_bulk_test_fold(hash, at + AOTX_BLOCK_HEADER_BYTES,
                                   (unsigned long long)header.byte_len
                                   - AOTX_BLOCK_HEADER_BYTES);
    }
    if (aotx_bulk_test_load(&block->block_seq) != first) {
        state->retries += 1ull;
        return 0;
    }
    if (first != state->expect_block) {
        state->gaps += 1ull;
    }
    state->expect_block = first + 1ull;
    state->blocks += 1ull;
    if (header.kind == AOTX_BLOCK_PAD) {
        state->pads += 1ull;
    } else if (state->count < AOTX_TEST_BLOCKS) {
        state->seen[state->count].block_seq = first;
        state->seen[state->count].handle = header.first_seq;
        state->seen[state->count].byte_len = header.byte_len;
        state->seen[state->count].hash = hash;
        state->count += 1u;
    }
    state->cursor += header.byte_len;
    return 1;
}

static void *aotx_bulk_test_drain(void *argument)
{
    aotx_bulk_test_consumer *state = (aotx_bulk_test_consumer *)argument;
    const aotx_host_ring_preamble *preamble = (const aotx_host_ring_preamble *)state->map;
    while (state->stop == 0) {
        if (state->paused != 0) {
            usleep(200);
            continue;
        }
        unsigned long long head = aotx_bulk_test_load(&preamble->head);
        int moved = 0;
        while (state->cursor < head) {
            int step = aotx_bulk_test_block(state);
            if (step <= 0) {
                break;
            }
            moved = 1;
        }
        if (moved != 0) {
            __atomic_store_n((unsigned long long *)&preamble->cursor, state->cursor,
                             __ATOMIC_RELEASE);
        } else {
            usleep(100);
        }
    }
    return NULL;
}

/* One tick, node by node, with the payload kernel where the tick load goes. */
static void aotx_bulk_test_tick(unsigned int payloads, unsigned long long *handles)
{
    aotx_sched_tick_start<<<1, 1>>>(0ull);
    if (payloads != 0u) {
        aotx_bulk_test_put<<<payloads, AOTX_TEST_THREADS>>>(payloads, handles);
    }
    aotx_sched_commit<<<1, 1>>>();
    aotx_seam_flush<<<1, AOTX_FLUSH_THREADS>>>();
    aotx_seam_bulk_flush<<<1, AOTX_FLUSH_THREADS>>>();
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
}

/* One tick with one payload of a stated length. */
static void aotx_bulk_test_one(const aotx_seam_rings *rings, unsigned long long *handles,
                               unsigned long long length);

/* The journal ring has no drain in this test, so the cursor follows the head and the tick
 * is never held. The bulk ring is the ring this test examines. */
static void aotx_bulk_test_follow(const aotx_seam_rings *rings)
{
    volatile aotx_host_ring_preamble *preamble =
        (volatile aotx_host_ring_preamble *)rings->host_map;
    unsigned long long head = __atomic_load_n((const unsigned long long *)&preamble->head,
                                              __ATOMIC_ACQUIRE);
    __atomic_store_n((unsigned long long *)&preamble->cursor, head, __ATOMIC_RELEASE);
}

static void aotx_bulk_test_one(const aotx_seam_rings *rings, unsigned long long *handles,
                               unsigned long long length)
{
    aotx_sched_tick_start<<<1, 1>>>(0ull);
    aotx_bulk_test_flood<<<1, 1u>>>(1u, length, handles);
    aotx_sched_commit<<<1, 1>>>();
    aotx_seam_flush<<<1, AOTX_FLUSH_THREADS>>>();
    aotx_seam_bulk_flush<<<1, AOTX_FLUSH_THREADS>>>();
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_bulk_test_follow(rings);
    const aotx_host_ring_preamble *preamble =
        (const aotx_host_ring_preamble *)rings->bulk_map;
    for (unsigned int w = 0u; w < 20000u; ++w) {
        if (__atomic_load_n(&preamble->cursor, __ATOMIC_ACQUIRE)
            >= __atomic_load_n(&preamble->head, __ATOMIC_ACQUIRE)) {
            break;
        }
        usleep(100);
    }
}

static void aotx_bulk_test_state(aotx_bulk_state *state)
{
    aotx_check_runtime(cudaMemcpyFromSymbol(state, aotx_bulk, sizeof *state),
                       "cudaMemcpyFromSymbol");
}

static unsigned long long aotx_bulk_test_refused(void)
{
    aotx_bulk_state state;
    aotx_bulk_test_state(&state);
    return state.refused;
}

/* Find the record that names one handle, and give back its length and its kind. */
static int aotx_bulk_test_record(const unsigned char *ring, unsigned long long slots,
                                 unsigned long long from, unsigned long long to,
                                 unsigned long long handle, aotx_bulk_body *body)
{
    for (unsigned long long seq = to; seq > from; --seq) {
        const aotx_record_header *record = (const aotx_record_header *)
            (ring + ((seq - 1ull) & (slots - 1ull)) * AOTX_SLOT_BYTES);
        if (record->seq != seq || record->type != AOTX_REC_BULK) {
            continue;
        }
        const aotx_bulk_body *found =
            (const aotx_bulk_body *)((const unsigned char *)record + AOTX_HEADER_BYTES);
        if (found->handle == handle) {
            *body = *found;
            return 0;
        }
    }
    return 1;
}

/* One pass at a payload count: one tick, then the blocks and the records must agree. */
static unsigned int aotx_bulk_test_pass(aotx_bulk_test_consumer *state,
                                        const aotx_seam_rings *rings,
                                        const aotx_mem_map *map, unsigned char *ring,
                                        unsigned long long *handles, unsigned int payloads,
                                        unsigned int *failed)
{
    unsigned long long slots = map->ring_bytes / AOTX_SLOT_BYTES;
    unsigned long long host_handles[AOTX_TEST_PAYLOADS];
    aotx_seam_state seam;
    unsigned int applied = 0u;
    unsigned int first_seen = state->count;

    aotx_check_runtime(cudaMemcpyFromSymbol(&seam, aotx_seam, sizeof seam),
                       "cudaMemcpyFromSymbol");
    unsigned long long from = seam.dev.tail;
    aotx_bulk_test_tick(payloads, handles);
    aotx_bulk_test_follow(rings);
    aotx_check_runtime(cudaMemcpy(host_handles, handles,
                                  (size_t)payloads * sizeof(unsigned long long),
                                  cudaMemcpyDeviceToHost), "cudaMemcpy");
    aotx_check_runtime(cudaMemcpyFromSymbol(&seam, aotx_seam, sizeof seam),
                       "cudaMemcpyFromSymbol");
    unsigned long long to = seam.dev.tail;
    aotx_check_runtime(cudaMemcpy(ring, (const void *)map->ring, (size_t)map->ring_bytes,
                                  cudaMemcpyDeviceToHost), "cudaMemcpy");

    for (unsigned int i = 0u; i < 20000u && state->count < first_seen + payloads; ++i) {
        usleep(200);
    }

    unsigned int matched = 0u;
    unsigned int wrong = 0u;
    for (unsigned int p = 0u; p < payloads; ++p) {
        unsigned long long length = aotx_bulk_test_length(p);
        unsigned long long padded = (length + 7ull) & ~7ull;
        unsigned long long handle = host_handles[p];
        if (handle == 0ull) {
            wrong += 1u;
            continue;
        }
        aotx_bulk_body body;
        if (aotx_bulk_test_record(ring, slots, from, to, handle, &body) != 0) {
            wrong += 1u;
            continue;
        }
        if (body.length != length || body.kind != AOTX_BULK_KIND_TEXT) {
            wrong += 1u;
            continue;
        }
        const aotx_bulk_test_seen *seen = 0;
        for (unsigned int s = first_seen; s < state->count; ++s) {
            if (state->seen[s].handle == handle) {
                seen = &state->seen[s];
                break;
            }
        }
        if (seen == 0 || seen->byte_len != AOTX_BLOCK_HEADER_BYTES + padded) {
            wrong += 1u;
            continue;
        }
        /* The content of the block must be the content the payload was given, with the
         * bytes of the rounding at zero. */
        unsigned char *want = (unsigned char *)calloc((size_t)padded, 1u);
        for (unsigned long long b = 0ull; b < length; ++b) {
            want[b] = aotx_bulk_test_byte(length, b);
        }
        unsigned long long hash = aotx_bulk_test_fold(AOTX_FNV_BASIS, want, padded);
        free(want);
        if (hash != seen->hash) {
            wrong += 1u;
            continue;
        }
        matched += 1u;
    }

    applied += 3u;
    if (matched != payloads) {
        printf("bulk: %u payloads of %u matched a record and a block\n", matched, payloads);
        *failed += 1u;
    }
    if (wrong != 0u) {
        printf("bulk: %u payloads of %u did not agree with their block\n", wrong, payloads);
        *failed += 1u;
    }
    if (state->gaps != 0ull || state->bad != 0ull) {
        printf("bulk: the block sequence has %llu gaps and %llu bad blocks\n",
               state->gaps, state->bad);
        *failed += 1u;
    }
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

    unsigned long long boot_id = 0xB01C0DEull;
    if (aotx_mem_reserve(&map) != 0 || aotx_seam_open(&rings, boot_id) != 0) {
        printf("bulk: the map or the rings did not open\n");
        return 1;
    }
    aotx_seam_bind(&rings, map.ring, map.ring_bytes, boot_id);
    if (aotx_seam_bind_bulk(&rings, map.scratch, AOTX_BULK_STAGE_BYTES) != 0) {
        printf("bulk: the bulk ring did not bind\n");
        return 1;
    }
    aotx_seam_note_boot<<<1, 1>>>(0ull, 0ull);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    unsigned long long *handles = 0;
    aotx_check_runtime(cudaMalloc(&handles, AOTX_TEST_PAYLOADS * sizeof(unsigned long long)),
                       "cudaMalloc");
    unsigned char *ring = (unsigned char *)malloc((size_t)map.ring_bytes);
    aotx_bulk_test_consumer *state =
        (aotx_bulk_test_consumer *)calloc(1, sizeof *state);
    state->map = rings.bulk_map;
    state->data = rings.bulk_map + sizeof(aotx_host_ring_preamble);
    state->data_bytes = AOTX_BULK_RING_DATA_BYTES;
    state->mask = AOTX_BULK_RING_DATA_BYTES - 1ull;
    state->boot_id = boot_id;
    state->expect_block = 1ull;
    pthread_t thread;
    pthread_create(&thread, NULL, aotx_bulk_test_drain, state);

    /* Case set 1: one payload, then 64 payloads of distinct length in one tick. */
    const unsigned int counts[2] = { 1u, AOTX_TEST_PAYLOADS };
    for (unsigned int c = 0u; c < 2u; ++c) {
        applied += aotx_bulk_test_pass(state, &rings, &map, ring, handles, counts[c],
                                       &failed);
    }

    /* Case set 2: a full ring refuses staging. The consumer stops, so the cursor stands
     * still and the room the tick start reads goes to zero. Nothing waits. */
    {
        const unsigned int flood = 7u;
        const unsigned long long big = 1048576ull;
        unsigned long long before = aotx_bulk_test_refused();
        unsigned long long host_handles[AOTX_TEST_PAYLOADS];
        unsigned int refusals = 0u;
        unsigned int ticks = 0u;
        state->paused = 1;
        for (unsigned int i = 0u; i < 128u && refusals == 0u; ++i) {
            aotx_sched_tick_start<<<1, 1>>>(0ull);
            aotx_bulk_test_flood<<<1, flood>>>(flood, big, handles);
            aotx_sched_commit<<<1, 1>>>();
            aotx_seam_flush<<<1, AOTX_FLUSH_THREADS>>>();
            aotx_seam_bulk_flush<<<1, AOTX_FLUSH_THREADS>>>();
            aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
            aotx_bulk_test_follow(&rings);
            aotx_check_runtime(cudaMemcpy(host_handles, handles,
                                          flood * sizeof(unsigned long long),
                                          cudaMemcpyDeviceToHost), "cudaMemcpy");
            ticks += 1u;
            for (unsigned int p = 0u; p < flood; ++p) {
                if (host_handles[p] == 0ull) {
                    refusals += 1u;
                }
            }
        }
        unsigned long long after = aotx_bulk_test_refused();
        applied += 3u;
        if (refusals == 0u) {
            printf("bulk: a full ring took every payload\n");
            failed += 1u;
        }
        if (after != before + refusals) {
            printf("bulk: the counter moved by %llu and %u payloads were refused\n",
                   after - before, refusals);
            failed += 1u;
        }
        if (ticks < 2u) {
            printf("bulk: the first tick already found no room, so the ring was not full\n");
            failed += 1u;
        }
        printf("bulk: %u payloads of one megabyte refused after %u ticks, counter %llu\n",
               refusals, ticks, after - before);
        state->paused = 0;
    }

    /* The consumer takes the blocks that the pause held back. */
    {
        const aotx_host_ring_preamble *preamble =
            (const aotx_host_ring_preamble *)rings.bulk_map;
        for (unsigned int i = 0u; i < 20000u; ++i) {
            if (state->cursor >= aotx_bulk_test_load(&preamble->head)) {
                break;
            }
            usleep(200);
        }
    }

    /* Case set 4: a pointer of an earlier tick. The staging region and the entry table
     * last one tick, so a commit that comes late names a payload that no block holds. The
     * commit refuses it and counts it. */
    {
        aotx_bulk_state bulk;
        unsigned long long *device_handle = 0;
        unsigned long long late = 0ull;
        aotx_check_runtime(cudaMalloc(&device_handle, sizeof late), "cudaMalloc");
        aotx_bulk_test_state(&bulk);
        unsigned long long stale = bulk.stale;
        unsigned long long blocks = state->blocks;

        /* The first tick stages and writes no record, so no block carries the payload. */
        aotx_sched_tick_start<<<1, 1>>>(0ull);
        aotx_bulk_test_hold<<<1, 1>>>(4096ull);
        aotx_sched_commit<<<1, 1>>>();
        aotx_seam_flush<<<1, AOTX_FLUSH_THREADS>>>();
        aotx_seam_bulk_flush<<<1, AOTX_FLUSH_THREADS>>>();
        aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        aotx_bulk_test_follow(&rings);

        /* The tick that follows tries to name that payload. */
        aotx_sched_tick_start<<<1, 1>>>(0ull);
        aotx_bulk_test_late<<<1, 1>>>(4096ull, device_handle);
        aotx_sched_commit<<<1, 1>>>();
        aotx_seam_flush<<<1, AOTX_FLUSH_THREADS>>>();
        aotx_seam_bulk_flush<<<1, AOTX_FLUSH_THREADS>>>();
        aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        aotx_bulk_test_follow(&rings);
        aotx_check_runtime(cudaMemcpy(&late, device_handle, sizeof late,
                                      cudaMemcpyDeviceToHost), "cudaMemcpy");
        aotx_bulk_test_state(&bulk);
        for (unsigned int w = 0u; w < 2000u; ++w) {
            usleep(100);
            if (state->blocks != blocks) {
                break;
            }
        }
        applied += 3u;
        if (late != 0ull) {
            printf("bulk: a commit of an earlier tick gave back the handle %llu\n", late);
            failed += 1u;
        }
        if (bulk.stale != stale + 1ull) {
            printf("bulk: the counter of late commits moved by %llu and not by 1\n",
                   bulk.stale - stale);
            failed += 1u;
        }
        if (state->blocks != blocks) {
            printf("bulk: %llu blocks were written for a payload that no record names\n",
                   state->blocks - blocks);
            failed += 1u;
        }
        cudaFree(device_handle);
    }

    /* Case set 5: the tail rule. A payload block rounds to 8 bytes, so a block can leave a
     * tail of fewer bytes than a block header. The flush writes a pad block first, because
     * a tail of that size has no place for one. */
    {
        const aotx_host_ring_preamble *preamble =
            (const aotx_host_ring_preamble *)rings.bulk_map;
        unsigned long long pads = 0ull;
        unsigned long long left = 0ull;
        for (unsigned int i = 0u; i < 512u; ++i) {
            left = AOTX_BULK_RING_DATA_BYTES
                 - (aotx_bulk_test_load(&preamble->head) & (AOTX_BULK_RING_DATA_BYTES - 1ull));
            if (left <= AOTX_BULK_STAGE_BYTES - 4096ull) {
                break;
            }
            aotx_bulk_test_one(&rings, handles, 1048576ull);
        }
        pads = state->pads;
        /* The block would leave 8 bytes, which is less than a block header. */
        aotx_bulk_test_one(&rings, handles, left - 8ull - AOTX_BLOCK_HEADER_BYTES);
        for (unsigned int w = 0u; w < 20000u; ++w) {
            if (state->cursor >= aotx_bulk_test_load(&preamble->head)) {
                break;
            }
            usleep(100);
        }
        applied += 2u;
        if (state->pads != pads + 1ull) {
            printf("bulk: the tail of %llu bytes took no pad block\n", left);
            failed += 1u;
        }
        if (state->tails != 0ull) {
            printf("bulk: %llu blocks left a tail with no place for a pad block\n",
                   state->tails);
            failed += 1u;
        }
        printf("bulk: a block that would leave 8 bytes of %llu took a pad block first\n",
               left);
    }

    /* Case set 6: the data area comes to its end and a pad block takes the tail. The block
     * that follows starts at offset zero and never wraps. */
    {
        unsigned long long pads = state->pads;
        for (unsigned int i = 0u; i < 64u && state->pads == pads; ++i) {
            aotx_sched_tick_start<<<1, 1>>>(0ull);
            aotx_bulk_test_flood<<<1, 7u>>>(7u, 1048576ull, handles);
            aotx_sched_commit<<<1, 1>>>();
            aotx_seam_flush<<<1, AOTX_FLUSH_THREADS>>>();
            aotx_seam_bulk_flush<<<1, AOTX_FLUSH_THREADS>>>();
            aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
            aotx_bulk_test_follow(&rings);
            const aotx_host_ring_preamble *preamble =
                (const aotx_host_ring_preamble *)rings.bulk_map;
            for (unsigned int w = 0u; w < 20000u; ++w) {
                if (state->cursor >= aotx_bulk_test_load(&preamble->head)) {
                    break;
                }
                usleep(100);
            }
        }
        applied += 1u;
        if (state->pads == pads) {
            printf("bulk: the data area did not come to its end and no pad block was made\n");
            failed += 1u;
        }
    }

    state->stop = 1;
    pthread_join(thread, NULL);
    printf("bulk: blocks %llu pads %llu retries %llu gaps %llu bad %llu\n",
           state->blocks, state->pads, state->retries, state->gaps, state->bad);
    applied += 1u;
    if (state->bad != 0ull) {
        printf("bulk: %llu blocks did not hold to the block rules\n", state->bad);
        failed += 1u;
    }

    free(state);
    free(ring);
    cudaFree(handles);
    aotx_seam_finish(&rings);
    aotx_seam_close(&rings);
    aotx_mem_release(&map);
    printf("bulk: %u cases applied, %u failed\n", applied, failed);
    return failed == 0u ? 0 : 1;
}
