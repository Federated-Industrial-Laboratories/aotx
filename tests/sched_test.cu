/* Purpose: Check the tick graph: its shape never changes, and a tick stays in its budget.
 * Owns: The test consumer of the host ring, the test graph and the counts of the cases.
 * Launch shape: One block for each bus writer; the consumer is a host thread.
 * Lifetime: One run of the test program. */
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "boot/check.h"
#include "bus/bus.cuh"
#include "mem/mem.cuh"
#include "sched/sched.cuh"
#include "seam/seam.cuh"

/* The nodes of one tick: tick start, apply, tick load, commit, flush, bulk flush. The say
 * path, the decode and the reply of the console add their own, and the pump counts them. */
#define AOTX_TICK_NODES    AOTX_TICK_NODES_TICK
#define AOTX_TEST_NODES    32u
#define AOTX_TEST_EDGES    64u

/* The tick check: 16 writers, 64 messages each, and the record load beside them. */
#define AOTX_TEST_WRITERS  16u
#define AOTX_TEST_EACH     64u
#define AOTX_TEST_LOAD     12000ull
#define AOTX_TEST_TICKS    500u
#define AOTX_TEST_SAMPLES  4096u

/* The budget of one tick, in nanoseconds. */
#define AOTX_TEST_P99_NS   10000000ull
#define AOTX_TEST_MEAN_NS  5000000ull

/* What the graph check reads before and after the run of 1,000 ticks. */
typedef struct aotx_sched_test_shape {
    unsigned int nodes;
    unsigned int edges;
    cudaGraphNode_t node[AOTX_TEST_NODES];
    cudaGraphNodeType kind[AOTX_TEST_NODES];
    void *func[AOTX_TEST_NODES];
    unsigned int grid[AOTX_TEST_NODES];
    cudaGraphNode_t from[AOTX_TEST_EDGES];
    cudaGraphNode_t to[AOTX_TEST_EDGES];
} aotx_sched_test_shape;

/* One block for each writer, one thread for each message. The writers append at once, so
 * the load is the load a bus of 16 agents makes in one tick. */
__global__ void aotx_sched_test_bus(unsigned int writers, unsigned int each)
{
    if (blockIdx.x >= writers || threadIdx.x >= each) {
        return;
    }
    char text[32];
    unsigned int writer = AOTX_WRITER_AGENT_BASE + blockIdx.x;
    for (unsigned int i = 0u; i < 32u; ++i) {
        text[i] = (char)('a' + ((blockIdx.x * 7u + threadIdx.x + i) % 26u));
    }
    aotx_bus_append(writer, AOTX_BUS_NOTE, 0u, text, 32u, 0ull, 0ull, 0.0f, aotx_time_tick);
}

typedef struct aotx_sched_test_consumer {
    const unsigned char *map;
    const unsigned char *data;
    unsigned long long data_bytes;
    unsigned long long mask;
    unsigned long long boot_id;
    volatile int stop;
    unsigned long long cursor;
    unsigned long long blocks;
    unsigned long long records;
    unsigned long long bad;
    unsigned int ticks;
    unsigned long long tick_ns[AOTX_TEST_SAMPLES];
    unsigned long long tick_records[AOTX_TEST_SAMPLES];
} aotx_sched_test_consumer;

static unsigned long long aotx_sched_test_load(const void *at)
{
    return __atomic_load_n((const unsigned long long *)at, __ATOMIC_ACQUIRE);
}

static long long aotx_sched_test_now_ns(void)
{
    struct timespec at;
    clock_gettime(CLOCK_MONOTONIC, &at);
    return (long long)at.tv_sec * 1000000000ll + (long long)at.tv_nsec;
}

/* Take one block and keep the statistics record of every complete tick. */
static int aotx_sched_test_block(aotx_sched_test_consumer *state)
{
    const unsigned char *at = state->data + (state->cursor & state->mask);
    const aotx_block_header *block = (const aotx_block_header *)at;
    unsigned long long first = aotx_sched_test_load(&block->block_seq);
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
        if (record->type != AOTX_REC_STATS) {
            continue;
        }
        const aotx_stats_body *body =
            (const aotx_stats_body *)((const unsigned char *)record + AOTX_HEADER_BYTES);
        if (state->ticks < AOTX_TEST_SAMPLES) {
            state->tick_ns[state->ticks] = body->tick_ns;
            state->tick_records[state->ticks] = body->records;
            state->ticks += 1u;
        }
    }
    if (aotx_sched_test_load(&block->block_seq) != first) {
        return 0;
    }
    state->blocks += 1ull;
    state->records += header.record_count;
    state->cursor += header.byte_len;
    return 1;
}

static void *aotx_sched_test_drain(void *argument)
{
    aotx_sched_test_consumer *state = (aotx_sched_test_consumer *)argument;
    const aotx_host_ring_preamble *preamble = (const aotx_host_ring_preamble *)state->map;
    while (state->stop == 0) {
        unsigned long long head = aotx_sched_test_load(&preamble->head);
        int moved = 0;
        while (state->cursor < head) {
            int step = aotx_sched_test_block(state);
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

/* Read the nodes, the node kinds, the kernels and the edges of a graph. */
static int aotx_sched_test_shape_of(cudaGraph_t graph, aotx_sched_test_shape *shape)
{
    size_t nodes = 0;
    size_t edges = 0;
    memset(shape, 0, sizeof *shape);
    aotx_check_runtime(cudaGraphGetNodes(graph, 0, &nodes), "cudaGraphGetNodes");
    aotx_check_runtime(cudaGraphGetEdges(graph, 0, 0, 0, &edges),
                       "cudaGraphGetEdges");
    if (nodes > AOTX_TEST_NODES || edges > AOTX_TEST_EDGES) {
        return 1;
    }
    aotx_check_runtime(cudaGraphGetNodes(graph, shape->node, &nodes), "cudaGraphGetNodes");
    aotx_check_runtime(cudaGraphGetEdges(graph, shape->from, shape->to, 0, &edges),
                       "cudaGraphGetEdges");
    shape->nodes = (unsigned int)nodes;
    shape->edges = (unsigned int)edges;
    for (unsigned int i = 0u; i < shape->nodes; ++i) {
        aotx_check_runtime(cudaGraphNodeGetType(shape->node[i], &shape->kind[i]),
                           "cudaGraphNodeGetType");
        if (shape->kind[i] == cudaGraphNodeTypeKernel) {
            cudaKernelNodeParams params;
            memset(&params, 0, sizeof params);
            aotx_check_runtime(cudaGraphKernelNodeGetParams(shape->node[i], &params),
                               "cudaGraphKernelNodeGetParams");
            shape->func[i] = params.func;
            shape->grid[i] = params.gridDim.x;
        }
    }
    return 0;
}

/* Two shapes agree when the node count, the edge count, every node, every kernel and every
 * edge agree. The order the driver gives the nodes in is stable for one graph. */
static unsigned int aotx_sched_test_same(const aotx_sched_test_shape *a,
                                         const aotx_sched_test_shape *b)
{
    unsigned int wrong = 0u;
    if (a->nodes != b->nodes || a->edges != b->edges) {
        return 1u;
    }
    for (unsigned int i = 0u; i < a->nodes; ++i) {
        if (a->node[i] != b->node[i] || a->kind[i] != b->kind[i]
            || a->func[i] != b->func[i] || a->grid[i] != b->grid[i]) {
            wrong += 1u;
        }
    }
    for (unsigned int i = 0u; i < a->edges; ++i) {
        if (a->from[i] != b->from[i] || a->to[i] != b->to[i]) {
            wrong += 1u;
        }
    }
    return wrong;
}

/* The tick shape of the check is the shape of the tick graph. The bus writers go where the
 * tick load goes, so the statistics record covers the load they make. */
static int aotx_sched_test_graph(cudaStream_t stream, cudaGraphExec_t *exec,
                                 unsigned long long reserve, unsigned long long load,
                                 unsigned int blocks)
{
    cudaGraph_t graph = 0;
    aotx_check_runtime(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
                       "cudaStreamBeginCapture");
    aotx_sched_tick_start<<<1, 1, 0, stream>>>(reserve);
    aotx_seam_apply_inbound<<<AOTX_APPLY_BLOCKS, AOTX_APPLY_THREADS, 0, stream>>>();
    aotx_sched_test_bus<<<AOTX_TEST_WRITERS, AOTX_TEST_EACH, 0, stream>>>(AOTX_TEST_WRITERS,
                                                                         AOTX_TEST_EACH);
    aotx_sched_workload<<<blocks, AOTX_WORKLOAD_THREADS, 0, stream>>>(load);
    aotx_sched_commit<<<1, 1, 0, stream>>>();
    aotx_seam_flush<<<1, AOTX_FLUSH_THREADS, 0, stream>>>();
    aotx_seam_bulk_flush<<<1, AOTX_FLUSH_THREADS, 0, stream>>>();
    aotx_check_runtime(cudaStreamEndCapture(stream, &graph), "cudaStreamEndCapture");
    aotx_check_runtime(cudaGraphInstantiate(exec, graph, 0), "cudaGraphInstantiate");
    cudaGraphDestroy(graph);
    return 0;
}

static int aotx_sched_test_compare(const void *a, const void *b)
{
    unsigned long long left = *(const unsigned long long *)a;
    unsigned long long right = *(const unsigned long long *)b;
    return (left < right) ? -1 : ((left > right) ? 1 : 0);
}

int main(void)
{
    CUdevice device;
    CUcontext context;
    aotx_mem_map map;
    aotx_seam_rings rings;
    aotx_pump pump;
    aotx_pump_report report;
    unsigned int applied = 0u;
    unsigned int failed = 0u;
    aotx_check_driver(cuInit(0), "cuInit");
    aotx_check_driver(cuDeviceGet(&device, 0), "cuDeviceGet");
    aotx_check_driver(cuDevicePrimaryCtxRetain(&context, device), "cuDevicePrimaryCtxRetain");
    aotx_check_driver(cuCtxSetCurrent(context), "cuCtxSetCurrent");

    unsigned long long boot_id = 0x5CEDull;
    if (aotx_mem_reserve(&map) != 0 || aotx_seam_open(&rings, boot_id) != 0) {
        printf("sched: the map or the rings did not open\n");
        return 1;
    }
    aotx_seam_bind(&rings, map.ring, map.ring_bytes, boot_id);
    aotx_seam_bind_bulk(&rings, map.scratch, AOTX_BULK_STAGE_BYTES);
    aotx_seam_note_boot<<<1, 1>>>(0ull, 0ull);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    aotx_sched_test_consumer *state =
        (aotx_sched_test_consumer *)calloc(1, sizeof *state);
    state->map = rings.host_map;
    state->data = rings.host_map + sizeof(aotx_host_ring_preamble);
    state->data_bytes = AOTX_HOST_RING_DATA_BYTES;
    state->mask = AOTX_HOST_RING_DATA_BYTES - 1ull;
    state->boot_id = boot_id;
    pthread_t thread;
    pthread_create(&thread, NULL, aotx_sched_test_drain, state);

    /* The graph check: the shape of the tick graph does not change over 1,000 ticks with a
     * new tick load parameter for every tick. */
    aotx_sched_test_shape *before = (aotx_sched_test_shape *)calloc(1, sizeof *before);
    aotx_sched_test_shape *after = (aotx_sched_test_shape *)calloc(1, sizeof *after);
    if (aotx_pump_build(&pump, 64ull, 1u) != 0) {
        printf("sched: the tick graph did not build\n");
        return 1;
    }
    if (aotx_sched_test_shape_of(pump.graph, before) != 0) {
        printf("sched: the graph holds more nodes than the check keeps\n");
        return 1;
    }
    applied += 2u;
    /* The documented list: the tick itself, the say path, the decode and the reply. The
     * decode holds no node when the run has no language model. */
    unsigned int decode_nodes = pump.decode ? AOTX_TICK_NODES_DECODE : 0u;
    unsigned int parts = AOTX_TICK_NODES + AOTX_TICK_NODES_SAY + decode_nodes
                       + AOTX_TICK_NODES_REPLY;
    applied += 1u;
    if (pump.decode_nodes != decode_nodes || pump.say_nodes != AOTX_TICK_NODES_SAY
        || pump.reply_nodes != AOTX_TICK_NODES_REPLY) {
        printf("sched: the parts put %u say, %u decode and %u reply nodes in the tick and "
               "the list has %u, %u and %u\n", pump.say_nodes, pump.decode_nodes,
               pump.reply_nodes, AOTX_TICK_NODES_SAY, decode_nodes,
               AOTX_TICK_NODES_REPLY);
        failed += 1u;
    }
    if (before->nodes != parts) {
        printf("sched: the tick graph holds %u nodes and its parts have %u\n",
               before->nodes, parts);
        failed += 1u;
    }
    if (before->edges != parts - 1u) {
        printf("sched: the tick graph holds %u edges and a chain of %u nodes has %u\n",
               before->edges, parts, parts - 1u);
        failed += 1u;
    }
    for (unsigned int t = 0u; t < 1000u; ++t) {
        aotx_pump_set(&pump, 16ull + (unsigned long long)(t % 97u) * 13ull,
                      1u + (t % 7u));
        aotx_pump_tick(&pump);
    }
    if (aotx_sched_test_shape_of(pump.graph, after) != 0) {
        printf("sched: the graph holds more nodes than the check keeps\n");
        return 1;
    }
    unsigned int wrong = aotx_sched_test_same(before, after);
    applied += 2u;
    if (wrong != 0u) {
        printf("sched: %u nodes or edges of the tick graph changed over 1,000 ticks\n",
               wrong);
        failed += 1u;
    }
    aotx_pump_read(&report);
    if (report.tick < 1000ull) {
        printf("sched: %llu ticks ran of 1,000\n", report.tick);
        failed += 1u;
    }
    printf("sched: the tick graph holds %u nodes and %u edges over %llu ticks\n",
           after->nodes, after->edges, report.tick);
    aotx_pump_close(&pump);

    /* The tick check: 16 bus writers of 64 messages each, beside the record load. The
     * statistics record of every complete tick states what the tick took. */
    cudaStream_t stream = 0;
    cudaGraphExec_t exec = 0;
    cudaEvent_t event = 0;
    cudaEvent_t opened = 0;
    cudaEvent_t closed = 0;
    aotx_check_runtime(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                       "cudaStreamCreateWithFlags");
    aotx_check_runtime(cudaEventCreateWithFlags(&event, cudaEventDisableTiming),
                       "cudaEventCreateWithFlags");
    aotx_check_runtime(cudaEventCreate(&opened), "cudaEventCreate");
    aotx_check_runtime(cudaEventCreate(&closed), "cudaEventCreate");
    unsigned long long *whole =
        (unsigned long long *)calloc(AOTX_TEST_TICKS, sizeof(unsigned long long));
    unsigned long long reserve = AOTX_TEST_LOAD
                               + (unsigned long long)AOTX_TEST_WRITERS * AOTX_TEST_EACH;
    aotx_sched_test_graph(stream, &exec, reserve, AOTX_TEST_LOAD, 64u);
    state->ticks = 0u;
    long long started = aotx_sched_test_now_ns();
    long long next = started;
    for (unsigned int t = 0u; t < AOTX_TEST_TICKS; ++t) {
        aotx_check_runtime(cudaEventRecord(opened, stream), "cudaEventRecord");
        aotx_check_runtime(cudaGraphLaunch(exec, stream), "cudaGraphLaunch");
        aotx_check_runtime(cudaEventRecord(closed, stream), "cudaEventRecord");
        aotx_check_runtime(cudaEventRecord(event, stream), "cudaEventRecord");
        aotx_check_runtime(cudaEventSynchronize(event), "cudaEventSynchronize");
        float spent_ms = 0.0f;
        aotx_check_runtime(cudaEventElapsedTime(&spent_ms, opened, closed),
                           "cudaEventElapsedTime");
        whole[t] = (unsigned long long)((double)spent_ms * 1e6);
        next += AOTX_TICK_PERIOD_NS;
        long long now = aotx_sched_test_now_ns();
        if (next > now) {
            struct timespec deadline;
            deadline.tv_sec = (time_t)(next / 1000000000ll);
            deadline.tv_nsec = (long)(next % 1000000000ll);
            clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &deadline, 0);
        }
    }
    long long spent = aotx_sched_test_now_ns() - started;
    for (unsigned int i = 0u; i < 10000u && state->ticks < AOTX_TEST_TICKS; ++i) {
        usleep(500);
    }

    unsigned int count = state->ticks;
    if (count > AOTX_TEST_TICKS) {
        count = AOTX_TEST_TICKS;
    }
    unsigned long long *sorted =
        (unsigned long long *)malloc((size_t)count * sizeof(unsigned long long));
    unsigned long long total = 0ull;
    unsigned long long records = 0ull;
    for (unsigned int i = 0u; i < count; ++i) {
        sorted[i] = state->tick_ns[i];
        total += state->tick_ns[i];
        records += state->tick_records[i];
    }
    qsort(sorted, count, sizeof *sorted, aotx_sched_test_compare);
    unsigned long long mean = (count > 0u) ? total / count : 0ull;
    unsigned long long p99 = (count > 0u) ? sorted[(count * 99u) / 100u] : 0ull;
    unsigned long long worst = (count > 0u) ? sorted[count - 1u] : 0ull;
    printf("sched: %u ticks measured, mean %llu us, p99 %llu us, worst %llu us\n",
           count, mean / 1000ull, p99 / 1000ull, worst / 1000ull);
    unsigned long long *whole_sorted =
        (unsigned long long *)malloc(AOTX_TEST_TICKS * sizeof(unsigned long long));
    unsigned long long whole_total = 0ull;
    for (unsigned int i = 0u; i < AOTX_TEST_TICKS; ++i) {
        whole_sorted[i] = whole[i];
        whole_total += whole[i];
    }
    qsort(whole_sorted, AOTX_TEST_TICKS, sizeof *whole_sorted, aotx_sched_test_compare);
    printf("sched: the whole graph takes mean %llu us, p99 %llu us, worst %llu us\n",
           whole_total / AOTX_TEST_TICKS / 1000ull,
           whole_sorted[(AOTX_TEST_TICKS * 99u) / 100u] / 1000ull,
           whole_sorted[AOTX_TEST_TICKS - 1u] / 1000ull);
    free(whole_sorted);
    free(whole);
    printf("sched: %llu records a tick, %u writers of %u messages, %lld ms of run\n",
           (count > 0u) ? records / count : 0ull, AOTX_TEST_WRITERS, AOTX_TEST_EACH,
           spent / 1000000ll);

    applied += 4u;
    if (count < AOTX_TEST_TICKS) {
        printf("sched: %u statistics records of %u ticks reached the host ring\n",
               count, AOTX_TEST_TICKS);
        failed += 1u;
    }
    if (p99 > AOTX_TEST_P99_NS) {
        printf("sched: the p99 tick is %llu us and the budget is %llu us\n",
               p99 / 1000ull, AOTX_TEST_P99_NS / 1000ull);
        failed += 1u;
    }
    if (mean > AOTX_TEST_MEAN_NS) {
        printf("sched: the mean tick is %llu us and the budget is %llu us\n",
               mean / 1000ull, AOTX_TEST_MEAN_NS / 1000ull);
        failed += 1u;
    }
    if (state->bad != 0ull) {
        printf("sched: %llu blocks did not hold to the block rules\n", state->bad);
        failed += 1u;
    }

    free(sorted);
    state->stop = 1;
    pthread_join(thread, NULL);
    cudaGraphExecDestroy(exec);
    cudaEventDestroy(event);
    cudaEventDestroy(opened);
    cudaEventDestroy(closed);
    cudaStreamDestroy(stream);
    free(before);
    free(after);
    free(state);
    aotx_seam_finish(&rings);
    aotx_seam_close(&rings);
    aotx_mem_release(&map);
    printf("sched: %u cases applied, %u failed\n", applied, failed);
    return failed == 0u ? 0 : 1;
}
