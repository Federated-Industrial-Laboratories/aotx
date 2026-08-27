/* Purpose: Capture the tick graph once and launch one tick for each pump step.
 * Owns: The stream, the event, the graph and the graph instance.
 * Launch shape: Host glue only; the graph holds the kernels.
 * Lifetime: From the capture at start to the close at exit. */
#include <cuda_runtime.h>
#include <stddef.h>
#include <string.h>
#include <time.h>

#include "boot/check.h"
#include "sched/sched.cuh"

static long long aotx_pump_now_ns(void)
{
    struct timespec at;
    clock_gettime(CLOCK_MONOTONIC, &at);
    return (long long)at.tv_sec * 1000000000ll + (long long)at.tv_nsec;
}

/* Find the two nodes that take a parameter for each tick. The graph shape never changes, so
 * the search runs once. */
static int aotx_pump_find(aotx_pump *pump)
{
    size_t count = 0;
    aotx_check_runtime(cudaGraphGetNodes(pump->graph, 0, &count), "cudaGraphGetNodes");
    if (count == 0u || count > 16u) {
        return 1;
    }
    cudaGraphNode_t nodes[16];
    aotx_check_runtime(cudaGraphGetNodes(pump->graph, nodes, &count), "cudaGraphGetNodes");
    pump->start_node = 0;
    pump->work_node = 0;
    for (size_t i = 0; i < count; ++i) {
        cudaKernelNodeParams params;
        memset(&params, 0, sizeof params);
        if (cudaGraphKernelNodeGetParams(nodes[i], &params) != cudaSuccess) {
            continue;
        }
        if (params.func == (void *)aotx_sched_tick_start) {
            pump->start_node = nodes[i];
        }
        if (params.func == (void *)aotx_sched_workload) {
            pump->work_node = nodes[i];
        }
    }
    return (pump->start_node == 0 || pump->work_node == 0) ? 1 : 0;
}

/* The node order is the tick: start, apply, load, commit, flush. The stream capture makes
 * one chain of nodes from the launch order. */
int aotx_pump_build(aotx_pump *pump, unsigned long long workload, unsigned int blocks)
{
    memset(pump, 0, sizeof *pump);
    pump->workload = workload;
    pump->blocks = (blocks == 0u) ? 1u : blocks;
    aotx_check_runtime(cudaStreamCreateWithFlags(&pump->stream, cudaStreamNonBlocking),
                       "cudaStreamCreateWithFlags");
    aotx_check_runtime(cudaEventCreateWithFlags(&pump->event, cudaEventDisableTiming),
                       "cudaEventCreateWithFlags");
    aotx_check_runtime(cudaStreamBeginCapture(pump->stream, cudaStreamCaptureModeGlobal),
                       "cudaStreamBeginCapture");
    aotx_sched_tick_start<<<1, 1, 0, pump->stream>>>(pump->workload);
    aotx_seam_apply_inbound<<<AOTX_APPLY_BLOCKS, AOTX_APPLY_THREADS, 0, pump->stream>>>();
    aotx_sched_workload<<<pump->blocks, AOTX_WORKLOAD_THREADS, 0, pump->stream>>>(pump->workload);
    aotx_sched_commit<<<1, 1, 0, pump->stream>>>();
    aotx_seam_flush<<<1, AOTX_FLUSH_THREADS, 0, pump->stream>>>();
    aotx_check_runtime(cudaStreamEndCapture(pump->stream, &pump->graph),
                       "cudaStreamEndCapture");
    aotx_check_runtime(cudaGraphInstantiate(&pump->exec, pump->graph, 0),
                       "cudaGraphInstantiate");
    if (aotx_pump_find(pump) != 0) {
        return 1;
    }
    pump->next_ns = 0ll;
    return 0;
}

int aotx_pump_set(aotx_pump *pump, unsigned long long workload, unsigned int blocks)
{
    pump->workload = workload;
    pump->blocks = (blocks == 0u) ? 1u : blocks;
    void *args[1];
    args[0] = &pump->workload;

    cudaKernelNodeParams start;
    memset(&start, 0, sizeof start);
    start.func = (void *)aotx_sched_tick_start;
    start.gridDim = dim3(1, 1, 1);
    start.blockDim = dim3(1, 1, 1);
    start.kernelParams = args;
    aotx_check_runtime(cudaGraphExecKernelNodeSetParams(pump->exec, pump->start_node, &start),
                       "cudaGraphExecKernelNodeSetParams");

    cudaKernelNodeParams work;
    memset(&work, 0, sizeof work);
    work.func = (void *)aotx_sched_workload;
    work.gridDim = dim3(pump->blocks, 1, 1);
    work.blockDim = dim3(AOTX_WORKLOAD_THREADS, 1, 1);
    work.kernelParams = args;
    aotx_check_runtime(cudaGraphExecKernelNodeSetParams(pump->exec, pump->work_node, &work),
                       "cudaGraphExecKernelNodeSetParams");
    return 0;
}

void aotx_pump_tick(aotx_pump *pump)
{
    aotx_check_runtime(cudaGraphLaunch(pump->exec, pump->stream), "cudaGraphLaunch");
    aotx_check_runtime(cudaEventRecord(pump->event, pump->stream), "cudaEventRecord");
    aotx_check_runtime(cudaEventSynchronize(pump->event), "cudaEventSynchronize");
}

/* The records that no block holds yet go to the host ring. The flush node is the only node
 * that runs, so no new record is made and the tick count does not change. */
void aotx_pump_flush(aotx_pump *pump)
{
    aotx_seam_flush<<<1, AOTX_FLUSH_THREADS, 0, pump->stream>>>();
    aotx_check_runtime(cudaEventRecord(pump->event, pump->stream), "cudaEventRecord");
    aotx_check_runtime(cudaEventSynchronize(pump->event), "cudaEventSynchronize");
}

/* The pace holds a schedule and not a delay, so a sleep that runs long does not push the
 * ticks that follow it. A tick that runs long gives the schedule a new start. */
void aotx_pump_pace(aotx_pump *pump)
{
    long long now = aotx_pump_now_ns();
    if (pump->next_ns == 0ll || pump->next_ns + AOTX_TICK_PERIOD_NS < now) {
        pump->next_ns = now;
    }
    pump->next_ns += AOTX_TICK_PERIOD_NS;
    if (pump->next_ns <= now) {
        return;
    }
    struct timespec deadline;
    deadline.tv_sec = (time_t)(pump->next_ns / 1000000000ll);
    deadline.tv_nsec = (long)(pump->next_ns % 1000000000ll);
    clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &deadline, 0);
}

void aotx_pump_read(aotx_pump_report *report)
{
    aotx_sched_state sched;
    aotx_seam_state seam;
    unsigned long long tick = 0ull;
    aotx_check_runtime(cudaMemcpyFromSymbol(&sched, aotx_sched, sizeof sched),
                       "cudaMemcpyFromSymbol");
    aotx_check_runtime(cudaMemcpyFromSymbol(&seam, aotx_seam, sizeof seam),
                       "cudaMemcpyFromSymbol");
    aotx_check_runtime(cudaMemcpyFromSymbol(&tick, aotx_time_tick, sizeof tick),
                       "cudaMemcpyFromSymbol");
    report->records = sched.records;
    report->blocks = sched.blocks;
    report->held = sched.held_count;
    report->tick = tick;
    report->state_hash = seam.apply.state_hash;
    report->applied = seam.apply.applied_count;
    report->rejected = seam.apply.rejected;
    report->tail = seam.dev.tail;
    report->flushed = seam.dev.flushed;
    report->consumed = seam.in.consumed;
    report->overrun = seam.dev.overrun;
}

void aotx_pump_close(aotx_pump *pump)
{
    if (pump->exec != 0) {
        cudaGraphExecDestroy(pump->exec);
        pump->exec = 0;
    }
    if (pump->graph != 0) {
        cudaGraphDestroy(pump->graph);
        pump->graph = 0;
    }
    if (pump->event != 0) {
        cudaEventDestroy(pump->event);
        pump->event = 0;
    }
    if (pump->stream != 0) {
        cudaStreamDestroy(pump->stream);
        pump->stream = 0;
    }
}
