/* Purpose: Capture the tick graph once and launch one tick for each pump step.
 * Owns: The stream, the event, the graph and the graph instance.
 * Launch shape: Host glue only; the graph holds the kernels.
 * Lifetime: From the capture at start to the close at exit. */
#include <cuda_runtime.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include "agent/agent_state.cuh"
#include "boot/check.h"
#include "cli/prompt.cuh"
#include "model/decode.cuh"
#include "model/graph_host.h"
#include "model/load.cuh"
#include "sched/sched.cuh"
#include "settings/settings.cuh"
#include "tool/tool_state.cuh"

int aotx_pump_build(aotx_pump *pump, unsigned long long workload, unsigned int blocks)
{
    memset(pump, 0, sizeof *pump);
    pump->workload = workload;
    pump->blocks = (blocks == 0u) ? 1u : blocks;
    aotx_check_runtime(cudaStreamCreateWithFlags(&pump->stream, cudaStreamNonBlocking),
                       "cudaStreamCreateWithFlags");
    aotx_check_runtime(cudaEventCreateWithFlags(&pump->event, cudaEventDisableTiming),
                       "cudaEventCreateWithFlags");

    /* The page range opens with the pump, because the pump answers the page requests of
     * the tick that went before. */
    if (aotx_kv_open(&pump->kv) != 0) {
        return 1;
    }

    /* The decode makes its buffers and captures its forward pass before the tick capture
     * starts. A capture does not take an allocation or a second capture. The pass of the
     * embedding role opens for the same reason. */
    aotx_decode_open();
    if (aotx_tool_open() != 0) {
        fprintf(stderr, "memory is not ready; install an embedding model and select "
                        "--roles embedding; other tools remain available\n");
    }
    if(aotx_model_hold_of(AOTX_MODEL_LANGUAGE_AUDIO)->ready && aotx_kv_reserve(&pump->kv))return 1;
    if (aotx_pump_capture(pump) != 0) {
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

/* The page requests of a tick are answered between two ticks, when no kernel of the tick
 * graph runs. The map and the unmap take the pump stream, so the stream that draws the
 * display never waits for them. */
void aotx_pump_tick(aotx_pump *pump)
{
    aotx_check_runtime(cudaGraphLaunch(pump->exec, pump->stream), "cudaGraphLaunch");
    aotx_check_runtime(cudaEventRecord(pump->event, pump->stream), "cudaEventRecord");
    aotx_check_runtime(cudaEventSynchronize(pump->event), "cudaEventSynchronize");
    aotx_kv_serve(&pump->kv, pump->stream);
    if (aotx_model_load_step(pump) != 0) {
        pump->model_refused = 1u;
    }
    /* An import or a remove of a device tool ends the graph of the tick. The capture runs
     * between two ticks, and the tick that follows launches the new instance. */
    if (aotx_pump_stale(pump) != 0) {
        aotx_pump_recapture(pump);
    }
    /* The agent of the console spawns in the tick that the import of its role lands. The
     * pump names that tick once and then reads the mark no more. */
    if (pump->console_agent == 0u) {
        aotx_check_runtime(cudaMemcpyFromSymbol(&pump->console_agent, aotx_agent_boot_mark,
                                                sizeof pump->console_agent),
                           "cudaMemcpyFromSymbol");
        if (pump->console_agent != 0u) {
            printf("catalog: the role of the console is installed and its agent holds "
                   "slot 0\n");
        }
    }
}

/* The records that no block holds yet go to the host ring, and the payloads that no block
 * holds yet go to the bulk ring. The two flush kernels are the only kernels that run, so no
 * new record is made and the tick count does not change. */
void aotx_pump_flush(aotx_pump *pump)
{
    aotx_seam_flush<<<1, AOTX_FLUSH_THREADS, 0, pump->stream>>>();
    aotx_seam_bulk_flush<<<1, AOTX_FLUSH_THREADS, 0, pump->stream>>>();
    aotx_check_runtime(cudaEventRecord(pump->event, pump->stream), "cudaEventRecord");
    aotx_check_runtime(cudaEventSynchronize(pump->event), "cudaEventSynchronize");
}

/* The pace holds a schedule and not a delay, so a sleep that runs long does not push the
 * ticks that follow it. A tick that runs long gives the schedule a new start.
 *
 * The period comes from the control page, which the tick commit node writes with a release
 * store. The pace takes one acquire load of it for each tick, so a change of the setting
 * takes effect at the next tick. The glue reads a number and parses nothing. */
void aotx_pump_pace(aotx_pump *pump)
{
    long long period = (long long)aotx_settings_period_ns();
    long long now = aotx_pump_now_ns();
    if (pump->next_ns == 0ll || pump->next_ns + period < now) {
        pump->next_ns = now;
    }
    pump->next_ns += period;
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
    report->held = AOTX_STALL_HELD(sched.held_count);
    report->tick = tick;
    report->state_hash = seam.apply.state_hash;
    report->applied = seam.apply.applied_count;
    report->rejected = seam.apply.rejected;
    report->tail = seam.dev.tail;
    report->flushed = seam.dev.flushed;
    report->consumed = seam.in.consumed;
    report->overrun = seam.dev.overrun;
    aotx_check_runtime(cudaMemcpyFromSymbol(&report->paced, aotx_seam_replay_holds,
                                            sizeof report->paced), "cudaMemcpyFromSymbol");

    /* The decode counters. The table is large, so the read takes the three fields alone. */
    unsigned int marks[2] = { 0u, 0u };
    unsigned int pages = 0u;
    aotx_check_runtime(cudaMemcpyFromSymbol(marks, aotx_seqs, sizeof marks,
                                            offsetof(aotx_seq_table, live)),
                       "cudaMemcpyFromSymbol");
    aotx_check_runtime(cudaMemcpyFromSymbol(&pages, aotx_kv, sizeof pages,
                                            offsetof(aotx_kv_table, mapped_pages)),
                       "cudaMemcpyFromSymbol");
    report->live = marks[0];
    report->refused = marks[1];
    report->pages = pages;

    unsigned int spawned = 0u;
    aotx_check_runtime(cudaMemcpyFromSymbol(&spawned, aotx_agent_boot_mark,
                                            sizeof spawned), "cudaMemcpyFromSymbol");
    report->console_agent = spawned;
    aotx_model_load_state load;
    aotx_check_runtime(cudaMemcpyFromSymbol(&load, aotx_model_load, sizeof load),
                       "cudaMemcpyFromSymbol");
    report->model_bytes = load.placed_bytes;
}

void aotx_pump_close(aotx_pump *pump)
{
    aotx_tool_close();
    aotx_decode_close();
    aotx_kv_close(&pump->kv);
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
