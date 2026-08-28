/* Purpose: Capture the tick graph once and launch one tick for each pump step.
 * Owns: The stream, the event, the graph and the graph instance.
 * Launch shape: Host glue only; the graph holds the kernels.
 * Lifetime: From the capture at start to the close at exit. */
#include <cuda_runtime.h>
#include <stddef.h>
#include <string.h>
#include <time.h>

#include "agent/agent_state.cuh"
#include "boot/check.h"
#include "cli/prompt.cuh"
#include "model/decode.cuh"
#include "model/graph_host.h"
#include "sched/sched.cuh"
#include "tool/tool_state.cuh"

static long long aotx_pump_now_ns(void)
{
    struct timespec at;
    clock_gettime(CLOCK_MONOTONIC, &at);
    return (long long)at.tv_sec * 1000000000ll + (long long)at.tv_nsec;
}

/* Nodes the capture holds so far. The count of each part of the tick comes from the change
 * of this number, so the check of the graph names every part. */
static unsigned int aotx_pump_count(cudaStream_t stream)
{
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    cudaGraph_t graph = 0;
    size_t count = 0;
    if (cudaStreamGetCaptureInfo(stream, &status, 0, &graph, 0, 0, 0) != cudaSuccess
        || graph == 0) {
        return 0u;
    }
    aotx_check_runtime(cudaGraphGetNodes(graph, 0, &count), "cudaGraphGetNodes");
    return (unsigned int)count;
}

/* Find the two nodes that take a parameter for each tick. The graph shape never changes, so
 * the search runs once. */
static int aotx_pump_find(aotx_pump *pump)
{
    size_t count = 0;
    aotx_check_runtime(cudaGraphGetNodes(pump->graph, 0, &count), "cudaGraphGetNodes");
    if (count == 0u || count > AOTX_TICK_NODES_MAX) {
        return 1;
    }
    pump->nodes = (unsigned int)count;
    cudaGraphNode_t nodes[AOTX_TICK_NODES_MAX];
    aotx_check_runtime(cudaGraphGetNodes(pump->graph, nodes, &count), "cudaGraphGetNodes");
    pump->start_node = 0;
    pump->work_node = 0;
    for (size_t i = 0; i < count; ++i) {
        /* The type is read first. A parameter read of a node that is not a kernel node
         * is an error, and that error stays until the next error read. */
        cudaGraphNodeType type = cudaGraphNodeTypeEmpty;
        aotx_check_runtime(cudaGraphNodeGetType(nodes[i], &type), "cudaGraphNodeGetType");
        if (type != cudaGraphNodeTypeKernel) {
            continue;
        }
        cudaKernelNodeParams params;
        memset(&params, 0, sizeof params);
        aotx_check_runtime(cudaGraphKernelNodeGetParams(nodes[i], &params),
                           "cudaGraphKernelNodeGetParams");
        if (params.func == (void *)aotx_sched_tick_start) {
            pump->start_node = nodes[i];
        }
        if (params.func == (void *)aotx_sched_workload) {
            pump->work_node = nodes[i];
        }
    }
    return (pump->start_node == 0 || pump->work_node == 0) ? 1 : 0;
}

/* The node order is the order of the tick. The first nodes are the tick start, the apply
 * and the say path. The decode follows with its plan, its forward pass as one child node,
 * and its commit.
 *
 * The tool path follows the decode. Its nodes are the fill step, the four tokenizer steps
 * and the plan. The pass of the embedding role is a second child node, and the search and
 * the tool step come after it.
 *
 * The agent step comes after the tool step, because an agent takes the result of its tool
 * in the tick that result arrives. The reply of the console comes after the agent step, so
 * a reply that no agent streams shows nothing. The last nodes are the tick load, the tick
 * commit, the record flush and the bulk flush.
 *
 * The stream capture makes one chain of nodes from the launch order. The shape of the
 * graph never changes. */
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
     * starts. A capture does not take an allocation or a second capture. */
    unsigned int decode = (aotx_decode_open() == 0) ? 1u : 0u;

    /* The pass of the embedding role is captured before the tick capture starts. The
     * reason is the reason of the decode. A capture takes no allocation and no second
     * capture. */
    aotx_tool_open();

    /* The conductor takes slot 0 before the first tick. The say command of that tick then
     * finds it, and a replay of the journal finds it as well. */
    aotx_agent_open();
    aotx_check_runtime(cudaStreamBeginCapture(pump->stream, cudaStreamCaptureModeGlobal),
                       "cudaStreamBeginCapture");
    aotx_sched_tick_start<<<1, 1, 0, pump->stream>>>(pump->workload);
    aotx_seam_apply_inbound<<<AOTX_APPLY_BLOCKS, AOTX_APPLY_THREADS, 0, pump->stream>>>();
    unsigned int at = aotx_pump_count(pump->stream);
    aotx_cli_say_capture(pump->stream);
    pump->say_nodes = aotx_pump_count(pump->stream) - at;
    at += pump->say_nodes;
    pump->decode = (decode != 0u && aotx_decode_capture(pump->stream) == 0) ? 1u : 0u;
    pump->decode_nodes = aotx_pump_count(pump->stream) - at;
    at += pump->decode_nodes;
    pump->embed = (aotx_tool_capture(pump->stream) == 0) ? 1u : 0u;
    pump->tool_nodes = aotx_pump_count(pump->stream) - at;
    at += pump->tool_nodes;
    aotx_agent_capture(pump->stream);
    pump->agent_nodes = aotx_pump_count(pump->stream) - at;
    at += pump->agent_nodes;
    aotx_cli_reply_capture(pump->stream);
    pump->reply_nodes = aotx_pump_count(pump->stream) - at;
    aotx_sched_workload<<<pump->blocks, AOTX_WORKLOAD_THREADS, 0, pump->stream>>>(pump->workload);
    aotx_sched_commit<<<1, 1, 0, pump->stream>>>();
    aotx_seam_flush<<<1, AOTX_FLUSH_THREADS, 0, pump->stream>>>();
    aotx_seam_bulk_flush<<<1, AOTX_FLUSH_THREADS, 0, pump->stream>>>();
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

/* The page requests of a tick are answered between two ticks, when no kernel of the tick
 * graph runs. The map and the unmap take the pump stream, so the stream that draws the
 * display never waits for them. */
void aotx_pump_tick(aotx_pump *pump)
{
    aotx_check_runtime(cudaGraphLaunch(pump->exec, pump->stream), "cudaGraphLaunch");
    aotx_check_runtime(cudaEventRecord(pump->event, pump->stream), "cudaEventRecord");
    aotx_check_runtime(cudaEventSynchronize(pump->event), "cudaEventSynchronize");
    aotx_kv_serve(&pump->kv, pump->stream);
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
