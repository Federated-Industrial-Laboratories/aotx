/* Purpose: Capture the tick graph, and capture it again when the device tools change.
 * Owns: Nothing; the pump holds the stream, the graph and the graph instance.
 * Launch shape: Host glue only; the graph holds the kernels.
 * Lifetime: From the first capture to the close at exit. */
#include <cuda_runtime.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include "agent/agent_state.cuh"
#include "boot/check.h"
#include "cognitive/checkpoint.cuh"
#include "cli/prompt.cuh"
#include "model/decode.cuh"
#include "sched/sched.cuh"
#include "tool/module.cuh"
#include "tool/tool_state.cuh"

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

/* Find the two nodes that take a parameter for each tick. The two kernels stand once in
 * the graph, so the search runs after every capture. */
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
        if (cudaGraphKernelNodeGetParams(nodes[i], &params) != cudaSuccess) {
            /* A node the driver added holds the kernel of a module, and the runtime knows
             * no such function. The error is taken and the search goes on. */
            cudaGetLastError();
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

/* The node order is the order of the tick. The first nodes are the tick start, the apply
 * and the say path. The decode follows with its plan, its forward pass as one child node,
 * and its commit.
 *
 * The tool path follows the decode. Its nodes are the fill step, the four tokenizer steps
 * and the plan. The pass of the embedding role is a second child node. The search and the
 * node of every device tool module come after it, and the tool step ends the path.
 *
 * The agent step comes after the tool step, because an agent takes the result of its tool
 * in the tick that result arrives. The reply of the console comes after the agent step, so
 * a reply that no agent streams shows nothing. The last nodes are the tick load, the tick
 * commit, the record flush and the bulk flush.
 *
 * The stream capture makes one chain of nodes from the launch order. The shape of the
 * graph changes only when a device tool comes in or goes out. */
int aotx_pump_capture(aotx_pump *pump)
{
    cudaGraph_t graph = 0;
    cudaGraphExec_t exec = 0;

    /* The built-in tools go in the catalog before the first tick, so a role that names
     * one of them finds it at the import. The module of every device tool is loaded here
     * as well, because a capture takes no launch and no allocation of its own. */
    /* The module path and the catalog launch on the stream of the pump and wait on its
     * event. A capture therefore holds that stream alone and the display is not held. */
    aotx_tool_module_on(pump->stream, pump->event);
    aotx_catalog_open();
    pump->modules = aotx_tool_module_open();
    aotx_check_runtime(cudaMemcpyFromSymbol(&pump->gen, aotx_catalog,
                                            sizeof pump->gen,
                                            offsetof(aotx_catalog_state, count)
                                            + offsetof(aotx_catalog_counts, device_gen)),
                       "cudaMemcpyFromSymbol");
    aotx_check_runtime(cudaStreamBeginCapture(pump->stream, cudaStreamCaptureModeGlobal),
                       "cudaStreamBeginCapture");
    aotx_sched_tick_start<<<1, 1, 0, pump->stream>>>(pump->workload);
    aotx_seam_apply_inbound<<<AOTX_APPLY_BLOCKS, AOTX_APPLY_THREADS, 0, pump->stream>>>();
    unsigned int at = aotx_pump_count(pump->stream);
    aotx_cli_say_capture(pump->stream);
    pump->say_nodes = aotx_pump_count(pump->stream) - at;
    at += pump->say_nodes;
    pump->decode = (aotx_decode_capture(pump->stream) == 0) ? 1u : 0u;
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
    aotx_sched_workload<<<pump->blocks, AOTX_WORKLOAD_THREADS, 0, pump->stream>>>(
        pump->workload);
    aotx_sched_commit<<<1, 1, 0, pump->stream>>>();
    aotx_checkpoint_capture(pump->stream);
    aotx_seam_flush<<<1, AOTX_FLUSH_THREADS, 0, pump->stream>>>();
    aotx_seam_bulk_flush<<<1, AOTX_FLUSH_THREADS, 0, pump->stream>>>();
    aotx_check_runtime(cudaStreamEndCapture(pump->stream, &graph), "cudaStreamEndCapture");
    aotx_check_runtime(cudaGraphInstantiate(&exec, graph, 0), "cudaGraphInstantiate");

    /* The instance the tick before this one used is given back here. That tick ended on
     * its event, so no launch of it stands. */
    if (pump->exec != 0) {
        cudaGraphExecDestroy(pump->exec);
    }
    if (pump->graph != 0) {
        cudaGraphDestroy(pump->graph);
    }
    pump->graph = graph;
    pump->exec = exec;
    return aotx_pump_find(pump);
}

int aotx_pump_recapture(aotx_pump *pump)
{
    unsigned int before = pump->nodes;
    long long start = aotx_pump_now_ns();
    if (aotx_pump_capture(pump) != 0) {
        return 1;
    }
    unsigned int took = (unsigned int)((aotx_pump_now_ns() - start) / 1000ll);
    pump->recaptures += 1u;
    pump->recapture_us = took;
    /* The record of the capture names the tick, the node count before and after, and the
     * microseconds it took. The kernel writes the console line and the bus note. The
     * figures thus reach the operator by the path every note of the catalog takes. */
    aotx_tool_module_note<<<1, 1, 0, pump->stream>>>(before, pump->nodes, took);
    aotx_check_runtime(cudaEventRecord(pump->event, pump->stream), "cudaEventRecord");
    aotx_check_runtime(cudaEventSynchronize(pump->event), "cudaEventSynchronize");
    /* The parameters of the two nodes of the tick belong to the new instance. */
    aotx_pump_set(pump, pump->workload, pump->blocks);
    return 0;
}

/* Report whether the catalog holds a set of device tools the graph was not built with. */
int aotx_pump_stale(const aotx_pump *pump)
{
    unsigned int gen = 0u;
    aotx_check_runtime(cudaMemcpyFromSymbol(&gen, aotx_catalog, sizeof gen,
                                            offsetof(aotx_catalog_state, count)
                                            + offsetof(aotx_catalog_counts, device_gen)),
                       "cudaMemcpyFromSymbol");
    return (gen != pump->gen) ? 1 : 0;
}
