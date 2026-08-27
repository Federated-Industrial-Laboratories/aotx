/* Purpose: Capture the panel kernels and the raster kernel as one graph and launch it.
 * Owns: The stream of the highest priority, the event, the graph and the graph instance.
 * Launch shape: Host glue only; the graph holds the kernels.
 * Lifetime: From the capture at the first frame to the close at exit. */
#include <cuda_runtime.h>
#include <string.h>

#include "boot/check.h"
#include "ui/ui.cuh"

/* The raster runs on the stream of the highest priority, so a long tick on the pump stream
 * does not hold the display. The graph shape never changes. */
int aotx_ui_graph_build(aotx_ui_graph *graph)
{
    int low = 0;
    int high = 0;
    memset(graph, 0, sizeof *graph);
    aotx_check_runtime(cudaDeviceGetStreamPriorityRange(&low, &high),
                       "cudaDeviceGetStreamPriorityRange");
    graph->priority = high;
    aotx_check_runtime(cudaStreamCreateWithPriority(&graph->stream, cudaStreamNonBlocking,
                                                    high),
                       "cudaStreamCreateWithPriority");
    aotx_check_runtime(cudaEventCreateWithFlags(&graph->event, cudaEventDisableTiming),
                       "cudaEventCreateWithFlags");
    aotx_check_runtime(cudaStreamBeginCapture(graph->stream, cudaStreamCaptureModeGlobal),
                       "cudaStreamBeginCapture");
    aotx_ui_console<<<1, AOTX_UI_PANEL_THREADS, 0, graph->stream>>>();
    aotx_ui_agents<<<1, AOTX_UI_PANEL_THREADS, 0, graph->stream>>>();
    aotx_ui_bus<<<1, AOTX_UI_PANEL_THREADS, 0, graph->stream>>>();
    aotx_ui_arena<<<1, AOTX_UI_PANEL_THREADS, 0, graph->stream>>>();
    aotx_ui_tick<<<1, AOTX_UI_PANEL_THREADS, 0, graph->stream>>>();
    aotx_ui_seam<<<1, AOTX_UI_PANEL_THREADS, 0, graph->stream>>>();
    aotx_ui_raster<<<AOTX_UI_RASTER_BLOCKS, AOTX_UI_RASTER_THREADS, 0, graph->stream>>>();
    aotx_check_runtime(cudaStreamEndCapture(graph->stream, &graph->graph),
                       "cudaStreamEndCapture");
    aotx_check_runtime(cudaGraphInstantiate(&graph->exec, graph->graph, 0),
                       "cudaGraphInstantiate");
    return 0;
}

void aotx_ui_graph_run(aotx_ui_graph *graph)
{
    aotx_check_runtime(cudaGraphLaunch(graph->exec, graph->stream), "cudaGraphLaunch");
    aotx_check_runtime(cudaEventRecord(graph->event, graph->stream), "cudaEventRecord");
    aotx_check_runtime(cudaEventSynchronize(graph->event), "cudaEventSynchronize");
}

void *aotx_ui_graph_pixels(void)
{
    void *at = 0;
    aotx_check_runtime(cudaGetSymbolAddress(&at, aotx_ui_pixel), "cudaGetSymbolAddress");
    return at;
}

void aotx_ui_graph_close(aotx_ui_graph *graph)
{
    if (graph->exec != 0) {
        cudaGraphExecDestroy(graph->exec);
        graph->exec = 0;
    }
    if (graph->graph != 0) {
        cudaGraphDestroy(graph->graph);
        graph->graph = 0;
    }
    if (graph->event != 0) {
        cudaEventDestroy(graph->event);
        graph->event = 0;
    }
    if (graph->stream != 0) {
        cudaStreamDestroy(graph->stream);
        graph->stream = 0;
    }
}
