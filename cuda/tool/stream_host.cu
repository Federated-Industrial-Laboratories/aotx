/* Purpose: Hold the stream and the event the module path launches and waits on.
 * Owns: The stream and the event, when the caller named none of its own.
 * Launch shape: Host glue only; no kernel stands in this file.
 * Lifetime: From the first wait to the close of the run.
 *
 * The pump gives its own stream and event, so a capture waits for the work of that stream
 * alone. The rule of the scheduler is that the pump thread never waits over the whole
 * device. Such a wait holds the stream that draws the display. */
#include <cuda_runtime.h>

#include "boot/check.h"
#include "tool/module.cuh"
#include "tool/module_host.h"

/* The stream and the event of the module path. The pump gives its own, so a capture waits
 * on the work of that stream alone and the stream of the display is not held. */
static cudaStream_t aotx_tool_module_line;
static cudaEvent_t aotx_tool_module_mark;
static int aotx_tool_module_own;

void aotx_tool_module_on(void *stream, void *event)
{
    if (aotx_tool_module_own != 0 && stream != 0) {
        cudaEventDestroy(aotx_tool_module_mark);
        cudaStreamDestroy(aotx_tool_module_line);
        aotx_tool_module_own = 0;
    }
    aotx_tool_module_line = (cudaStream_t)stream;
    aotx_tool_module_mark = (cudaEvent_t)event;
}

/* The stream of this path. A caller that named none gets one of its own, made here at the
 * first call. Every launch and every wait of this path then stands on one stream. */
cudaStream_t aotx_tool_module_line_of(void)
{
    if (aotx_tool_module_mark == 0) {
        aotx_check_runtime(cudaStreamCreateWithFlags(&aotx_tool_module_line,
                                                     cudaStreamNonBlocking),
                           "cudaStreamCreateWithFlags");
        aotx_check_runtime(cudaEventCreateWithFlags(&aotx_tool_module_mark,
                                                    cudaEventDisableTiming),
                           "cudaEventCreateWithFlags");
        aotx_tool_module_own = 1;
    }
    return aotx_tool_module_line;
}

/* Wait for the work this path put on its stream. The wait stands on that stream alone, so
 * the stream that draws the display is not held. */
void aotx_tool_module_wait(void)
{
    cudaStream_t on = aotx_tool_module_line_of();
    aotx_check_runtime(cudaEventRecord(aotx_tool_module_mark, on), "cudaEventRecord");
    aotx_check_runtime(cudaEventSynchronize(aotx_tool_module_mark),
                       "cudaEventSynchronize");
}
