/* Purpose: Put the agent step in the tick graph.
 * Owns: Nothing; the agent state lives on the device.
 * Launch shape: Host glue only; the graph holds the kernel.
 * Lifetime: From the boot of the run to the end of it. */
#include <cuda_runtime.h>

#include "agent/agent_state.cuh"

/* The step is one block of one thread for each agent. One block lets the first thread give
 * the pending tasks to idle agents before the threads step their agents. The assignment of
 * a run therefore does not depend on which thread arrives first. */
int aotx_agent_capture(void *stream)
{
    cudaStream_t on = (cudaStream_t)stream;
    aotx_agent_step<<<1, AOTX_SLOTS, 0, on>>>(0ull);
    return 0;
}
