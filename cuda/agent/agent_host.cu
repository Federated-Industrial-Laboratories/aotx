/* Purpose: Put the conductor in the table and the agent step in the tick graph.
 * Owns: Nothing; the agent state lives on the device.
 * Launch shape: Host glue only; the graph holds the kernel.
 * Lifetime: From the boot of the run to the end of it. */
#include <cuda_runtime.h>

#include "agent/agent_state.cuh"
#include "boot/check.h"

int aotx_agent_open(void)
{
    /* The conductor stands before the first tick, so a say command of that tick finds it.
     * A restore takes the same path. The conductor is in the table before the replay
     * starts, and a replayed say finds the agent it named when it was live. */
    aotx_agent_boot<<<1, 1>>>();
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    unsigned int state = AOTX_AGENT_STATE_FREE;
    aotx_check_runtime(cudaMemcpyFromSymbol(&state, aotx_agents, sizeof state,
                                            offsetof(aotx_agent_table, agent)
                                            + offsetof(aotx_agent, state)),
                       "cudaMemcpyFromSymbol");
    return (state != AOTX_AGENT_STATE_FREE) ? 0 : 1;
}

/* The step is one block of one thread for each agent. One block lets the first thread give
 * the pending tasks to idle agents before the threads step their agents. The assignment of
 * a run therefore does not depend on which thread arrives first. */
int aotx_agent_capture(void *stream)
{
    cudaStream_t on = (cudaStream_t)stream;
    aotx_agent_step<<<1, AOTX_SLOTS, 0, on>>>(0ull);
    return 0;
}
