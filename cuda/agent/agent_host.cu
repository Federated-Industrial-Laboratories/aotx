/* Purpose: Put the agent step in the tick graph.
 * Owns: Nothing; the agent state lives on the device.
 * Launch shape: Host glue only; the graph holds the kernel.
 * Lifetime: From the boot of the run to the end of it. */
#include <cuda_runtime.h>

#include "agent/agent_state.cuh"
#include "cognitive/live.cuh"
#include "cognitive/checkpoint.cuh"
#include "cognitive/maintenance.cuh"
#ifdef AOTX_AFFECT
#include "affect/affect.cuh"
#endif

/* The step is one block of one thread for each agent. One block lets the first thread give
 * the pending tasks to idle agents before the threads step their agents. The assignment of
 * a run therefore does not depend on which thread arrives first. */
int aotx_agent_capture(void *stream)
{
    cudaStream_t on = (cudaStream_t)stream;
    aotx_live_stage<<<1, 64, 0, on>>>();
    aotx_memory_seed<<<128, 256, 0, on>>>();
    aotx_memory_plan<<<1, 256, 0, on>>>();
    aotx_memory_offsets<<<128, 256, 0, on>>>();
    aotx_memory_copy<<<128, 256, 0, on>>>();
    aotx_memory_install<<<128, 256, 0, on>>>();
    aotx_memory_publish<<<1, 64, 0, on>>>();
    aotx_live_prepare<<<1, 64, 0, on>>>();
    aotx_live_search<<<AOTX_RECALL_BATCH, 64, 0, on>>>();
    aotx_live_decide<<<1, 64, 0, on>>>();
    aotx_live_commit<<<1, 64, 0, on>>>();
    aotx_agent_step<<<1, AOTX_SLOTS, 0, on>>>(0ull);
#ifdef AOTX_AFFECT
    /* The turn node follows the step, so it reads the turns that ended in this tick. */
    aotx_affect_capture(stream);
#endif
    aotx_checkpoint_capture(stream);
    return 0;
}
