/* Purpose: Hold the tick counter that the tick start kernel advances.
 * Owns: The tick counter.
 * Launch shape: No kernels; storage only.
 * Lifetime: The whole run. */
#include "time/time.cuh"

/* The first tick is tick 1, because the tick start kernel adds one before it writes a record. */
__device__ unsigned long long aotx_time_tick = 0;
