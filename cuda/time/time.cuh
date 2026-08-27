/* Purpose: Give each record its tick and its clock sample.
 * Owns: The tick counter.
 * Launch shape: One thread reads the clock.
 * Lifetime: The whole run. */
#ifndef AOTX_TIME_CUH
#define AOTX_TIME_CUH

/* The device clock in nanoseconds from an arbitrary origin. The driver resets the origin at
 * load. The value measures lag only; order comes from the tick and the record sequence. */
__device__ __forceinline__ unsigned long long aotx_time_globaltimer(void)
{
    unsigned long long ns;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ns));
    return ns;
}

/* The tick counter. The tick start kernel adds one to it; every other kernel reads it. */
extern __device__ unsigned long long aotx_time_tick;

__device__ __forceinline__ unsigned long long aotx_time_now(void)
{
    return aotx_time_tick;
}

#endif
