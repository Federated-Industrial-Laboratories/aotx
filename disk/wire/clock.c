/* Purpose: Read the wall clock and pause with a delay that doubles.
 * Owns: Nothing; the caller holds the backoff state.
 * Threading: One thread; the functions hold no state between calls.
 * Lifetime: The life of the process. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "disk/wire/diskwire.h"

#include <time.h>

#define AOTX_PAUSE_FIRST 50000u    /* 50 microseconds */
#define AOTX_PAUSE_LIMIT 2000000u  /* 2 milliseconds */

uint64_t aotx_wall_ns(void)
{
    struct timespec ts;
    if (clock_gettime(CLOCK_REALTIME, &ts) != 0) {
        return 0;
    }
    return (uint64_t)ts.tv_sec * 1000000000u + (uint64_t)ts.tv_nsec;
}

void aotx_pause(uint64_t *state)
{
    struct timespec ts;
    uint64_t ns = (*state == 0) ? AOTX_PAUSE_FIRST : *state;
    /* A hot spin holds a core against the producer. The delay doubles to a limit, so an
     * idle reader costs almost nothing and a busy reader still answers fast. */
    if (ns > AOTX_PAUSE_LIMIT) {
        ns = AOTX_PAUSE_LIMIT;
    }
    *state = ns * 2;
    ts.tv_sec = (time_t)(ns / 1000000000u);
    ts.tv_nsec = (long)(ns % 1000000000u);
    nanosleep(&ts, NULL);
}
