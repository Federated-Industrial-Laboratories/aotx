/* Purpose: Bind the mirror and run the raster graph for a terminal when no window is open.
 * Owns: The thread of the mirror, its stop flag and its counts.
 * Launch shape: Host glue only; the raster graph holds the kernels.
 * Lifetime: From the bind at the start to the stop at the close. */
#include <cuda.h>
#include <cuda_runtime.h>
#include <pthread.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "boot/check.h"
#include "settings/settings.cuh"
#include "ui/mirror.cuh"

/* Microseconds the thread waits between two reads of the attached count. A terminal that
 * attaches waits this long at the most for the first frame. */
#define AOTX_MIRROR_IDLE_US 20000

/* What the thread and the caller share. */
typedef struct aotx_mirror_run {
    aotx_ui_graph graph;
    CUcontext context;
    const volatile aotx_mirror_preamble *preamble;
    volatile int stop;
    unsigned long long frames;
    unsigned long long idle;
    unsigned int hz;
    int running;
} aotx_mirror_run;

static aotx_mirror_run aotx_mirror_thread_run;
static pthread_t aotx_mirror_thread_id;

static long long aotx_mirror_now_ns(void)
{
    struct timespec at;
    clock_gettime(CLOCK_MONOTONIC, &at);
    return (long long)at.tv_sec * 1000000000ll + (long long)at.tv_nsec;
}

int aotx_mirror_bind(const aotx_seam_rings *rings)
{
    void *device = 0;
    aotx_mirror_state state;
    if (rings->mirror_map == 0) {
        return 1;
    }
    aotx_check_runtime(cudaHostGetDevicePointer(&device, rings->mirror_map, 0),
                       "cudaHostGetDevicePointer");
    const aotx_mirror_preamble *preamble = (const aotx_mirror_preamble *)rings->mirror_map;
    memset(&state, 0, sizeof state);
    state.preamble = (unsigned char *)device;
    state.slot = (unsigned char *)device + sizeof(aotx_mirror_preamble);
    state.slot_bytes = preamble->slot_bytes;
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_mirror, &state, sizeof state),
                       "cudaMemcpyToSymbol");
    return 0;
}

unsigned int aotx_mirror_attached(const aotx_seam_rings *rings)
{
    if (rings->mirror_map == 0) {
        return 0u;
    }
    const aotx_mirror_preamble *preamble = (const aotx_mirror_preamble *)rings->mirror_map;
    return __atomic_load_n((const unsigned int *)&preamble->attached, __ATOMIC_ACQUIRE);
}

/* The thread runs the raster graph at the frame rate of the settings while the feeder says
 * a terminal is attached. It launches nothing while no terminal reads. */
static void *aotx_mirror_frames(void *state)
{
    aotx_mirror_run *run = (aotx_mirror_run *)state;
    long long next_ns = aotx_mirror_now_ns();
    aotx_check_driver(cuCtxSetCurrent(run->context), "cuCtxSetCurrent");
    while (run->stop == 0) {
        if (__atomic_load_n((const unsigned int *)&run->preamble->attached,
                            __ATOMIC_ACQUIRE) == 0u) {
            run->idle += 1ull;
            usleep(AOTX_MIRROR_IDLE_US);
            next_ns = aotx_mirror_now_ns();
            continue;
        }
        unsigned long long hz = aotx_settings_mirror_hz();
        if (hz == 0ull) {
            hz = 1ull;
        }
        run->hz = (unsigned int)hz;
        aotx_ui_graph_run(&run->graph);
        run->frames += 1ull;

        /* The pace holds the frame rate. A frame that took longer than its period starts
         * the next period at the clock, so the thread does not run behind for ever. The
         * wait is in slices, so a stop at a low frame rate does not hold the close. */
        next_ns += (long long)(1000000000ull / hz);
        long long now = aotx_mirror_now_ns();
        while (run->stop == 0 && next_ns - now >= 1000ll) {
            long long left = next_ns - now;
            long long slice = (long long)AOTX_MIRROR_IDLE_US * 1000ll;
            usleep((useconds_t)(((left < slice) ? left : slice) / 1000ll));
            now = aotx_mirror_now_ns();
        }
        if (next_ns < now) {
            next_ns = now;
        }
    }
    return NULL;
}

int aotx_mirror_start(const aotx_seam_rings *rings)
{
    if (aotx_mirror_thread_run.running != 0) {
        return 0;
    }
    if (rings->mirror_map == 0) {
        return 1;
    }
    memset(&aotx_mirror_thread_run, 0, sizeof aotx_mirror_thread_run);
    aotx_mirror_thread_run.preamble = (const volatile aotx_mirror_preamble *)rings->mirror_map;
    aotx_check_driver(cuCtxGetCurrent(&aotx_mirror_thread_run.context), "cuCtxGetCurrent");
    /* The graph is captured on this thread, before the thread starts. A capture in the
     * global mode refuses the launches of another thread while it runs. */
    if (aotx_ui_graph_build(&aotx_mirror_thread_run.graph) != 0) {
        return 1;
    }
    if (pthread_create(&aotx_mirror_thread_id, NULL, aotx_mirror_frames,
                       &aotx_mirror_thread_run) != 0) {
        aotx_ui_graph_close(&aotx_mirror_thread_run.graph);
        return 1;
    }
    aotx_mirror_thread_run.running = 1;
    return 0;
}

void aotx_mirror_stop(void)
{
    if (aotx_mirror_thread_run.running == 0) {
        return;
    }
    aotx_mirror_thread_run.stop = 1;
    pthread_join(aotx_mirror_thread_id, NULL);
    aotx_ui_graph_close(&aotx_mirror_thread_run.graph);
    aotx_mirror_thread_run.running = 0;
}

/* Write the cost of the node in the report of the frames. A run that published no frame
 * writes nothing. */
void aotx_mirror_report_line(void)
{
    aotx_mirror_report report;
    aotx_mirror_read(&report);
    if (report.frames == 0ull) {
        return;
    }
    double each = (double)report.node_ns / (double)report.frames / 1000.0;
    if (report.launched == 0ull) {
        printf("mirror: %llu frames from the raster of the window, node %.1f us\n",
               report.frames, each);
        return;
    }
    printf("mirror: %llu frames, %llu of them from the thread at %u Hz, node %.1f us\n",
           report.frames, report.launched, report.hz, each);
}

void aotx_mirror_read(aotx_mirror_report *report)
{
    aotx_mirror_state state;
    memset(&state, 0, sizeof state);
    memset(report, 0, sizeof *report);
    if (cudaMemcpyFromSymbol(&state, aotx_mirror, sizeof state) != cudaSuccess) {
        return;
    }
    report->frames = state.frame;
    report->launched = aotx_mirror_thread_run.frames;
    report->node_ns = state.node_ns;
    report->idle = aotx_mirror_thread_run.idle;
    report->hz = aotx_mirror_thread_run.hz;
}
