/* Purpose: Run the tick pump on its own thread while the window draws on this thread.
 * Owns: The pump thread, its stop flag and the read of the flag the quit command sets.
 * Launch shape: Host glue only; the tick graph and the raster graph hold the kernels.
 * Lifetime: From the bind of the window to the close of the window. */
#include <cuda.h>
#include <cuda_runtime.h>
#include <pthread.h>
#include <stdio.h>

#include "boot/boot.cuh"
#include "boot/check.h"
#include "ui/ui.cuh"

/* What the pump thread and the window thread share. The stop flag ends the run. */
typedef struct aotx_boot_run {
    aotx_pump *pump;
    CUcontext context;
    volatile int stop;
    unsigned long long ticks;
} aotx_boot_run;

unsigned int aotx_boot_quit(void)
{
    unsigned int quit = 0u;
    aotx_check_runtime(cudaMemcpyFromSymbol(&quit, aotx_cli_quit, sizeof quit),
                       "cudaMemcpyFromSymbol");
    return quit;
}

/* The pump thread makes ticks at the tick period and stops at the flag or at the window. */
static void *aotx_boot_pump(void *state)
{
    aotx_boot_run *run = (aotx_boot_run *)state;
    aotx_check_driver(cuCtxSetCurrent(run->context), "cuCtxSetCurrent");
    while (run->stop == 0) {
        aotx_pump_tick(run->pump);
        run->ticks += 1ull;
        if (aotx_boot_quit() != 0u) {
            run->stop = 1;
            break;
        }
        aotx_pump_pace(run->pump);
    }
    return NULL;
}

int aotx_boot_window_run(aotx_pump *pump, int keys_fd)
{
    aotx_boot_run run;
    pthread_t thread;

    run.pump = pump;
    run.context = 0;
    run.stop = 0;
    run.ticks = 0ull;
    aotx_check_driver(cuCtxGetCurrent(&run.context), "cuCtxGetCurrent");
    if (aotx_ui_window_bind(keys_fd) != 0) {
        aotx_ui_window_close();
        return 1;
    }
    if (pthread_create(&thread, NULL, aotx_boot_pump, &run) != 0) {
        fprintf(stderr, "the pump thread did not start\n");
        aotx_ui_window_close();
        return 1;
    }
    while (run.stop == 0 && aotx_ui_window_frame() != 0) {
        /* The window draws at the rate of the display while the pump makes the ticks. */
    }
    run.stop = 1;
    pthread_join(thread, NULL);

    /* The report states what the display gave while the ticks ran. The frame rate under a
     * tick load is therefore a measured figure and not a claim. */
    aotx_ui_frame_report report;
    aotx_ui_window_report(&report);
    printf("window: %llu ticks, %llu frames, raster priority %d\n", run.ticks,
           report.frames, aotx_ui_window_priority());
    printf("window: frame mean %.2f ms, worst %.2f ms, %llu over %.1f ms, %llu keys dropped\n",
           (double)report.mean_ns / 1e6, (double)report.worst_ns / 1e6, report.late,
           (double)AOTX_UI_LATE_NS / 1e6, report.dropped);
    aotx_ui_window_close();
    return 0;
}
