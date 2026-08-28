/* Purpose: Start the system and run ticks until the run ends.
 * Owns: The context, the memory map, the rings, the pump and the disk side programs.
 * Launch shape: Host glue only; the graph holds the kernels.
 * Lifetime: The program. */
#include <cuda.h>
#include <cuda_runtime.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "boot/boot.cuh"
#include "boot/check.h"
#include "mem/mem.cuh"
#include "ui/ui.cuh"

/* The number of the signal that asks the run to stop. A handler may set a flag of this type
 * and do nothing else, so the flag is all that the handler sets. */
static volatile sig_atomic_t aotx_boot_signal_number;

static void aotx_boot_on_signal(int number)
{
    /* The first signal stops the run and the report names it; a later one changes nothing. */
    if (aotx_boot_signal_number == 0) {
        aotx_boot_signal_number = (sig_atomic_t)number;
    }
}

int aotx_boot_signal(void)
{
    return (int)aotx_boot_signal_number;
}

/* Take the stop signals. The run then ends the way the quit command ends it. The path is the
 * last flush, the closed rings, the wait for the disk side programs, and the reports. The
 * handler stays in place, so a second signal changes nothing and the run keeps its close.
 * A run that holds a drawing context must never end at the default action. The display
 * server keeps the window of a program that stops in the middle of a frame. */
static void aotx_boot_take_signals(void)
{
    struct sigaction action;
    memset(&action, 0, sizeof action);
    action.sa_handler = aotx_boot_on_signal;
    sigemptyset(&action.sa_mask);
    sigaction(SIGTERM, &action, NULL);
    sigaction(SIGINT, &action, NULL);
}

static unsigned long long aotx_boot_wall_ns(void)
{
    struct timespec at;
    clock_gettime(CLOCK_REALTIME, &at);
    return (unsigned long long)at.tv_sec * 1000000000ull + (unsigned long long)at.tv_nsec;
}

static long long aotx_boot_now_ns(void)
{
    struct timespec at;
    clock_gettime(CLOCK_MONOTONIC, &at);
    return (long long)at.tv_sec * 1000000000ll + (long long)at.tv_nsec;
}

/* The last flush moves every record that no block holds into the host ring. The flush runs
 * alone, so the tick count of the run does not change. */
static void aotx_boot_last_flush(aotx_pump *pump)
{
    aotx_pump_report report;
    for (unsigned int i = 0u; i < 256u; ++i) {
        aotx_pump_flush(pump);
        aotx_pump_read(&report);
        if (report.tail == report.flushed) {
            return;
        }
        usleep(2000);
    }
}

int main(int argc, char **argv)
{
    aotx_boot_options options;
    aotx_boot_children children;
    aotx_seam_rings rings;
    aotx_mem_map map;
    aotx_pump pump;
    aotx_pump_report report;
    unsigned long long sample = 0ull;

    int bad = aotx_boot_parse(argc, argv, &options);
    if (bad != 0) {
        return bad;
    }
    memset(&children, 0, sizeof children);

    /* The window and its drawing context come before the first driver call. The context
     * then binds to the device that drives the display. */
    if (options.window && aotx_ui_window_open() != 0) {
        return 1;
    }
    /* The handlers come after the window opens. A start that stops before this point holds
     * no rings and no programs, and one signal ends it through the default action. */
    aotx_boot_take_signals();

    CUdevice device;
    CUcontext context;
    aotx_check_driver(cuInit(0), "cuInit");
    aotx_check_driver(cuDeviceGet(&device, 0), "cuDeviceGet");
    /* The primary context is the one the display path shares. */
    aotx_check_driver(cuDevicePrimaryCtxRetain(&context, device), "cuDevicePrimaryCtxRetain");
    aotx_check_driver(cuCtxSetCurrent(context), "cuCtxSetCurrent");

    /* The clock module check is the first act of every start. */
    if (aotx_boot_clock_check(&sample) != 0) {
        fprintf(stderr, "the clock module did not give a sample\n");
        return 1;
    }
    printf("clock module: globaltimer %llu ns\n", sample);
    if (options.clock_only) {
        cuDevicePrimaryCtxRelease(device);
        return 0;
    }
    if (options.journal == NULL && options.solo == 0) {
        fprintf(stderr, "a journal directory is needed\n");
        aotx_boot_usage();
        return 2;
    }

    unsigned long long boot_id = aotx_boot_wall_ns() ^ ((unsigned long long)getpid() << 48);
    if (aotx_mem_reserve(&map) != 0) {
        fprintf(stderr, "the memory map did not open\n");
        return 1;
    }
    if (aotx_seam_open(&rings, boot_id) != 0) {
        fprintf(stderr, "the rings did not open\n");
        return 1;
    }
    aotx_seam_bind(&rings, map.ring, map.ring_bytes, boot_id);
    /* The staging area of the bulk channel is the first bytes of the scratch arena. */
    if (aotx_seam_bind_bulk(&rings, map.scratch, AOTX_BULK_STAGE_BYTES) != 0) {
        fprintf(stderr, "the bulk ring did not bind\n");
        return 1;
    }
    printf("boot: id %llx ring %llu MB scratch %llu MB host ring %llu MB\n",
           boot_id, map.ring_bytes >> 20, map.scratch_bytes >> 20,
           (unsigned long long)AOTX_HOST_RING_DATA_BYTES >> 20);

    /* The model files come in before the first tick, because the vocabulary and the
     * weights are state that every later step reads. */
    if (options.models != NULL) {
        int state = aotx_boot_models(options.models, options.roles, aotx_boot_signal);
        if (state != 0) {
            return state;
        }
    }

    aotx_seam_note_boot<<<1, 1>>>(0ull, aotx_boot_wall_ns());
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    if (aotx_pump_build(&pump, options.workload, options.blocks) != 0) {
        fprintf(stderr, "the tick graph did not build\n");
        return 1;
    }

    if (options.solo == 0
        && aotx_boot_start_drain(&children, &rings, options.journal, options.derive) != 0) {
        return 1;
    }
    if (options.restore
        && aotx_boot_replay(&children, &rings, options.journal, &pump) != 0) {
        fprintf(stderr, "the replay did not finish\n");
    }
    /* The window writes each key event as a 16-byte frame into the pipe. The feeder reads
     * the frames from the read end and makes a key record of each one. */
    int keys[2] = { -1, -1 };
    if (options.window && pipe(keys) != 0) {
        fprintf(stderr, "the key pipe did not open\n");
        return 1;
    }
    if (options.solo == 0 && aotx_boot_start_feed(&children, &rings, keys[0]) != 0) {
        return 1;
    }
    aotx_pump_set(&pump, options.workload, options.blocks);
    unsigned long long first_record = 0ull;
    aotx_pump_read(&report);
    first_record = report.records;
    long long started = aotx_boot_now_ns();
    unsigned long long made = 0ull;
    if (options.window) {
        aotx_boot_window_run(&pump, keys[1], options.derive);
    } else {
        for (unsigned long long tick = 0ull; options.ticks == 0ull || tick < options.ticks;
             ++tick) {
            aotx_pump_tick(&pump);
            if (options.ticks == 0ull) {
                aotx_pump_read(&report);
                made = report.records - first_record;
                if (options.workload != 0ull && made >= options.records) {
                    break;
                }
            }
            if (aotx_boot_quit() != 0u || aotx_boot_signal() != 0) {
                break;
            }
            aotx_pump_pace(&pump);
        }
    }
    long long spent = aotx_boot_now_ns() - started;

    aotx_boot_last_flush(&pump);
    aotx_seam_finish(&rings);
    aotx_boot_stop(&children);
    aotx_pump_read(&report);
    if (options.ticks == 0ull && options.workload != 0ull && spent > 0ll) {
        printf("rate: %llu records in %lld ms, %.0f records a second\n",
               made, spent / 1000000ll, (double)made * 1e9 / (double)spent);
    }
    if (aotx_boot_signal() != 0) {
        printf("boot: signal %d stops the run\n", aotx_boot_signal());
    }
    printf("ticks %llu records %llu blocks %llu held %llu applied %llu hash %llx\n",
           report.tick, report.records, report.blocks, report.held,
           report.applied, report.state_hash);

    aotx_pump_close(&pump);
    aotx_seam_close(&rings);
    aotx_mem_release(&map);
    cuDevicePrimaryCtxRelease(device);
    return 0;
}
