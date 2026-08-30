/* Purpose: Start the system and run ticks until the run ends.
 * Owns: The context, the memory map, the rings, the pump and the disk side programs.
 * Launch shape: Host glue only; the graph holds the kernels.
 * Lifetime: The program. */
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "boot/boot.cuh"
#include "boot/check.h"
#include "disk/settings/settings.h"
#include "mem/mem.cuh"
#include "settings/settings.cuh"
#include "tool/module.cuh"
#include "ui/mirror.cuh"
#include "ui/ui.cuh"

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
    if (options.version) {
        aotx_boot_version();
        return 0;
    }
    memset(&children, 0, sizeof children);

    /* The settings file comes before the surfaces, because it names them. A command line
     * option wins over the file for the same key. */
    char settings_path[512];
    aotx_settings *file = (aotx_settings *)calloc(1, sizeof *file);
    if (file == NULL) {
        return 1;
    }
    settings_path[0] = '\0';
    if (aotx_boot_settings(&options, file, settings_path,
                           (unsigned int)sizeof settings_path) != 0) {
        fprintf(stderr, "the settings file %s cannot be read\n", settings_path);
        return 2;
    }

    /* The window and its drawing context come before the first driver call. The context
     * then binds to the device that drives the display. */
    if (options.window && aotx_ui_window_open() != 0) {
        return 1;
    }
    /* The handlers come after the window opens. A start that stops before this point holds
     * no rings and no programs, and one signal ends it through the default action. */
    aotx_boot_signals_open();

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
    /* The card is read before any placement. A profile the free memory cannot hold stops
     * the run with the figures. */
    int held = aotx_boot_card_check();
    if (held != 0) {
        cuDevicePrimaryCtxRelease(device);
        return held;
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
    if (aotx_mirror_bind(&rings) != 0) {
        fprintf(stderr, "the mirror did not bind\n");
        return 1;
    }
    if (aotx_seam_bind_bulk(&rings, map.scratch, AOTX_BULK_STAGE_BYTES) != 0) {
        fprintf(stderr, "the bulk ring did not bind\n");
        return 1;
    }
    printf("boot: id %llx ring %llu MB scratch %llu MB host ring %llu MB\n",
           boot_id, map.ring_bytes >> 20, map.scratch_bytes >> 20,
           (unsigned long long)AOTX_HOST_RING_DATA_BYTES >> 20);

    if (aotx_boot_phase_open(options.journal) != 0) {
        return 1;
    }
    if (options.models != NULL) {
        int state = aotx_boot_models(options.models, options.roles, aotx_boot_signal);
        if (state != 0) {
            return state;
        }
    }

    aotx_seam_note_boot<<<1, 1>>>(0ull, aotx_boot_wall_ns());
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_boot_card_note();

    if (aotx_settings_page_open() != 0) {
        fprintf(stderr, "the control page did not open\n");
        return 1;
    }
    aotx_tool_module_root(options.modules);
    aotx_tool_module_journal(options.journal);
    if (aotx_pump_build(&pump, options.workload, options.blocks) != 0) {
        fprintf(stderr, "the tick graph did not build\n");
        return 1;
    }

    if (options.solo == 0
        && aotx_boot_start_drain(&children, &rings, options.journal, options.derive) != 0) {
        return 1;
    }
    if (options.restore
        && aotx_boot_phase_set("replaying") != 0) {
        return 1;
    }
    if (options.restore
        && aotx_boot_replay(&children, &rings, options.journal, &pump) != 0) {
        fprintf(stderr, "the replay did not finish\n");
        return 1;
    }
    /* The window writes each key event as a 16-byte frame into the pipe. The feeder reads
     * the frames from the read end and makes a key record of each one. */
    int keys[2] = { -1, -1 };
    if (options.window && pipe(keys) != 0) {
        fprintf(stderr, "the key pipe did not open\n");
        return 1;
    }
    /* A restore replays the setting records of the journal, so the run it restores keeps
     * its settings and the feeder reads no file. A fresh boot gives the feeder the file. */
    if (options.solo == 0
        && aotx_boot_start_feed(&children, &rings, keys[0], options.root, options.journal,
                                options.restore ? NULL : settings_path,
                                options.restore ? NULL : options.modules) != 0) {
        return 1;
    }
    /* Without a window the raster graph has no thread, so the mirror runs it on one of
     * its own. The thread launches nothing while no terminal reads. */
    if (options.window == 0 && aotx_mirror_start(&rings) != 0) {
        fprintf(stderr, "the mirror thread did not start\n");
        return 1;
    }
    /* The terminal program starts after the feeder, because it attaches to the socket the
     * feeder makes. A run that a terminal started already opens no second one. */
    if (options.tui && options.tui_attached == 0
        && aotx_boot_start_tui(&children, options.journal) != 0) {
        return 1;
    }
    if (aotx_boot_phase_set("running") != 0) {
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

    aotx_mirror_stop();
    aotx_boot_last_flush(&pump);
    aotx_seam_finish(&rings);
    aotx_boot_stop(&children);
    aotx_boot_phase_close();
    aotx_pump_read(&report);
    if (options.ticks == 0ull && options.workload != 0ull && spent > 0ll) {
        printf("rate: %llu records in %lld ms, %.0f records a second\n",
               made, spent / 1000000ll, (double)made * 1e9 / (double)spent);
    }
    if (aotx_boot_signal() != 0) {
        printf("boot: signal %d stops the run\n", aotx_boot_signal());
    }
    printf("ticks %llu records %llu blocks %llu held %llu applied %llu hash %llx "
           "model MB %llu\n",
           report.tick, report.records, report.blocks, report.held,
           report.applied, report.state_hash, report.model_bytes >> 20);
    aotx_mirror_report_line();
    aotx_pump_close(&pump);
    aotx_settings_page_close();
    free(file);
    aotx_seam_close(&rings);
    aotx_mem_release(&map);
    cuDevicePrimaryCtxRelease(device);
    return 0;
}
