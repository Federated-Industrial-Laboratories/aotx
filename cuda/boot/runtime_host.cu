/* Purpose: Bind a validated runtime file to boot options and checkpoint transport.
 * Owns: The disk metadata activation handle until process exit.
 * Launch shape: Host glue only; file validation is C and state capture is CUDA.
 * Lifetime: One optional CCIR activation. */
#include "boot/boot.cuh"
#include "boot/check.h"
#include "cognitive/checkpoint.cuh"
#include "disk/runtime/activate.h"
#include "shared/host.h"
#include <cuda_runtime.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static aotx_runtime_boot aotx_runtime_boot_data;
static void aotx_runtime_boot_close(void) { aotx_runtime_release(&aotx_runtime_boot_data); }
unsigned int aotx_boot_runtime_features(void) { return aotx_runtime_boot_data.features; }
int aotx_boot_runtime_shared_open(void) {
    if (!(aotx_runtime_boot_data.features & AOTX_RUNTIME_SHARED)) return 0;
    if (!aotx_shared_open(&aotx_runtime_boot_data.shared)) return 0;
    fprintf(stderr, "shared runtime: the required device tables did not open\n");
    return 1;
}
int aotx_boot_runtime_open(aotx_boot_options *options) {
    if (!options->ccir) return 0;
    if (!options->journal || options->models || options->roles || options->modules ||
        options->settings || options->policy || options->memory_mirror || options->restore || options->solo || options->workload) {
        fprintf(stderr, "a runtime file requires a journal and its own component settings\n");
        return 2;
    }
    int rc = aotx_runtime_prepare(options->ccir, options->journal, AOTX_ARCH, &aotx_runtime_boot_data);
    if (rc) {
        fprintf(stderr, "runtime file refused: %s (%d)\n", aotx_ccir_status_text(rc), rc);
        return 2;
    }
    if (atexit(aotx_runtime_boot_close)) { aotx_runtime_boot_close(); return 1; }
    options->models = options->ccir; options->roles = aotx_runtime_boot_data.roles;
    options->modules = aotx_runtime_boot_data.modules; options->settings = aotx_runtime_boot_data.settings;
    options->memory_mirror = options->ccir;
    options->policy = aotx_runtime_boot_data.policy[0] ? aotx_runtime_boot_data.policy : NULL;
    options->restore = aotx_runtime_boot_data.mode == 2;
    options->runtime_seed = options->restore ? NULL : options->ccir;
    return 0;
}
void aotx_boot_runtime_bind(const aotx_boot_options *options, aotx_seam_rings *rings) {
    unsigned int enabled = options->ccir != NULL;
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_runtime_enabled, &enabled, sizeof(enabled)), "cudaMemcpyToSymbol");
    if (enabled) {
        aotx_checkpoint_ring *ring = (aotx_checkpoint_ring *)rings->checkpoint_map;
        ring->reserved[0] = 1;
        memcpy(ring->pad_head, aotx_runtime_boot_data.revision, 32);
    }
}

int aotx_boot_runtime_ready(const aotx_boot_options *options, const aotx_seam_rings *rings,
    aotx_pump *pump, int (*stopped)(void), aotx_boot_children *children) {
    if (!options->ccir) return 0;
    const aotx_checkpoint_ring *ring = (const aotx_checkpoint_ring *)rings->checkpoint_map;
    while (children && children->drain > 0 && (!stopped || !stopped())) {
        int ended = 0, status = 0;
        if (aotx_seam_poll(children->drain, &ended, &status)) {
            if (errno == EINTR) continue;
            break;
        }
        if (ended) {
            children->drain = 0;
            fprintf(stderr, "runtime: the disk writer ended before input was ready (code %d)\n", status);
            break;
        }
        if (__atomic_load_n(&ring->error, __ATOMIC_ACQUIRE)) break;
        uint64_t serial = __atomic_load_n(&ring->ack_serial, __ATOMIC_ACQUIRE);
        uint64_t consumed = __atomic_load_n(&ring->consumed, __ATOMIC_ACQUIRE);
        uint64_t ack_boot = __atomic_load_n(&ring->ack_boot, __ATOMIC_ACQUIRE);
        uint64_t generation = __atomic_load_n(&ring->generation, __ATOMIC_ACQUIRE);
        uint64_t incarnation = __atomic_load_n(ring->incarnation, __ATOMIC_ACQUIRE) |
            __atomic_load_n(ring->incarnation + 1, __ATOMIC_ACQUIRE);
        uint64_t digest = __atomic_load_n(ring->commit_digest, __ATOMIC_ACQUIRE) |
            __atomic_load_n(ring->commit_digest + 1, __ATOMIC_ACQUIRE) |
            __atomic_load_n(ring->commit_digest + 2, __ATOMIC_ACQUIRE) |
            __atomic_load_n(ring->commit_digest + 3, __ATOMIC_ACQUIRE);
        uint64_t after = __atomic_load_n(&ring->ack_serial, __ATOMIC_ACQUIRE);
        if (serial == after && !(serial & 1) && consumed && serial / 2 == consumed &&
            consumed <= __atomic_load_n(&ring->head, __ATOMIC_ACQUIRE) &&
            ring->layout == AOTX_CP_LAYOUT && ack_boot == ring->boot && generation && incarnation && digest) {
            printf("runtime: recovered state is durable; input is ready\n");
            return 0;
        }
        aotx_pump_tick(pump);
        aotx_pump_pace(pump);
    }
    fprintf(stderr, "runtime: the recovered state did not reach a complete durable checkpoint\n");
    return 1;
}
