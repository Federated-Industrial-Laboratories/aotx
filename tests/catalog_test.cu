#define _GNU_SOURCE
/* Purpose: Check the catalog: the import path, the manifest reader, the arena and remove.
 * Owns: The counts of the cases and the module texts each case builds.
 * Launch shape: The cases drive the tick graph; the kernels take one thread.
 * Lifetime: One run of the test program. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <cuda.h>

#include "agent/agent_state.cuh"
#include "boot/check.h"
#include "catalog/catalog.cuh"
#include "mem/mem.cuh"
#include "sched/sched.cuh"
#include "seam/seam.cuh"
#include "tool/tool_state.cuh"

#include "catalog_feed.h"
#include "catalog_cases.h"

int main(int argc, char **argv)
{
    const char *modules = (argc > 1) ? argv[1] : AOTX_MODULES_DIR;
    unsigned int applied = 0u;
    unsigned int failed = 0u;

    CUdevice device;
    CUcontext context;
    aotx_mem_map map;
    aotx_seam_rings rings;
    aotx_pump pump;
    aotx_check_driver(cuInit(0), "cuInit");
    aotx_check_driver(cuDeviceGet(&device, 0), "cuDeviceGet");
    aotx_check_driver(cuDevicePrimaryCtxRetain(&context, device),
                      "cuDevicePrimaryCtxRetain");
    aotx_check_driver(cuCtxSetCurrent(context), "cuCtxSetCurrent");
    unsigned long long boot_id = 0xCA7A106ull;
    if (aotx_mem_reserve(&map) != 0 || aotx_seam_open(&rings, boot_id) != 0) {
        printf("catalog: the map or the rings did not open\n");
        return 1;
    }
    aotx_seam_bind(&rings, map.ring, map.ring_bytes, boot_id);
    aotx_seam_bind_bulk(&rings, map.scratch, AOTX_BULK_STAGE_BYTES);
    aotx_seam_note_boot<<<1, 1>>>(0ull, 0ull);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    /* The built-in tools stand before the first tick. The check names the count, because
     * every later count of entries holds them. */
    applied += 1u;
    if (aotx_catalog_open() != 0) {
        printf("catalog: the built-in tools did not go in the catalog\n");
        return 1;
    }
    aotx_catalog_state *state = aotx_test_catalog_read();
    unsigned int built = 0u;
    unsigned int stat = 0u;
    for (unsigned int i = 0u; i < AOTX_MODULE_SLOTS; ++i) {
        built += (state->entry[i].state == AOTX_CATALOG_INSTALLED
                  && state->entry[i].tool.side == AOTX_CATALOG_SIDE_BUILT) ? 1u : 0u;
        stat += (state->entry[i].state == AOTX_CATALOG_INSTALLED
                 && state->entry[i].name_len == 7u
                 && memcmp(state->entry[i].name, "fs_stat", 7u) == 0
                 && state->entry[i].tool.built_in == AOTX_TOOL_FS_STAT) ? 1u : 0u;
    }
    free(state);
    if (built != AOTX_CATALOG_BUILT_IN) {
        printf("catalog: %u of %u built-in tools went in\n", built,
               (unsigned int)AOTX_CATALOG_BUILT_IN);
        failed += 1u;
    } else {
        printf("catalog: %u built-in tools stand before the first tick\n", built);
    }
    applied += 1u;
    if (stat != 1u) {
        printf("catalog: fs_stat has %u built-in entries\n", stat);
        failed += 1u;
    }

    /* The reader and the list run with no ring and no tick, because both are device
     * functions over the arena. The import path needs the tick graph. */
    aotx_catalog_test_reader(&applied, &failed);
    aotx_catalog_test_head(&applied, &failed);

    if (aotx_pump_build(&pump, 0ull, 1u) != 0) {
        printf("catalog: the tick graph did not build\n");
        return 1;
    }
    printf("catalog: %u nodes in the tick graph\n", pump.nodes);

    aotx_catalog_test_import(&pump, &rings, boot_id, 1u, &applied, &failed);
    aotx_catalog_test_import(&pump, &rings, boot_id, AOTX_MODULE_SLOTS - built,
                             &applied, &failed);
    aotx_catalog_test_kinds(&pump, &rings, boot_id, &applied, &failed);
    aotx_catalog_test_refusals(&pump, &rings, boot_id, &applied, &failed);
    aotx_catalog_test_replace(&pump, &rings, boot_id, &applied, &failed);
    aotx_catalog_test_remove(&pump, &rings, boot_id, &applied, &failed);
    aotx_catalog_test_roles(&pump, &rings, boot_id, modules, &applied, &failed);
    aotx_catalog_test_lists(&pump, &rings, boot_id, &applied, &failed);
    aotx_catalog_test_skill_use(&pump, &rings, boot_id, 1u, &applied, &failed);
    aotx_catalog_test_skill_use(&pump, &rings, boot_id, AOTX_SLOTS, &applied, &failed);
    aotx_catalog_test_prompt_tick(&pump, &rings, boot_id, modules, &applied, &failed);
    aotx_catalog_test_skill_file(&pump, &rings, boot_id, &applied, &failed);
    aotx_catalog_test_arriving(&pump, &rings, boot_id, &applied, &failed);
    aotx_catalog_test_same_batch(&pump, &rings, boot_id, modules, &applied, &failed);
    aotx_catalog_test_import_line(&pump, &rings, boot_id, &applied, &failed);
    aotx_catalog_test_reasons(&pump, &rings, boot_id, &applied, &failed);
    aotx_catalog_test_restore(&pump, &rings, boot_id, &applied, &failed);
    aotx_catalog_test_room(&pump, &rings, boot_id, &applied, &failed);

    aotx_pump_close(&pump);
    aotx_seam_close(&rings);
    aotx_mem_release(&map);
    cuDevicePrimaryCtxRelease(device);
    printf("catalog: %u cases applied, %u failed\n", applied, failed);
    return (failed == 0u) ? 0 : 1;
}
